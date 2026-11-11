import httpx
from conftest import completion

from agentserve.agents.sql import SQLAgent
from agentserve.application.service import TaskService
from agentserve.domain.models import TaskInput
from agentserve.llm.client import VLLMClient
from agentserve.tools.sqlite import SQLTool


async def test_v21_repairs_denied_function_with_existing_budget(settings):
    settings.sql_prompt_version = "evidence_v21"
    settings.sql_function_profile = "with_replace_v1"
    replies = [
        completion(sql="SELECT trim(name,'-') FROM customers"),
        completion(sql="SELECT replace(name,'-','') FROM customers"),
        completion(answer="已返回名称。"),
    ]
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=replies[len(calls) - 1])

    tool = SQLTool(
        {"demo_orders": settings.databases["demo_orders"].path}, function_profile="with_replace_v1"
    )
    try:
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            report = await TaskService(
                SQLAgent(VLLMClient(client, settings.model), tool, settings)
            ).run(TaskInput(database_id="demo_orders", question="删除姓名中的所有连字符"))
        assert report.status == "COMPLETED" and report.sql_repairs == 1 and report.model_calls == 3
        assert report.query_result.rows == [["Alice"], ["Bob"]]
    finally:
        tool.close()


async def test_v21_does_not_retry_denied_write(settings):
    settings.sql_prompt_version = "evidence_v21"
    tool = SQLTool({"demo_orders": settings.databases["demo_orders"].path})
    try:
        async with httpx.AsyncClient(
            base_url="http://test/v1/",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=completion(sql="DELETE FROM customers"))
            ),
        ) as client:
            report = await TaskService(
                SQLAgent(VLLMClient(client, settings.model), tool, settings)
            ).run(TaskInput(database_id="demo_orders", question="删除客户"))
        assert report.status == "FAILED" and report.error["code"] == "POLICY_REJECTED"
        assert report.model_calls == 1 and report.sql_repairs == 0
    finally:
        tool.close()
