import asyncio
import json

import httpx
import pytest
from conftest import completion

from agentserve.agents.sql import SQLAgent
from agentserve.application.service import TaskService
from agentserve.domain.models import AgentError, TaskInput
from agentserve.llm.client import VLLMClient
from agentserve.tools.sqlite import SQLTool

TASK = TaskInput(database_id="demo_orders", question="一共有多少位客户？")


async def run_script(settings, replies, *, task=TASK):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        reply = replies[len(requests) - 1]
        return httpx.Response(200, json=reply)

    tool = SQLTool({key: db.path for key, db in settings.databases.items()})
    try:
        async with httpx.AsyncClient(
            base_url="http://test/v1/", transport=httpx.MockTransport(handle)
        ) as client:
            report = await TaskService(
                SQLAgent(VLLMClient(client, settings.model), tool, settings)
            ).run(task)
        return report, requests
    finally:
        tool.close()


async def test_success_through_real_sql_and_protocol(settings):
    report, requests = await run_script(settings, [completion(), completion(answer="有2位客户。")])
    assert report.status == "COMPLETED" and report.query_result.rows == [[2]]
    assert report.evaluation_status == "not_evaluated"
    assert report.model_calls == 2 and report.sql_repairs == 0
    assert requests[1]["tool_choice"] == "none"
    feedback = requests[1]["messages"][-1]
    assert feedback["role"] == "tool" and json.loads(feedback["content"])["rows"] == [[2]]
    assert [e["type"] for e in report.events] == [
        "TASK_STARTED",
        "MODEL_QUEUED",
        "MODEL_STARTED",
        "MODEL_COMPLETED",
        "SQL_STARTED",
        "SQL_COMPLETED",
        "MODEL_QUEUED",
        "MODEL_STARTED",
        "MODEL_COMPLETED",
        "TASK_COMPLETED",
    ]


async def test_sql_error_repair_is_bounded_and_feedback_is_visible(settings):
    report, requests = await run_script(
        settings,
        [
            completion("SELECT missing FROM customers"),
            completion(),
            completion(answer="有2位客户。"),
        ],
    )
    assert report.status == "COMPLETED" and report.sql_repairs == 1
    assert "no such column" in requests[1]["messages"][-1]["content"]
    assert report.query_result.rows == [[2]]


async def test_repair_budget_exhausts_after_three_attempts(settings):
    report, requests = await run_script(settings, [completion("SELECT missing FROM customers")] * 3)
    assert report.error["code"] == "SQL_REPAIR_EXHAUSTED"
    assert report.model_calls == 3 and report.sql_repairs == 2 and len(requests) == 3


async def test_policy_failure_does_not_trigger_repair(settings):
    report, requests = await run_script(settings, [completion("DELETE FROM customers")])
    assert report.error["code"] == "POLICY_REJECTED"
    assert report.sql_repairs == 0 and len(requests) == 1


@pytest.mark.parametrize(
    "bad",
    [
        completion(calls=0),
        completion(calls=2),
        completion(arguments='{"sql":"SELECT 1","path":"/tmp/arbitrary"}'),
        completion(arguments="not JSON"),
        {"id": "bad", "choices": []},
    ],
)
async def test_invalid_model_actions_get_one_format_repair(settings, bad):
    report, _ = await run_script(settings, [bad, completion(), completion(answer="有2位客户。")])
    assert report.status == "COMPLETED" and report.format_repairs == 1


async def test_format_budget_exhaustion(settings):
    report, requests = await run_script(settings, [completion(calls=2)] * 2)
    assert report.error["code"] == "MODEL_PROTOCOL_ERROR" and len(requests) == 2


async def test_output_limit_is_explicit(settings):
    report, _ = await run_script(settings, [completion(finish_reason="length")])
    assert report.error["code"] == "MODEL_OUTPUT_LIMIT"


async def test_summary_failure_preserves_sql_result(settings):
    report, _ = await run_script(settings, [completion(), completion(answer="")])
    assert report.status == "FAILED" and report.error["stage"] == "summary"
    assert report.query_result.rows == [[2]] and report.sql is not None


async def test_global_call_budget_includes_format_and_sql_repairs(settings):
    settings.max_model_calls = 2
    report, requests = await run_script(settings, [completion(calls=2), completion()])
    assert report.error["code"] == "MODEL_BUDGET_EXHAUSTED" and len(requests) == 2
    assert report.query_result.rows == [[2]]


async def test_unknown_database_never_calls_model(settings):
    with pytest.raises(AgentError, match="not configured"):
        await run_script(settings, [], task=TaskInput(database_id="../../arbitrary", question="x"))


async def test_transport_failure_has_no_implicit_retry(settings):
    class OfflineBackend:
        calls = 0

        async def generate(self, *args, **kwargs):
            self.calls += 1
            raise AgentError("MODEL_TRANSPORT_ERROR", "unavailable")

    backend = OfflineBackend()
    tool = SQLTool({key: db.path for key, db in settings.databases.items()})
    try:
        report = await TaskService(SQLAgent(backend, tool, settings)).run(TASK)
        assert report.error["code"] == "MODEL_TRANSPORT_ERROR" and backend.calls == 1
    finally:
        tool.close()


async def test_total_deadline_and_external_cancellation(settings):
    class SlowBackend:
        async def generate(self, *args, **kwargs):
            await asyncio.sleep(10)

    settings.task_timeout_seconds = 0.02
    tool = SQLTool({key: db.path for key, db in settings.databases.items()})
    try:
        service = TaskService(SQLAgent(SlowBackend(), tool, settings))
        report = await service.run(TASK)
        assert report.error["code"] == "TASK_TIMEOUT"
        pending = asyncio.create_task(service.run(TASK))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    finally:
        tool.close()


async def test_independent_tasks_do_not_share_conversation(settings):
    report1, report2 = await asyncio.gather(
        run_script(settings, [completion(), completion(answer="2")]),
        run_script(
            settings,
            [
                completion("SELECT name FROM customers ORDER BY customer_id"),
                completion(answer="Alice, Bob"),
            ],
        ),
    )
    assert report1[0].task_id != report2[0].task_id
    assert report1[0].query_result.rows == [[2]]
    assert report2[0].query_result.rows == [["Alice"], ["Bob"]]
