import json
import shutil
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentserve.application.bootstrap import open_task_service
from agentserve.config import MetricSettings, Settings
from agentserve.domain.models import TaskInput
from agentserve.knowledge.metrics import MetricCatalog

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "data/knowledge_v32/registry.json"
DATABASE = "business_orders_v1"
BODY = {
    "database_id": DATABASE,
    "mode": "metric",
    "metric": {"name": " ＣＯＭＰＬＥＴＥＤ　 ０rders ", "start": "2026-01-01", "end": "2026-02-01"},
}
pytestmark = pytest.mark.asyncio


async def test_metric_parameters_and_empty_result():
    settings = Settings.load(ROOT / "configs/app.v221.C1.json")
    settings.memory = None
    settings.metric = MetricSettings(registry_path=RULES)
    city = "Beijing' OR 1=1 --"
    async with open_task_service(settings) as service:
        completed = await service.run(
            TaskInput.model_validate({**BODY, "metric": {**BODY["metric"], "name": "ＣＯＭＰＬＥＴＥＤ　 Orders"}})
        )
        assert completed.status == "COMPLETED"
        assert completed.query_result.rows[0][0] > 0
        assert completed.metric["rule"]["unit"] == "笔"
        empty = await service.run(
            TaskInput.model_validate(
                {**BODY, "metric": {**BODY["metric"], "name": "成交营收", "city": city}}
            )
        )
        assert empty.status == "COMPLETED"
        assert empty.query_result.rows == [[None]]
        assert empty.metric["citation"]["anchor"] == "L3-L3"
        assert empty.model_calls == 0 and city not in empty.sql
        assert [event["type"] for event in empty.events].count("SQL_STARTED") == 1


async def test_metric_rejects_invalid_requests():
    for body in (
        {**BODY, "session_id": "session"},
        {**BODY, "question": "write to orders"},
        {**BODY, "metric": {**BODY["metric"], "start": "2026-02-30"}},
        {**BODY, "metric": {**BODY["metric"], "end": "2026-01-01"}},
        {**BODY, "metric": {**BODY["metric"], "group_by": "status"}},
        {"database_id": DATABASE},
    ):
        with pytest.raises(ValidationError):
            TaskInput.model_validate(body)


async def test_approved_rule_rejects_document_only_change(tmp_path):
    registry = shutil.copytree(RULES.parent, tmp_path / "rules") / RULES.name
    document = registry.with_name("settled-revenue.md")
    document.write_bytes(document.read_bytes().replace(b"completed", b"cancelled"))
    manifest_path = registry.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    manifest["documents"][0]["source_sha256"] = sha256(document.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source binding changed"):
        MetricCatalog(registry, purpose="product", database_ids={DATABASE})
