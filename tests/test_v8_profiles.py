import json

import pytest
from pydantic import ValidationError

from agentserve.api.web import web_settings
from agentserve.application.execution import execution_snapshot
from agentserve.config import MemorySettings, Settings


@pytest.mark.parametrize(("override", "profile", "model", "sam"), [
    (None, "suffix", "agent-qwen-suffix", False),
    ("suffix-agspec", "suffix-agspec", "agent-qwen-suffix-agspec", False),
    ("session-sam", "session-sam", "agent-qwen-sam", True),
])
def test_v8_profile_pairs_the_app_model_and_drafting_policy(
    settings, tmp_path, monkeypatch, override, profile, model, sam,
):
    settings.memory = MemorySettings()
    config = tmp_path / "app.v8.json"
    config.write_text(json.dumps({**settings.model_dump(mode="json"), "serving_profile": "suffix"}))
    monkeypatch.setenv("AGENTSERVE_CONFIG", str(config))
    monkeypatch.setenv("AGENTSERVE_WEB_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENTSERVE_WEB_SAM_ALL_PHASES", "1")
    if override is None:
        monkeypatch.delenv("AGENTSERVE_SERVING_PROFILE", raising=False)
    else:
        monkeypatch.setenv("AGENTSERVE_SERVING_PROFILE", override)
    for resolved in (Settings.load(config), web_settings("demo")):
        assert (resolved.serving_profile, resolved.model) == (profile, model)
        assert resolved.session_sam is resolved.session_sam_all_phases is sam
        snapshot = execution_snapshot(resolved)
        assert (snapshot["serving_profile"], snapshot["model"]) == (profile, model)
    assert web_settings("demo").state_database == tmp_path / "v6-demo.sqlite"


def test_v8_override_preserves_legacy_loading_and_session_identity(settings, tmp_path, monkeypatch):
    settings.memory = MemorySettings()
    settings.model = "agent-qwen-sam"
    settings.session_sam = True
    settings.session_sam_all_phases = True
    legacy = tmp_path / "app.v7.json"
    legacy.write_text(settings.model_dump_json())
    v8 = tmp_path / "app.v8.json"
    v8.write_text(json.dumps({**settings.model_dump(mode="json"), "serving_profile": "suffix"}))
    monkeypatch.setenv("AGENTSERVE_SERVING_PROFILE", "suffix-agspec")
    before = execution_snapshot(Settings.load(legacy))
    after = execution_snapshot(Settings.load(v8))
    assert "serving_profile" not in before
    assert before["model"] == "agent-qwen-sam"
    assert after["model"] == "agent-qwen-suffix-agspec"
    key = "memory_policy_sha256"
    assert before["execution_identity"][key] == after["execution_identity"][key]
    monkeypatch.setenv("AGENTSERVE_SERVING_PROFILE", "unknown")
    with pytest.raises(ValidationError, match="serving_profile"):
        Settings.load(v8)
