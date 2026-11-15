"""Explicit queue/config boundaries, synthetic raw input, no model calls."""
import json

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.data.ingest import import_topic
from app.errors import ServiceError
from app.main import create_app
from app.repository import Repository


def bundle_settings(tmp_path, **overrides):
    return Settings(_env_file=None, data_dir=tmp_path, model_name="", model_api_key="",
        **({"model_source_bundle_enabled": True, "strict_evidence_max_requests": 10,
            "strict_evidence_token_stop": 30000, **overrides}))


def loaded_repo(tmp_path):
    settings = bundle_settings(tmp_path)
    repo = Repository(settings)
    imported = import_topic(settings.database_path,
        [{"doc_id": "synthetic", "title": "合成原文", "content": "原始记录。",
          "paragraphs": [{"para_id": "opaque", "char_start": 0, "char_end": 5}]}],
        [{"question_id": "one", "question": "记录是什么？"}, {"question_id": "two", "question": "有明确限定吗？"}], "合成配置")
    return repo, imported["topic_id"]


def test_bundle_strategy_is_opt_in_and_frozen_in_run_config(tmp_path):
    default = Repository(Settings(_env_file=None, data_dir=tmp_path / "default", model_name="", model_api_key=""))
    assert "source_bundle_enabled" not in default.config()
    repo, topic_id = loaded_repo(tmp_path / "bundle")
    cfg = repo.config()
    assert cfg["source_bundle_protocol"] == "document-source-bundle-v1"
    assert cfg["source_bundle_enabled"] is True and cfg["strict_evidence_enabled"] is False
    assert cfg["strict_scope_enabled"] is False
    assert cfg["source_strategy"] == "dsh_source_bundles"
    assert cfg["prompt_version"] == "trusted-analysis-dsh-source-bundles-v1"
    assert cfg["source_bundle_config"]["front_token_stop"] == 12000
    assert cfg["source_bundle_config"]["tail_reserved_tokens"] == 18000
    assert cfg["source_bundle_config"]["navigation_max_output_tokens"] == 1024
    assert cfg["source_bundle_config"]["submit_max_output_tokens"] == 4096
    manifest = cfg["source_bundle_code_manifest"]
    assert "app/workflows/bundle_workbench.py" in manifest
    assert "integrations/dsh/bundle_workbench_launcher.mjs" in manifest
    assert "scripts/source_bundle_tool.py" in manifest
    assert not any("fixture" in name or name.endswith(".test.mjs") for name in manifest)
    assert all(len(value) == 64 for value in manifest.values())
    question = repo.questions(topic_id)[0]
    task = repo.create_run(topic_id, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "queued-only", model_available=True)
    run = repo.run(task["run_id"])
    assert run["source_bundle_enabled"] is True and run["status"] == "queued"
    assert run["usage"] is None
    assert json.loads(repo.one("SELECT model_config_json FROM runs WHERE id=?", (task["run_id"],))["model_config_json"])["source_bundle_config"] == cfg["source_bundle_config"]


@pytest.mark.parametrize("field", ["model_strict_evidence_enabled", "model_paragraph_ranked_enabled", "model_premise_bank_enabled"])
def test_bundle_cannot_mix_with_another_evidence_strategy(tmp_path, field):
    with pytest.raises(ValueError, match="mutually exclusive"):
        bundle_settings(tmp_path, **{field: True})


@pytest.mark.parametrize("overrides,message", [
    ({"strict_evidence_max_requests": 8}, "reserve both independent"),
    ({"strict_evidence_token_stop": 20000}, "front and tail reservations"),
    ({"source_bundle_submit_max_output_tokens": 8192}, "phase output capacity"),
])
def test_impossible_whole_question_config_is_rejected_before_queue_or_model(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        bundle_settings(tmp_path, **overrides)


@pytest.mark.parametrize("include_baseline", [False, True])
def test_bundle_queue_refuses_extra_questions_or_extra_baseline(tmp_path, include_baseline):
    repo, topic_id = loaded_repo(tmp_path)
    ids = [q["id"] for q in repo.questions(topic_id)]
    with pytest.raises(ServiceError) as error:
        repo.create_run(topic_id, {"question_ids": ids[:1] if include_baseline else ids, "base_revision_id": None,
            "include_baseline": include_baseline, "mode": "initial"}, "must-not-queue", model_available=True)
    assert error.value.code == "strict_question_limit"
    assert repo.all("SELECT id FROM runs") == []


def test_health_reports_actual_framework_mode_without_credentials(tmp_path):
    with TestClient(create_app(bundle_settings(tmp_path))) as client:
        health = client.get("/api/health").json()
        assert health["source_bundle_enabled"] is True
        assert health["provider"] == "dsh-selection-langgraph-analysis"
        assert health["model_configured"] is False


def test_compact_handles_are_explicit_and_bound_to_the_queued_configuration(tmp_path):
    repo, topic_id = loaded_repo(tmp_path)
    assert "compact_handles" not in repo.config()["source_bundle_config"]
    compact = Repository(repo.settings.model_copy(update={"source_bundle_compact_handles": True}))
    assert compact.config()["source_bundle_config"]["compact_handles"] is True
    q = compact.questions(topic_id)[0]
    queued = compact.create_run(topic_id, {"question_ids": [q["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "compact-config-only", model_available=True)
    stored = json.loads(compact.one("SELECT model_config_json FROM runs WHERE id=?", (queued["run_id"],))["model_config_json"])
    assert stored["source_bundle_config"]["compact_handles"] is True
    assert compact.run(queued["run_id"])["usage"] is None


def test_compact_handles_cannot_be_enabled_for_a_different_workflow():
    with pytest.raises(ValueError, match="require the explicit source bundle"):
        Settings(_env_file=None, source_bundle_compact_handles=True)


def test_langchain_source_framework_is_explicit_and_reserves_one_front_and_three_tail_calls(tmp_path):
    settings = bundle_settings(tmp_path, source_bundle_front_framework="langchain", strict_evidence_max_requests=4)
    assert settings.model_prompt_version == "trusted-analysis-langchain-source-bundles-v1"
    assert settings.source_bundle_config["front_framework"] == "langchain"
    assert "compact_handles" not in settings.source_bundle_config
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
        assert health["source_bundle_front_framework"] == "langchain"
        assert health["provider"] == "langchain-selection-langgraph-analysis"
    with pytest.raises(ValueError, match="reserve both independent"):
        bundle_settings(tmp_path / "too-few", source_bundle_front_framework="langchain", strict_evidence_max_requests=3)


def test_langchain_source_framework_rejects_incompatible_workflows_and_dsh_handles(tmp_path):
    with pytest.raises(ValueError, match="requires the explicit source bundle"):
        Settings(_env_file=None, source_bundle_front_framework="langchain")
    with pytest.raises(ValueError, match="DSH short handles"):
        bundle_settings(tmp_path, source_bundle_front_framework="langchain", source_bundle_compact_handles=True)
