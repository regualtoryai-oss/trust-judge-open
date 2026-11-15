"""Strategy opt-in is explicit, persisted and never a silent fallback."""
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.agents.provider import LangChainOpenAIProvider, ModelOutputInvalid
from app.config import Settings
from app.errors import ServiceError
from app.main import create_app
from app.utils import digest
from app.worker import Worker
from app.workflows.analysis import run_analysis
from test_deepseek_provider import intercepted_adapter
from test_premise_bank import Q, paragraphs, source
from test_premise_bank_repository import setup, MeteredProductionDouble
from test_premise_bank_workflow import V18Provider
from test_repository import create


V17 = "trusted-analysis-json-v17-paragraph-recall-recovery"
V18 = "trusted-analysis-json-v18-quote-bank-purpose"


def test_default_is_v17_and_environment_setting_is_explicit(monkeypatch):
    monkeypatch.delenv("TRUST_MODEL_PREMISE_BANK_ENABLED", raising=False)
    defaults = Settings(_env_file=None, model_name="", model_api_key="")
    assert defaults.model_premise_bank_enabled is False and defaults.model_prompt_version == V17
    assert LangChainOpenAIProvider.premise_bank_enabled is False
    assert LangChainOpenAIProvider.prompt_version == V17
    monkeypatch.setenv("TRUST_MODEL_PREMISE_BANK_ENABLED", "true")
    enabled = Settings(_env_file=None, model_name="", model_api_key="")
    assert enabled.model_premise_bank_enabled is True and enabled.model_prompt_version == V18
    assert Settings(_env_file=None, model_premise_bank_enabled=False).model_premise_bank_enabled is False


@pytest.mark.parametrize("enabled", [False, True])
async def test_real_adapter_instance_reports_actual_selected_strategy_without_changing_budgets(enabled):
    async with intercepted_adapter(premise_bank_enabled=enabled) as (provider, requests):
        assert provider.premise_bank_enabled is enabled
        assert provider.prompt_version == (V18 if enabled else V17)
        assert provider.model_name == "deepseek-v4-pro"
        assert provider.capacity_policy("extract")["initial_tokens"] == 8192
        assert callable(provider.review_fact_purpose) and callable(provider.repair_fact_purpose)
        assert requests == []


@pytest.mark.parametrize("enabled", [False, True])
async def test_worker_queued_config_execution_and_initial_refine_progress_match_strategy(tmp_path, enabled):
    repo, topic, rows = setup(tmp_path)
    repo.settings = repo.settings.model_copy(update={"model_premise_bank_enabled": enabled})
    provider = MeteredProductionDouble(rows, repo)
    provider.premise_bank_enabled = enabled
    provider.prompt_version = repo.settings.model_prompt_version
    config = repo.config()
    assert config["premise_bank_enabled"] is enabled
    assert config["prompt_version"] == provider.prompt_version
    worker = Worker(repo, provider)
    initial = create(repo, topic)
    assert initial["premise_bank_enabled"] is enabled
    await worker.execute(repo.claim())
    finished = repo.run(initial["id"])
    assert finished["status"] == "succeeded", finished["error"]
    initial_activity = repo.activity(initial["id"])
    assert initial_activity["progress"]["completed_count"] == initial_activity["progress"]["total_count"] == 9 + int(enabled)
    result = next(iter(repo.snapshot(topic, finished["target_revision_id"])["question_results"].values()))
    assert ("purpose_review" in result) is enabled
    assert any(e["type"] == "node_completed" and e["data"].get("node") == "purpose_review"
               for e in repo.events(initial["id"])) is enabled
    assert bool(provider.purpose_calls) is enabled
    # This double's original scope explicitly rejects target classification;
    # only the opt-in path may license its separate necessary premise.
    assert len(result["evidence"]) == int(enabled)
    calls_before = len(provider.purpose_calls)
    refined = repo.create_run(topic, {"question_ids": initial["question_ids"], "mode": "quality-refine",
        "base_revision_id": finished["target_revision_id"], "source_revision_id": finished["target_revision_id"],
        "include_baseline": False}, "strategy-refine", model_available=True)
    await worker.execute(repo.claim())
    assert repo.run(refined["id"])["status"] == "succeeded", repo.run(refined["id"])["error"]
    activity = repo.activity(refined["id"])
    assert activity["progress"]["completed_count"] == activity["progress"]["total_count"] == 7 + int(enabled)
    assert len(provider.purpose_calls) - calls_before == int(enabled)


@pytest.mark.parametrize("configured", [False, True])
async def test_worker_refuses_either_direction_of_protocol_mismatch_before_any_request(tmp_path, configured):
    repo, topic, rows = setup(tmp_path)
    repo.settings = repo.settings.model_copy(update={"model_premise_bank_enabled": configured})
    provider = MeteredProductionDouble(rows, repo)
    provider.premise_bank_enabled = not configured
    async def forbidden(*_args): pytest.fail("protocol mismatch must fail before extraction")
    provider.extract = forbidden
    queued = create(repo, topic)
    await Worker(repo, provider).execute(repo.claim())
    failed = repo.run(queued["id"])
    assert failed["status"] == "failed" and failed["error"]["code"] == "worker_protocol_mismatch"
    assert provider.purpose_calls == provider.analyst_inputs == []


@pytest.mark.parametrize("original", [False, True])
def test_strategy_change_changes_trusted_config_and_blocks_resume(tmp_path, original):
    repo, topic, _rows = setup(tmp_path)
    repo.settings = repo.settings.model_copy(update={"model_premise_bank_enabled": original})
    before = repo.config()
    queued = create(repo, topic)
    repo.pause(queued["id"])
    repo.settings = repo.settings.model_copy(update={"model_premise_bank_enabled": not original})
    after = repo.config()
    assert before["prompt_version"] != after["prompt_version"] and digest(before) != digest(after)
    assert not repo.run(queued["id"])["resumable"]
    with pytest.raises(ServiceError) as failure:
        repo.resume(queued["id"], "cross-strategy-resume", model_available=True)
    assert failure.value.code == "resume_not_available"
    assert len(repo.runs(topic)) == 1


def test_setting_source_file_is_fingerprinted_and_source_change_blocks_resume(tmp_path, monkeypatch):
    repo, topic, _rows = setup(tmp_path)
    queued = create(repo, topic)
    repo.pause(queued["id"])
    before = repo.config()
    target = Path(__file__).resolve().parents[1] / "app/config.py"
    read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: read(path) + (b"\n# synthetic strategy default change" if path == target else b""))
    assert repo.config()["prompt_sha256"] != before["prompt_sha256"]
    with pytest.raises(ServiceError) as failure:
        repo.resume(queued["id"], "changed-code-resume", model_available=True)
    assert failure.value.code == "resume_not_available"


async def test_direct_checkpoint_fork_cannot_switch_workflow_strategy(tmp_path):
    rows = [source()]
    first = V18Provider(rows)
    first.premise_bank_enabled = False
    await run_analysis(Q, paragraphs(rows), first, checkpoint_path=tmp_path / "checkpoints.sqlite3",
        thread_id="v17-source", cancel_check=lambda: False, emit=lambda *_: None)
    second = V18Provider(rows)
    with pytest.raises(ModelOutputInvalid) as failure:
        await run_analysis(Q, paragraphs(rows), second, checkpoint_path=tmp_path / "checkpoints.sqlite3",
            thread_id="v18-target", resume_from_thread_id="v17-source", cancel_check=lambda: False, emit=lambda *_: None)
    assert failure.value.code == "resume_checkpoint_invalid"
    assert failure.value.details["check"] == "source_input_fingerprint"
    assert second.purpose_calls == second.analyst_inputs == []


@pytest.mark.parametrize("enabled", [False, True])
def test_health_does_not_expose_model_configuration_or_credentials(tmp_path, enabled):
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="", model_api_key="explicit-placeholder",
        model_premise_bank_enabled=enabled)
    with TestClient(create_app(settings=settings)) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert not {"model_config", "model_api_key", "model_base_url", "output_budgets"} & set(response.json())
    assert "explicit-placeholder" not in response.text
