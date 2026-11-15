"""Explicit synthetic v18 Worker/SQLite/publication/export lifecycle, no network."""
from copy import deepcopy
import json
from zipfile import ZipFile

import pytest

from app.agents.provider import LangChainOpenAIProvider
from app.config import Settings
from app.data.ingest import import_topic
from app.domain.premise_bank import VERSION
from app.errors import ServiceError
from app.repository import Repository
from app.utils import digest
from app.worker import Worker
from app.recovery import recover_completed_run
from test_premise_bank import Q, source
from test_premise_bank_workflow import V18Provider
from test_repository import create


def setup(tmp_path, *, question_count=1):
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="", model_premise_bank_enabled=True)
    repo = Repository(settings)
    texts = [source()["quote"], "场馆外墙为蓝色。"]
    docs = [{"doc_id": f"raw-source-{i}", "title": "合成原始资料", "content": text,
        "paragraphs": [{"para_id": "p1", "char_start": 0, "char_end": len(text)}]}
        for i, text in enumerate(texts)]
    questions = [{"question_id": f"raw-question-{i}", "question": Q["question"]} for i in range(question_count)]
    topic = import_topic(settings.database_path, docs, questions, "合成 v18 生命周期")["topic_id"]
    rows = []
    for p in repo.paragraphs(topic):
        row = source(p["id"], text=p["text"])
        row.update(paragraph_id=p["id"], raw_doc_id=p["raw_doc_id"], raw_para_id=p["raw_para_id"])
        rows.append(row)
    return repo, topic, rows


class MeteredProductionDouble(V18Provider):
    model_name = "explicit-v18-production-double"

    def __init__(self, rows, repo, **options):
        super().__init__(rows, **options)
        self.repo, self.observed_activity = repo, []

    async def review_fact_purpose(self, question, bank, ids):
        active = self.repo.one("SELECT id FROM runs WHERE status='running'")["id"]
        self.observed_activity.append(self.repo.activity(active))
        LangChainOpenAIProvider._record_usage(self, "review_fact_purpose", {
            "input_tokens": 13, "output_tokens": 7, "total_tokens": 20})
        return await super().review_fact_purpose(question, bank, ids)


async def published(tmp_path):
    repo, topic, rows = setup(tmp_path)
    provider = MeteredProductionDouble(rows, repo)
    queued = create(repo, topic)
    assert queued["status"] == "queued" and queued["premise_bank_enabled"] is True
    claimed = repo.claim()
    assert claimed["id"] == queued["id"]
    worker = Worker(repo, provider)
    await worker.execute(claimed)
    result = repo.run(queued["id"])
    assert result["status"] == "succeeded", result["error"]
    return repo, topic, result, provider, worker


async def test_v18_queue_worker_stage_usage_publish_and_real_four_file_export(tmp_path):
    repo, topic, run, provider, worker = await published(tmp_path)
    observed = provider.observed_activity[0]
    assert observed["progress"]["current_stage"] == "purpose_review"
    assert observed["progress"]["active_stages"] == [{"question_id": run["question_ids"][0], "stage": "purpose_review"}]
    done = repo.activity(run["id"])
    assert done["progress"]["completed_count"] == done["progress"]["total_count"] == 10
    assert {"question_id": run["question_ids"][0], "stage": "purpose_review"} in done["progress"]["completed_stages"]
    calls = [c for c in done["tool_calls"] if c["name"] == "review_fact_purpose"]
    assert len(calls) == 1 and calls[0]["status"] == "completed"
    assert calls[0]["stage"] == "purpose_review"
    assert run["usage"][run["question_ids"][0]][0]["operation"] == "review_fact_purpose"
    snapshot = repo.snapshot(topic, run["target_revision_id"])
    question_result = snapshot["question_results"][run["question_ids"][0]]
    assert question_result["verification_protocol"] == VERSION
    assert question_result["strict_scope_review"]["selected_fact_view_ids"] == []
    assert len(question_result["quote_faithful_bank"]["rows"]) == 2
    assert len(snapshot["evidence"]) == 1
    assert repo.export_readiness(topic, run["target_revision_id"])["can_export"]
    job = repo.create_export(topic, run["target_revision_id"], "v18-export")
    assert worker.export_one()
    assert repo.export(job["id"])["status"] == "succeeded"
    with ZipFile(repo.settings.export_dir / (job["id"] + ".zip")) as archive:
        assert {"evidence.json", "conclusions.json", "confidence.json", "conflicts.json", "manifest.json"} <= set(archive.namelist())
        facts = json.loads(archive.read("evidence.json"))
        assert len(facts) == 1 and "外墙" not in facts[0]["quote"]
    assert provider.analyst_inputs[0][1:] == provider.analyst_inputs[1][1:]


def strip_v18(result):
    """Attack all untrusted payload markers, leaving ordinary v17 material audit."""
    for key in ("verification_protocol", "quote_faithful_bank", "purpose_review"):
        result.pop(key, None)
    for row in result["evidence"]:
        row.pop("purpose_authorization", None)
    material = result["material_eligibility"]
    material.pop("purpose_authorization", None)
    for row in material.get("input_evidence", []):
        row.pop("purpose_authorization", None)
    material.pop("audit_sha256", None)
    material["audit_sha256"] = digest(material)


@pytest.mark.parametrize("attack", ["whole_bank", "missing_purpose", "strip_all_markers", "flattened_only"])
async def test_v18_actual_publication_refuses_bad_authorization_and_keeps_head_unpublished(tmp_path, monkeypatch, attack):
    repo, topic, rows = setup(tmp_path)
    provider = MeteredProductionDouble(rows, repo)
    queued = create(repo, topic)
    original_publish = repo.publish
    def tampered_publish(run_id, snapshot, fresh):
        qid = next(iter(fresh))
        result = deepcopy(fresh[qid])
        if attack == "whole_bank": result["evidence"] = deepcopy(result["quote_faithful_bank"]["rows"])
        elif attack == "missing_purpose": result.pop("purpose_review")
        elif attack == "strip_all_markers": strip_v18(result)
        if attack == "flattened_only":
            snapshot["evidence"] = deepcopy(result["quote_faithful_bank"]["rows"])
        else:
            snapshot["question_results"][qid] = fresh[qid] = result
            snapshot["evidence"] = deepcopy(result["evidence"])
        original_publish(run_id, snapshot, fresh)
    monkeypatch.setattr(repo, "publish", tampered_publish)
    await Worker(repo, provider).execute(repo.claim())
    failed = repo.run(queued["id"])
    assert failed["status"] == "failed" and failed["error"]["code"] == "publish_validation_failed"
    assert repo.topic(topic)["published_revision_id"] is None
    assert not repo.all("SELECT * FROM revision_artifacts")


@pytest.mark.parametrize("attack", ["whole_bank", "missing_purpose", "strip_all_markers", "flattened_only"])
async def test_v18_actual_export_replays_trusted_origin_config_and_authorized_pool(tmp_path, monkeypatch, attack):
    repo, topic, run, _provider, worker = await published(tmp_path)
    original_snapshot = repo.snapshot
    snapshot = deepcopy(original_snapshot(topic, run["target_revision_id"]))
    result = next(iter(snapshot["question_results"].values()))
    if attack == "whole_bank": result["evidence"] = deepcopy(result["quote_faithful_bank"]["rows"])
    elif attack == "missing_purpose": result.pop("purpose_review")
    elif attack == "strip_all_markers": strip_v18(result)
    if attack == "flattened_only": snapshot["evidence"] = deepcopy(result["quote_faithful_bank"]["rows"])
    else: snapshot["evidence"] = deepcopy(result["evidence"])
    # Ready artifacts themselves are immutable. Inject corrupted read projection
    # to exercise the real download boundary without disabling SQLite triggers.
    monkeypatch.setattr(repo, "snapshot", lambda *_args: deepcopy(snapshot))
    with pytest.raises(ServiceError) as failure:
        repo.formal_submission_files(topic, run["target_revision_id"])
    assert failure.value.code == "export_validation_failed"
    assert not repo.export_readiness(topic, run["target_revision_id"])["can_export"]
    with pytest.raises(ServiceError): repo.create_export(topic, run["target_revision_id"], "blocked")


async def test_v18_purpose_failure_is_recorded_as_real_failed_stage_not_snapshot(tmp_path):
    repo, topic, rows = setup(tmp_path)
    provider = MeteredProductionDouble(rows, repo, corrupt=True)
    queued = create(repo, topic)
    await Worker(repo, provider).execute(repo.claim())
    failed = repo.run(queued["id"])
    assert failed["status"] == "failed" and failed["error"]["stage"] == "purpose_review"
    activity = repo.activity(queued["id"])
    assert activity["progress"]["failed_stages"][0]["stage"] == "purpose_review"
    assert len(activity["usage"][queued["question_ids"][0]]) == 1
    assert activity["usage"][queued["question_ids"][0]][0]["operation"] == "review_fact_purpose"


async def test_v18_protocol_capability_mismatch_fails_before_any_provider_call(tmp_path):
    repo, topic, rows = setup(tmp_path)
    provider = MeteredProductionDouble(rows, repo)
    provider.premise_bank_enabled = False
    queued = create(repo, topic)
    await Worker(repo, provider).execute(repo.claim())
    failed = repo.run(queued["id"])
    assert failed["status"] == "failed" and failed["error"]["code"] == "worker_protocol_mismatch"
    assert provider.purpose_calls == provider.analyst_inputs == []
    assert not any(e["type"] == "tool_started" for e in repo.events(queued["id"]))


async def test_v18_completed_checkpoint_recovery_retains_all_permissions_without_new_calls(tmp_path, monkeypatch):
    repo, topic, rows = setup(tmp_path)
    provider = MeteredProductionDouble(rows, repo)
    queued = create(repo, topic)
    original_publish = repo.publish
    def fail_publication(*_args): raise ValueError("explicit synthetic publication interruption")
    monkeypatch.setattr(repo, "publish", fail_publication)
    await Worker(repo, provider).execute(repo.claim())
    assert repo.run(queued["id"])["error"]["stage"] == "publish"
    monkeypatch.setattr(repo, "publish", original_publish)
    calls = deepcopy(provider.purpose_calls)
    usage = deepcopy(repo.run(queued["id"])["usage"])
    def forbid_provider(*_args, **_kwargs): pytest.fail("completed recovery must not construct a provider")
    monkeypatch.setattr(LangChainOpenAIProvider, "__init__", forbid_provider)
    recovered = await recover_completed_run(repo, queued["id"], "v18-completed-recovery")
    assert recovered["status"] == "succeeded"
    result = next(iter(repo.snapshot(topic, recovered["target_revision_id"])["question_results"].values()))
    assert result["verification_protocol"] == VERSION
    assert result["quote_faithful_bank"]["formal_evidence"] is False
    assert result["purpose_review"]["complete"] and result["material_eligibility"]["purpose_authorization"]
    assert provider.purpose_calls == calls and repo.run(queued["id"])["usage"] == usage
    assert recovered["usage"] == {queued["question_ids"][0]: []}
    assert repo.formal_submission_files(topic, recovered["target_revision_id"])["evidence.json"]


async def test_v18_export_cannot_spoof_a_legacy_origin_to_downgrade_current_question(tmp_path, monkeypatch):
    repo, topic, rows = setup(tmp_path)
    original_config = repo.config
    repo.config = lambda: {**original_config(), "premise_bank_enabled": False}
    legacy = create(repo, topic)
    repo.cancel(legacy["id"])
    repo.config = original_config
    queued = create(repo, topic)
    await Worker(repo, MeteredProductionDouble(rows, repo)).execute(repo.claim())
    run = repo.run(queued["id"])
    assert run["status"] == "succeeded", run["error"]
    snapshot = repo.snapshot(topic, run["target_revision_id"])
    result = next(iter(snapshot["question_results"].values()))
    strip_v18(result)
    result["origin_run_id"] = legacy["id"]
    snapshot["evidence"] = deepcopy(result["evidence"])
    monkeypatch.setattr(repo, "snapshot", lambda *_args: deepcopy(snapshot))
    with pytest.raises(ServiceError, match="完整性"):
        repo.formal_submission_files(topic, run["target_revision_id"])


@pytest.mark.parametrize("mode", ["together", "inherited"])
async def test_shared_fact_across_questions_preserves_each_permission_and_exports_once(tmp_path, monkeypatch, mode):
    repo, topic, rows = setup(tmp_path, question_count=2)
    provider = MeteredProductionDouble(rows, repo)
    questions = repo.questions(topic)
    batches = [[q["id"] for q in questions]] if mode == "together" else [[q["id"]] for q in questions]
    for selected in batches:
        queued = create(repo, topic, selected=selected)
        await Worker(repo, provider).execute(repo.claim())
        latest = repo.run(queued["id"])
        assert latest["status"] == "succeeded", latest["error"]
    snapshot = repo.snapshot(topic, latest["target_revision_id"])
    results = list(snapshot["question_results"].values())
    assert len(results) == 2 and len(snapshot["evidence"]) == 1
    assert results[0]["evidence"][0]["id"] == results[1]["evidence"][0]["id"]
    assert results[0]["purpose_review"]["bank_id"] != results[1]["purpose_review"]["bank_id"]
    assert len(repo.formal_submission_files(topic, latest["target_revision_id"])["evidence.json"]) == 1
    roundtrip = json.loads(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
    roundtrip["question_results"] = dict(reversed(list(roundtrip["question_results"].items())))
    monkeypatch.setattr(repo, "snapshot", lambda *_args: deepcopy(roundtrip))
    assert len(repo.formal_submission_files(topic, latest["target_revision_id"])["evidence.json"]) == 1
