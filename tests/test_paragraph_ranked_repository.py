"""v19 real SQLite/Worker/publication/export tests with explicit zero-network doubles."""
from copy import deepcopy
import json
from zipfile import ZipFile

import pytest

from app.config import Settings
from app.data.ingest import import_topic
from app.errors import ServiceError
from app.repository import Repository
from app.worker import Worker
from app.recovery import recover_completed_run
from app.domain import paragraph_ranked as p
from test_paragraph_ranked_workflow import Q, sources, RankedProvider
from test_repository import create


def setup(tmp_path, *, question_count=1, enabled=True):
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="",
                        model_paragraph_ranked_enabled=enabled)
    repo = Repository(settings)
    docs = [{"doc_id": row["raw_doc_id"], "title": row["title"], "content": row["text"],
             "publish_time": row["raw_publish_time"], "source_credibility": .95,
             "paragraphs": [{"para_id": row["raw_para_id"], "char_start": 0, "char_end": len(row["text"])}]} for row in sources()]
    questions = [{"question_id": f"q{i}", "question": Q["question"]} for i in range(question_count)]
    topic = import_topic(settings.database_path, docs, questions, "合成v19生命周期")["topic_id"]
    return repo, topic


async def published(tmp_path, *, question_count=1):
    repo, topic = setup(tmp_path, question_count=question_count)
    provider = RankedProvider(); worker = Worker(repo, provider)
    queued = create(repo, topic); await worker.execute(repo.claim())
    done = repo.run(queued["id"])
    assert done["status"] == "succeeded", done["error"]
    return repo, topic, done, provider, worker


async def test_actual_worker_publication_progress_tokens_full_quote_and_export(tmp_path):
    repo, topic, done, provider, worker = await published(tmp_path)
    activity = repo.activity(done["id"])
    assert activity["progress"]["completed_count"] == activity["progress"]["total_count"] == 10
    assert {s["stage"] for s in activity["progress"]["completed_stages"]} >= {"paragraph_queries", "paragraph_recall", "paragraph_rank", "paragraph_annotate", "paragraph_fields"}
    records = done["usage"][done["question_ids"][0]]
    assert len(records) == 7 and {r["operation"] for r in records} >= {"paragraph_queries", "paragraph_rank", "paragraph_annotate", "paragraph_fields"}
    assert sum(r["total_tokens"] for r in records) == 84
    snapshot = repo.snapshot(topic, done["target_revision_id"])
    result = snapshot["question_results"][done["question_ids"][0]]
    assert result["verification_protocol"] == p.VERSION and "strict_scope_review" not in result
    assert all(r["grade_score"] == .95 for r in result["evidence"])
    assert repo.export_readiness(topic, done["target_revision_id"])["can_export"]
    job = repo.create_export(topic, done["target_revision_id"], "v19-export"); assert worker.export_one()
    assert repo.export(job["id"])["status"] == "succeeded"
    with ZipFile(repo.settings.export_dir / (job["id"] + ".zip")) as archive:
        evidence = json.loads(archive.read("evidence.json"))
        assert len(evidence) == 2 and {row["quote"] for row in evidence} == {row["text"] for row in sources()}
    assert len(provider.inputs) == 2 and provider.inputs[0][1:] == provider.inputs[1][1:]


async def test_two_questions_have_distinct_internal_annotations_and_stable_one_source_export(tmp_path):
    repo, topic, done, _provider, _worker = await published(tmp_path, question_count=2)
    results = list(repo.snapshot(topic, done["target_revision_id"])["question_results"].values())
    assert {r["id"] for r in results[0]["evidence"]}.isdisjoint(r["id"] for r in results[1]["evidence"])
    files = repo.formal_submission_files(topic, done["target_revision_id"])
    assert len(files["evidence.json"]) == 2


@pytest.mark.parametrize("attack", ["strip_protocol", "strip_all_markers", "changed_fact", "missing_fields"])
async def test_actual_publication_blocks_protocol_downgrade_or_same_source_changed_annotation(tmp_path, monkeypatch, attack):
    repo, topic = setup(tmp_path); provider = RankedProvider(); queued = create(repo, topic)
    original = repo.publish
    def corrupted(run_id, snapshot, fresh):
        qid = next(iter(fresh)); row = deepcopy(fresh[qid])
        if attack in {"strip_protocol", "strip_all_markers"}:
            row.pop("verification_protocol", None); row.pop("paragraph_ranked_proof", None)
            if attack == "strip_all_markers":
                for e in row["evidence"]: e.pop("paragraph_fields_verified", None); e.pop("span_location_method", None)
                row.pop("source_coverage", None)
        elif attack == "changed_fact": row["evidence"][0]["fact_des"] = "主管部门已确认批准。"
        else: row["paragraph_ranked_proof"]["field_reviews"] = []
        snapshot["question_results"][qid] = fresh[qid] = row; snapshot["evidence"] = deepcopy(row["evidence"])
        return original(run_id, snapshot, fresh)
    monkeypatch.setattr(repo, "publish", corrupted)
    await Worker(repo, provider).execute(repo.claim())
    failed = repo.run(queued["id"])
    assert failed["status"] == "failed" and repo.topic(topic)["published_revision_id"] is None
    assert not repo.all("SELECT * FROM revision_artifacts")


async def test_download_uses_origin_strategy_not_current_settings_and_rejects_corrupted_read(tmp_path, monkeypatch):
    repo, topic, done, _provider, _worker = await published(tmp_path)
    repo.settings = repo.settings.model_copy(update={"model_paragraph_ranked_enabled": False})
    assert repo.formal_submission_files(topic, done["target_revision_id"])
    original = repo.snapshot; snap = deepcopy(original(topic, done["target_revision_id"]))
    next(iter(snap["question_results"].values())).pop("paragraph_ranked_proof")
    monkeypatch.setattr(repo, "snapshot", lambda *_args: deepcopy(snap))
    with pytest.raises(ServiceError): repo.formal_submission_files(topic, done["target_revision_id"])


@pytest.mark.parametrize("enabled", [True, False])
async def test_worker_capability_mismatch_both_directions_is_pre_model(tmp_path, enabled):
    repo, topic = setup(tmp_path, enabled=enabled); provider = RankedProvider()
    provider.paragraph_ranked_enabled = not enabled
    queued = create(repo, topic); await Worker(repo, provider).execute(repo.claim())
    assert repo.run(queued["id"])["error"]["code"] == "worker_protocol_mismatch" and provider.calls == []


async def test_field_failure_is_visible_with_exact_stage_and_billable_counters(tmp_path):
    repo, topic = setup(tmp_path); provider = RankedProvider(reject_fields=True)
    queued = create(repo, topic); await Worker(repo, provider).execute(repo.claim())
    failed = repo.run(queued["id"]); activity = repo.activity(queued["id"])
    assert failed["error"]["stage"] == "paragraph_fields"
    assert activity["progress"]["failed_stages"][0]["stage"] == "paragraph_fields"
    assert len(failed["usage"][failed["question_ids"][0]]) == 4 and provider.calls.count("paragraph_fields") == 1


async def test_completed_v19_checkpoint_recovery_replays_proof_without_new_provider_calls(tmp_path, monkeypatch):
    repo, topic = setup(tmp_path); provider = RankedProvider(); queued = create(repo, topic)
    original = repo.publish
    def fail_publish(*args, **kwargs): raise ValueError("synthetic failure after completed graph")
    monkeypatch.setattr(repo, "publish", fail_publish)
    await Worker(repo, provider).execute(repo.claim())
    assert repo.run(queued["id"])["status"] == "failed"
    calls = list(provider.calls); monkeypatch.setattr(repo, "publish", original)
    recovered = await recover_completed_run(repo, queued["id"], idempotency_key="v19-completed-recovery")
    assert recovered["status"] == "succeeded" and provider.calls == calls
    result = next(iter(repo.snapshot(topic, recovered["target_revision_id"])["question_results"].values()))
    assert result["verification_protocol"] == p.VERSION and isinstance(result["paragraph_ranked_proof"], dict)
    assert result["usage"] == []
