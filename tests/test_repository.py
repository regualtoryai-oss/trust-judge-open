"""Real SQLite repository integration using explicitly synthetic workflow output.

These tests never instantiate or call a model. Their purpose is persistence,
revision, provenance and export integrity, not model-quality demonstration.
"""

from copy import deepcopy
import json
import sqlite3
from uuid import NAMESPACE_URL, uuid4, uuid5
from zipfile import ZipFile

import pytest

from app.config import Settings
from app.data.ingest import import_topic
from app.domain.snapshot import build_snapshot
from app.errors import ServiceError
from app.repository import Repository
from app.utils import loads
from app.worker import Worker, WorkerLease


def stable(value):
    return str(uuid5(NAMESPACE_URL, "explicit-test-fixture:" + value))


def setup_repo(tmp_path, *, question_count=1):
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="", model_premise_bank_enabled=False)
    repo = Repository(settings)
    # These synthetic publications model the legacy, pre-purpose protocol.
    original_config = repo.config
    repo.config = lambda: {**original_config(), "premise_bank_enabled": False}
    content = "🦊通报称失联3人。"
    documents = [{"doc_id": "doc_001", "title": "显式合成测试通报", "content": content, "paragraphs": [{"para_id": "doc_001#p1", "char_start": 0, "char_end": len(content)}], "publish_time": None, "content_url": None, "source_credibility": None}]
    questions = [{"question_id": f"q_{index:03}", "question": "这份测试通报中的失联人数是多少？"} for index in range(1, question_count + 1)]
    topic_id = import_topic(settings.database_path, documents, questions, "显式合成测试专题")["topic_id"]
    return repo, topic_id


def create(repo, topic_id, *, selected=None, key=None):
    qids = selected or [q["id"] for q in repo.questions(topic_id)]
    body = {"base_revision_id": repo.topic(topic_id)["head_revision_id"], "question_ids": qids, "mode": "initial", "include_baseline": False}
    # model_available=True authorizes only the test persistence path. There is
    # deliberately no provider instance and no model invocation in this fixture.
    return repo.create_run(topic_id, body, key or str(uuid4()), model_available=True)


def output(repo, topic_id, question, run_id, *, quote="失联3人"):
    para = repo.paragraphs(topic_id)[0]
    start = para["text"].index("失联3人")
    eid = stable(f"evidence:{para['id']}:{start}:{quote}")
    evidence = {"id": eid, "paragraph_id": para["id"], "raw_doc_id": para["raw_doc_id"], "raw_para_id": para["raw_para_id"], "start_cp": start, "end_cp": start + len(quote), "quote": quote, "fact_des": "测试通报称" + quote, "argument": {"时间": None, "主体": "失联人员", "地点": None, "内容": quote, "影响": None, "数值口径": quote}, "entities": ["失联人员"], "source_group_id": stable("source:" + para["document_id"]), "source_group_basis": "same_document_only_not_complete_origin_detection", "grade_score": None, "source_level": "unknown"}
    final = {"decision": "asserted", "conclusion": "测试通报称" + quote + "。", "judgments": [{"id": stable(question["id"] + ":judgment"), "statement": "该测试通报直接记载了人数", "evidence_ids": [eid], "prerequisite_judgment_ids": []}], "supporting_evidence_ids": [eid], "conflicting_evidence_ids": [], "conflicts": []}
    sid = stable(run_id + ":" + question["id"] + ":snapshot")
    analyses = [{"role": role, "evidence_snapshot_id": sid, **{key: deepcopy(value) for key, value in final.items() if key != "conflicts"}} for role in ("authority_analyst", "cross_validation_analyst")]
    return {"evidence": [evidence], "analyses": analyses, "final": final, "usage": [], "evidence_snapshot_id": sid, "origin_run_id": run_id}


def candidate(repo, topic_id, run, *, quote="失联3人"):
    all_questions = repo.questions(topic_id)
    selected = run["question_ids"] if "question_ids" in run else loads(run["question_ids_json"])
    questions = [q for q in all_questions if q["id"] in selected]
    results = {q["id"]: output(repo, topic_id, q, run["id"], quote=quote) for q in questions}
    revision = repo.revision(topic_id, run["target_revision_id"])
    snapshot = build_snapshot(repo.topic(topic_id), questions, results, revision_id=revision["id"], graph_revision=revision["revision_no"], run_id=run["id"])
    snapshot["scope"] = {"completed_question_ids": list(results), "all_question_count": len(all_questions), "is_complete_topic": len(results) == len(all_questions)}
    snapshot["overlays"] = {"excluded": [], "manual_evidence": [], "interpretation_overrides": {}}
    return snapshot, results


def publish(repo, topic_id, *, selected=None):
    run = create(repo, topic_id, selected=selected)
    claimed = repo.claim()
    assert claimed["id"] == run["id"]
    snapshot, results = candidate(repo, topic_id, run)
    repo.publish(run["id"], snapshot, results)
    return repo.run(run["id"]), snapshot


def test_ready_revision_and_all_persisted_content_are_immutable(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    run, snapshot = publish(repo, topic_id)
    rid = run["target_revision_id"]
    assert repo.topic(topic_id)["published_revision_id"] == rid
    assert repo.run(run["id"])["status"] == "succeeded"
    assert repo.snapshot(topic_id, rid)["question_results"] == snapshot["question_results"]
    assert repo.all("SELECT COUNT(*) n FROM agent_executions WHERE run_id=?", (run["id"],))[0]["n"] == 3
    with repo.connection() as db:
        for statement in ["UPDATE node_states SET content_json='{}' WHERE revision_id=?", "DELETE FROM graph_edges WHERE revision_id=?", "UPDATE source_groups SET basis_json='{}' WHERE revision_id=?", "DELETE FROM revision_artifacts WHERE revision_id=?", "UPDATE graph_revisions SET state='failed' WHERE id=?"]:
            with pytest.raises(sqlite3.IntegrityError, match="immutable_ready_revision"):
                db.execute(statement, (rid,))
        with pytest.raises(sqlite3.IntegrityError, match="immutable_ready_revision"):
            db.execute("UPDATE agent_executions SET output_json='{}' WHERE run_id=?", (run["id"],))
    events = repo.events(run["id"])
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert events[-1]["type"] == "run_succeeded"


def test_bad_source_span_rolls_back_without_changing_published_result(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    first, _ = publish(repo, topic_id)
    old_published = first["target_revision_id"]
    next_run = create(repo, topic_id)
    assert repo.claim()["id"] == next_run["id"]
    # Internally consistent model output deliberately contradicts the actual
    # source's character at the exact specified span. Repository must reject it.
    snapshot, results = candidate(repo, topic_id, next_run, quote="失联4人")
    with pytest.raises(ValueError):
        repo.publish(next_run["id"], snapshot, results)
    assert repo.topic(topic_id)["published_revision_id"] == old_published
    assert repo.all("SELECT COUNT(*) n FROM evidence")[0]["n"] == 1
    assert repo.all("SELECT COUNT(*) n FROM node_states WHERE revision_id=?", (next_run["target_revision_id"],))[0]["n"] == 0
    assert repo.all("SELECT COUNT(*) n FROM revision_artifacts WHERE revision_id=?", (next_run["target_revision_id"],))[0]["n"] == 0
    assert repo.revision(topic_id, next_run["target_revision_id"])["state"] == "building"


def test_preview_commit_hash_and_idempotent_replay(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    first, snapshot = publish(repo, topic_id)
    base = first["target_revision_id"]
    evidence_id = snapshot["evidence"][0]["id"]
    operations = [{"op": "exclude_evidence", "evidence_id": evidence_id}]
    preview = repo.preview_change(topic_id, base, operations, "核对测试引用")
    assert preview["affected_question_ids"] == [repo.questions(topic_id)[0]["id"]]
    assert snapshot["conclusions"][0]["conclusion_node_id"] in preview["affected_node_ids"]
    saved = repo.one("SELECT * FROM change_previews WHERE id=?", (preview["preview_id"],))
    assert loads(saved["operations_json"]) == operations
    assert saved["operations_sha256"] == preview["content_hash"]
    assert repo.topic(topic_id)["head_revision_id"] == base
    body = {"preview_id": preview["preview_id"], "base_revision_id": base, "preview_content_hash": preview["content_hash"]}
    with pytest.raises(ServiceError) as error:
        repo.commit_change(topic_id, {**body, "preview_content_hash": "0" * 64}, "bad-hash", model_available=True)
    assert error.value.status == 409
    assert repo.one("SELECT consumed_by_run_id FROM change_previews WHERE id=?", (preview["preview_id"],))["consumed_by_run_id"] is None
    committed = repo.commit_change(topic_id, body, "commit-key", model_available=True)
    replay = repo.commit_change(topic_id, body, "commit-key", model_available=True)
    assert committed["id"] == replay["id"]
    assert repo.all("SELECT COUNT(*) n FROM changesets")[0]["n"] == 1
    assert repo.one("SELECT consumed_by_run_id FROM change_previews WHERE id=?", (preview["preview_id"],))["consumed_by_run_id"] == committed["id"]
    options = loads(repo.one("SELECT options_json FROM run_options WHERE run_id=?", (committed["id"],))["options_json"])
    assert options["overlays"]["excluded"] == [[snapshot["evidence"][0][key] for key in ("paragraph_id", "start_cp", "end_cp", "quote")]]
    with pytest.raises(ServiceError) as error:
        repo.commit_change(topic_id, {**body, "preview_content_hash": "f" * 64}, "commit-key", model_available=True)
    assert error.value.code == "idempotency_conflict"


def test_stale_preview_is_rejected_after_head_changes(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    first, snapshot = publish(repo, topic_id)
    base = first["target_revision_id"]
    preview = repo.preview_change(topic_id, base, [{"op": "exclude_evidence", "evidence_id": snapshot["evidence"][0]["id"]}], "测试过期版本")
    second = create(repo, topic_id)
    repo.cancel(second["id"])
    body = {"preview_id": preview["preview_id"], "base_revision_id": base, "preview_content_hash": preview["content_hash"]}
    with pytest.raises(ServiceError) as error:
        repo.commit_change(topic_id, body, "stale-preview", model_available=True)
    assert error.value.status == 409 and error.value.code == "stale_base"
    assert repo.all("SELECT COUNT(*) n FROM changesets")[0]["n"] == 0


def test_queued_cancel_is_terminal_and_never_claimed(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    run = create(repo, topic_id)
    cancelled = repo.cancel(run["id"])
    assert cancelled["status"] == "cancelled" and cancelled["cancel_requested"]
    assert repo.claim() is None
    assert repo.revision(topic_id, run["target_revision_id"])["state"] == "cancelled"
    assert repo.topic(topic_id)["published_revision_id"] is None
    assert repo.cancel(run["id"])["status"] == "cancelled"


def test_late_worker_cannot_publish_after_running_cancellation(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    first, _ = publish(repo, topic_id)
    late = create(repo, topic_id)
    assert repo.claim()["id"] == late["id"]
    snapshot, results = candidate(repo, topic_id, late)
    assert repo.cancel(late["id"])["status"] == "running"
    with pytest.raises(ServiceError) as error:
        repo.publish(late["id"], snapshot, results)
    assert error.value.code == "publish_rejected"
    assert repo.topic(topic_id)["published_revision_id"] == first["target_revision_id"]
    repo.finish_failed(late["id"], "cancelled", "explicit test cancellation", cancelled=True)
    assert repo.run(late["id"])["status"] == "cancelled"


def test_restart_recovery_preserves_ready_snapshot_and_checkpoint_file(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    first, _ = publish(repo, topic_id)
    export = repo.create_export(topic_id, first["target_revision_id"], "interrupted-export")
    with repo.connection() as db:
        db.execute("UPDATE export_jobs SET status='running' WHERE id=?", (export["id"],))
    interrupted = create(repo, topic_id)
    assert repo.claim()["id"] == interrupted["id"]
    # An explicit marker confirms recovery preserves the checkpoint file;
    # meaningful LangGraph checkpoint contents are tested in test_workflow.py.
    repo.settings.checkpoint_path.write_bytes(b"explicit checkpoint-preservation test marker")
    before = repo.settings.checkpoint_path.read_bytes()
    restarted = Repository(repo.settings)
    restarted.recover()
    assert restarted.run(interrupted["id"])["status"] == "failed"
    assert restarted.run(interrupted["id"])["error"]["code"] == "worker_interrupted"
    assert restarted.export(export["id"])["status"] == "failed"
    assert restarted.topic(topic_id)["published_revision_id"] == first["target_revision_id"]
    assert restarted.snapshot(topic_id, first["target_revision_id"])
    assert restarted.settings.checkpoint_path.read_bytes() == before
    assert create(restarted, topic_id)["status"] == "queued"


def test_worker_lease_prevents_second_worker_recovery(tmp_path):
    first = WorkerLease(tmp_path / "worker.lock")
    second = WorkerLease(tmp_path / "worker.lock")
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="worker"):
            second.acquire()
    finally:
        first.close()
        second.close()
    second.acquire()
    second.close()


def test_partial_question_scope_cannot_create_a_successful_formal_export(tmp_path):
    repo, topic_id = setup_repo(tmp_path, question_count=2)
    selected = [repo.questions(topic_id)[0]["id"]]
    run, snapshot = publish(repo, topic_id, selected=selected)
    assert not snapshot["scope"]["is_complete_topic"]
    try:
        job = repo.create_export(topic_id, run["target_revision_id"], "partial-export")
    except ServiceError as error:
        assert error.status == 409
        return
    worker = Worker(repo)
    assert worker.export_one()
    assert repo.export(job["id"])["status"] == "failed"
    assert not (repo.settings.export_dir / (job["id"] + ".zip")).exists()


def test_complete_export_uses_ready_version_and_contains_verified_files(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    run, _ = publish(repo, topic_id)
    first = repo.create_export(topic_id, run["target_revision_id"], "complete-export")
    assert repo.create_export(topic_id, run["target_revision_id"], "complete-export")["id"] == first["id"]
    assert Worker(repo).export_one()
    result = repo.export(first["id"])
    assert result["status"] == "succeeded"
    with ZipFile(repo.settings.export_dir / (first["id"] + ".zip")) as archive:
        assert {"evidence.json", "conclusions.json", "confidence.json", "conflicts.json", "manifest.json"} <= set(archive.namelist())
        assert json.loads(archive.read("conclusions.json"))[0]["question_id"] == "q_001"
        assert json.loads(archive.read("evidence.json"))[0]["quote"] == repo.paragraphs(topic_id)[0]["text"]
        assert json.loads(archive.read("manifest.json"))["revision_id"] == run["target_revision_id"]


def test_wrong_complete_scope_label_cannot_bypass_actual_question_coverage(tmp_path):
    repo, topic_id = setup_repo(tmp_path, question_count=2)
    selected = [repo.questions(topic_id)[0]["id"]]
    run = create(repo, topic_id, selected=selected)
    assert repo.claim()["id"] == run["id"]
    snapshot, results = candidate(repo, topic_id, run)
    # Mislabelled internal metadata must not turn a partial result into an
    # officially complete topic package. The authoritative set lives in SQL.
    snapshot["scope"]["is_complete_topic"] = True
    snapshot["scope"]["all_question_count"] = 1
    try:
        repo.publish(run["id"], snapshot, results)
    except (ValueError, ServiceError):
        assert repo.topic(topic_id)["published_revision_id"] is None
        return
    try:
        job = repo.create_export(topic_id, run["target_revision_id"], "mislabelled-partial")
    except ServiceError as error:
        assert error.status == 409
        return
    assert Worker(repo).export_one()
    assert repo.export(job["id"])["status"] == "failed"


@pytest.mark.parametrize("field", ["topic_id", "run_id", "revision_id"])
def test_publish_rejects_snapshot_from_another_resource_scope(tmp_path, field):
    repo, topic_id = setup_repo(tmp_path)
    run = create(repo, topic_id)
    assert repo.claim()["id"] == run["id"]
    snapshot, results = candidate(repo, topic_id, run)
    snapshot[field] = str(uuid4())
    with pytest.raises((ValueError, ServiceError)):
        repo.publish(run["id"], snapshot, results)
    assert repo.topic(topic_id)["published_revision_id"] is None
    assert repo.all("SELECT COUNT(*) n FROM revision_artifacts")[0]["n"] == 0


def interpretation():
    return {"fact_description": "测试人工修订后的事实解释", "stance": "neutral", "argument": {"time": "", "subject": "失联人员", "location": "测试地点", "content": "按原文重新审查口径", "impact": "", "numeric_scope": ""}, "entities": ["失联人员"]}


def test_failed_recompute_retry_inherits_its_unpublished_overlays(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    initial, initial_snapshot = publish(repo, topic_id)
    base = initial["target_revision_id"]
    eid = initial_snapshot["evidence"][0]["id"]
    operations = [{"op": "set_interpretation", "evidence_id": eid, "interpretation": interpretation()}, {"op": "exclude_evidence", "evidence_id": eid}]
    preview = repo.preview_change(topic_id, base, operations, "测试重算失败后的人工修订保留")
    committed = repo.commit_change(topic_id, {"preview_id": preview["preview_id"], "base_revision_id": base, "preview_content_hash": preview["content_hash"]}, "failing-recompute", model_available=True)
    failed_options = loads(repo.one("SELECT options_json FROM run_options WHERE run_id=?", (committed["id"],))["options_json"])
    assert failed_options["overlays"]["excluded"]
    assert failed_options["overlays"]["manual_evidence"]
    assert failed_options["overlays"]["interpretation_overrides"][eid]["argument"]["地点"] == "测试地点"
    assert repo.claim()["id"] == committed["id"]
    repo.finish_failed(committed["id"], "validation_failed", "explicit synthetic recompute failure")
    assert repo.run(committed["id"])["status"] == "failed"
    assert repo.topic(topic_id)["published_revision_id"] == base
    retry_request = {"base_revision_id": committed["target_revision_id"], "question_ids": committed["question_ids"], "mode": "retry", "include_baseline": True}
    retry = repo.create_run(topic_id, retry_request, "retry-failed-recompute", model_available=True)
    retried_options = loads(repo.one("SELECT options_json FROM run_options WHERE run_id=?", (retry["id"],))["options_json"])
    assert retried_options["overlays"] == failed_options["overlays"]
    assert retried_options["include_baseline"] is True
    assert repo.claim()["options"]["overlays"] == failed_options["overlays"]
    # A retry must not silently reconstruct overlays from the last ready
    # snapshot, which predates the user's newly committed intervention.
    assert not initial_snapshot["overlays"]["excluded"]
    assert repo.topic(topic_id)["published_revision_id"] == base


@pytest.mark.parametrize("operation", ["exclude_evidence", "set_interpretation"])
def test_preview_invalidates_every_question_that_received_changed_candidate(tmp_path, operation):
    repo, topic_id = setup_repo(tmp_path, question_count=2)
    run = create(repo, topic_id)
    assert repo.claim()["id"] == run["id"]
    questions = repo.questions(topic_id)
    results = {q["id"]: output(repo, topic_id, q, run["id"]) for q in questions}
    first_id, second_id = questions[0]["id"], questions[1]["id"]
    shared = results[first_id]["evidence"][0]
    other = deepcopy(shared)
    para = repo.paragraphs(topic_id)[0]
    quote = "通报称"
    start = para["text"].index(quote)
    other.update(id=stable("unused-candidate-regression:" + para["id"]), start_cp=start, end_cp=start + len(quote), quote=quote, fact_des="显式测试通报的引述位置")
    other["argument"] = {"时间": None, "主体": "通报", "地点": None, "内容": quote, "影响": None, "数值口径": None}
    second = results[second_id]
    second["evidence"].append(other)
    second["final"].update(decision="undetermined", conclusion="当前材料仅用于核对引述口径。", supporting_evidence_ids=[other["id"]])
    second["final"]["judgments"][0]["evidence_ids"] = [other["id"]]
    revision = repo.revision(topic_id, run["target_revision_id"])
    snapshot = build_snapshot(repo.topic(topic_id), questions, results, revision_id=revision["id"], graph_revision=revision["revision_no"], run_id=run["id"], source_paragraphs=repo.paragraphs(topic_id))
    snapshot["scope"] = {"completed_question_ids": list(results), "all_question_count": 2, "is_complete_topic": True}
    snapshot["overlays"] = {"excluded": [], "manual_evidence": [], "interpretation_overrides": {}}
    repo.publish(run["id"], snapshot, results)
    second_nodes = {node["id"] for node in snapshot["nodes"] if node["question_id"] == second_id}
    assert any(row["id"] == shared["id"] for row in snapshot["question_results"][second_id]["evidence"])
    assert not any(edge["from_node_id"] == shared["id"] and edge["to_node_id"] in second_nodes for edge in snapshot["edges"])
    op = {"op": operation, "evidence_id": shared["id"]}
    if operation == "set_interpretation":
        op["interpretation"] = interpretation()
    preview = repo.preview_change(topic_id, revision["id"], [op], "修改旧图未使用但曾提供给研判的问题候选")
    assert set(preview["affected_question_ids"]) == {first_id, second_id}
