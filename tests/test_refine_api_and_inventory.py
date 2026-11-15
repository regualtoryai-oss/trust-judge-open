"""Isolated synthetic API/worker/CAS/export and controlled inventory checks."""
import asyncio
from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.domain.program_inventory import inventory_proof, strip_verified_inventory_statements, validate_published_inventory
from app.domain.snapshot import build_snapshot
from app.domain.scope_batches import deduplicate_verified_facts
from app.domain.strict_verification import build_scope_request, resolve_scope_verification
from app.main import create_app as _create_app
from app.worker import Worker
from app.workflows.analysis import _validate_analysis
from test_repository import setup_repo, create, output
from test_workflow import FakeProvider


def create_app(settings, **kwargs):
    """This module's explicit old scope double does not implement v18 purpose."""
    application = _create_app(settings, **kwargs)
    original_config = application.state.repo.config
    application.state.repo.config = lambda: {**original_config(), "premise_bank_enabled": False}
    return application


class StrictProvider(FakeProvider):
    test_only_legacy_precision = False

    async def verify_evidence_scope(self, question, evidence):
        self.calls.append(("scope", deepcopy(question), deepcopy(evidence)))
        payload, _, views = build_scope_request(question, evidence)
        decisions = {f"C{index+1}": {"fact_entailment": "entailed", "target_relation": "direct_answer",
            "target_question_text": question["question"], "qualifier_dimension": None,
            "requires_unstated_bridge": False, "target_fact_present_but_annotation_incomplete": False,
            "reason": "Explicit synthetic test labels; no model-quality claim."} for index in range(len(views))}
        return resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"], "decisions": decisions}, question, evidence)


async def ready(tmp_path):
    repo, tid = setup_repo(tmp_path)
    provider = StrictProvider()
    initial = create(repo, tid)
    await Worker(repo, provider).execute(repo.claim())
    assert repo.run(initial["id"])["status"] == "succeeded", repo.run(initial["id"])["error"]
    return repo, tid, provider, initial


async def test_quality_refine_api_new_task_reuses_complete_candidate_not_paid_scan(tmp_path):
    repo, tid, provider, origin = await ready(tmp_path)
    old_snapshot = repo.snapshot(tid, origin["target_revision_id"])
    qid = repo.questions(tid)[0]["id"]
    client = TestClient(create_app(repo.settings, provider=provider))
    body = {"mode": "quality-refine", "question_ids": [qid], "base_revision_id": repo.topic(tid)["head_revision_id"],
            "source_revision_id": origin["target_revision_id"], "include_baseline": False}
    provider.calls.clear()
    response = client.post(f"/api/topics/{tid}/runs", json=body, headers={"Idempotency-Key": "new-quality-job"})
    assert response.status_code == 202, response.json()
    job = response.json()
    assert job["run_id"] != origin["id"] and job["mode"] == "quality-refine"
    assert job["derived_inputs"][0]["source_run_id"] == origin["id"]
    again = client.post(f"/api/topics/{tid}/runs", json=body, headers={"Idempotency-Key": "new-quality-job"})
    assert again.json()["run_id"] == job["run_id"]
    await Worker(repo, provider).execute(repo.claim())
    result = repo.run(job["run_id"])
    assert result["status"] == "succeeded", result["error"]
    assert not any(call[0] in {"extract", "select_evidence"} for call in provider.calls)
    assert sum(call[0] == "scope" for call in provider.calls) == 1
    new = repo.snapshot(tid, result["target_revision_id"])["question_results"][qid]
    assert new["candidate_evidence"] == old_snapshot["question_results"][qid]["candidate_evidence"]
    assert new["candidate_import_provenance"]["new_source_scan_performed"] is False
    events = repo.events(job["run_id"])
    assert any(row["type"] == "candidate_artifact_imported" for row in events)
    assert not any(row["type"] in {"source_coverage_planned", "source_coverage_progress", "checkpoint_reused"} for row in events)
    assert repo.snapshot(tid, origin["target_revision_id"]) == old_snapshot


async def test_queued_refine_pause_resume_preserves_internal_input_without_cross_version_flags(tmp_path):
    repo, tid, provider, origin = await ready(tmp_path)
    qid = repo.questions(tid)[0]["id"]
    request = {"mode": "quality-refine", "question_ids": [qid], "base_revision_id": repo.topic(tid)["head_revision_id"],
               "source_revision_id": origin["target_revision_id"], "include_baseline": False}
    job = repo.create_run(tid, request, "quality-paused", model_available=True)
    repo.pause(job["run_id"])
    continued = repo.resume(job["run_id"], "same-version-resume", model_available=True)
    assert continued["mode"] == "quality-refine"
    assert continued["derived_inputs"] == job["derived_inputs"]
    assert continued["resume_mode"] == "start_unstarted"
    provider.calls.clear()
    await Worker(repo, provider).execute(repo.claim())
    assert repo.run(continued["run_id"])["status"] == "succeeded"
    assert not any(call[0] == "extract" for call in provider.calls)


@pytest.mark.parametrize("block_import_checkpoint", [False, True], ids=["normal", "blocked-checkpoint-write"])
async def test_started_refine_resume_counts_only_executed_graph_stages(tmp_path, monkeypatch, block_import_checkpoint):
    repo, tid, provider, origin = await ready(tmp_path)
    qid = repo.questions(tid)[0]["id"]
    job = repo.create_run(tid, {"mode": "quality-refine", "question_ids": [qid],
        "base_revision_id": repo.topic(tid)["head_revision_id"], "source_revision_id": origin["target_revision_id"]},
        "started-quality-pause", model_available=True)
    original = provider.verify_evidence_scope
    scope_entered = asyncio.Event()
    write_started, release_write = asyncio.Event(), asyncio.Event()
    source_thread = job["run_id"] + ":" + qid
    native_aput = AsyncSqliteSaver.aput

    async def gated_aput(saver, config, checkpoint, metadata, new_versions):
        if (block_import_checkpoint and config["configurable"]["thread_id"] == source_thread
                and "candidate_import_provenance" in checkpoint["channel_values"] and not write_started.is_set()):
            # A real SQLite write is held pending. The next paid scope request
            # must not enter until this completed import is durable.
            write_started.set()
            await release_write.wait()
        return await native_aput(saver, config, checkpoint, metadata, new_versions)

    monkeypatch.setattr(AsyncSqliteSaver, "aput", gated_aput)

    async def pause_after_scope(question, evidence):
        scope_entered.set()
        result = await original(question, evidence)
        repo.pause(job["run_id"])
        return result

    provider.verify_evidence_scope = pause_after_scope
    execution = asyncio.create_task(Worker(repo, provider).execute(repo.claim()))
    try:
        if block_import_checkpoint:
            await asyncio.wait_for(write_started.wait(), 5)
            # Observe an actual model-entry event while the writer is blocked,
            # rather than sleeping in the provider and assuming it is durable.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(scope_entered.wait(), 0.05)
    finally:
        release_write.set()
        await execution
    assert repo.run(job["run_id"])["view_status"] == "paused"
    async with AsyncSqliteSaver.from_conn_string(str(repo.settings.checkpoint_path)) as saver:
        stored = await saver.aget_tuple({"configurable": {"thread_id": source_thread}})
    assert "candidate_import_provenance" in stored.checkpoint["channel_values"]
    provider.verify_evidence_scope = original
    provider.calls.clear()
    continued = repo.resume(job["run_id"], "started-quality-continue", model_available=True)
    await Worker(repo, provider).execute(repo.claim())
    assert repo.run(continued["run_id"])["status"] == "succeeded"
    activity = repo.activity(continued["run_id"])
    assert activity["progress"]["completed_count"] == activity["progress"]["total_count"] == 7
    assert not {"retrieve", "extract", "verify"} & {row["stage"] for row in activity["progress"]["completed_stages"]}
    assert not any(call[0] == "extract" for call in provider.calls)
    reused = next(row["data"] for row in activity["events"] if row["type"] == "checkpoint_reused")
    assert reused["reused_nodes"] == ["candidate_import"]
    assert not any(row["type"] == "node_started" and row["data"].get("node") == "candidate_import"
        for row in activity["events"])


async def test_refine_cancel_cannot_publish_over_ready_source(tmp_path):
    repo, tid, provider, origin = await ready(tmp_path)
    qid = repo.questions(tid)[0]["id"]
    job = repo.create_run(tid, {"mode": "quality-refine", "question_ids": [qid],
        "base_revision_id": repo.topic(tid)["head_revision_id"], "source_revision_id": origin["target_revision_id"]},
        "cancel-quality", model_available=True)
    original = provider.verify_evidence_scope
    async def cancel_after_paid_result(question, evidence):
        result = await original(question, evidence)
        repo.cancel(job["run_id"])
        return result
    provider.verify_evidence_scope = cancel_after_paid_result
    await Worker(repo, provider).execute(repo.claim())
    assert repo.run(job["run_id"])["status"] == "cancelled"
    assert repo.topic(tid)["published_revision_id"] == origin["target_revision_id"]


def test_api_requires_only_new_refine_mode_to_accept_source_version(tmp_path):
    repo, tid = setup_repo(tmp_path)
    client = TestClient(create_app(repo.settings, provider=StrictProvider()))
    base = {"base_revision_id": None, "question_ids": [repo.questions(tid)[0]["id"]]}
    for extra in ({"mode": "quality-refine"}, {"mode": "initial", "source_revision_id": str(uuid4())},
                  {"mode": "initial", "test_only_legacy_precision": True}):
        result = client.post(f"/api/topics/{tid}/runs", json={**base, **extra}, headers={"Idempotency-Key": str(uuid4())})
        assert result.status_code == 422
    assert not repo.runs(tid)


def test_unverified_manual_fact_kept_in_workbench_but_formal_export_api_blocked(tmp_path):
    repo, tid = setup_repo(tmp_path)
    q = repo.questions(tid)[0]
    job = create(repo, tid)
    repo.claim()
    result = output(repo, tid, q, job["id"])
    row = result["evidence"][0]
    row.update({"interpretation_origin": "human", "scope_verified": False, "manual_review_required": True,
                "fact_des": "人工声称4人，原文仍记载3人；这项解释未通过严审。"})
    result["strict_scope_review"] = {"applied": True, "unresolved_count": 1, "manual_unverified_count": 1}
    revision = repo.revision(tid, job["target_revision_id"])
    snapshot = build_snapshot(repo.topic(tid), [q], {q["id"]: result}, revision_id=revision["id"],
        graph_revision=revision["revision_no"], run_id=job["id"])
    snapshot["overlays"] = {"excluded": [], "manual_evidence": [], "interpretation_overrides": {}}
    snapshot["scope"] = {"completed_question_ids": [q["id"]], "all_question_count": 1, "is_complete_topic": True}
    repo.publish(job["id"], snapshot, {q["id"]: result})
    client = TestClient(create_app(repo.settings, provider=StrictProvider()))
    readiness = client.get(f"/api/topics/{tid}/revisions/{revision['id']}/export-readiness")
    assert readiness.status_code == 200 and readiness.json()["can_export"] is False
    assert any(row["code"] == "export_manual_fact_unverified" for row in readiness.json()["blockers"])
    blocked = client.post(f"/api/topics/{tid}/exports", json={"revision_id": revision["id"]}, headers={"Idempotency-Key": "manual-export"})
    assert blocked.status_code == 409 and blocked.json()["detail"]["code"] == "export_manual_fact_unverified"
    preview = repo.submission_preview(tid, revision["id"])
    assert preview["formal_submission"] is False and preview["manual_review_warnings"][0]["manual_review_required"]
    assert repo.snapshot(tid, revision["id"])["evidence"][0]["fact_des"] == row["fact_des"]


async def test_program_inventory_only_allows_fixed_count_statements_not_bare_numbers(tmp_path):
    repo, tid, provider, initial = await ready(tmp_path)
    q = repo.questions(tid)[0]
    result = repo.snapshot(tid, initial["target_revision_id"])["question_results"][q["id"]]
    evidence = result["evidence"]
    proof = inventory_proof(evidence, result["candidate_evidence"], repo.paragraphs(tid))
    base = {"decision": "asserted", "conclusion": proof["allowed_statements"][0], "judgments": [],
            "supporting_evidence_ids": [row["id"] for row in evidence], "conflicting_evidence_ids": [], "conflicts": []}
    accepted = _validate_analysis(base, evidence, q["id"], "arbitrate", arbitration=True,
        source_inventory_resolver=lambda ids: proof)
    assert accepted["program_source_inventory"] == proof
    assert strip_verified_inventory_statements(base["conclusion"], proof, set(proof["cited_evidence_ids"])) == "[程序来源盘点]"
    with pytest.raises(ValueError):
        _validate_analysis({**base, "conclusion": "已经核定1人失联。"}, evidence, q["id"], "arbitrate", arbitration=True,
            source_inventory_resolver=lambda ids: proof)
    tampered = deepcopy(result)
    tampered["final"] = accepted
    tampered["final"]["program_source_inventory"]["active_source_documents"] = 99
    with pytest.raises(ValueError, match="invalid_program_source_inventory"):
        validate_published_inventory(tampered, repo.paragraphs(tid))


def test_exact_annotation_dedup_only_within_same_paragraph_and_only_verified_automatic_facts():
    base = {"id": "long", "paragraph_id": "p1", "raw_doc_id": "d1", "raw_para_id": "r1", "quote": "有关部门通报，设备已经停机。",
        "fact_des": "设备已经停机。", "argument": {"主体": "设备"}, "entities": ["设备", "部门"],
        "scope_verified": True, "interpretation_origin": "model", "fact_view_id": "F-long"}
    short = {**deepcopy(base), "id": "short", "quote": "设备已经停机。", "fact_view_id": "F-short", "entities": ["部门", "设备"]}
    other_doc = {**deepcopy(short), "id": "other-doc", "paragraph_id": "p2", "raw_doc_id": "d2"}
    other_fact = {**deepcopy(short), "id": "different-fact", "fact_des": "有人声称设备已经停机。"}
    manual = {**deepcopy(short), "id": "manual", "interpretation_origin": "human"}
    unchecked = {**deepcopy(short), "id": "unchecked", "scope_verified": False}
    originals = [base, short, other_doc, other_fact, manual, unchecked]
    frozen = deepcopy(originals)
    rows, audit = deduplicate_verified_facts(originals)
    assert {row["id"] for row in rows} == {"short", "other-doc", "different-fact", "manual", "unchecked"}
    assert audit == [{"fact_evidence_id": "long", "fact_view_id": "F-long", "representative_evidence_id": "short",
        "basis": "same_paragraph_exact_annotation_shortest_independently_verified_quote"}]
    assert originals == frozen
