"""Synthetic immutable pause/cost ancestry; no network or reference material."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.capture_bundle_workbench_real import capture_episode, capture_resume_lineage, save
import scripts.score_bundle_workbench_real as scorer


def native_front(tmp_path, name, count):
    folder = tmp_path / name
    counts = {"input_tokens": count - 2, "output_tokens": 2, "total_tokens": count}
    wire = {"model": "deepseek-v4-pro", "thinking": {"type": "disabled"}, "max_tokens": 8192,
        "tools": [{"function": {"strict": True}}]}
    bindings = {"wire": save(folder / "wire.json", wire), "actual_wire": save(folder / "actual.json", wire)}
    bindings.update(preflight=save(folder / "pre.json", {"operation": "strict_selection", "admitted": True,
        "wire": bindings["wire"], "actual_wire": bindings["actual_wire"], "reserved_output_tokens": 8192}),
        finished=save(folder / "finished.json", {"operation": "strict_selection", "request_started": True,
            "outcome": "invalid_output" if name == "failed" else "success", "unknown_usage_requests": 0}),
        settlement=save(folder / "settled.json", {**counts, "operation": "strict_selection", "usage_known": True,
            "actual_request_started": True}), response=save(folder / "response.json", {"usage": counts}))
    record = {**counts, "operation": "strict_selection", "usage_source": "provider"}
    receipt = save(folder / "closed.json", {"usage_records": [record], "status": "failed" if name == "failed" else "succeeded", "known_usage_complete": True})
    files = {"episode_directory": str(folder), "front": {"closed-receipt.json": receipt, "langchain_requests": [bindings], "budget_forecasts": []},
        "tail": {"requests": [], "budget_forecasts": []}}
    return record, files


def lineage_fixture(tmp_path):
    failed, failed_files = native_front(tmp_path, "failed", 7)
    source, source_files = native_front(tmp_path, "source", 11)
    provenance = {"kind": "checkpoint_continuation", "source_thread_id": "source:q", "checkpoint_id": "checkpoint",
        "reused_nodes": ["source_bundle_selection"], "source_evidence_snapshot_id": None}
    response = {"proof": {"closed_receipt": source_files["front"]["closed-receipt.json"]}}
    source_files["front"]["provider-response.json"] = save(tmp_path / "source/provider.json", response)
    result = {"execution_provenance": provenance, "source_bundle_proof": {"response": response}}
    current = {"id": "current", "question_ids": ["q"], "topic_id": "topic", "usage": {"q": []}, "resumed_from_run_id": "source"}
    rows, entries = {}, []
    for rid, record, files, status in [("failed", failed, failed_files, "failed"), ("source", source, source_files, "cancelled")]:
        run = {"id": rid, "topic_id": "topic", "question_ids": ["q"], "status": status,
            "completed_at": "2026-10-21", "usage": {"q": [record]}, "error": {"code": "paused"} if rid == "source" else {"code": "invalid"}}
        entries.append({"run_id": rid, "run": save(tmp_path / f"{rid}/run.json", run),
            "activity": save(tmp_path / f"{rid}/activity.json", {"usage": run["usage"]}),
            "native_manifest": save(tmp_path / f"{rid}/native.json", files)})
        rows[rid] = {"id": rid, "topic_id": "topic", "question_ids": ["q"], "input_manifest_sha256": "manifest",
            "model_config_sha256": "config", "usage": run["usage"], "cost_parent_run_ids": ["failed"] if rid == "source" else []}
    rows["current"] = {**current, "input_manifest_sha256": "manifest", "model_config_sha256": "config",
        "resume_from_run_id": "source", "resume_checkpoint_sources": {"q": "source"}, "cost_parent_run_ids": ["failed"]}
    lineage = {"format": "explicit-checkpoint-source-ancestor-files-v1", "ancestor_files_are_current_requests": False,
        "new_model_calls": 0, "execution_provenance": provenance, "ancestors": entries,
        "checkpoint_identity": save(tmp_path / "checkpoint-identity.json", {"format": "readonly-original-checkpoint-identity-v1",
            "thread_id": "source:q", "checkpoint_ns": "", "checkpoint_id": "checkpoint", "serialized_checkpoint_state_read": False}),
        "database_projection": save(tmp_path / "projection.json", {"format": "readonly-run-cost-checkpoint-lineage-v1",
            "current_run_id": "current", "runs": rows})}
    activity = {"events": [{"type": "checkpoint_reused", "data": {"question_id": "q", **provenance}},
        {"type": "cost_lineage_seeded", "data": {"prior_run_ids": ["failed", "source"], "question_id": "q",
            "source_manifest_sha256": "manifest", "prior_calls": 2, "prior_known_tokens": 18}}]}
    native = {"front": {"budget_forecasts": []}, "tail": {"budget_forecasts": []}}
    return lineage, current, activity, result, native


@pytest.mark.parametrize("tamper", [None, "missing", "duplicate", "other_question", "thread", "reused_tail", "receipt", "event", "wire_reuse", "cycle"])
def test_resume_distinguishes_original_front_from_current_tail_and_rejects_unproven_ancestors(tmp_path, tamper):
    lineage, run, activity, result, native = lineage_fixture(tmp_path)
    if tamper == "missing":
        lineage["ancestors"].pop()
    elif tamper == "duplicate":
        lineage["ancestors"][1] = deepcopy(lineage["ancestors"][0])
    elif tamper == "other_question":
        value = json.loads(Path(lineage["database_projection"]["file"]).read_text())
        value["runs"]["failed"]["question_ids"] = ["another"]
        lineage["database_projection"] = save(tmp_path / "wrong-question.json", value)
    elif tamper == "thread":
        result["execution_provenance"]["source_thread_id"] = "failed:q"
    elif tamper == "reused_tail":
        result["execution_provenance"]["reused_nodes"].append("authority_analyst")
    elif tamper == "receipt":
        result["source_bundle_proof"]["response"]["proof"]["closed_receipt"] = {"file": "fake", "sha256": "f" * 64}
    elif tamper == "event":
        activity["events"].pop()
    elif tamper == "wire_reuse":
        lineage["ancestors"][1]["native_manifest"] = lineage["ancestors"][0]["native_manifest"]
    elif tamper == "cycle":
        value = json.loads(Path(lineage["database_projection"]["file"]).read_text())
        value["runs"]["failed"]["cost_parent_run_ids"] = ["current"]
        lineage["database_projection"] = save(tmp_path / "cycle.json", value)
    if tamper is not None:
        with pytest.raises(ValueError):
            scorer._resume_source_files(lineage, run, activity, result, native, [])
    else:
        checked = scorer._resume_source_files(lineage, run, activity, result, native, [])
        assert checked["audit"]["ancestor_known_tokens"] == 18
        assert checked["audit"]["ancestor_calls"] == 2
        assert checked["audit"]["ancestor_files_are_current_requests"] is False
        assert checked["native"]["front"] == native["front"]
        assert checked["native"]["source_front"]["closed-receipt.json"] == result["source_bundle_proof"]["response"]["proof"]["closed_receipt"]


@pytest.mark.asyncio
async def test_actual_worker_checkpoint_files_are_audited_without_paid_calls(tmp_path, monkeypatch, model, infer):
    from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
    from app.repository import Repository
    from app.worker import Worker
    from test_context_workbench import _import
    from test_lc_context_case import raw_case
    from test_paragraph_quality_workbench import settings_for, patch_quality_http
    settings = settings_for(tmp_path / "runtime", model, compact=True).model_copy(update={"strict_evidence_max_requests": 5})
    repo = Repository(settings)
    topic = _import(repo, raw_case(["原始程序记录仍待最终确认。", "另一份记录提供相反观点。"])); qid = repo.questions(topic)[0]["id"]
    first = LCBundleWorkbenchProvider(settings); observed_first = patch_quality_http(monkeypatch, first)
    task = repo.create_run(topic, {"question_ids": [qid], "base_revision_id": None, "include_baseline": False, "mode": "initial"}, "first", model_available=True)
    event = repo.event
    def pause(rid, kind, data):
        event(rid, kind, data)
        if rid == task["run_id"] and kind == "node_started" and data.get("node") == "freeze":
            repo.pause(rid)
    monkeypatch.setattr(repo, "event", pause)
    await Worker(repo, first).execute(repo.claim())
    resumed = repo.resume(task["run_id"], "resume", model_available=True)
    second = LCBundleWorkbenchProvider(settings); observed_second = patch_quality_http(monkeypatch, second)
    await Worker(repo, second).execute(repo.claim())
    run = repo.run(resumed["run_id"])
    assert run["status"] == "succeeded", run["error"]
    result = repo.snapshot(topic, run["target_revision_id"])["question_results"][qid]
    getter = lambda path: repo.activity(path.split("/")[2]) if path.endswith("/activity") else repo.run(path.split("/")[2])
    captured = capture_resume_lineage(settings.database_path, run, result, tmp_path / "closed", {task["run_id"]: first.directory}, getter)
    checked = scorer._resume_source_files(captured, run, repo.activity(run["id"]), result, capture_episode(second.directory), [])
    front = {"previous_records": checked["previous_records"], "new_records": checked["source_records"]}
    from app.workflows.analysis import _project_question
    audit = scorer._tail(result, _project_question(repo.questions(topic)[0]),
        checked["native"], run["usage"][qid], front, {"engineering_episode_stop": 80000, "max_total_requests": 5}, [])
    assert audit["whole_episode_known_tokens"] == 480
    assert audit["whole_episode_actual_requests"] == 4
    assert len(observed_first) == 1 and len(observed_second) == 3
    assert audit["actual_tail_requests"] == 3


from test_lc_local_rank_integration import model, infer  # noqa: E402,F401 - only synthetic local fixture


def test_whole_chain_forecasts_stay_bound_to_separate_original_ancestor_file(tmp_path):
    from app.agents.episode_budget import POLICY, plan_source_episode, plan_tail, planning_sources
    from app.agents.strict_transport import LIMITS
    from app.domain.lc_bundle_case import prepare_lc_case
    from test_source_bundle_tool_bridge import raw_case
    raw = raw_case(); prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    args = {"known_tokens": 18, "known_calls": 2, "threshold": 45000, "max_calls": 8,
        "model_name": "deepseek-v4-pro", "output_limits": {**LIMITS, "strict_selection": 2048},
        "compact_analyses": True, "counter": lambda strings: [10] * len(strings)}
    evidence = planning_sources(prepared, counter=args["counter"])
    analyses = [{"role": role, "evidence_snapshot_id": "synthetic", "decision": "undetermined", "conclusion": "待核验。",
        "judgments": [], "supporting_evidence_ids": [], "conflicting_evidence_ids": []}
        for role in ("authority_analyst", "cross_validation_analyst")]
    source = save(tmp_path / "original-source/budget-before-first-paid.json", plan_source_episode(raw["question"], prepared, **args))
    selected = save(tmp_path / "new-tail/budget-selected-sources.json", plan_tail(raw["question"], evidence, **args))
    actual = save(tmp_path / "new-tail/budget-actual-analyses.json", plan_tail(raw["question"], evidence, analyses=analyses, **args))
    native = {"front": {}, "tail": {"budget_forecasts": [selected, actual]}, "ancestor_budget_forecasts": [source]}
    metadata = {"source_bundle_config": {"whole_chain_planning": POLICY}}
    freeze = {"engineering_episode_stop": 45000, "max_total_requests": 8}
    audits = scorer._budget_forecasts(native, metadata, freeze, [])
    assert {a["phase"] for a in audits} == {"before_first_paid_request", "selected_sources", "actual_analyses"}
    assert [a for a in audits if a["phase"] == "before_first_paid_request"][0]["bound_receipt"] == source
    with pytest.raises(ValueError, match="forecast_binding_changed"):
        scorer._budget_forecasts({**native, "ancestor_budget_forecasts": [source, source]}, metadata, freeze, [])


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("tamper", [None, "response", "new_tokens", "missing_event", "claim_completed", "manifest", "config"])
def test_sealed_paid_front_owner_is_distinct_from_initial_or_completed_pause_checkpoint(tmp_path, completed, tamper):
    lineage, run, activity, result, native = lineage_fixture(tmp_path)
    projection = json.loads(Path(lineage["database_projection"]["file"]).read_text())
    entries = {e["run_id"]: e for e in lineage["ancestors"]}
    origin = json.loads(Path(entries["source"]["run"]["file"]).read_text()); origin["status"] = "failed"; origin["error"] = {"code": "validation_failed"}
    entries["source"]["run"] = save(tmp_path / "origin-failed-run.json", origin)
    paused = {"id": "checkpoint", "topic_id": "topic", "question_ids": ["q"], "status": "cancelled", "error": {"code": "paused"},
        "completed_at": "2026-10-21", "usage": {}}
    paused_activity = {"usage": {}, "events": []}
    archive = {"app/workflows/bundle_workbench.py": save(tmp_path / "recovery-source.py", {"synthetic": True})}
    native["tail"]["source-bundle-episode.json"] = save(tmp_path / "current-metadata.json", {"source_archives": archive})
    paused_files = {"episode_directory": str(tmp_path / "checkpoint"), "front": {}, "tail": {"requests": [], "budget_forecasts": [],
        "source-bundle-episode.json": save(tmp_path / "paused-metadata.json", {"source_archives": archive})}}
    result["execution_provenance"]["source_thread_id"] = "checkpoint:q"
    result["execution_provenance"]["reused_nodes"] = ["source_bundle_selection"] if completed else []
    result["evidence"] = []
    run["resumed_from_run_id"] = "checkpoint"
    question, documents, paragraphs = {"id": "q", "question": "原始问题。"}, [], []
    source_files = json.loads(Path(entries["source"]["native_manifest"]["file"]).read_text())
    receipt = {"format": "lc-sealed-front-recovery-v1", "origin_run_id": "source", "origin_topic_id": "topic", "origin_question_id": "q",
        "recovery_run_id": "checkpoint" if completed else "current", "source_manifest_sha256": "manifest",
        "response": source_files["front"]["provider-response.json"], "origin_native_outer_episode": "source",
        "original_front_proof": source_files["front"]["closed-receipt.json"], "current_recovery_source_archives": archive,
        "source_proof_replay_verified": True, "current_canonical_projection_verified": True, "not_a_new_native_front": True,
        "original_response_modified": False, "fields_modified": False, "new_model_calls": 0, "new_tokens": 0,
        "prior_known_tokens": 18, "evidence_count": 0, "current_question_sha256": scorer.canonical_digest(question),
        "current_sources_sha256": scorer.canonical_digest({"documents": documents, "paragraphs": paragraphs})}
    config = {"origin_run_id": "source", "response": receipt["response"], "source_model_calls": 0, "original_response_modified": False}
    if tamper == "response": receipt["response"] = {"file": "different", "sha256": "a" * 64}
    elif tamper == "new_tokens": receipt["new_tokens"] = 1
    elif tamper == "manifest": receipt["source_manifest_sha256"] = "other"
    elif tamper == "config": config["origin_run_id"] = "failed"
    receipt_binding = save(tmp_path / "sealed-front-recovery.json", receipt)
    target_files = paused_files if completed else native
    target_files["front"]["sealed-front-recovery.json"] = receipt_binding
    event = {"type": "sealed_front_recovery_completed", "data": {"origin_run_id": "source", "recovery_receipt": receipt_binding,
        "actual_request_started": False, "new_model_calls": 0, "new_tokens": 0}}
    if tamper != "missing_event":
        (paused_activity if completed else activity).setdefault("events", []).append(event)
    if tamper == "claim_completed":
        result["execution_provenance"]["reused_nodes"] = [] if completed else ["source_bundle_selection"]
    projection["runs"]["checkpoint"] = {"id": "checkpoint", "topic_id": "topic", "question_ids": ["q"], "usage": {},
        "input_manifest_sha256": "manifest", "model_config_sha256": "config", "cost_parent_run_ids": ["source"]}
    projection["runs"]["current"].update(resume_from_run_id="checkpoint", resume_checkpoint_sources={"q": "checkpoint"}, sealed_front_recovery=config)
    entries["checkpoint"] = {"run_id": "checkpoint", "run": save(tmp_path / "paused-run.json", paused),
        "activity": save(tmp_path / "paused-activity.json", paused_activity), "native_manifest": save(tmp_path / "paused-native.json", paused_files)}
    lineage["ancestors"] = list(entries.values())
    lineage["database_projection"] = save(tmp_path / "sealed-projection.json", projection)
    lineage["checkpoint_identity"] = save(tmp_path / "sealed-checkpoint.json", {"format": "readonly-original-checkpoint-identity-v1",
        "thread_id": "checkpoint:q", "checkpoint_ns": "", "checkpoint_id": "checkpoint", "serialized_checkpoint_state_read": False})
    activity["events"][0]["data"] = {"question_id": "q", **result["execution_provenance"]}
    activity["events"][1]["data"]["prior_run_ids"] = ["checkpoint", "failed", "source"]
    if tamper is not None:
        with pytest.raises(ValueError):
            scorer._resume_source_files(lineage, run, activity, result, native, [], question=question, documents=documents, paragraphs=paragraphs)
    else:
        checked = scorer._resume_source_files(lineage, run, activity, result, native, [], question=question, documents=documents, paragraphs=paragraphs)
        assert checked["audit"]["source_run_id"] == "source"
        assert checked["audit"]["checkpoint_owner_run_id"] == "checkpoint"
        assert checked["audit"]["completed_source_checkpoint_reused"] is completed
        assert checked["audit"]["ancestor_known_tokens"] == 18
        assert checked["audit"]["sealed_source_recovery"]["new_model_calls"] == 0
