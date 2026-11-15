"""Fresh model generation retains the real failed bill without borrowing output."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.capture_bundle_workbench_real import FORMAT, save
import scripts.score_bundle_workbench_real as scorer
from test_bundle_resume_capture_score import native_front


def fixture(tmp_path):
    previous, old_files = native_front(tmp_path, "failed", 7)
    source, own = native_front(tmp_path, "new-source", 11)
    records = [source]
    for operation in sorted(scorer.TAIL):
        record, one = native_front(tmp_path, operation, 13)
        item = one["front"]["langchain_requests"][0]
        for key in ("preflight", "finished", "settlement"):
            value = json.loads(Path(item[key]["file"]).read_text())
            value["operation"] = operation
            item[key] = save(tmp_path / operation / ("rebound-" + key + ".json"), value)
        record["operation"] = operation
        records.append(record)
        own["tail"]["requests"].append(item)
    response = {"proof": {"closed_receipt": own["front"]["closed-receipt.json"]}}
    own["front"]["provider-response.json"] = save(tmp_path / "new-source/provider-response.json", response)
    result = {"source_bundle_proof": {"response": response}}
    run = {"id": "current", "topic_id": "topic", "question_ids": ["q"], "usage": {"q": records}}
    parent = {"id": "failed", "topic_id": "topic", "question_ids": ["q"], "usage": {"q": [previous]},
        "status": "failed", "completed_at": "2026-10-21"}
    run_binding = save(tmp_path / "parent-run.json", parent)
    native_binding = save(tmp_path / "parent-native.json", old_files)
    entry = {"run_id": "failed", "run": run_binding,
        "activity": save(tmp_path / "parent-activity.json", {"usage": parent["usage"]}),
        "native_manifest": native_binding}
    rows = {"failed": {**parent, "input_manifest_sha256": "sources", "cost_parent_run_ids": []},
        "current": {**run, "input_manifest_sha256": "sources", "cost_parent_run_ids": ["failed"]}}
    closure = save(tmp_path / "original-failed-closure.json", {"format": FORMAT, "closed": True, "fixture": False,
        "real_provider": True, "status": "failed", "run_id": "failed",
        "artifacts": {"native_episode_manifest": native_binding, "run": run_binding}})
    frozen = {"format": "explicit-prelaunch-cost-parent-lineage-v1", "direct_cost_parent_run_ids": ["failed"],
        "runs": {"failed": rows["failed"]}, "failed_capture_bindings": [{"run_id": "failed", "failed_closure": closure,
            "native_episode_manifest": native_binding, "run": run_binding}], "known_usage": scorer.known_usage([previous]),
        "known_calls": 1, "ancestor_files_are_current_requests": False, "new_generation_reuses_old_responses": False,
        "new_model_calls": 0}
    freeze = {"cost_parent_run_ids": ["failed"], "prior_known_tokens": 7, "prior_actual_requests": 1,
        "new_generation_max_provider_calls": 4, "max_total_requests": 5,
        "prelaunch_cost_parent_lineage": save(tmp_path / "prelaunch.json", frozen)}
    lineage = {"format": "explicit-checkpoint-source-ancestor-files-v1", "ancestor_files_are_current_requests": False,
        "new_model_calls": 0, "ancestors": [entry],
        "database_projection": save(tmp_path / "lineage.json", {"format": "readonly-run-cost-checkpoint-lineage-v1",
            "current_run_id": "current", "runs": rows})}
    activity = {"events": [{"type": "cost_lineage_seeded", "data": {"prior_run_ids": ["failed"], "question_id": "q",
        "source_manifest_sha256": "sources", "prior_calls": 1, "prior_known_tokens": 7}}]}
    return lineage, run, activity, result, own, freeze, {"previous_records": [previous], "new_records": [source]}


@pytest.mark.parametrize("tamper", [None, "duplicate_parent", "duplicate_wire", "other_question", "cycle", "seed_tokens",
    "reset_cost", "three_new_calls", "borrowed_front", "cache", "checkpoint", "frozen_cost", "parent_success"])
def test_failed_cost_parent_is_paid_once_while_all_four_outputs_are_new(tmp_path, tamper):
    lineage, run, activity, result, native, freeze, front = fixture(tmp_path)
    if tamper == "duplicate_parent":
        lineage["ancestors"].append(deepcopy(lineage["ancestors"][0]))
    elif tamper == "duplicate_wire":
        native["tail"]["requests"][1] = deepcopy(native["tail"]["requests"][0])
    elif tamper in {"other_question", "cycle"}:
        value = json.loads(Path(lineage["database_projection"]["file"]).read_text())
        value["runs"]["failed"]["question_ids" if tamper == "other_question" else "cost_parent_run_ids"] = ["other" if tamper == "other_question" else "current"]
        lineage["database_projection"] = save(tmp_path / "tampered-projection.json", value)
    elif tamper == "seed_tokens":
        activity["events"][0]["data"]["prior_known_tokens"] = 0
    elif tamper == "reset_cost":
        front["previous_records"] = []
    elif tamper == "three_new_calls":
        native["tail"]["requests"].pop()
    elif tamper == "borrowed_front":
        result["source_bundle_proof"]["response"]["proof"]["closed_receipt"] = {"file": "old", "sha256": "a" * 64}
    elif tamper == "cache":
        native["tail"]["cached_requests"] = [{}]
    elif tamper == "checkpoint":
        result["execution_provenance"] = {"kind": "checkpoint_continuation"}
    elif tamper == "frozen_cost":
        freeze["max_total_requests"] = 4
    elif tamper == "parent_success":
        entry = lineage["ancestors"][0]
        parent = json.loads(Path(entry["run"]["file"]).read_text()); parent["status"] = "succeeded"
        entry["run"] = save(tmp_path / "successful-parent.json", parent)
    if tamper:
        with pytest.raises(ValueError):
            scorer._native_failed_cost_parents(lineage, run, activity, result, native, freeze, front, [])
    else:
        audit = scorer._native_failed_cost_parents(lineage, run, activity, result, native, freeze, front, [])
        assert audit["ancestor_known_tokens"] == 7 and audit["ancestor_actual_requests"] == 1
        assert audit["current_actual_requests"] == 4 and audit["current_known_tokens"] == 50
        assert audit["whole_actual_requests"] == 5 and audit["whole_known_tokens"] == 57
        assert audit["old_model_outputs_reused"] is False
