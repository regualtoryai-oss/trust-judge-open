"""Verify closed UI bundle/native/publication/download bytes BEFORE reference IO.

Zero model calls. A-ID E/S/C diagnostics use actual exported rows and the same
offline scoring engine as prior real episodes. Semantic B remains unknown.
"""
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL, SOURCE_CONTEXT_PROTOCOL, competition_files
from app.workflows.bundle_workbench import PROMPT_VERSION, validate_result
from scripts.audit_dsh_closed_episode import bound
from scripts.capture_bundle_workbench_real import FORMAT, verify_ui_start
from scripts.final_evidence_scoring import normalize_legacy
from scripts.offline_reference_tools import load_reference_tools
from scripts.score_real_quality_episode import read_bound, score_normalized
from scripts.score_workbench_real_episode import (
    canonical_digest, descriptor, known_usage, remap_aliases, verify_legacy_sources,
    audit_cached_requests,
    verify_scoped_relations, verify_source_rows, verify_syntax_recovery,
    verify_tail_schema, verify_visible_tool_wire, wire_payload,
)
from scripts.workbench_source_archive import SourceArchive, relative_source


ROOT = Path(__file__).resolve().parents[1]
TAIL = {"authority_analyst", "cross_validation_analyst", "arbitrate"}
LC_PROOF_FORMAT = "langchain-source-bundle-workbench-proof-v1"
LC_PROMPT_VERSION = "trusted-analysis-langchain-source-bundles-v1"


def _counter(records):
    known_usage(records)
    return Counter((r["input_tokens"], r["output_tokens"], r["total_tokens"]) for r in records)


def _lines(binding, guards):
    raw, item = bound(binding["file"], binding["sha256"]); guards.append(item)
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def _proof_guards(value, guards):
    if isinstance(value, dict):
        if isinstance(value.get("file"), str) and isinstance(value.get("sha256"), str):
            name = Path(value["file"]).name.lower()
            if name in {".env", "evidence_samples.json", "conclusions.json"}:
                raise ValueError("runtime_proof_cannot_reference_gold_or_credentials")
            guards.append(bound(value["file"], value["sha256"])[1])
        else:
            for child in value.values(): _proof_guards(child, guards)
    elif isinstance(value, list):
        for child in value: _proof_guards(child, guards)


def _front_usage(result, run_usage, guards):
    response = result["source_bundle_proof"]["response"]
    proof, report = response["proof"], response["usage_report"]
    closed = descriptor(proof["closed_receipt"], guards)
    if closed.get("execution_mode") != "workbench" or report.get("real_api_requests") != report["new_request_count"] or report.get("simulated_usage_not_bill") is not False:
        raise ValueError("real_bundle_scoring_forbids_fixture_native_requests")
    if proof.get("format") == LC_PROOF_FORMAT:
        records = closed.get("usage_records", [])
        if len(records) != 1 or records[0].get("operation") != "strict_selection":
            raise ValueError("real_langchain_front_requires_one_settled_provider_call")
    else:
        rows = _lines(proof["usage_ledger"], guards)
        records = [{**r["provider_usage"], "usage_source": "provider"} for r in rows if r.get("status") == "usage_settled"]
    actual = [r for r in run_usage if r.get("operation") not in TAIL]
    if _counter(actual) != _counter(records): raise ValueError("front_actual_usage_must_equal_workbench_records")
    previous = descriptor(proof["previous_usage"], guards)["records"] if proof.get("previous_usage") else []
    return {"new_records": records, "previous_records": previous, "report": report, "closed": closed}


def _usage_signature(records):
    known_usage(records)
    return Counter((r.get("operation"), r["input_tokens"], r["output_tokens"], r["total_tokens"]) for r in records)


def _remaining_usage_records(combined, removed):
    """Subtract the source call once while preserving other calls by its owner."""
    remaining, previous = _usage_signature(removed), []
    for record in combined:
        key = next(iter(_usage_signature([record])))
        if remaining[key]:
            remaining[key] -= 1
        else:
            previous.append(record)
    if any(remaining.values()):
        raise ValueError("cached_source_receipt_not_in_actual_ancestor_usage")
    return previous


def _ancestor_native_usage(native, guards, used_requests):
    """Audit billable ancestor wires even if their business output was invalid."""
    items = [*native.get("tail", {}).get("requests", []),
             *native.get("front", {}).get("langchain_requests", [])]
    records = []
    for item in items:
        identity = (str(Path(item["actual-wire" if "actual-wire" in item else "actual_wire"]["file"]).resolve()),
                    item["actual-wire" if "actual-wire" in item else "actual_wire"]["sha256"])
        if identity in used_requests:
            raise ValueError("resume_duplicate_actual_ancestor_request")
        used_requests.add(identity)
        pre, wire, actual, finish, settlement, response = (descriptor(item[key], guards) for key in
            ("preflight", "wire", "actual-wire" if "actual-wire" in item else "actual_wire", "finished", "settlement", "response"))
        op = pre.get("operation")
        tools = wire.get("tools", [])
        if (actual != wire or pre.get("admitted") is not True or finish.get("request_started") is not True
                or finish.get("operation") != op or settlement.get("operation") != op
                or settlement.get("actual_request_started") is not True or settlement.get("usage_known") is not True
                or finish.get("unknown_usage_requests") != 0 or pre.get("wire") != item["wire"]
                or pre.get("actual_wire") != item["actual-wire" if "actual-wire" in item else "actual_wire"]
                or wire.get("model") != "deepseek-v4-pro" or wire.get("thinking") != {"type": "disabled"}
                or len(tools) != 1 or tools[0].get("function", {}).get("strict") is not True
                or wire.get("tool_choice", {}).get("function", {}).get("name") != tools[0]["function"].get("name")
                or wire.get("max_tokens") != pre.get("reserved_output_tokens")):
            raise ValueError("resume_ancestor_actual_wire_or_settlement_changed")
        counts = {key: settlement.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")}
        record = {**counts, "operation": op, "usage_source": "provider"}
        known_usage([record])
        if any(response.get("usage", {}).get(key) != value for key, value in counts.items()):
            raise ValueError("resume_ancestor_provider_usage_changed")
        records.append(record)
    return records


def _native_failed_cost_parents(lineage, run, activity, result, native, freeze, front, guards):
    """A fresh four-call generation may inherit fees, never old model output."""
    if (lineage.get("format") != "explicit-checkpoint-source-ancestor-files-v1"
            or lineage.get("ancestor_files_are_current_requests") is not False or lineage.get("new_model_calls") != 0
            or lineage.get("execution_provenance") or "checkpoint_identity" in lineage
            or run.get("resumed_from_run_id") or result.get("execution_provenance")
            or native.get("tail", {}).get("cached_requests") or native.get("front", {}).get("sealed-front-recovery.json")):
        raise ValueError("fresh_generation_cost_parents_cannot_reuse_checkpoint_or_responses")
    projection = descriptor(lineage["database_projection"], guards)
    rows, rid = projection.get("runs", {}), run["id"]
    current = rows.get(rid, {})
    qid, tid = run["question_ids"][0], run["topic_id"]
    direct = freeze.get("cost_parent_run_ids")
    if (projection.get("format") != "readonly-run-cost-checkpoint-lineage-v1" or projection.get("current_run_id") != rid
            or not isinstance(direct, list) or not direct or len(set(direct)) != len(direct)
            or current.get("cost_parent_run_ids") != direct or current.get("usage") != run["usage"]
            or current.get("resume_from_run_id") or current.get("sealed_front_recovery")):
        raise ValueError("fresh_generation_explicit_cost_parent_projection_changed")
    visiting, seen = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError("fresh_generation_cost_parent_cycle")
        if key in seen:
            return
        row = rows.get(key, {})
        parents = row.get("cost_parent_run_ids", [])
        if (row.get("id") != key or row.get("topic_id") != tid or row.get("question_ids") != [qid]
                or row.get("input_manifest_sha256") != current.get("input_manifest_sha256")
                or row.get("resume_from_run_id") or row.get("sealed_front_recovery")
                or not isinstance(parents, list) or len(set(parents)) != len(parents)):
            raise ValueError("fresh_generation_cost_parent_identity_source_or_duplicate_changed")
        visiting.add(key)
        for parent in parents:
            visit(parent)
        visiting.remove(key)
        seen.add(key)
    visit(rid)
    entries = lineage.get("ancestors", [])
    if (set(rows) != seen or len(entries) != len(seen) - 1
            or {e.get("run_id") for e in entries} != seen - {rid}):
        raise ValueError("fresh_generation_missing_duplicate_or_unrelated_cost_parent")
    used, previous, parent_files = set(), [], {}
    for entry in entries:
        pid = entry["run_id"]
        parent, parent_activity, files = (descriptor(entry[key], guards) for key in ("run", "activity", "native_manifest"))
        if (parent.get("id") != pid or parent.get("topic_id") != tid or parent.get("question_ids") != [qid]
                or parent.get("status") != "failed" or not parent.get("completed_at")
                or parent.get("usage") != rows[pid].get("usage") or parent_activity.get("usage") != parent["usage"]
                or files.get("tail", {}).get("cached_requests") or files.get("front", {}).get("sealed-front-recovery.json")):
            raise ValueError("fresh_generation_requires_explicit_failed_paid_ancestor")
        records = _ancestor_native_usage(files, guards, used)
        if _usage_signature(records) != _usage_signature(parent["usage"].get(qid, [])):
            raise ValueError("fresh_generation_ancestor_usage_not_exactly_once")
        previous.extend(records)
        parent_files[pid] = files
    old = descriptor(freeze["prelaunch_cost_parent_lineage"], guards)
    bindings = old.get("failed_capture_bindings", [])
    if (old.get("format") != "explicit-prelaunch-cost-parent-lineage-v1"
            or old.get("direct_cost_parent_run_ids") != direct
            or old.get("runs") != {key: rows[key] for key in seen - {rid}}
            or old.get("ancestor_files_are_current_requests") is not False
            or old.get("new_generation_reuses_old_responses") is not False or old.get("new_model_calls") != 0
            or len(bindings) != len(direct) or {b.get("run_id") for b in bindings} != set(direct)):
        raise ValueError("fresh_generation_frozen_failed_parent_descriptors_changed")
    for binding in bindings:
        parent_closure = descriptor(binding["failed_closure"], guards)
        if (parent_closure.get("format") != FORMAT or parent_closure.get("closed") is not True
                or parent_closure.get("fixture") is not False or parent_closure.get("real_provider") is not True
                or parent_closure.get("status") != "failed" or parent_closure.get("run_id") != binding["run_id"]
                or parent_closure.get("artifacts", {}).get("native_episode_manifest") != binding.get("native_episode_manifest")
                or parent_closure.get("artifacts", {}).get("run") != binding.get("run")
                or descriptor(binding["native_episode_manifest"], guards) != parent_files[binding["run_id"]]):
            raise ValueError("fresh_generation_parent_must_bind_original_failed_capture")
    own = _ancestor_native_usage(native, guards, used)
    if (len(own) != 4 or Counter(r["operation"] for r in own) != Counter(["strict_selection", *TAIL])
            or _usage_signature(own) != _usage_signature(run["usage"].get(qid, []))
            or len(native.get("front", {}).get("langchain_requests", [])) != 1
            or len(native.get("tail", {}).get("requests", [])) != 3):
        raise ValueError("fresh_generation_requires_four_new_native_requests")
    proof = result["source_bundle_proof"]["response"]["proof"]
    if (proof.get("closed_receipt") != native["front"].get("closed-receipt.json")
            or descriptor(native["front"]["provider-response.json"], guards) != result["source_bundle_proof"]["response"]):
        raise ValueError("fresh_generation_must_use_current_native_front_not_ancestor_result")
    total = known_usage(previous)
    seeded = [e.get("data", {}) for e in activity.get("events", []) if e.get("type") == "cost_lineage_seeded"]
    if (len(seeded) != 1 or seeded[0].get("prior_run_ids") != sorted(seen - {rid})
            or seeded[0].get("question_id") != qid or seeded[0].get("source_manifest_sha256") != current["input_manifest_sha256"]
            or seeded[0].get("prior_calls") != len(previous) or seeded[0].get("prior_known_tokens") != total["total_tokens"]
            or _usage_signature(front["previous_records"]) != _usage_signature(previous)
            or old.get("known_usage") != total or old.get("known_calls") != len(previous)
            or freeze.get("prior_known_tokens") != total["total_tokens"] or freeze.get("prior_actual_requests") != len(previous)
            or freeze.get("new_generation_max_provider_calls") != 4 or freeze.get("max_total_requests") != len(previous) + 4):
        raise ValueError("fresh_generation_cost_seed_dropped_or_duplicated_requests")
    return {"kind": "new_native_generation_with_failed_cost_parents", "cost_ancestor_run_ids": sorted(seen - {rid}),
        "ancestor_known_tokens": total["total_tokens"], "ancestor_actual_requests": len(previous),
        "current_actual_requests": 4, "current_known_tokens": known_usage(own)["total_tokens"],
        "whole_actual_requests": len(previous) + 4, "whole_known_tokens": total["total_tokens"] + known_usage(own)["total_tokens"],
        "ancestor_files_are_current_requests": False, "old_model_outputs_reused": False,
        "each_actual_ancestor_request_counted_once": True}


def _sealed_recovery_audit(receipt, binding, source, recovery_activity, current, qid, tid, result, guards,
                           *, recovery_run_id, recovery_native, question, documents, paragraphs, all_prior_records):
    config = current.get("sealed_front_recovery")
    front = source["files"]["front"]
    proof = result["source_bundle_proof"]["response"]["proof"]
    if (receipt.get("format") != "lc-sealed-front-recovery-v1" or not isinstance(config, dict)
            or config.get("origin_run_id") != source["run_id"] or config.get("response") != front.get("provider-response.json")
            or source["run"].get("status") != "failed"
            or receipt.get("origin_run_id") != source["run_id"] or receipt.get("origin_topic_id") != tid
            or receipt.get("origin_question_id") != qid or receipt.get("recovery_run_id") != recovery_run_id
            or receipt.get("source_manifest_sha256") != current.get("input_manifest_sha256")
            or receipt.get("response") != front.get("provider-response.json")
            or receipt.get("original_front_proof") != proof.get("closed_receipt")
            or receipt.get("origin_native_outer_episode") != Path(source["files"]["episode_directory"]).name
            or any(receipt.get(key) is not True for key in ("source_proof_replay_verified", "current_canonical_projection_verified", "not_a_new_native_front"))
            or any(receipt.get(key) is not False for key in ("original_response_modified", "fields_modified"))
            or receipt.get("new_model_calls") != 0 or receipt.get("new_tokens") != 0
            or receipt.get("prior_known_tokens") != known_usage(all_prior_records)["total_tokens"]
            or receipt.get("evidence_count") != len(result["evidence"])
            or question is None or documents is None or paragraphs is None
            or receipt.get("current_question_sha256") != canonical_digest(question)
            or receipt.get("current_sources_sha256") != canonical_digest({"documents": documents, "paragraphs": paragraphs})):
        raise ValueError("sealed_front_recovery_identity_source_or_cost_changed")
    closed = descriptor(front["closed-receipt.json"], guards)
    if closed.get("status") != "succeeded" or closed.get("known_usage_complete") is not True:
        raise ValueError("sealed_front_recovery_requires_original_successful_complete_front")
    metadata = descriptor(recovery_native["tail"]["source-bundle-episode.json"], guards)
    if (not isinstance(receipt.get("current_recovery_source_archives"), dict)
            or not receipt["current_recovery_source_archives"]
            or receipt["current_recovery_source_archives"] != metadata.get("source_archives")):
        raise ValueError("sealed_front_recovery_runtime_archive_changed")
    _proof_guards(receipt.get("current_recovery_source_archives"), guards)
    events = [e.get("data", {}) for e in recovery_activity.get("events", []) if e.get("type") == "sealed_front_recovery_completed"]
    if (len(events) != 1 or events[0].get("origin_run_id") != source["run_id"]
            or events[0].get("recovery_receipt") != binding or events[0].get("actual_request_started") is not False
            or events[0].get("new_model_calls") != 0 or events[0].get("new_tokens") != 0):
        raise ValueError("sealed_front_recovery_actual_zero_call_event_missing")
    return {"origin_run_id": source["run_id"], "recovery_run_id": recovery_run_id,
        "receipt": binding, "new_model_calls": 0, "new_tokens": 0, "original_response_modified": False}


def _resume_source_files(lineage, run, activity, result, native, guards, *, question=None, documents=None, paragraphs=None):
    """A v1 resume has one completed source ancestor and three new tail calls.

    All cost parents are explicit, unique original manifests. Partial-tail or
    cross-topic checkpoint reuse remains fail closed rather than guessed.
    """
    if (lineage.get("format") != "explicit-checkpoint-source-ancestor-files-v1"
            or lineage.get("ancestor_files_are_current_requests") is not False or lineage.get("new_model_calls") != 0):
        raise ValueError("resume_explicit_ancestor_descriptor_required")
    projection = descriptor(lineage["database_projection"], guards)
    rows = projection.get("runs", {})
    current = rows.get(run["id"], {})
    provenance = result.get("execution_provenance", {})
    qid, tid = run["question_ids"][0], run["topic_id"]
    if (projection.get("format") != "readonly-run-cost-checkpoint-lineage-v1" or projection.get("current_run_id") != run["id"]
            or current.get("resume_from_run_id") != run.get("resumed_from_run_id")
            or not run.get("resumed_from_run_id") or current.get("usage") != run["usage"]
            or lineage.get("execution_provenance") != provenance or provenance.get("kind") != "checkpoint_continuation"
            or not isinstance(provenance.get("checkpoint_id"), str)
            or TAIL.intersection(provenance.get("reused_nodes", []))):
        raise ValueError("resume_checkpoint_source_identity_changed")
    source_id = current.get("resume_checkpoint_sources", {}).get(qid)
    if provenance.get("source_thread_id") != str(source_id) + ":" + qid:
        raise ValueError("resume_checkpoint_thread_does_not_match_original_run")
    checkpoint = descriptor(lineage["checkpoint_identity"], guards)
    if (checkpoint.get("format") != "readonly-original-checkpoint-identity-v1"
            or checkpoint.get("thread_id") != provenance["source_thread_id"]
            or checkpoint.get("checkpoint_ns") != "" or checkpoint.get("checkpoint_id") != provenance["checkpoint_id"]
            or checkpoint.get("serialized_checkpoint_state_read") is not False):
        raise ValueError("resume_original_checkpoint_identity_changed")
    visiting, visited = set(), set()
    def visit(rid):
        if rid in visiting:
            raise ValueError("resume_ancestor_cycle")
        if rid in visited:
            return
        row = rows.get(rid, {})
        parents = row.get("cost_parent_run_ids", [])
        if not isinstance(parents, list) or len(set(parents)) != len(parents):
            raise ValueError("resume_duplicate_cost_parent_id")
        visiting.add(rid)
        for parent in parents + ([row["resume_from_run_id"]] if row.get("resume_from_run_id") else []):
            visit(parent)
        visiting.remove(rid)
        visited.add(rid)
    visit(run["id"])
    seen, pending = set(), [run["id"]]
    while pending:
        rid = pending.pop()
        if rid in seen:
            continue
        row = rows.get(rid)
        if (not isinstance(row, dict) or row.get("id") != rid or row.get("topic_id") != tid
                or row.get("question_ids") != [qid] or row.get("input_manifest_sha256") != current.get("input_manifest_sha256")):
            raise ValueError("resume_ancestor_other_question_or_source")
        seen.add(rid)
        pending.extend(row.get("cost_parent_run_ids", []))
        if row.get("resume_from_run_id"):
            pending.append(row["resume_from_run_id"])
    entries = lineage.get("ancestors", [])
    if (set(rows) != seen or len(entries) != len(seen) - 1
            or {e.get("run_id") for e in entries} != seen - {run["id"]}):
        raise ValueError("resume_missing_duplicate_or_unrelated_ancestor")
    reused_events = [e.get("data") for e in activity.get("events", []) if e.get("type") == "checkpoint_reused"]
    seeded_events = [e.get("data", {}) for e in activity.get("events", []) if e.get("type") == "cost_lineage_seeded"]
    if (len(reused_events) != 1 or any(reused_events[0].get(k) != v for k, v in provenance.items())
            or len(seeded_events) != 1 or seeded_events[0].get("prior_run_ids") != sorted(seen - {run["id"]})):
        raise ValueError("resume_actual_checkpoint_or_cost_events_missing")
    used_requests, previous, source, parents = set(), [], None, {}
    for entry in entries:
        rid = entry["run_id"]
        prior_run, prior_activity, files = (descriptor(entry[key], guards) for key in ("run", "activity", "native_manifest"))
        if (prior_run.get("id") != rid or prior_run.get("topic_id") != tid or prior_run.get("question_ids") != [qid]
                or prior_run.get("status") not in {"failed", "cancelled"} or not prior_run.get("completed_at")
                or prior_run.get("usage") != rows[rid].get("usage") or prior_activity.get("usage") != prior_run.get("usage")):
            raise ValueError("resume_ancestor_terminal_run_usage_changed")
        records = _ancestor_native_usage(files, guards, used_requests)
        if _usage_signature(records) != _usage_signature(prior_run["usage"].get(qid, [])):
            raise ValueError("resume_ancestor_new_usage_not_exactly_once")
        parents[rid] = {"run_id": rid, "files": files, "records": records, "run": prior_run, "activity": prior_activity}
    checkpoint_owner = parents.get(source_id)
    if checkpoint_owner is None:
        raise ValueError("resume_source_ancestor_missing")
    paused = checkpoint_owner["run"]
    if (paused.get("status") != "cancelled" or paused.get("error", {}).get("code") != "paused"
            or rows[source_id].get("model_config_sha256") != current.get("model_config_sha256")
            or checkpoint_owner["files"]["tail"].get("requests")
            or any(r.get("operation") in TAIL for r in checkpoint_owner["records"])):
        raise ValueError("resume_requires_paused_same_input_checkpoint")
    completed_source_reused = "source_bundle_selection" in provenance.get("reused_nodes", [])
    recovery_bindings = [(rid, files["front"]["sealed-front-recovery.json"])
        for rid, files in [(run["id"], native), (source_id, checkpoint_owner["files"])]
        if "sealed-front-recovery.json" in files.get("front", {})]
    recovery_audit = None
    if recovery_bindings:
        expected_recovery_owner = source_id if completed_source_reused else run["id"]
        if len(recovery_bindings) != 1 or recovery_bindings[0][0] != expected_recovery_owner:
            raise ValueError("sealed_recovery_owner_must_match_actual_checkpoint_progress")
        recovery_binding = recovery_bindings[0][1]
        recovery = descriptor(recovery_binding, guards)
        origin_id = recovery.get("origin_run_id")
        source = parents.get(origin_id)
        if (source is None or origin_id == source_id or checkpoint_owner["records"]
                or native.get("front", {}).get("langchain_requests") or native.get("front", {}).get("native_requests")):
            raise ValueError("sealed_recovery_must_reuse_separate_paid_source_without_new_front_HTTP")
        recovery_audit = _sealed_recovery_audit(recovery, recovery_binding, source,
            checkpoint_owner["activity"] if completed_source_reused else activity,
            current, qid, tid, result, guards, recovery_run_id=expected_recovery_owner,
            recovery_native=checkpoint_owner["files"] if completed_source_reused else native,
            question=question, documents=documents, paragraphs=paragraphs,
            all_prior_records=[record for parent in parents.values() for record in parent["records"]])
    else:
        if not completed_source_reused:
            raise ValueError("initial_checkpoint_needs_explicit_sealed_source_recovery")
        source = checkpoint_owner
    proof = result["source_bundle_proof"]["response"]["proof"]
    if (source["files"]["front"].get("closed-receipt.json") != proof.get("closed_receipt")
            or descriptor(source["files"]["front"]["provider-response.json"], guards) != result["source_bundle_proof"]["response"]):
        raise ValueError("resume_front_must_be_original_immutable_checkpoint_response")
    for rid, parent in parents.items():
        if rid != source["run_id"]:
            previous.extend(parent["records"])
    combined = previous + source["records"]
    if (seeded_events[0].get("question_id") != qid
            or seeded_events[0].get("source_manifest_sha256") != current.get("input_manifest_sha256")
            or seeded_events[0].get("prior_calls") != len(combined)
            or seeded_events[0].get("prior_known_tokens") != known_usage(combined)["total_tokens"]):
        raise ValueError("resume_actual_cost_seed_dropped_or_duplicated_requests")
    native = {**native, "source_front": source["files"]["front"],
        "ancestor_budget_forecasts": [*source["files"]["tail"].get("budget_forecasts", []),
            *source["files"]["front"].get("budget_forecasts", [])]}
    return {"native": native, "source_records": source["records"], "previous_records": previous,
        "audit": {"source_run_id": source["run_id"], "checkpoint_owner_run_id": source_id,
            "checkpoint_kind": "completed_source" if completed_source_reused else "initial_or_pre_source",
            "completed_source_checkpoint_reused": completed_source_reused,
            "sealed_source_recovery": recovery_audit,
            "cost_ancestor_run_ids": sorted(seen - {run["id"]}),
            "checkpoint_id": provenance["checkpoint_id"], "ancestor_calls": len(used_requests),
            "ancestor_known_tokens": known_usage(previous + source["records"])["total_tokens"],
            "ancestor_files_are_current_requests": False, "each_actual_ancestor_request_counted_once": True}}


def _verify_original_documents(raw, documents, paragraphs):
    originals = {d["doc_id"]: d for d in raw["documents"]}
    if len(originals) != len(raw["documents"]) or {d["raw_doc_id"] for d in documents} != set(originals) or len(documents) != len(originals):
        raise ValueError("bundle_complete_original_documents_required")
    for doc in documents:
        original = originals[doc["raw_doc_id"]]
        if doc["content"] != original["content"] or doc["content_sha256"] != sha256(original["content"].encode("utf-8")).hexdigest():
            raise ValueError("bundle_full_original_document_including_gaps_changed")
    bounds = {(d["doc_id"], p["para_id"]): (p["char_start"], p["char_end"]) for d in raw["documents"] for p in d["paragraphs"]}
    for row in paragraphs:
        if (row["document_start_cp"], row["document_end_cp"]) != bounds[(row["raw_doc_id"], row["raw_para_id"])]:
            raise ValueError("bundle_original_paragraph_document_coordinates_changed")


def _cached_source_files(lineage, run, activity, result, native, guards, *, question, documents, paragraphs):
    """Use existing paid-cache parents; immutable requests are never rebilled."""
    cached = native["tail"].get("cached_requests", [])
    requests = native["tail"].get("requests", [])
    partial = (len(cached) == 2 and {row.get("operation") for row in cached} == TAIL - {"arbitrate"}
        and len(requests) == 1 and descriptor(requests[0]["preflight"], guards).get("operation") == "arbitrate")
    complete = not requests and len(cached) == 3 and {row.get("operation") for row in cached} == TAIL
    if (lineage.get("format") != "explicit-checkpoint-source-ancestor-files-v1"
            or lineage.get("ancestor_files_are_current_requests") is not False or lineage.get("new_model_calls") != 0
            or run.get("resumed_from_run_id") or result.get("execution_provenance")
            or not (partial or complete)):
        raise ValueError("explicit_sealed_source_requires_cached_AB_plus_one_new_arb_or_complete_cached_tail")
    projection = descriptor(lineage["database_projection"], guards)
    rows = projection.get("runs", {}); current = rows.get(run["id"], {})
    if (projection.get("format") != "readonly-run-cost-checkpoint-lineage-v1" or projection.get("current_run_id") != run["id"]
            or current.get("resume_from_run_id") is not None):
        raise ValueError("cached_source_current_run_projection_changed")
    qid, tid = question["id"], run["topic_id"]
    seen, pending = set(), [run["id"]]
    while pending:
        rid = pending.pop()
        if rid in seen: continue
        row = rows.get(rid, {})
        if (row.get("id") != rid or row.get("topic_id") != tid or row.get("question_ids") != [qid]
                or row.get("input_manifest_sha256") != current.get("input_manifest_sha256")):
            raise ValueError("cached_source_cost_parent_other_question_or_source")
        seen.add(rid)
        parents = row.get("cost_parent_run_ids", []) + ([row["resume_from_run_id"]] if row.get("resume_from_run_id") else [])
        if run["id"] in parents: raise ValueError("cached_source_cost_parent_cycle")
        pending.extend(parents)
    entries = lineage.get("ancestors", [])
    if (set(rows) != seen or len(entries) != len(seen) - 1 or {e.get("run_id") for e in entries} != seen - {run["id"]}
            or current.get("usage") != run["usage"]):
        raise ValueError("cached_source_missing_duplicate_cost_ancestor")
    checked, used = {}, set()
    for entry in entries:
        rid = entry["run_id"]
        previous, previous_activity, files = (descriptor(entry[key], guards) for key in ("run", "activity", "native_manifest"))
        if (previous.get("id") != rid or previous.get("topic_id") != tid or previous.get("question_ids") != [qid]
                or previous.get("status") not in {"failed", "cancelled"} or not previous.get("completed_at")
                or previous.get("usage") != rows[rid].get("usage") or previous_activity.get("usage") != previous.get("usage")):
            raise ValueError("cached_source_terminal_ancestor_usage_changed")
        records = _ancestor_native_usage(files, guards, used)
        if _usage_signature(records) != _usage_signature(previous["usage"].get(qid, [])):
            raise ValueError("cached_source_ancestor_wire_usage_not_exactly_once")
        checked[rid] = {"run_id": rid, "run": previous, "files": files, "records": records}
    receipt_binding = native["front"].get("sealed-front-recovery.json")
    if receipt_binding is None: raise ValueError("cached_source_sealed_receipt_missing")
    receipt = descriptor(receipt_binding, guards)
    source = checked.get(receipt.get("origin_run_id"))
    if source is None: raise ValueError("cached_source_original_front_owner_missing")
    if descriptor(source["files"]["front"]["provider-response.json"], guards) != result["source_bundle_proof"]["response"]:
        raise ValueError("cached_source_original_response_bytes_changed")
    combined = [record for parent in checked.values() for record in parent["records"]]
    new_records = _ancestor_native_usage(native, guards, used)
    if (len(new_records) != (1 if partial else 0)
            or partial and new_records[0].get("operation") != "arbitrate"
            or _usage_signature(new_records) != _usage_signature(run["usage"].get(qid, []))):
        raise ValueError("cached_AB_requires_exactly_one_new_paid_arbitration")
    local = _sealed_recovery_audit(receipt, receipt_binding, source, activity, current, qid, tid, result, guards,
        recovery_run_id=run["id"], recovery_native=native, question=question, documents=documents, paragraphs=paragraphs, all_prior_records=combined)
    events = [e.get("data", {}) for e in activity.get("events", []) if e.get("type") == "cost_lineage_seeded"]
    if (len(events) != 1 or events[0].get("prior_run_ids") != sorted(checked)
            or events[0].get("prior_calls") != len(combined) or events[0].get("prior_known_tokens") != known_usage(combined)["total_tokens"]):
        raise ValueError("cached_source_complete_cost_seed_missing")
    cache_parents = {descriptor(row["cache_hit"], guards).get("parent_run_id") for row in native["tail"]["cached_requests"]}
    if not cache_parents <= set(checked): raise ValueError("cached_source_unbound_tail_parent")
    # A current paid arbitration must use its current forecast. Historical
    # phases fill only missing stages; never relabel the old forecast as new.
    phases = {descriptor(row, guards).get("phase") for row in
        [*native["tail"].get("budget_forecasts", []), *native["front"].get("budget_forecasts", [])]}
    budgets, budget_owners = [], {}
    for parent in [source, *[checked[key] for key in sorted(cache_parents) if key != source["run_id"]]]:
        for row in [*parent["files"]["tail"].get("budget_forecasts", []), *parent["files"]["front"].get("budget_forecasts", [])]:
            phase = descriptor(row, guards).get("phase")
            if phase not in phases:
                budgets.append(row)
                budget_owners[row["file"]] = parent["files"]["tail"]["source-bundle-episode.json"]
                phases.add(phase)
    source_records = descriptor(source["files"]["front"]["closed-receipt.json"], guards).get("usage_records", [])
    if len(source_records) != 1 or source_records[0].get("operation") != "strict_selection":
        raise ValueError("cached_source_requires_one_original_source_receipt")
    previous = _remaining_usage_records(combined, source_records)
    native = {**native, "ancestor_budget_forecasts": budgets,
        "ancestor_budget_forecast_metadata": budget_owners,
        "cost_parent_native_files": {rid: parent["files"]["tail"] for rid, parent in checked.items()}}
    return {"native": native, "source_records": source_records, "previous_records": previous,
        "audit": {"source_run_id": source["run_id"], "cost_ancestor_run_ids": sorted(checked),
            "new_model_calls": len(new_records), "cached_analyses_with_new_arbitration": partial,
            "ancestor_known_tokens": known_usage(combined)["total_tokens"], "ancestor_calls": len(combined), "sealed_source_recovery": local,
            "tail_reused_from_actual_paid_parent_runs": sorted(cache_parents), "cached_result_is_new_generation": False}}


def _selection_mode(proof, frozen_config, run):
    """Bind front grades/projection flags to the frozen worker, not final roles."""
    for key, public in (("selection_only", "source_bundle_selection_only"),
                        ("graded_selection", "source_bundle_graded_selection"),
                        ("compact_analyses", "source_bundle_compact_analyses"),
                        ("issue_frame", "source_bundle_issue_frame"),
                        ("context_passages", "source_bundle_context_passages"),
                        ("complete_grading", "source_bundle_complete_grading"),
                        ("hybrid_retrieval", "source_bundle_hybrid_retrieval"),
                        ("paragraph_quality", "source_bundle_paragraph_quality"),
                        ("paragraph_quality_compact", "source_bundle_paragraph_quality_compact"),
                        ("parent_token_pack", "source_bundle_parent_token_pack")):
        actual, expected = proof.get(key, False), frozen_config.get(key, False)
        if (type(actual) is not bool or type(expected) is not bool or actual != expected
                or public in run and run[public] is not actual
                or actual and public not in run and (key != "selection_only"
                    or proof.get("graded_selection") is True or proof.get("compact_analyses") is True)):
            raise ValueError("bundle_frozen_selection_mode_changed")
    if proof.get("graded_selection") is True:
        if (proof.get("selection_only") is not True or frozen_config.get("selection_output_tokens") != 2048
                or proof.get("selection_output_tokens") != 2048
                or proof.get("graded_selection_policy") != frozen_config.get("graded_selection_policy")):
            raise ValueError("bundle_frozen_graded_selection_policy_changed")
    if proof.get("paragraph_quality") is True:
        if (any(proof.get(key, False) for key in
                ("selection_only", "graded_selection", "context_passages", "complete_grading", "hybrid_retrieval"))
                or type(proof.get("selection_output_tokens")) is not int
                or proof["selection_output_tokens"] <= 0
                or proof.get("selection_output_tokens") != frozen_config.get("selection_output_tokens")
                or proof.get("paragraph_quality_policy") != frozen_config.get("paragraph_quality_policy")
                or not isinstance(proof.get("paragraph_quality_policy"), dict)):
            raise ValueError("bundle_frozen_paragraph_quality_policy_changed")
    if proof.get("paragraph_quality_compact") is True:
        if (proof.get("paragraph_quality") is not True
                or proof.get("paragraph_quality_compact_policy") != frozen_config.get("paragraph_quality_compact_policy")
                or not isinstance(proof.get("paragraph_quality_compact_policy"), dict)):
            raise ValueError("bundle_frozen_paragraph_quality_compact_policy_changed")
    if proof.get("parent_token_pack") is True:
        if (proof.get("paragraph_quality") is not True or proof.get("paragraph_quality_compact") is not True
                or not isinstance(proof.get("parent_token_pack_policy"), dict)
                or proof["parent_token_pack_policy"] != frozen_config.get("parent_token_pack_policy")):
            raise ValueError("bundle_frozen_parent_token_pack_policy_changed")
    for feature in ("complete_grading", "hybrid_retrieval"):
        if proof.get(feature, False) and (proof.get("context_passages") is not True
                or proof.get(feature + "_policy") != frozen_config.get(feature + "_policy")):
            raise ValueError("bundle_frozen_context_feature_policy_changed")
    if proof.get("whole_chain_planning") != frozen_config.get("whole_chain_planning"):
        raise ValueError("bundle_frozen_whole_chain_planning_changed")
    return {"graded_selection": proof.get("graded_selection", False),
        "paragraph_quality": proof.get("paragraph_quality", False),
        "paragraph_quality_compact": proof.get("paragraph_quality_compact", False),
        "parent_token_pack": proof.get("parent_token_pack", False),
        "compact_analyses": proof.get("compact_analyses", False), "front_use_type_is_final_conflict": False}


def _verify_analysis_input_projection(receipt, payload, analyses, aliases, *, codec):
    """Check archived lossless A/B aliases and actual arbitration input bytes."""
    if (not isinstance(receipt, dict) or receipt.get("format") != "strict-analysis-input-projection-v1"
            or receipt.get("operation") != "arbitrate" or receipt.get("new_model_calls") != 0
            or receipt.get("trusted_original") != analyses or receipt.get("evidence_aliases") != aliases
            or receipt.get("projected_payload_sha256") != canonical_digest(payload)):
        raise ValueError("bundle_analysis_input_projection_binding_changed")
    audit = codec.validate_projection(analyses, receipt.get("projected"), receipt.get("audit"))
    if (remap_aliases(receipt["projected"], aliases) != receipt.get("wire_analyses")
            or receipt["wire_analyses"] != payload.get("analyses")
            or codec.restore_analyses(receipt["projected"], receipt["audit"], trusted_original=analyses) != analyses):
        raise ValueError("bundle_analysis_input_projection_wire_changed")
    return audit


def _budget_forecasts(native_files, metadata, freeze, guards):
    policy = metadata.get("source_bundle_config", {}).get("whole_chain_planning")
    bindings = [*native_files["tail"].get("budget_forecasts", []), *native_files["front"].get("budget_forecasts", []),
                *native_files.get("ancestor_budget_forecasts", [])]
    if policy is None:
        if bindings:
            raise ValueError("bundle_unconfigured_budget_forecasts")
        return []
    rows, phases = [], set()
    for binding in bindings:
        row = descriptor(binding, guards)
        original_owner = native_files.get("ancestor_budget_forecast_metadata", {}).get(binding["file"])
        owner = descriptor(original_owner, guards) if original_owner is not None else None
        expected_stop = owner.get("whole_token_stop") if owner else freeze["engineering_episode_stop"]
        expected_calls = owner.get("max_total_requests") if owner else freeze["max_total_requests"]
        if owner and owner.get("source_bundle_config", {}).get("whole_chain_planning") != policy:
            raise ValueError("bundle_original_budget_owner_policy_changed")
        phase = row.get("phase")
        requests = list(row.get("requests", []))
        if row.get("front_request") is not None:
            requests.insert(0, row["front_request"])
        if (row.get("format") != policy.get("version") or row.get("policy") != policy
                or phase not in {"before_first_paid_request", "selected_sources", "actual_analyses"} or phase in phases
                or row.get("is_actual_usage") is not False or row.get("is_guaranteed_global_upper_bound") is not False
                or row.get("admitted") is not True or row.get("threshold") != expected_stop
                or row.get("max_calls") != expected_calls):
            raise ValueError("bundle_budget_forecast_binding_changed")
        components = [r.get(key) for r in requests for key in
            ("serialized_json_input_estimate", "reserved_output_tokens", "safety_tokens")]
        if (any(type(n) is not int or n < 0 for n in [*components, row.get("known_tokens"), row.get("future_analysis_wire_allowance")])
                or row.get("future_analysis_wire_allowance") != (0 if phase == "actual_analyses" else policy.get("future_analysis_wire_allowance"))
                or [r.get("operation") for r in requests] != (["strict_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"]
                    if phase == "before_first_paid_request" else ["arbitrate"] if phase == "actual_analyses"
                    else ["authority_analyst", "cross_validation_analyst", "arbitrate"])
                or any(r.get("safety_tokens") != policy.get("safety_per_request") for r in requests)
                or row.get("required_remaining_planning_tokens") != sum(components) + row["future_analysis_wire_allowance"]
                or row.get("planned_whole_tokens") != row["known_tokens"] + row["required_remaining_planning_tokens"]
                or row["planned_whole_tokens"] > row["threshold"]):
            raise ValueError("bundle_budget_forecast_math_changed")
        bodies = row.get("wire_bodies")
        if (not isinstance(bodies, list) or len(bodies) != len(requests)
                or any(sha256((json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
                    != request.get("native_wire_sha256") for body, request in zip(bodies, requests))):
            raise ValueError("bundle_budget_forecast_native_body_changed")
        rows.append({"phase": phase, "bound_receipt": binding, "planned_whole_tokens": row["planned_whole_tokens"],
            "is_actual_usage": False, "is_guaranteed_global_upper_bound": False})
        phases.add(phase)
    if phases != {"before_first_paid_request", "selected_sources", "actual_analyses"}:
        raise ValueError("bundle_complete_budget_forecasts_missing")
    cached = native_files["tail"].get("cached_requests", [])
    actual_requests = native_files["tail"].get("requests", [])
    if (len(cached) == 2 and {row.get("operation") for row in cached} == TAIL - {"arbitrate"}
            and len(actual_requests) == 1):
        # Only this new arbitration can consume provider budget. Its receipt
        # must be owned by the current run and match the exact actual wire;
        # a historical parent's plan cannot stand in for this admission.
        current = [binding for binding in native_files["tail"].get("budget_forecasts", [])
            if descriptor(binding, guards).get("phase") == "actual_analyses"]
        if len(current) != 1:
            raise ValueError("cached_AB_requires_current_paid_arb_forecast")
        plan = descriptor(current[0], guards)
        item = actual_requests[0]
        pre, wire, actual = (descriptor(item[key], guards) for key in ("preflight", "wire", "actual-wire"))
        request = plan["requests"][0] if len(plan.get("requests", [])) == 1 else {}
        if (not metadata.get("source_bundle_config", {}).get("sealed_front_recovery")
                or pre.get("operation") != "arbitrate" or pre.get("admitted") is not True
                or pre.get("wire") != item["wire"] or pre.get("actual_wire") != item["actual-wire"] or wire != actual
                or plan.get("wire_bodies") != [actual] or request.get("operation") != "arbitrate"
                or plan.get("known_calls") != freeze.get("prior_actual_requests")
                or plan.get("known_tokens") != freeze.get("prior_known_tokens")
                or plan.get("known_calls") + 1 != freeze["max_total_requests"]
                or pre.get("known_tokens_before") != plan.get("known_tokens")
                or pre.get("experimental_stop_threshold") != plan.get("threshold")
                or request.get("serialized_json_input_estimate") != pre.get("serialized_json_token_estimate")
                or request.get("reserved_output_tokens") != pre.get("reserved_output_tokens")
                or request.get("safety_tokens") != pre.get("safety_tokens")):
            raise ValueError("cached_AB_current_arb_forecast_not_bound_to_actual_wire_and_whole_cost")
        for row in rows:
            if row["bound_receipt"] == current[0]:
                row["current_actual_arbitration_wire_verified"] = True
    return rows


def _tail(result, question, native_files, run_usage, front, freeze, guards):
    manifest = native_files["tail"]
    episode = descriptor(manifest["episode.json"], guards)
    metadata = descriptor(manifest["source-bundle-episode.json"], guards)
    budget_forecasts = _budget_forecasts(native_files, metadata, freeze, guards)
    lc = result["source_bundle_proof"]["response"]["proof"].get("format") == LC_PROOF_FORMAT
    expected_metadata = "langchain-source-bundle-workbench-episode-v1" if lc else "dsh-source-bundle-workbench-episode-v1"
    expected_prompt = LC_PROMPT_VERSION if lc else PROMPT_VERSION
    expected_selection = "langchain_single_source_selection" if lc else "official_dsh_source_selection"
    if (metadata.get("format") != expected_metadata
            or metadata.get("base_episode_scope") != "inherited_strict_tail_only"
            or metadata.get("base_episode") != manifest["episode.json"]
            or metadata.get("prompt_version") != expected_prompt
            or metadata.get("source_bundle_enabled") is not True or metadata.get("strict_evidence_enabled") is not False
            or metadata.get("tail_scoped_arbitration") is not True
            or metadata.get("pipeline") != [expected_selection, "authority_analyst", "cross_validation_analyst", "arbitrate"]
            or metadata.get("whole_token_stop") != freeze["engineering_episode_stop"]
            or metadata.get("max_total_requests") != freeze["max_total_requests"]):
        raise ValueError("bundle_metadata_must_bind_inherited_tail_scope")
    if (episode.get("strict") is not True or episode.get("model") != "deepseek-v4-pro" or episode.get("thinking") != "disabled"
            or episode.get("temperature") != 0 or episode.get("automatic_retries") != 0
            or episode.get("experimental_stop_threshold") != freeze["engineering_episode_stop"]):
        raise ValueError("real_strict_tail_frozen_contract_required")
    prior = descriptor(manifest["resume-prior-usage.json"], guards)
    seeded = front["previous_records"] + front["new_records"]
    if (_counter(prior["records"]) != _counter(seeded) or prior.get("prior_known_tokens") != known_usage(seeded)["total_tokens"]
            or prior.get("prior_calls") != len(seeded) or prior.get("not_new_usage") is not True):
        raise ValueError("tail_seed_must_include_each_parent_and_DSH_request_once")
    raw_archives, raw_sources = episode.get("source_archive", {}), episode.get("sources", {})
    archives = {relative_source(k): v for k, v in raw_archives.items()}
    sources = {relative_source(k): v for k, v in raw_sources.items()}
    if len(archives) != len(raw_archives) or len(sources) != len(raw_sources):
        raise ValueError("duplicate_normalized_tail_source_archive_path")
    if not archives or set(archives) != set(sources): raise ValueError("tail_full_source_archive_required")
    for key, item in archives.items():
        if item["sha256"] != sources[key]: raise ValueError("tail_source_archive_binding_changed")
        data, guard = bound(item["file"], sources[key]); guards.append(guard)
    wires, responses, operations, relations, response_shas, schema_audits, recoveries, records = [], [], [], [], [], [], [], []
    input_projection_audits = []
    # Workflow adds already-validated continuation metadata after model calls.
    # It was not present in the A/B objects sent to arbitration.
    native_analyses = ([{k: v for k, v in row.items() if k != "execution_provenance"} for row in result["analyses"]]
        if result.get("execution_provenance") else result["analyses"])
    known, parent = 0, known_usage(seeded)["total_tokens"]
    with SourceArchive(archives):
        cached = manifest.get("cached_requests", [])
        cache_audits = []
        if cached:
            partial = (len(cached) == 2 and {row.get("operation") for row in cached} == TAIL - {"arbitrate"}
                and len(manifest["requests"]) == 1
                and descriptor(manifest["requests"][0]["preflight"], guards).get("operation") == "arbitrate")
            if (not lc or not metadata.get("source_bundle_config", {}).get("sealed_front_recovery")
                    or manifest["requests"] and not partial):
                raise ValueError("cached_bundle_tail_requires_explicit_sealed_source_and_exact_remaining_HTTP")
            _, _, _, _, cache_audits = audit_cached_requests(cached, native_files.get("cost_parent_native_files", {}), guards,
                syntax_decoder_sha256=sources.get("app/agents/strict_json_syntax.py"), local_schema_sha256=sources.get("app/agents/schemas.py"),
                scoped_schema_sha256=sources.get("app/agents/scoped_arbitration.py"), allow_bundle_tail=True,
                allow_bundle_partial_analyses=partial)
        for item in [*cached, *manifest["requests"]]:
            reused = "cache_hit" in item
            wire, finished = (descriptor(item[key], guards) for key in ("wire", "finished"))
            pre = descriptor(item["preflight"], guards) if not reused else {"operation": item["operation"], "reserved_output_tokens": wire.get("max_tokens")}
            if lc and not reused:
                actual = descriptor(item["actual-wire"], guards)
                if pre.get("actual_wire") != item["actual-wire"] or actual != wire:
                    raise ValueError("langchain_tail_actual_HTTP_bytes_not_bound")
            op = pre.get("operation")
            if not reused and (op not in TAIL or pre.get("admitted") is not True or finished.get("request_started") is not True
                    or finished.get("outcome") != "success" or finished.get("unknown_usage_requests") != 0
                    or pre["wire"]["sha256"] != item["wire"]["sha256"] or pre.get("known_tokens_before") != parent + known
                    or pre.get("experimental_stop_threshold") != freeze["engineering_episode_stop"]):
                raise ValueError("three_actual_successful_tail_calls_required")
            if not reused:
                reservation = parent + known + pre["serialized_json_token_estimate"] + pre["reserved_output_tokens"] + pre["safety_tokens"]
                if reservation > freeze["engineering_episode_stop"]: raise ValueError("tail_request_exceeded_whole_episode_stop")
            tools = wire.get("tools", [])
            if (wire.get("model") != episode["model"] or wire.get("thinking") != {"type": "disabled"}
                    or wire.get("max_tokens") != pre["reserved_output_tokens"] or len(tools) != 1
                    or tools[0].get("function", {}).get("strict") is not True
                    or wire.get("tool_choice", {}).get("function", {}).get("name") != tools[0]["function"]["name"]):
                raise ValueError("tail_actual_native_strict_wire_required")
            response = descriptor(item["response"], guards)
            settlement = descriptor(item["settlement"], guards) if not reused else None
            if "syntax-recovery" in item:
                audit = verify_syntax_recovery(response, wire, descriptor(item["syntax-recovery"], guards), operation=op,
                    response_sha256=item["response"]["sha256"], decoder_sha256=sources.get("app/agents/strict_json_syntax.py"),
                    local_schema_sha256=sources.get("app/agents/schemas.py"), scoped_schema_sha256=sources.get("app/agents/scoped_arbitration.py"))
                from scripts.score_workbench_real_episode import bound_code_module
                decoded, _ = bound_code_module("app/agents/strict_json_syntax.py", sources["app/agents/strict_json_syntax.py"]).decode_visible_arguments(response["tool_calls_wire"][0]["function"]["arguments"])
                recoveries.append(audit)
            else:
                verify_visible_tool_wire(response)
                decoded = json.loads(response["tool_calls_wire"][0]["function"]["arguments"])
            schema_audits.append(verify_tail_schema(decoded, wire, operation=op, local_schema_sha256=sources.get("app/agents/schemas.py"),
                scoped_schema_sha256=sources.get("app/agents/scoped_arbitration.py")))
            counts = {key: settlement.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")} if not reused else None
            if not reused: known_usage([{**counts, "usage_source": "provider"}])
            if not reused and (settlement.get("usage_known") is not True or settlement.get("actual_request_started") is not True or settlement.get("operation") != op
                    or settlement.get("known_tokens_before") != known or finished.get("operation") != op
                    or any(response.get("usage", {}).get(k) != v for k, v in counts.items())):
                raise ValueError("tail_usage_settlement_and_visible_response_disagree")
            if not reused: known += counts["total_tokens"]
            if not reused and (settlement.get("known_total_tokens_after") != known or finished.get("known_total_tokens") != known
                    or finished.get("known_episode_tokens_with_ancestors") != parent + known):
                raise ValueError("tail_usage_accumulator_reset_or_disagrees")
            payload = wire_payload(wire)
            if payload.get("question") != question: raise ValueError("tail_original_question_changed")
            if op != "arbitrate" and ("analyses" in payload or payload.get("evidence_snapshot_id") != result["evidence_snapshot_id"]):
                raise ValueError("independent_analyst_received_peer_or_wrong_snapshot")
            original = {(r["raw_doc_id"], r["raw_para_id"], r["quote"]): r for r in result["evidence"]}
            aliases = {}
            for row in payload["evidence"]:
                fact = original.get((row.get("raw_doc_id"), row.get("raw_para_id"), row.get("quote")))
                if fact is None or row.get("fact_des") != fact["fact_des"] or row.get("grade_score") != fact["grade_score"] or row.get("id") in aliases:
                    raise ValueError("tail_full_frozen_evidence_differed")
                aliases[row["id"]] = fact["id"]
            if len(aliases) != len(original) or len(set(aliases.values())) != len(original): raise ValueError("tail_incomplete_frozen_evidence_inventory")
            forward_aliases = {original_id: alias for alias, original_id in aliases.items()}
            compact = metadata.get("source_bundle_config", {}).get("compact_analyses", False)
            if "analysis-input-projection" in item:
                if compact is not True or op != "arbitrate":
                    raise ValueError("bundle_unexpected_analysis_input_projection")
                from scripts.score_workbench_real_episode import bound_code_module
                codec = bound_code_module("app/domain/analysis_wire_projection.py", sources.get("app/domain/analysis_wire_projection.py"))
                input_projection_audits.append(_verify_analysis_input_projection(
                    descriptor(item["analysis-input-projection"], guards), payload, native_analyses, forward_aliases, codec=codec))
            elif op == "arbitrate":
                if compact is True or payload.get("analyses") != remap_aliases(native_analyses, forward_aliases):
                    raise ValueError("bundle_complete_analysis_input_or_projection_missing")
            if op != "arbitrate":
                canonical = remap_aliases(decoded, aliases)
                ids = {j["id"]: str(uuid5(NAMESPACE_URL, f"trust-judge:judgment:{question['id']}:{op}:{j['id']}")) for j in canonical["judgments"]}
                for judgment in canonical["judgments"]:
                    judgment["id"] = ids[judgment["id"]]
                    judgment["prerequisite_judgment_ids"] = [ids[i] for i in judgment["prerequisite_judgment_ids"]]
                canonical.update(role=op, evidence_snapshot_id=result["evidence_snapshot_id"])
                published = [a for a in native_analyses if a.get("role") == op]
                if len(published) == 1 and published[0].get("relation_quality") is not None:
                    from scripts.score_workbench_real_episode import bound_code_module
                    codec = bound_code_module("app/domain/provisional_relations.py", sources.get("app/domain/provisional_relations.py"))
                    canonical, quality = codec.quarantine_provisional_relations(canonical, role=op)
                    if quality is None:
                        raise ValueError("provisional_relation_quality_requires_original_overlap")
                    canonical["relation_quality"] = quality
                if len(published) != 1 or published[0] != canonical: raise ValueError("published_independent_analysis_changed_from_actual_wire")
            wires.append(wire); responses.append(response); operations.append(op)
            relations.append(descriptor(item["relation-projection"], guards) if "relation-projection" in item else None)
            response_shas.append(item["response"]["sha256"])
            if not reused: records.append({**counts, "usage_source": "provider", "operation": op})
        if len(operations) != 3 or set(operations) != TAIL or operations[-1] != "arbitrate" or _counter([r for r in run_usage if r.get("operation") in TAIL]) != _counter(records):
            raise ValueError("exact_A_B_arbitration_actual_tail_required")
        native = {"logical_operations": operations, "relation_projections": relations, "response_file_sha256": response_shas,
            "runtime_sources_sha256": sources, "actual_request_syntax_recoveries": recoveries}
        adapter = {**result, "verification_protocol": "strict-workbench-evidence-balanced-v2"}
        relation_audit = verify_scoped_relations(adapter, wires, responses, native, question_internal_id=question["id"])
    total_calls, total_tokens = len(seeded) + len(records), parent + known
    if total_calls > freeze["max_total_requests"] or total_calls > episode["max_requests"] or total_tokens > freeze["engineering_episode_stop"]:
        raise ValueError("whole_episode_cost_exceeded_frozen_limit")
    return {"actual_tail_requests": len(records), "cached_tail_deliveries": len(cached), "cached_tail_audits": cache_audits,
        "tail_usage": known_usage(records), "whole_episode_actual_requests": total_calls,
        "whole_episode_known_tokens": total_tokens, "front_and_parent_tokens": parent, "tail_schema_audits": schema_audits,
        "native_relation_audit": relation_audit, "analysis_input_projection_audits": input_projection_audits,
        "whole_chain_forecast_receipts": budget_forecasts,
        "independent_A_B_inputs_verified": True, "costs_reset_between_stages": False}


def _normalized_export(preview, question, result):
    files = preview["files"]
    keys = {(r["raw_doc_id"], r["raw_para_id"], r["quote"]) for r in result["evidence"]}
    evidence = [deepcopy(r) for r in files["evidence.json"] if (r["doc_id"], r["para_id"], r["quote"]) in keys]
    if len(evidence) != len(keys) or len({(r["doc_id"], r["para_id"]) for r in evidence}) != len(evidence):
        raise ValueError("final_export_requires_exact_once_per_selected_anchor")
    conclusions = [r for r in files["conclusions.json"] if r.get("question_id") == question["question_id"]]
    if len(conclusions) != 1 or conclusions[0]["question"] != question["question"]: raise ValueError("one_actual_exported_original_question_required")
    conclusion, by_id = conclusions[0], {r["fact_id"]: r for r in evidence}
    sides = {}
    for kind in ("supporting_facts", "conflicting_facts"):
        sides[kind] = []
        for ref in conclusion[kind]:
            fact = by_id.get(ref["fact_id"])
            if fact is None or any(fact[k] != ref[k] for k in ("doc_id", "para_id", "quote")): raise ValueError("final_export_relation_not_in_question_evidence")
            sides[kind].append({**deepcopy(fact), "fact_des": ref["fact_des"]})
    return {"question_id": question["question_id"], "conclusion": conclusion["conclusion"], "evidence": evidence, **sides,
            "normalization": "actual-browser-competition-export-rows-no-repair-v1", "contains_reference_material": False}


def _delivery_sets(scope, proof, paragraphs):
    """Separate local recall, actual complete delivery and final exported rows."""
    exposed = set(scope["exposed_paragraph_ids"])
    read_ids = set(scope["returned_complete_paragraph_ids"])
    by_id = {p["id"]: (p["raw_doc_id"], p["raw_para_id"]) for p in paragraphs}
    extra = {}
    candidate_ids = exposed | read_ids
    if proof.get("parent_token_pack") is True:
        pool = scope.get("recall_pool_paragraph_ids")
        if (scope.get("parent_token_pack") is not True
                or scope.get("parent_token_pack_policy") != proof.get("parent_token_pack_policy")
                or not isinstance(pool, list) or len(set(pool)) != len(pool)
                or scope.get("recall_pool_count") != len(pool) or not read_ids <= set(pool)
                or exposed != read_ids or not set(pool) <= set(by_id)):
            raise ValueError("token_pack_recall_and_actual_complete_delivery_changed")
        candidate_ids = set(pool)
        extra = {"parent_token_pack": True, "program_recall_pool_parents": len(pool),
            "actually_delivered_complete_parents": len(read_ids),
            "recalled_but_not_delivered_parents": len(candidate_ids - read_ids),
            "program_recall_is_model_reading": False, "model_received_candidate_previews": False,
            "parent_text_truncated": False}
    if not candidate_ids <= set(by_id) or not read_ids <= set(by_id):
        raise ValueError("source_delivery_ids_outside_immutable_originals")
    return {by_id[i] for i in candidate_ids}, {by_id[i] for i in read_ids}, exposed, read_ids, extra


def _historical_baseline(historical, question, raw, contexts, guards):
    """An unavailable historical run never fabricates a before score or cost."""
    if not isinstance(historical, dict) or not isinstance(historical.get("result"), dict):
        return None, None, {"available": False, "reason": "historical_result_not_supplied"}
    if historical["result"].get("exists") is False:
        return None, None, {"available": False, "reason": "historical_result_missing_in_prelaunch_metadata",
            "prelaunch_result_metadata": historical["result"], "baseline_generated_for_comparison": False}
    bindings = {}
    for key in ("closed_receipt", "result", "raw_input"):
        if key not in historical: continue
        row = historical[key]
        if "sha256" not in row:
            if (row.get("exists") is not True or row.get("hash_deferred_until_output_closed") is not True
                    or row.get("body_parsed") is not False or row.get("reference_answer_read") is not False):
                raise ValueError("historical_unhashed_binding_requires_explicit_prelaunch_metadata")
            data, closed = bound(row["file"])
            if len(data) != row.get("bytes"):
                raise ValueError("historical_metadata_file_size_changed")
            row = closed
        bindings[key] = row
    terminal, old_result = descriptor(bindings["closed_receipt"], guards), descriptor(bindings["result"], guards)
    if terminal.get("status") != "succeeded" or old_result.get("origin_run_id") != terminal.get("id", terminal.get("run_id")):
        raise ValueError("closed_historical_v15_result_required")
    if terminal.get("artifacts", {}).get("result") and terminal["artifacts"]["result"]["sha256"] != bindings["result"]["sha256"]:
        raise ValueError("historical_terminal_result_hash_disagrees")
    if bindings.get("raw_input") and descriptor(bindings["raw_input"], guards) != raw:
        raise ValueError("historical_original_raw_sources_changed")
    verify_legacy_sources(old_result, contexts)
    return normalize_legacy(old_result, question), known_usage(old_result["usage"]), {"available": True, "bindings": bindings}


def prepare_closed(path, expected_sha256, browser_binding, browser_sha256):
    guards = []
    closure = read_bound(path, guards, expected_sha256)
    if closure.get("format") != FORMAT or closure.get("closed") is not True or closure.get("fixture") is not False or closure.get("real_provider") is not True:
        raise ValueError("real_UI_bundle_terminal_closure_required")
    artifacts = {k: descriptor(v, guards) for k, v in closure["artifacts"].items()}
    run, activity = artifacts["run"], artifacts["activity"]
    if run.get("id") != closure["run_id"] or not run.get("completed_at") or run.get("status") != closure["status"]: raise ValueError("terminal_run_identity_changed")
    ui = descriptor(closure["ui_start_proof"], guards)
    verify_ui_start(ui, run, closure["input_freeze"]["sha256"])
    guards.append(bound(ui["browser_observation"]["file"], ui["browser_observation"]["sha256"])[1])
    if closure["status"] != "succeeded":
        if "question_result" in artifacts: raise ValueError("failed_episode_cannot_borrow_old_final")
        return {"failed": True, "closure": closure, "run": run, "guards": guards}
    freeze = descriptor(closure["input_freeze"], guards)
    expected_prompt = (LC_PROMPT_VERSION if artifacts["question_result"].get("source_bundle_proof", {}).get("response", {}).get("proof", {}).get("format") == LC_PROOF_FORMAT else PROMPT_VERSION)
    if (freeze.get("format") != "bundle-workbench-input-freeze-v1" or freeze.get("source_bundle_protocol") not in {SOURCE_BUNDLE_PROTOCOL, SOURCE_CONTEXT_PROTOCOL}
            or freeze.get("prompt_version") != expected_prompt or freeze.get("raw_topic_id") != closure["raw_topic_id"]
            or freeze["question"]["question_id"] != closure["raw_question_id"]): raise ValueError("bundle_prelaunch_freeze_contract_changed")
    result, snapshot, revision, docs, paragraphs = (artifacts[k] for k in ("question_result", "snapshot", "revision", "source_documents", "source_paragraphs"))
    if result.get("source_bundle_protocol") != freeze["source_bundle_protocol"]:
        raise ValueError("bundle_frozen_source_geometry_protocol_changed")
    qid, tid = closure["question_internal_id"], closure["topic_id"]
    questions = [q for q in artifacts["questions"] if q.get("id") == qid]
    if (run.get("question_ids") != [qid] or run.get("topic_id") != tid or run.get("cancel_requested") is not False
            or revision.get("state") != "ready" or revision.get("id") != run.get("target_revision_id") or revision.get("topic_id") != tid
            or snapshot.get("run_id") != run["id"] or snapshot.get("revision_id") != revision["id"] or snapshot.get("topic_id") != tid
            or snapshot.get("question_results", {}).get(qid) != result or result.get("origin_run_id") != run["id"]
            or len(questions) != 1 or questions[0].get("raw_question_id") != freeze["question"]["question_id"] or questions[0].get("question") != freeze["question"]["question"]):
        raise ValueError("published_snapshot_must_be_this_terminal_UI_task")
    from app.workflows.analysis import _project_question
    question = _project_question(questions[0])
    _proof_guards(result.get("source_bundle_proof"), guards)
    validate_result(result, question, paragraphs, docs)
    current_usage = run["usage"].get(qid, [])
    if result.get("usage") != current_usage or activity.get("usage") != run["usage"] or activity.get("run", {}).get("id") != run["id"]:
        raise ValueError("run_activity_and_published_usage_mismatch")
    raw = descriptor(freeze["raw_sources"], guards)
    contexts, _ = verify_source_rows(raw, paragraphs)
    _verify_original_documents(raw, docs, paragraphs)
    native_files = artifacts["native_episode_manifest"]
    resume, failed_cost_audit = None, None
    if run.get("resumed_from_run_id"):
        if "resume_lineage" not in artifacts:
            raise ValueError("resumed_run_requires_explicit_original_ancestor_files")
        resume = _resume_source_files(artifacts["resume_lineage"], run, activity, result, native_files, guards,
            question=question, documents=docs, paragraphs=paragraphs)
        native_files = resume["native"]
        if any(row.get("operation") not in TAIL for row in current_usage):
            raise ValueError("resumed_complete_source_must_not_repeat_paid_front")
        front = _front_usage(result, resume["source_records"], guards)
        if _usage_signature(front["previous_records"]) != _usage_signature(resume["previous_records"]):
            raise ValueError("resume_front_historical_usage_dropped_or_duplicated")
        provenance = result["execution_provenance"]
        for row in [*result["analyses"], result["final"]]:
            role = row.get("role", "arbitrate")
            if row.get("execution_provenance") != {**provenance, "reused": role in provenance["reused_nodes"]}:
                raise ValueError("resume_published_role_provenance_changed")
    elif native_files["tail"].get("cached_requests"):
        if "resume_lineage" not in artifacts:
            raise ValueError("cached_tail_requires_explicit_cost_lineage")
        resume = _cached_source_files(artifacts["resume_lineage"], run, activity, result, native_files, guards,
            question=question, documents=docs, paragraphs=paragraphs)
        native_files = resume["native"]
        front = _front_usage(result, resume["source_records"], guards)
        if _usage_signature(front["previous_records"]) - _usage_signature(resume["previous_records"]):
            raise ValueError("cached_tail_dropped_original_source_cost_parents")
        front["historical_front_previous_records"] = front["previous_records"]
        front["previous_records"] = resume["previous_records"]
    else:
        if result.get("execution_provenance"):
            raise ValueError("non_resumed_run_cannot_borrow_checkpoint_front")
        front = _front_usage(result, current_usage, guards)
        if "resume_lineage" in artifacts:
            failed_cost_audit = _native_failed_cost_parents(artifacts["resume_lineage"], run, activity, result,
                native_files, freeze, front, guards)
        elif front["previous_records"] or freeze.get("cost_parent_run_ids"):
            raise ValueError("new_native_generation_requires_explicit_failed_cost_parent_files")
    metadata = descriptor(artifacts["native_episode_manifest"]["tail"]["source-bundle-episode.json"], guards)
    selection_mode = _selection_mode(result["source_bundle_proof"]["response"]["proof"], metadata.get("source_bundle_config", {}), run)
    tail = _tail(result, question, native_files, current_usage, front, freeze, guards)
    tail["selection_mode"] = selection_mode
    if resume is not None:
        tail["checkpoint_continuation_audit"] = resume["audit"]
    if failed_cost_audit is not None:
        tail["failed_cost_parent_audit"] = failed_cost_audit
    preview = artifacts["api_submission_preview"]
    required_api = {"graph": {"revision_id", "graph_revision", "nodes", "edges", "scope"},
        "artifacts": {"revision_id", "conclusions", "conflicts", "confidence", "agent_outputs", "source_groups", "scope"}}
    for key, required in required_api.items():
        section = artifacts["api_publication"].get(key)
        if not isinstance(section, dict) or not required <= set(section) or any(section[k] != snapshot[k] for k in required):
            raise ValueError("API_publication_must_equal_closed_snapshot")
    readiness = artifacts["export_readiness"]
    if readiness.get("revision_id") != revision["id"] or readiness.get("topic_id") != tid:
        raise ValueError("export_readiness_must_bind_closed_revision")
    if preview.get("topic_id") != tid or preview.get("revision_id") != revision["id"] or preview["files"] != competition_files(snapshot, source_documents=docs, source_paragraphs=paragraphs):
        raise ValueError("actual_API_export_differs_from_published_source_snapshot")
    binding = read_bound(browser_binding, guards, browser_sha256)
    if binding.get("format") != "bundle-actual-browser-download-binding-v1" or binding.get("closed_inputs", {}).get("sha256") != expected_sha256:
        raise ValueError("actual_browser_download_binding_required_before_reference_IO")
    proof = descriptor(binding["ui_export_proof"], guards)
    downloaded, download_guard = bound(binding["download_file"]["file"], binding["download_file"]["sha256"]); guards.append(download_guard)
    api_bytes, _ = bound(closure["artifacts"]["api_submission_preview"]["file"], closure["artifacts"]["api_submission_preview"]["sha256"])
    if (downloaded != api_bytes or proof.get("actual_browser_UI_download") is not True or proof.get("download_file") != binding["download_file"]
            or binding.get("exact_API_preview_bytes_equal") is not True or proof.get("exact_API_preview_bytes_equal") is not True
            or proof.get("run_id") != run["id"] or proof.get("revision_id") != revision["id"] or proof.get("whole_question_tokens") != tail["whole_episode_known_tokens"]
            or proof.get("exact_API_preview_sha256") != sha256(api_bytes).hexdigest()): raise ValueError("actual_browser_download_bytes_or_identity_changed")
    baseline, baseline_usage, baseline_availability = _historical_baseline(freeze.get("historical_v15"), freeze["question"], raw, contexts, guards)
    scope = result["source_bundle_proof"]["response"]["proof"]
    if scope.get("format") == LC_PROOF_FORMAT:
        delivered_scope = descriptor(scope["scope"], guards)
    else:
        delivered_scope = descriptor(scope["tool_submission"], guards)["scope"]
    candidates, read_keys, exposed, read_ids, delivery_counts = _delivery_sets(delivered_scope, scope, paragraphs)
    # Only the CLOSED, native-message/geometry-replayed context scope can
    # authorize a partial quote. These ranges never inflate full-parent reads.
    delivered_context_ranges = (deepcopy(delivered_scope["delivered_source_passages"])
        if scope.get("context_passages") is True else None)
    return {"failed": False, "closure": closure, "run": run, "snapshot": snapshot, "result": result, "question": freeze["question"],
        "contexts": contexts, "raw": raw, "candidates": candidates, "read_keys": read_keys, "normalized": _normalized_export(preview, freeze["question"], result),
        "delivered_context_ranges": delivered_context_ranges,
        "baseline": baseline, "baseline_usage": baseline_usage, "baseline_availability": baseline_availability,
        "front": front, "tail": tail, "guards": guards,
        "browser_download": binding, "source_scope": {"search_preview_paragraphs": 0 if scope.get("parent_token_pack") is True else len(exposed),
            "native_source_anchor_paragraphs": len(exposed), "native_full_read_paragraphs": len(read_ids),
            "native_source_passage_count": len(delivered_scope.get("delivered_source_passages", [])),
            "native_partial_context_paragraphs": len(set(delivered_scope.get("partially_provided_paragraph_ids", []))),
            "partial_parent_scope_is_unknown_outside_delivered_ranges": scope.get("context_passages", False),
            "final_exported_anchor_records": len(result["evidence"]), "preview_is_full_read": False, "read_is_comprehension": False,
            **delivery_counts}}


def score_episode(path, expected_sha256, reference_evidence, reference_conclusions, *, browser_binding, browser_sha256):
    prepared = prepare_closed(path, expected_sha256, browser_binding, browser_sha256)
    guards = prepared["guards"]
    if prepared["failed"]:
        return {"format": "closed-bundle-workbench-quality-score-v1", "status": prepared["closure"]["status"],
            "run_id": prepared["run"]["id"], "after": None, "before": None, "final_precision_recall": None,
            "usage_observation": prepared["closure"]["usage_observation"], "references_read": False, "new_model_calls": 0, "input_guards": guards}
    # FIRST reference access: this task, source proof, 3 real tail calls, whole
    # cost, published export and actual browser bytes have all passed checks.
    evidence = read_bound(reference_evidence, guards)
    conclusions = read_bound(reference_conclusions, guards)
    question = prepared["question"]
    references = [q for q in conclusions if q.get("question_id") == question["question_id"] and q.get("question") == question["question"]]
    if len(references) != 1: raise ValueError("one_same_original_reference_question_required")
    _, engine, mapper_guards = load_reference_tools(); guards.extend(mapper_guards)
    after = score_normalized(question, prepared["normalized"], evidence, references[0], prepared["contexts"], prepared["candidates"], prepared["read_keys"],
        engine=engine, delivered_context_ranges=prepared.get("delivered_context_ranges"))
    before = (score_normalized(question, prepared["baseline"], evidence, references[0], prepared["contexts"], set(prepared["contexts"]), set(prepared["contexts"]), engine=engine)
        if prepared.get("baseline") is not None else None)
    for item in guards: bound(item["file"], item["sha256"])
    return {"format": "closed-bundle-workbench-quality-score-v1", "status": "succeeded", "question": question,
        "after": after, "before": before, "after_source_scope": prepared["source_scope"], "after_tail_audit": prepared["tail"],
        "after_front_usage": prepared["front"]["report"], "before_usage": prepared["baseline_usage"], "before_usage_is_new_cost": False,
        "before_availability": prepared.get("baseline_availability", {"available": before is not None}),
        "actual_browser_download_audit": prepared["browser_download"], "published_document_bundle_quotes_verified": True,
        "legacy_literal_diagnostic_limit": "The unchanged scorer checks quote literals inside original anchor paragraphs. Its literal_quote_issues are parent-only diagnostics and cannot invalidate already verified document-scope bundle spans.",
        "actual_export_rows_scored": True, "proposed_or_candidate_rows_scored_as_final": False,
        "historical_before_actual_after_comparison": before is not None, "same_input_framework_causal_pair": False,
        "official_score": False, "semantic_B_point_score": None, "new_model_calls": 0, "new_paid_judge_calls": 0,
        "references_written_to_runtime": False, "input_guards": guards, "input_guards_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("closed-inputs", "expected-sha256", "reference-evidence", "reference-conclusions", "browser-binding", "browser-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / "docs/reports"): parser.error("offline scores must be saved under docs/reports")
    result = score_episode(args.closed_inputs, args.expected_sha256, args.reference_evidence, args.reference_conclusions,
        browser_binding=args.browser_binding, browser_sha256=args.browser_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2); stream.write("\n")
    print(json.dumps({"output": str(output), "status": result["status"], "new_model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__": main()
