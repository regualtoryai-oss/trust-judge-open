"""Offline source-ID diagnostics for failed real arms, never final quality scores.

Only call after both actual generations are CLOSED. This diagnostic does not
repair, replay or accept either tool attempt and never writes runtime inputs.
"""
import argparse
import json
from pathlib import Path

from scripts.audit_dsh_closed_episode import audit_closed_episode, bound
from scripts.score_real_quality_episode import (ROOT, full_context, original_sources,
    read_bound, read_json_lines, parse_tool_args, union_rows, validate_case)
from scripts.offline_reference_tools import load_reference_tools
from scripts.two_stage_evidence_matching import location


def diagnose(before_receipt, before_manifest, after_receipt, after_manifest, raw_input, sample_root):
    guards, arms = [], {}
    for name, rp, mp, tool in (("before", before_receipt, before_manifest, "submit_evidence"),
                               ("after", after_receipt, after_manifest, "read_selected_sources")):
        receipt = read_bound(rp, guards)
        usage = audit_closed_episode(rp, bound(rp)[1]["sha256"], mp, root=ROOT)
        guards.extend(usage["input_guards"])
        if usage["has_final_artifact"] or receipt["status"] not in {"failed_model_turn", "budget_blocked"}:
            raise ValueError("failed_real_closed_arms_without_final_required")
        manifest = read_bound(mp, guards, receipt["manifest_sha256"])
        case = read_bound(manifest["casePath"], guards, receipt["case_sha256"])
        sdk = read_bound(Path(rp).parent / "sdk-summary.json", guards, receipt["sdk_summary_sha256"])
        log = receipt["official_session_log"]
        events = read_json_lines(log["file"], guards, log["sha256"])
        calls = [row for row in sdk["trace"] if row.get("event_type") == "tool/call"]
        official = [row for row in events if row.get("type") == "tool/call"]
        if (len(calls) != 1 or len(official) != 1 or calls[0].get("tool") != tool
                or official[0]["data"].get("name") != tool
                or calls[0]["arguments"] != official[0]["data"]["arguments"]):
            raise ValueError("same_single_original_bound_tool_attempt_required")
        args = parse_tool_args(calls[0]["arguments"])
        if name == "before":
            if not all(isinstance(row, list) and len(row) == 5 and isinstance(row[0], str)
                       for row in args["selected"]):
                raise ValueError("diagnostic_requires_unmodified_exact_tuple_attempt")
            refs = [row[0] for row in args["selected"]]
        else:
            refs = args["refs"]
        by_ref = {row["paragraph_ref"]: row for row in case["catalog"]}
        if len(refs) != len(set(refs)) or not set(refs) <= set(by_ref):
            raise ValueError("unique_original_candidate_addresses_required")
        arms[name] = {"receipt": receipt, "usage": usage, "manifest": manifest,
            "case": case, "refs": refs, "by_ref": by_ref}
    before, after = arms["before"], arms["after"]
    if (after["receipt"]["model_requests"] != 1 or after["receipt"]["status"] != "budget_blocked"
            or [row["request_sequence"] for row in after["usage"]["blocked_before_request"]] != [2]):
        raise ValueError("after_second_request_must_be_blocked_before_full_delivery")
    prior = after["manifest"]["priorClosedReceipt"]
    if (Path(prior["file"]).resolve() != Path(before_receipt).resolve()
            or prior["sha256"] != bound(before_receipt)[1]["sha256"]
            or after["manifest"]["priorKnownTokens"] != before["usage"]["known_total_tokens"]
            or before["case"]["question"] != after["case"]["question"]
            or before["case"]["topic_title"] != after["case"]["topic_title"]):
        raise ValueError("same_question_and_actual_prior_cost_required")
    raw = read_bound(raw_input, guards)
    contexts, documents = original_sources(raw)
    for arm in arms.values():
        validate_case(arm["case"], raw, contexts, documents)
    complete = read_bound(Path(before_receipt).parent / "complete-context.txt", guards,
                          before["receipt"]["complete_context_sha256"])
    if complete != full_context(before["case"]):
        raise ValueError("before_full_source_context_changed")
    question = before["case"]["question"]
    # FIRST reference access after both real closures, usage and raw binding.
    folder = Path(sample_root) / before["case"]["topic_title"]
    evidence = read_bound(folder / "evidence_samples.json", guards)
    conclusions = read_bound(folder / "conclusions.json", guards)
    reference = [row for row in conclusions if row["question_id"] == question["question_id"]]
    if len(reference) != 1 or reference[0]["question"] != question["question"]:
        raise ValueError("same_reference_question_required")
    _, engine, mapper_guards = load_reference_tools()
    guards.extend(mapper_guards)
    issues = []
    sides = {kind: engine._relation_rows(reference[0][kind], evidence, submitted=False,
        issues=issues, context=[question["question_id"], kind])
        for kind in ("supporting_facts", "conflicting_facts")}
    target = union_rows(*sides.values())
    diagnostics = {}
    for name, arm in arms.items():
        rows = [arm["by_ref"][p] for p in arm["refs"]]
        metrics = engine.compare_rows(question["question_id"], rows, target)["unique_paragraphs"]["A_id_only"]["metrics"]
        scope = list(arm["by_ref"].values())
        pool = engine.compare_rows(question["question_id"], scope, target)["unique_paragraphs"]["A_id_only"]["metrics"]
        diagnostics[name] = {"stage": "rejected_original_submit_attempt" if name == "before" else "requested_full_read_addresses",
            "not_accepted_final_output": True, "source_id_diagnostic_only": metrics,
            "candidate_pool_source_id_diagnostic_only": pool,
            "source_ids": [{"p": p, "doc_id": arm["by_ref"][p]["doc_id"], "para_id": arm["by_ref"][p]["para_id"]} for p in arm["refs"]],
            "usage": arm["usage"], "final_precision": None, "final_recall": None,
            "S_C_final_quality_available": False, "semantic_B_point_score": None}
    guards.append(bound(__file__)[1])
    for descriptor in guards:
        bound(descriptor["file"], descriptor["sha256"])
    return {"format": "failed-real-pair-source-stage-diagnostic-v1", "question": question,
        "diagnostics": diagnostics, "reference_issues": issues, "input_guards": guards,
        "both_real_arms_closed_before_reference_access": True,
        "whole_question_tokens": after["usage"]["known_total_tokens"],
        "after_full_read_delivered_to_started_model_request": False,
        "raw_attempt_vs_read_request_is_quality_improvement_comparison": False,
        "new_model_calls": 0, "replay_performed": False, "runtime_reference_write": False,
        "official_score": False, "published": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before-receipt", "before-manifest", "after-receipt", "after-manifest", "raw-input", "sample-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = diagnose(args.before_receipt, args.before_manifest, args.after_receipt, args.after_manifest, args.raw_input, args.sample_root)
    with Path(args.output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: v["source_id_diagnostic_only"] for k, v in result["diagnostics"].items()}, ensure_ascii=False))
