"""CLOSED-only OFFLINE DSH/legacy comparison. No model, settings or credential IO."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from scripts.audit_dsh_closed_episode import audit_closed_episode, bound
from scripts.audit_historical_final_baseline import audit as audit_legacy
from scripts.final_evidence_scoring import normalize_dsh, normalize_legacy, score_final, score_reference_conclusion
from scripts.offline_reference_tools import load_reference_tools
from scripts.two_stage_evidence_matching import location

ROOT = Path(__file__).resolve().parents[1]


def read(path, guards, expected=None):
    data, descriptor = bound(path, expected)
    guards.append(descriptor)
    return json.loads(data.decode("utf-8-sig"))


def cache_decisions(guards):
    from scripts.evaluate_semantics import _import_closed_caches
    directories = [ROOT / "docs/reports/v18-purpose-pilot-20261019" / relative for relative in (
        "independent-comparison/judge-cache", "fixed-candidate-purpose/independent-comparison/judge-cache")]
    index, audits, verified = _import_closed_caches(directories)
    guards.extend(verified)
    decisions = []
    for entries in index.values():
        for record in entries:
            row = record["entry"]
            decisions.append({key: row[key] for key in ("payload_sha256", "judge", "prompt_version", "prompt_sha256")}
                             | {"verdict": row["verdict"]["verdict"], "sampling": row.get("sampling"),
                                "closed_provenance": [record["source"]]})
    return decisions, audits


def examples(engine, submitted, reference, scope, contexts):
    comparison = engine.compare_rows("offline", submitted, reference)["unique_paragraphs"]["A_id_only"]
    left, right = engine._units(submitted, True), engine._units(reference, True)
    reference_keys = {location(row) for row in reference if location(row)}
    selected_keys = {location(row) for row in submitted if location(row)}
    def mirrors(key, candidates):
        text = contexts.get(key)
        return [{"doc_id": k[0], "para_id": k[1]} for k in sorted(candidates)
                if k != key and text is not None and contexts.get(k) == text]
    nonreference = []
    for index in comparison["unmatched_submitted_indices"]:
        key = left[index]["key"]
        alternate = mirrors(key, reference_keys) if key else []
        nonreference.append({"doc_id": key[0] if key else None, "para_id": key[1] if key else None,
            "reason": "exact_parent_mirror_of_reference_semantics_unknown" if alternate else "source_id_not_in_reference_semantics_unknown",
            "literal_parent_mirror_candidates": alternate, "semantic_false_fact_asserted": False})
    missed = []
    for index in comparison["unmatched_reference_indices"]:
        key = right[index]["key"]
        alternate = mirrors(key, selected_keys) if key else []
        missed.append({"doc_id": key[0] if key else None, "para_id": key[1] if key else None,
            "reason": "unlocated_reference_retained" if key is None else "outside_frozen_24_source_pool" if key not in scope else "in_source_pool_not_selected_for_this_category_by_id",
            "literal_parent_mirror_candidates": alternate, "cross_id_semantic_match": "unknown"})
    return {"all_nonreference_source_ids": nonreference, "all_missed_reference_source_ids": missed,
            "examples_are_source_id_proxies_not_semantic_false_positives": True}


def score_closed(receipt_path, expected_sha256, manifest_path, *, baseline_root, case_id, raw_input, sample_root):
    usage = audit_closed_episode(receipt_path, expected_sha256, manifest_path, root=ROOT)
    guards = list(usage["input_guards"])
    receipt = read(receipt_path, guards, expected_sha256)
    if (receipt["status"] != "closed_proposed_unverified" or receipt["turn_end"] != {"kind": "completed"}
            or receipt["error_present"] or receipt["cleanup_error"] is not None or not usage["has_final_artifact"]):
        raise ValueError("complete_proposed_submission_required_before_reference_access")
    baseline = audit_legacy(baseline_root, [case_id])
    old = baseline["cases"][0]
    source = read(old["frozen_input"]["file"], guards, old["frozen_input"]["sha256"])
    case_path = next(g["file"] for g in usage["input_guards"] if g["sha256"] == receipt["case_sha256"])
    case = read(case_path, guards, receipt["case_sha256"])
    if (case["question"] != {key: source["question"][key] for key in ("question_id", "question")}
            or case["topic_title"] != source["title"] or len(case["catalog"]) != len(source["paragraphs"])):
        raise ValueError("same_original_question_and_complete_source_scope_required")
    raw = read(raw_input, guards, old["original_raw_sha256"])
    contexts, documents = {}, {}
    for doc in raw["documents"]:
        documents[doc["doc_id"]] = doc
        for p in doc["paragraphs"]:
            text = doc["content"][p["char_start"]:p["char_end"]]
            if text != p["text"] or (doc["doc_id"], p["para_id"]) in contexts:
                raise ValueError("original_raw_literal_source_required")
            contexts[(doc["doc_id"], p["para_id"])] = text
    for index, (parent, frozen) in enumerate(zip(case["catalog"], source["paragraphs"]), 1):
        meta = case["source_metadata"][parent["paragraph_ref"]]
        key = frozen["raw_doc_id"], frozen["raw_para_id"]
        if (parent["paragraph_ref"] != f"P{index}" or location(parent) != key or parent["text"] != frozen["text"]
                or contexts.get(key) != parent["text"] or parent["title"] != frozen["title"]
                or documents[key[0]]["content"][frozen["start_cp"]:frozen["end_cp"]] != parent["text"]
                or meta["source_text_sha256_utf8"] != sha256(parent["text"].encode()).hexdigest()):
            raise ValueError("raw_case_original_ids_fulltext_title_order_or_spans_changed")
    context_path = Path(receipt_path).parent / "complete-context.txt"
    data, descriptor = bound(context_path, receipt["complete_context_sha256"]); guards.append(descriptor)
    visible = json.loads(data)
    titles, aliases, paragraphs = [], {}, []
    for parent in case["catalog"]:
        if parent["doc_id"] not in aliases:
            alias = f"D{len(aliases)+1}"; aliases[parent["doc_id"]] = alias; titles.append([alias, parent["title"]])
        paragraphs.append([parent["paragraph_ref"], aliases[parent["doc_id"]], parent["text"]])
    if visible != {"question": case["question"]["question"], "titles": titles, "paragraph_columns": ["p", "d", "text"],
                   "paragraphs": paragraphs, "candidate_scope_only": True, "outside_context_unknown": True}:
        raise ValueError("full_original_context_binding_changed")
    submission = read(Path(receipt_path).parent / "submission.json", guards, receipt["submission_sha256"])
    current = normalize_dsh(submission, case["question"])
    for row in current["evidence"]:
        if location(row) not in {location(r) for r in case["catalog"]} or row["quote"] != contexts[location(row)]:
            raise ValueError("submitted_full_parent_quote_or_source_changed")
    old_result = read(old["artifacts"]["result"]["file"], guards, old["artifacts"]["result"]["sha256"])
    legacy = normalize_legacy(old_result, case["question"])
    for row in old_result["evidence"]:
        p = next(p for p in source["paragraphs"] if p["id"] == row["paragraph_id"])
        if p["text"][row["start_cp"]:row["end_cp"]] != row["quote"]:
            raise ValueError("historical_quote_span_changed")
    # FIRST answer-reference access: every generation/usage/raw/source gate above.
    folder = Path(sample_root).resolve() / case["topic_title"]
    evidence = read(folder / "evidence_samples.json", guards)
    conclusions = read(folder / "conclusions.json", guards)
    refs = [q for q in conclusions if q["question_id"] == case["question"]["question_id"]]
    if len(refs) != 1 or refs[0]["question"] != case["question"]["question"]:
        raise ValueError("same_original_reference_question_required")
    _, engine, mapper_guards = load_reference_tools(); guards.extend(mapper_guards)
    decisions, cache_audits = cache_decisions(guards)
    source_scope = {location(row) for row in case["catalog"]}
    issues = []
    sides = {side: engine._relation_rows(refs[0][side], evidence, submitted=False, issues=issues, context=[case_id, side])
             for side in ("supporting_facts", "conflicting_facts")}
    reference = {"evidence": evidence, **sides}
    scoped = lambda rows: [r for r in rows if location(r) is None or location(r) in source_scope]
    runs = {}
    for name, normalized in (("dsh", current), ("v18", legacy)):
        full = score_reference_conclusion(case["question"], normalized, evidence, refs[0], contexts, closed_decisions=decisions)
        within = score_final(case["question"], normalized, scoped(evidence), scoped(sides["supporting_facts"]),
                             scoped(sides["conflicting_facts"]), contexts, closed_decisions=decisions, engine=engine)
        runs[name] = {"full_reference": full, "within_frozen_source_scope_reference": within,
                      "id_error_examples": {kind: examples(engine, normalized[kind], reference[kind], source_scope, contexts)
                                            for kind in reference}}
    implementation = [ROOT / "scripts" / name for name in ("score_dsh_closed_comparison.py", "audit_dsh_closed_episode.py",
        "audit_historical_final_baseline.py", "final_evidence_scoring.py", "two_stage_evidence_matching.py", "evaluate_semantics.py")]
    guards.extend(bound(path)[1] for path in implementation)
    for desc in guards: bound(desc["file"], desc["sha256"])
    return {"format": "closed-dsh-historical-route-final-comparison-v1", "case_id": case_id,
        "question": case["question"], "source_scope_paragraphs": len(source_scope), "whole_topic_paragraphs": len(contexts),
        "comparison_kind": "historical_architecture_route_not_pure_framework_causal_effect", "dsh_usage_audit": usage,
        "historical_usage": old["actual_usage_totals"], "historical_calls": len(old["actual_usage_by_operation"]),
        "historical_elapsed_seconds": old["elapsed_seconds"], "runs": runs, "reference_issues": issues,
        "semantic_cache_imports": cache_audits, "cached_votes_are_legacy_single_not_three": True,
        "full_question_not_blind_unseen": True, "official_score": False, "new_paid_judge_calls": 0,
        "new_model_calls": 0, "reference_written_to_runtime": False, "input_guards": guards, "input_guards_unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "expected-sha256", "manifest", "baseline-root", "case-id", "raw-input", "sample-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = score_closed(args.receipt, args.expected_sha256, args.manifest, baseline_root=args.baseline_root,
        case_id=args.case_id, raw_input=args.raw_input, sample_root=args.sample_root)
    with Path(args.output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({"case_id": result["case_id"], "new_model_calls": 0,
        "output_sha256": sha256(Path(args.output).read_bytes()).hexdigest()}))
