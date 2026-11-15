"""CLOSED-only offline quality diagnostics. Never loads credentials or calls a model.

Question selection targets are that question's reference S-union-C, not all topic
evidence. ID-only figures are proxies; semantic uncertainty remains explicit.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from scripts.audit_dsh_closed_episode import audit_closed_episode, bound
from scripts.final_evidence_scoring import normalize_dsh, normalize_legacy, score_final
from scripts.offline_reference_tools import load_reference_tools
from scripts.two_stage_evidence_matching import location

ROOT = Path(__file__).resolve().parents[1]


def read_bound(path, guards, expected=None):
    data, descriptor = bound(path, expected)
    guards.append(descriptor)
    return json.loads(data.decode("utf-8-sig"))


def union_rows(*groups):
    """Deduplicate located parents; unresolved reference facts stay denominators."""
    rows, seen = [], set()
    for group in groups:
        for row in group:
            key = ("located", *location(row)) if location(row) else (
                "unlocated", json.dumps(row, ensure_ascii=False, sort_keys=True))
            if key not in seen:
                seen.add(key)
                rows.append(deepcopy(row))
    return rows


def brief_category(category):
    """Keep both official readings without huge unknown pair payload dumps."""
    bounds = category["two_interpretation_bounds"]["unique_paragraphs"]
    return {
        "A_id_only_proxy": category["A_source_id_proxy"]["unique_paragraphs"]["A_id_only"]["metrics"],
        "raw_record_A_id_only_proxy": category["A_source_id_proxy"]["raw_records"]["A_id_only"]["metrics"],
        "semantic_interpretations": {kind: {
            "lower": values["lower"]["metrics"], "upper": values["upper"]["metrics"],
            "point_estimate_available": values["point_estimate_available"]}
            for kind, values in bounds["interpretations"].items()},
        "semantic_counts": bounds["counts"], "semantic_B_point_score": None,
        "upper_bound_is_measured_score": False,
    }


def _context_delivery(contexts, candidates, read_keys, delivered_context_ranges):
    """Validate explicit, already native-verified parent-local deliveries.

    This is a range guard, not a substitute for the caller's CLOSED native
    message/geometry replay. Adjacent or overlapping rows are never unioned.
    In particular, local retrieval windows do not become complete-parent reads.
    """
    if not isinstance(delivered_context_ranges, list):
        raise ValueError("delivered_context_ranges_must_be_explicit_list")
    ranges = {}
    for row in delivered_context_ranges:
        if not isinstance(row, dict):
            raise ValueError("delivered_context_range_requires_original_source_and_local_cp")
        key = (row.get("raw_doc_id"), row.get("raw_para_id"))
        if (not all(isinstance(x, str) and x for x in key)
                or (location(row) is not None and location(row) != key)
                or key not in candidates):
            raise ValueError("delivered_context_range_requires_candidate_original_parent")
        parent, quote = contexts.get(key), row.get("quote")
        start, end = row.get("local_start_cp"), row.get("local_end_cp")
        if (not isinstance(parent, str) or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(parent) or not isinstance(quote, str)
                or parent[start:end] != quote):
            raise ValueError("delivered_context_range_local_cp_quote_not_literal")
        ranges.setdefault(key, []).append({"local_start_cp": start, "local_end_cp": end, "quote": quote})
    return ranges


def _selected_context_delivery(selected, contexts, read_keys, ranges):
    for row in selected:
        key, quote = location(row), row.get("quote")
        # An independently verified full read still authorizes its full parent.
        # Otherwise one single actual range must contain this entire quote;
        # do not join ranges, infer a gap, or borrow another parent's text.
        texts = ([contexts.get(key)] if key in read_keys else []) + [r["quote"] for r in ranges.get(key, [])]
        if not isinstance(quote, str) or not quote or not any(isinstance(t, str) and quote in t for t in texts):
            raise ValueError("selected_quote_requires_single_actually_delivered_source_range")


def source_errors(selected, target, candidates, read_keys, contexts, *, context_ranges=None):
    """Locate ID attrition, never call every nonreference source a false fact."""
    context_ranges = context_ranges or {}
    selected_keys = {location(r) for r in selected if location(r)}
    target_keys = {location(r) for r in target if location(r)}
    def mirrors(key, other):
        text = contexts.get(key)
        return [{"doc_id": a, "para_id": b} for a, b in sorted(other)
                if (a, b) != key and text is not None and contexts.get((a, b)) == text]
    missed = []
    for row in union_rows(target):
        key = location(row)
        if key in selected_keys:
            continue
        reason = ("unlocated_reference_retained" if key is None else
                  "outside_candidate_pool" if key not in candidates else
                  "partial_context_delivered_not_selected_outside_ranges_unknown" if key in context_ranges and key not in read_keys else
                  "candidate_not_completely_delivered" if key not in read_keys else
                  "complete_parent_delivered_not_selected")
        item = {"doc_id": key[0] if key else None, "para_id": key[1] if key else None,
                       "reason": reason, "literal_parent_mirrors_selected": mirrors(key, selected_keys),
                       "cross_id_semantic_match": "unknown"}
        if key in context_ranges and key not in read_keys:
            quote = row.get("quote")
            item.update(delivered_local_cp_ranges=[
                [r["local_start_cp"], r["local_end_cp"]] for r in context_ranges[key]],
                reference_quote_within_single_delivered_range=(
                    any(quote in r["quote"] for r in context_ranges[key]) if isinstance(quote, str) and quote else None),
                outside_delivered_context_semantics="unknown")
        missed.append(item)
    nonreference = []
    for row in union_rows(selected):
        key = location(row)
        if key in target_keys:
            continue
        nonreference.append({"doc_id": key[0] if key else None, "para_id": key[1] if key else None,
                             "reason": "source_id_not_in_this_question_reference_union_semantics_unknown",
                             "literal_parent_mirrors_of_reference": mirrors(key, target_keys),
                             "semantic_false_fact_asserted": False})
    return {"missed_reference_sources": missed, "nonreference_selected_sources": nonreference,
            "nonreference_is_semantic_false_positive": False}


def score_normalized(question, normalized, reference_evidence, reference_conclusion,
                     contexts, candidates, read_keys, *, engine=None, delivered_context_ranges=None):
    """Pure after-closure scoring function; no reference paths or API access."""
    if reference_conclusion.get("question_id") != question["question_id"] or reference_conclusion.get("question") != question["question"]:
        raise ValueError("same_original_reference_question_required")
    if not read_keys <= candidates:
        raise ValueError("selected_must_have_completely_delivered_candidate_parent")
    ranges = {}
    if delivered_context_ranges is None:
        if not {location(r) for r in normalized["evidence"]} <= read_keys:
            raise ValueError("selected_must_have_completely_delivered_candidate_parent")
    else:
        ranges = _context_delivery(contexts, candidates, read_keys, delivered_context_ranges)
        for kind in ("evidence", "supporting_facts", "conflicting_facts"):
            _selected_context_delivery(normalized[kind], contexts, read_keys, ranges)
    if engine is None:
        _, engine, _ = load_reference_tools()
    issues = []
    sides = {kind: engine._relation_rows(reference_conclusion[kind], reference_evidence,
        submitted=False, issues=issues, context=[question["question_id"], kind])
        for kind in ("supporting_facts", "conflicting_facts")}
    target = union_rows(sides["supporting_facts"], sides["conflicting_facts"])
    result = score_final(question, normalized, target, sides["supporting_facts"],
                         sides["conflicting_facts"], contexts, engine=engine)
    diagnostic = score_final(question, normalized, reference_evidence, sides["supporting_facts"],
                             sides["conflicting_facts"], contexts, engine=engine)
    selected = {location(row) for row in normalized["evidence"] if location(row)}
    located_target = {location(row) for row in target if location(row)}
    errors = source_errors(normalized["evidence"], target, candidates, read_keys, contexts, context_ranges=ranges)
    relation_errors = {kind: source_errors(normalized[kind], sides[kind], candidates, read_keys, contexts, context_ranges=ranges)
                       for kind in sides}
    for kind, error_group in relation_errors.items():
        for row in error_group["missed_reference_sources"]:
            if (row["doc_id"], row["para_id"]) in selected:
                row["reason"] = "selected_parent_not_linked_to_this_final_relation"
    return {
        "question_selection": brief_category(result["categories"]["evidence"]),
        "supporting": brief_category(result["categories"]["supporting_facts"]),
        "conflicting": brief_category(result["categories"]["conflicting_facts"]),
        "pipeline": {"candidate_parents": len(candidates), "completely_delivered_parents": len(read_keys),
            "explicit_delivered_context_ranges": sum(len(rows) for rows in ranges.values()),
            "context_delivered_parents": len(ranges), "partial_context_parents": len(set(ranges) - read_keys),
            "selected_parents": len(selected), "question_reference_union_units": len(target),
            "located_question_reference_union": len(located_target),
            "reference_union_in_candidates": len(located_target & candidates),
            "reference_union_completely_delivered": len(located_target & read_keys),
            "reference_union_parent_ids_with_some_delivered_context": len(located_target & set(ranges)),
            "reference_union_parent_ids_with_partial_context_only": len(located_target & (set(ranges) - read_keys)),
            "reference_union_selected": len(located_target & selected),
            "context_ranges_are_complete_parent_reads": False,
            "reference_quote_delivery_or_entailment_unknown": True,
            "delivery_is_comprehension_or_entailment": False},
        "question_selection_errors": errors, "relation_errors": relation_errors,
        "single_question_vs_complete_topic_diagnostic": {
            "score": brief_category(diagnostic["categories"]["evidence"]),
            "not_complete_topic_submission_or_official_T1": True},
        "literal_quote_issues": result["literal_quote_issues"], "reference_issues": issues,
        "conclusion_correctness": "not_semantically_judged", "new_paid_judge_calls": 0,
    }


def original_sources(raw):
    contexts, documents = {}, {}
    for doc in raw["documents"]:
        if doc["doc_id"] in documents:
            raise ValueError("unique_raw_documents_required")
        documents[doc["doc_id"]] = doc
        for paragraph in doc["paragraphs"]:
            key = (doc["doc_id"], paragraph["para_id"])
            text = doc["content"][paragraph["char_start"]:paragraph["char_end"]]
            if key in contexts or text != paragraph["text"]:
                raise ValueError("original_raw_literal_paragraph_required")
            contexts[key] = text
    return contexts, documents


def validate_case(case, raw, contexts, documents):
    by_ref = {}
    if len([q for q in raw["questions"] if {k: q[k] for k in ("question_id", "question")} == case["question"]]) != 1:
        raise ValueError("case_original_raw_question_required")
    seen = set()
    for parent in case["catalog"]:
        key, ref = location(parent), parent["paragraph_ref"]
        meta = case["source_metadata"][ref]
        doc = documents.get(parent["doc_id"])
        if (key is None or ref in by_ref or key in seen or parent.get("paragraph_complete") is not True
                or parent["text"] != contexts.get(key) or doc is None or parent["title"] != doc["title"]
                or meta["source_text_sha256_utf8"] != sha256(parent["text"].encode()).hexdigest()
                or meta.get("raw_source_coordinate_binding_verified") is not True
                or doc["content"][meta["document_start_cp"]:meta["document_end_cp"]] != parent["text"]):
            raise ValueError("case_original_identity_full_parent_title_span_binding_changed")
        by_ref[ref] = key
        seen.add(key)
    return by_ref


def full_context(case):
    aliases, titles, paragraphs = {}, [], []
    for parent in case["catalog"]:
        if parent["doc_id"] not in aliases:
            alias = f"D{len(aliases) + 1}"
            aliases[parent["doc_id"]] = alias
            titles.append([alias, parent["title"]])
        paragraphs.append([parent["paragraph_ref"], aliases[parent["doc_id"]], parent["text"]])
    return {"question": case["question"]["question"], "titles": titles,
        "paragraph_columns": ["p", "d", "text"], "paragraphs": paragraphs,
        "candidate_scope_only": True, "outside_context_unknown": True}


def verify_complete_delivery(receipt_path, receipt, case, by_ref, submission, guards):
    """Verify program context artifacts; never claim model comprehension."""
    if receipt["context_mode"] != "full-candidate-context":
        raise ValueError("this_initial_real_quality_scorer_requires_full_context_episode")
    delivered = read_bound(Path(receipt_path).parent / "complete-context.txt", guards, receipt["complete_context_sha256"])
    if delivered != full_context(case):
        raise ValueError("complete_context_original_parents_changed")
    scope = submission["scope"]
    if (scope.get("context_mode") != receipt["context_mode"] or scope.get("source_context_prepared") is not True
            or set(scope["source_context_prepared_refs"]) != set(by_ref)
            or scope["source_count"] != len(by_ref)):
        raise ValueError("submission_complete_context_scope_changed")
    # The source-bound native guard checked exact string inclusion before fetch.
    return set(by_ref.values())


def read_json_lines(path, guards, expected):
    data, descriptor = bound(path, expected)
    guards.append(descriptor)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line]


def parse_tool_args(value):
    return json.loads(value) if isinstance(value, str) else value


def verify_staged_delivery(receipt_path, receipt, case, by_ref, submission, guards):
    """Bind original read results to the subsequent native-presence proof.

    A local fetch is insufficient. The final native guard must have verified the
    same complete text in the successful tool result before the second request.
    Official session and compact SDK logs must both be bound by the CLOSED receipt.
    """
    folder = Path(receipt_path).resolve().parent
    if not all(receipt.get(key) for key in (
            "staged_presence_audit_sha256", "sdk_summary_sha256", "official_session_log")):
        raise ValueError("staged_delivery_requires_closed_bound_native_sdk_session_logs")
    proof_rows = read_json_lines(folder / "staged-presence.jsonl", guards,
                                receipt["staged_presence_audit_sha256"])
    if (receipt.get("model_requests") != 2 or len(proof_rows) != 2
            or [row.get("phase") for row in proof_rows] != ["preview", "final"]
            or any(row.get("native_presence_verified_before_admission") is not True
                   or row.get("semantic_verified") is not False for row in proof_rows)):
        raise ValueError("two_successful_staged_native_proofs_required")
    preview, final = proof_rows
    refs = final.get("read_refs")
    if (preview.get("read_refs") != [] or preview.get("full_read_result_sha256") is not None
            or not isinstance(refs, list) or len(set(refs)) != len(refs)
            or not set(refs) <= set(by_ref)):
        raise ValueError("unique_staged_original_read_refs_required")
    sdk = read_bound(folder / "sdk-summary.json", guards, receipt["sdk_summary_sha256"])
    descriptor = receipt["official_session_log"]
    session_path = Path(descriptor["file"]).resolve()
    if not session_path.is_relative_to(folder / "sessions"):
        raise ValueError("bound_official_session_must_belong_to_closed_episode")
    events = read_json_lines(session_path, guards, descriptor["sha256"])
    if (not events or events[0].get("type") != "session"
            or events[0].get("id") != sdk.get("sessionId")):
        raise ValueError("official_session_identity_changed")
    calls = [row for row in events if row.get("type") == "tool/call"]
    sdk_calls = [row for row in sdk["trace"] if row.get("event_type") == "tool/call"]
    if ([row["data"].get("name") for row in calls] != ["read_selected_sources", "submit_evidence"]
            or [row.get("tool") for row in sdk_calls] != ["read_selected_sources", "submit_evidence"]
            or any(parse_tool_args(a["data"].get("arguments")) != parse_tool_args(b.get("arguments"))
                   for a, b in zip(calls, sdk_calls))):
        raise ValueError("one_bound_staged_read_then_submit_required")
    read_call, submit_call = calls
    if parse_tool_args(read_call["data"]["arguments"]) != {"refs": refs}:
        raise ValueError("staged_read_args_differ_from_native_delivered_refs")
    read_id = read_call["data"].get("callId")
    results = [row for row in events if row.get("type") == "tool/result"
               and row.get("data", {}).get("message", {}).get("toolCallId") == read_id]
    if (not isinstance(read_id, str) or len(results) != 1
            or results[0]["data"]["message"].get("isError") is not False
            or not read_call["seq"] < results[0]["seq"] < submit_call["seq"]):
        raise ValueError("one_successful_original_full_read_result_required")
    content = results[0]["data"]["message"].get("content")
    if not isinstance(content, list) or len(content) != 1 or content[0].get("type") != "text":
        raise ValueError("one_exact_full_read_result_text_required")
    literal = content[0].get("text")
    if not isinstance(literal, str) or sha256(literal.encode()).hexdigest() != final.get("full_read_result_sha256"):
        raise ValueError("full_read_result_not_identical_to_native_proof")
    source = {row["paragraph_ref"]: row for row in case["catalog"]}
    expected = {"status": "complete_selected_original_paragraphs", "paragraph_columns": ["p", "title", "text"],
        "paragraphs": [[p, source[p]["title"], source[p]["text"]] for p in refs],
        "paragraph_complete": True, "candidate_scope_only": True, "source_ready": False,
        "model_context_delivery_verified": False, "model_comprehension_verified": False,
        "semantic_verified": False, "publishable": False}
    if json.loads(literal) != expected:
        raise ValueError("staged_complete_original_title_text_or_order_changed")
    sdk_results = [row for row in sdk["trace"] if row.get("event_type") == "tool/result"
                   and row.get("call_id") == read_id]
    if len(sdk_results) != 1 or sdk_results[0].get("is_error") is not False:
        raise ValueError("bound_sdk_read_success_required")
    scope = submission["scope"]
    if (scope.get("context_mode") != "staged-short-window" or scope.get("source_count") != len(by_ref)
            or scope.get("candidate_preview_refs") != list(by_ref)
            or scope.get("local_fetched_full_refs") != refs
            or scope.get("source_context_prepared_refs") != refs
            or scope.get("source_context_prepared") is not True):
        raise ValueError("staged_submission_scope_not_same_complete_read")
    return {by_ref[p] for p in refs}, {
        "mode": "staged-short-window", "bound_native_presence_and_official_read_result_verified": True,
        "candidate_previews": len(by_ref), "complete_original_parents_delivered": len(refs),
        "complete_original_parent_refs": refs, "model_comprehension_or_entailment_verified": False,
        "sdk_and_official_session_bound_by_closed_receipt": True,
    }


def read_historical_baselines(path, expected, guards, question, raw_sha, contexts):
    audit = read_bound(path, guards, expected)
    if (audit["reference_answers_read"] is not False or audit["old_original_raw_sha256"] != raw_sha
            or audit["new_original_raw_sha256"] != raw_sha):
        raise ValueError("same_raw_historical_baseline_required")
    frozen = read_bound(audit["old_raw"]["file"], guards, audit["old_raw"]["sha256"])
    if {k: frozen["question"][k] for k in ("question_id", "question")} != question:
        raise ValueError("same_question_historical_baseline_required")
    old_sources = {(p["raw_doc_id"], p["raw_para_id"]): p for p in frozen["paragraphs"]}
    if len(old_sources) != len(frozen["paragraphs"]) or any(contexts.get(k) != p["text"] for k, p in old_sources.items()):
        raise ValueError("old_complete_raw_sources_changed")
    results = []
    for row in audit["historic_baselines"]:
        terminal = read_bound(row["terminal"]["file"], guards, row["terminal"]["sha256"])
        result = read_bound(row["result"]["file"], guards, row["result"]["sha256"])
        if terminal["status"] != "succeeded" or terminal["input_sha256"] != audit["old_raw"]["sha256"]:
            raise ValueError("closed_complete_historical_result_required")
        if terminal["artifacts"]["result"]["sha256"] != row["result"]["sha256"]:
            raise ValueError("historical_result_not_bound_to_terminal")
        normalized = normalize_legacy(result, question)
        for fact in result["evidence"]:
            parent = old_sources.get((fact["raw_doc_id"], fact["raw_para_id"]))
            if parent is None or parent["text"][fact["start_cp"]:fact["end_cp"]] != fact["quote"]:
                raise ValueError("historical_literal_quote_or_scope_changed")
        results.append((row, normalized, set(old_sources)))
    return results


def validate_offline_recovery(path, expected, receipt_path, receipt, manifest_path,
                              usage, case, guards):
    """Admit transparent mechanical replay, never turn the failed run into success.

Only nonliteral time can be emptied in accepted annotations. Excluded rows must
be unlinked by the original model; final relations and conclusion stay exact.
"""
    recovery = read_bound(path, guards, expected)
    if (recovery.get("format") != "offline-recovered-real-tool-v1"
            or recovery.get("source_kind") != "offline_recovered_from_failed_real_tool"
            or recovery.get("closed") is not True or recovery.get("references_read_during_recovery") is not False
            or receipt.get("status") != "failed_model_turn" or usage["has_final_artifact"]
            or recovery.get("known_episode_tokens") != usage["known_total_tokens"]
            or recovery.get("published") is not False or recovery.get("semantic_verified") is not False
            or recovery.get("transformation", {}).get("program_only") is not True
            or recovery.get("transformation", {}).get("model_calls") != 0):
        raise ValueError("explicit_zero_call_offline_recovery_required")
    required = {"original_closed_receipt": (receipt_path, bound(receipt_path)[1]["sha256"]),
                "manifest": (manifest_path, receipt["manifest_sha256"]),
                "sdk_summary": (Path(receipt_path).parent / "sdk-summary.json", None)}
    loaded = {}
    for name, (original_path, original_sha) in required.items():
        descriptor = recovery[name]
        if Path(descriptor["file"]).resolve() != Path(original_path).resolve() or (original_sha and descriptor["sha256"] != original_sha):
            raise ValueError("recovery_original_generation_binding_changed")
        loaded[name] = read_bound(descriptor["file"], guards, descriptor["sha256"])
    source_calls = [event for event in loaded["sdk_summary"]["trace"] if event.get("event_type") == "tool/call"]
    if len(source_calls) != 1 or source_calls[0].get("tool") != "submit_evidence":
        raise ValueError("one_original_failed_submit_call_required")
    original = json.loads(source_calls[0]["arguments"]) if isinstance(source_calls[0].get("arguments"), str) else source_calls[0]["arguments"]
    if ("original_tool_arguments_sha256" in recovery and (not isinstance(source_calls[0]["arguments"], str)
            or sha256(source_calls[0]["arguments"].encode()).hexdigest() != recovery["original_tool_arguments_sha256"])):
        raise ValueError("original_tool_argument_literal_hash_changed")
    def payload(name):
        descriptor = recovery[name]
        return read_bound(descriptor["file"], guards, descriptor["sha256"])
    raw_args, accepted = payload("raw_tool_arguments"), payload("accepted_tool_arguments")
    quarantine = payload("quarantine")
    implementation = recovery["recovery_implementation"]
    guards.append(bound(implementation["file"], implementation["sha256"])[1])
    for descriptor in recovery.get("implementation_files", []):
        guards.append(bound(descriptor["file"], descriptor["sha256"])[1])
    if raw_args != original or not isinstance(accepted, dict):
        raise ValueError("recovery_exact_original_and_accepted_args_required")
    if {k: v for k, v in original.items() if k != "selected"} != {k: v for k, v in accepted.items() if k != "selected"}:
        raise ValueError("recovery_must_preserve_conclusion_and_all_original_relations")
    originals = {row["p"]: row for row in original["selected"]}
    old_order = [row["p"] for row in original["selected"]]
    new_order = [row["p"] for row in accepted["selected"]]
    if len(originals) != len(old_order) or len(set(new_order)) != len(new_order) or new_order != [p for p in old_order if p in set(new_order)]:
        raise ValueError("recovery_original_selected_order_subsequence_required")
    linked = set(original["supporting"]) | {row["p"] for row in original["conflicting"]}
    if not linked <= set(new_order):
        raise ValueError("recovery_cannot_drop_any_original_relation_parent")
    sources = {row["paragraph_ref"]: row for row in case["catalog"]}
    changed_times = []
    for row in accepted["selected"]:
        old = deepcopy(originals[row["p"]])
        if old != row:
            value = old.get("argument", {}).get("时间")
            if not isinstance(value, str) or not value or value in sources[row["p"]]["text"]:
                raise ValueError("recovery_only_clears_original_nonliteral_time")
            old["argument"]["时间"] = ""
            if old != row:
                raise ValueError("recovery_annotation_changed_beyond_nonliteral_time")
            changed_times.append(row["p"])
    artifact = payload("recovered_submission")
    if [row["paragraph_ref"] for row in artifact["evidence"]] != new_order:
        raise ValueError("replayed_artifact_selection_differs_from_accepted_args")
    final = artifact["conclusion"]
    if any(final.get(key) != accepted[key] for key in ("conclusion", "basis", "verdict_label")):
        raise ValueError("replayed_conclusion_changed")
    fact_refs = {row["fact_id"]: row["paragraph_ref"] for row in artifact["evidence"]}
    if ([fact_refs[row["fact_id"]] for row in final["supporting_facts"]] != accepted["supporting"]
            or [{"p": fact_refs[row["fact_id"]], "resolution_type": row["resolution_type"]} for row in final["conflicting_facts"]] != accepted["conflicting"]):
        raise ValueError("replayed_relation_changed")
    for fact, args in zip(artifact["evidence"], accepted["selected"]):
        if (any(fact.get(key) != args[key] for key in ("fact_des", "argument", "entities"))
                or set(fact.get("proposed_roles", [])) != set(args["roles"])):
            raise ValueError("replayed_annotations_differ_from_accepted_args")
    initial_rows = [{"doc_id": sources[row["p"]]["doc_id"], "para_id": sources[row["p"]]["para_id"],
                     "quote": sources[row["p"]]["text"], "fact_des": row.get("fact_des", ""),
                     "argument": row.get("argument", {}), "entities": row.get("entities", [])}
                    for row in original["selected"]]
    return artifact, {"source_kind": recovery["source_kind"], "original_end_to_end_success": False,
        "original_failed_status": receipt["status"], "original_model_selected_count": len(old_order),
        "replayed_accepted_count": len(new_order), "excluded_unlinked_original_refs": [p for p in old_order if p not in set(new_order)],
        "cleared_nonliteral_time_refs": changed_times, "initial_model_selected_rows": initial_rows,
        "quarantine_bound": True, "quarantine_audit": quarantine, "new_model_calls": 0,
        "known_episode_tokens_not_reset": usage["known_total_tokens"], "recovery_receipt": bound(path, expected)[1]}


def score_episode(receipt_path, expected_sha256, manifest_path, *, raw_input,
                  raw_sha256, sample_root, baseline_path=None, baseline_sha256=None,
                  recovery_path=None, recovery_sha256=None):
    usage = audit_closed_episode(receipt_path, expected_sha256, manifest_path, root=ROOT)
    guards = list(usage["input_guards"])
    receipt = read_bound(receipt_path, guards, expected_sha256)
    if not recovery_path and (receipt["status"] != "closed_proposed_unverified" or receipt["turn_end"] != {"kind": "completed"}
            or receipt["error_present"] or receipt["cleanup_error"] is not None or not usage["has_final_artifact"]):
        raise ValueError("complete_proposed_submission_required_before_reference_access")
    manifest = read_bound(manifest_path, guards, receipt["manifest_sha256"])
    case = read_bound(ROOT / manifest["casePath"], guards, receipt["case_sha256"])
    raw = read_bound(raw_input, guards, raw_sha256)
    contexts, documents = original_sources(raw)
    by_ref = validate_case(case, raw, contexts, documents)
    recovery = None
    if recovery_path:
        submission, recovery = validate_offline_recovery(recovery_path, recovery_sha256, receipt_path,
            receipt, manifest_path, usage, case, guards)
    else:
        submission = read_bound(Path(receipt_path).parent / "submission.json", guards, receipt["submission_sha256"])
    normalized = normalize_dsh(submission, case["question"])
    if receipt.get("context_mode") == "staged-short-window":
        read_keys, delivery_audit = verify_staged_delivery(receipt_path, receipt, case, by_ref, submission, guards)
    else:
        read_keys = verify_complete_delivery(receipt_path, receipt, case, by_ref, submission, guards)
        delivery_audit = {"mode": "full-candidate-context", "complete_original_parents_delivered": len(read_keys),
            "native_guard_checked_canonical_context_before_fetch": True,
            "model_comprehension_or_entailment_verified": False}
    for row in submission["evidence"]:
        if (by_ref.get(row["paragraph_ref"]) != location(row) or row["quote"] != contexts.get(location(row))
                or row.get("parent_start_cp") != 0 or row.get("parent_end_cp") != len(row["quote"])
                or row["source_text_sha256_utf8"] != sha256(row["quote"].encode()).hexdigest()):
            raise ValueError("submitted_source_quote_span_or_hash_changed")
    old = read_historical_baselines(baseline_path, baseline_sha256, guards, case["question"], raw_sha256, contexts) if baseline_path else []
    # FIRST reference access: all requested generations and immutable source checks above.
    folder = Path(sample_root).resolve() / case["topic_title"]
    evidence = read_bound(folder / "evidence_samples.json", guards)
    conclusions = read_bound(folder / "conclusions.json", guards)
    refs = [row for row in conclusions if row["question_id"] == case["question"]["question_id"]]
    if len(refs) != 1:
        raise ValueError("one_reference_conclusion_required")
    _, engine, mapper_guards = load_reference_tools()
    guards.extend(mapper_guards)
    candidates = set(by_ref.values())
    current = score_normalized(case["question"], normalized, evidence, refs[0], contexts, candidates, read_keys, engine=engine)
    if recovery:
        initial_normalized = {**normalized, "evidence": recovery["initial_model_selected_rows"]}
        current["original_failed_tool_selection_diagnostic"] = score_normalized(case["question"], initial_normalized,
            evidence, refs[0], contexts, candidates, read_keys, engine=engine)["question_selection"]
    baselines = {}
    for meta, normalized_old, scope in old:
        baselines[meta["version"]] = {"quality": score_normalized(case["question"], normalized_old,
            evidence, refs[0], contexts, scope, scope, engine=engine), "usage": meta["usage_summary"],
            "elapsed_seconds": meta["elapsed_seconds"], "same_source_scope": scope == candidates,
            "source_scope_intersection": len(scope & candidates), "historical_route_not_framework_causal": True}
    for name in ("score_real_quality_episode.py", "final_evidence_scoring.py", "two_stage_evidence_matching.py", "audit_dsh_closed_episode.py"):
        guards.append(bound(ROOT / "scripts" / name)[1])
    for descriptor in guards:
        bound(descriptor["file"], descriptor["sha256"])
    return {"format": "closed-real-question-quality-episode-v1", "question": case["question"],
        "topic_title": case["topic_title"], "current": current, "historical_baselines": baselines,
        "usage": usage, "delivery_audit": delivery_audit, "input_guards": guards, "input_guards_unchanged": True,
        "selection_policy": receipt.get("selection_policy"), "policy_sha256": receipt.get("selection_policy_sha256"),
        "source_kind": recovery["source_kind"] if recovery else "closed_real_submission",
        "offline_recovery": recovery, "original_end_to_end_success": recovery is None,
        "same_model_framework_paired_comparison": False, "sample_is_unseen_blind_test": False,
        "published": False, "official_score": False, "annotation_semantics_verified": False,
        "new_model_calls": 0, "new_paid_judge_calls": 0, "reference_written_to_runtime": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "expected-sha256", "manifest", "raw-input", "raw-sha256", "sample-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--baseline-path")
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--recovery-path")
    parser.add_argument("--recovery-sha256")
    args = parser.parse_args()
    result = score_episode(args.receipt, args.expected_sha256, args.manifest, raw_input=args.raw_input,
        raw_sha256=args.raw_sha256, sample_root=args.sample_root, baseline_path=args.baseline_path,
        baseline_sha256=args.baseline_sha256, recovery_path=args.recovery_path, recovery_sha256=args.recovery_sha256)
    with Path(args.output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output_sha256": sha256(Path(args.output).read_bytes()).hexdigest(),
        "current_A_id_only": result["current"]["question_selection"]["A_id_only_proxy"], "new_model_calls": 0}))
