"""Deterministic, offline paragraph diagnostics; no model or runtime imports.

``evaluate_bundle`` accepts complete reference topics and explicit submitted
evidence/conclusion lists. A is an ID-only diagnostic, B is an interval using
source-bound, already closed v2 semantic judgments. Neither is an official score.
Original rows, repeated relations and unlocatable references remain auditable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

FORMAT = "paragraph-quality-offline-v1"
SEMANTIC_PROMPT = "same-paragraph-four-dimensions-v2-derived-verdict"
SEMANTIC_PROMPT_SHA = "13ae5c1fc9be262c34aafb6599c572d7a521f15e68f72ebb93ab5808e0258fff"
SIDES = ("supporting_facts", "conflicting_facts")
DIMENSIONS = ("subject", "time", "value", "statistical_scope")
FACT_FIELDS = ("doc_id", "para_id", "quote", "fact_des", "argument", "entities",
               "start_cp", "end_cp", "char_start", "char_end")


class ParagraphEvaluationError(ValueError):
    pass


class _ClosedSemanticIndex(dict):
    """Internal output of full frozen-artifact validation, never a bare label API."""


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_descriptor(path, expected=None):
    path = Path(path).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and actual != expected:
        raise ParagraphEvaluationError("input_file_sha256_mismatch")
    return {"file": str(path), "sha256": actual}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _list(value, name):
    if not isinstance(value, list):
        raise ParagraphEvaluationError(name + "_must_be_list")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ParagraphEvaluationError(name + "_must_be_nonempty_string")
    return value


def location(row):
    if not isinstance(row, dict):
        return None
    values = row.get("doc_id"), row.get("para_id")
    return values if all(isinstance(v, str) and v.strip() for v in values) else None


def project_fact(row):
    return {key: deepcopy(row[key]) for key in FACT_FIELDS if key in row}


def metrics(matches, submitted, reference):
    return {"matches": matches, "submitted_records": submitted, "reference_records": reference,
            "precision": matches / submitted if submitted else None,
            "recall": matches / reference if reference else None,
            "f1": 2 * matches / (submitted + reference) if submitted or reference else None}


def maximum_matching(submitted_count, reference_count, edges):
    """Maximum 1:1 diagnostic matching; duplicate rows never get Cartesian credit."""
    adjacent, paired = defaultdict(set), {}
    for left, right in edges:
        if (type(left) is not int or type(right) is not int or
                not 0 <= left < submitted_count or not 0 <= right < reference_count):
            raise ParagraphEvaluationError("invalid_matching_edge")
        adjacent[left].add(right)

    def augment(left, seen):
        for right in sorted(adjacent[left]):
            if right in seen:
                continue
            seen.add(right)
            if right not in paired or augment(paired[right], seen):
                paired[right] = left
                return True
        return False

    for left in range(submitted_count):
        augment(left, set())
    return [{"submitted_index": left, "reference_index": right}
            for right, left in sorted(paired.items())]


def _scoped_gold(reference):
    """Same relation-ID union resolution as the closed v2 evaluator, no repairs."""
    ids = {row.get("reference", {}).get("fact_id") for row in reference["question_references"]
           if isinstance(row, dict) and isinstance(row.get("reference"), dict)}
    ids.discard(None)
    rows = [row for row in reference["facts"] if isinstance(row, dict) and row.get("fact_id") in ids]
    resolved = {row.get("fact_id") for row in rows}
    for identity in sorted(ids - resolved):
        rows.append({"fact_id": identity, "_missing_reference": True})
    return rows


def _validated_verdict(value):
    fields = {"verdict", "same_fact", *DIMENSIONS, "submitted_quote_supports_fact",
              "reference_quote_supports_fact", "reason"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ParagraphEvaluationError("semantic_verdict_shape_invalid")
    for name in DIMENSIONS:
        if value[name] not in {"compatible", "incompatible", "not_applicable", "uncertain"}:
            raise ParagraphEvaluationError("semantic_dimension_invalid")
    for name in ("same_fact", "submitted_quote_supports_fact", "reference_quote_supports_fact"):
        if value[name] not in {"yes", "no", "uncertain"}:
            raise ParagraphEvaluationError("semantic_judgment_invalid")
    if not isinstance(value["reason"], str) or not 0 < len(value["reason"].strip()) <= 500:
        raise ParagraphEvaluationError("semantic_reason_invalid")
    no = value["same_fact"] == "no" or any(value[x] == "incompatible" for x in DIMENSIONS)
    no |= any(value[x] == "no" for x in ("submitted_quote_supports_fact", "reference_quote_supports_fact"))
    unknown = value["same_fact"] == "uncertain" or any(value[x] == "uncertain" for x in DIMENSIONS)
    unknown |= any(value[x] == "uncertain" for x in ("submitted_quote_supports_fact", "reference_quote_supports_fact"))
    derived = "no_match" if no else "uncertain" if unknown else "match"
    if value["verdict"] != derived:
        raise ParagraphEvaluationError("semantic_verdict_contradicts_dimensions")
    return derived


def _semantic_index(entries, topics):
    """Accept no bare labels: exact frozen artifact, paragraph and pair hashes bind B."""
    index, audits = _ClosedSemanticIndex(), []
    for entry in entries:
        tid = _text(entry.get("topic_id"), "semantic_topic_id")
        if tid not in topics:
            raise ParagraphEvaluationError("semantic_topic_unknown")
        artifact, report = entry.get("artifact"), entry.get("report")
        if not isinstance(artifact, dict) or not isinstance(report, dict):
            raise ParagraphEvaluationError("semantic_artifact_and_report_required")
        if (artifact.get("format") != "semantic-eval-input-v1" or
                report.get("format") != "semantic-development-proxy-v1" or
                report.get("status") not in {"completed", "completed_with_uncertainty", "incomplete_supplier_failure"} or
                report.get("input_unchanged") is not True or
                report.get("frozen_input_sha256") != fingerprint(artifact) or
                report.get("prompt_version") != SEMANTIC_PROMPT or
                report.get("prompt_sha256") != SEMANTIC_PROMPT_SHA):
            raise ParagraphEvaluationError("semantic_closed_input_or_rubric_binding_invalid")
        qid = artifact["scope"]["question_id"]
        questions = {q["question_id"]: q for q in topics[tid]["conclusions"]}
        if qid not in questions or artifact["scope"]["question"] != questions[qid]["question"]:
            raise ParagraphEvaluationError("semantic_question_binding_invalid")
        source = {location(r): r["text"] for r in topics[tid]["source_paragraphs"]}
        declared = artifact["source_paragraphs"]
        if len({location(r) for r in declared}) != len(declared):
            raise ParagraphEvaluationError("semantic_duplicate_source_paragraph")
        for row in declared:
            if location(row) is None or source.get(location(row)) != row.get("text"):
                raise ParagraphEvaluationError("semantic_source_text_binding_invalid")
        declared_source = {location(row): row["text"] for row in declared}
        predicted, gold = artifact["submission"]["evidence"], _scoped_gold(artifact["reference"])
        gold_projections = Counter(fingerprint(project_fact(r)) for r in topics[tid]["evidence"])
        if any(fingerprint(project_fact(row)) not in gold_projections for row in gold
               if not row.get("_missing_reference")):
            raise ParagraphEvaluationError("semantic_reference_not_original_topic_evidence")
        seen = set()
        for pair in _list(report.get("pairs"), "semantic_pairs"):
            left, right = pair.get("submitted_index"), pair.get("reference_index")
            if (type(left) is not int or type(right) is not int or not 0 <= left < len(predicted)
                    or not 0 <= right < len(gold) or (left, right) in seen):
                raise ParagraphEvaluationError("semantic_pair_indices_invalid_or_duplicate")
            seen.add((left, right))
            p, g = predicted[left], gold[right]
            key = location(p)
            if key is None or location(g) != key:
                raise ParagraphEvaluationError("semantic_pair_cross_paragraph")
            if (pair.get("submitted_record_sha256") != fingerprint(p) or
                    pair.get("reference_record_sha256") != fingerprint(g)):
                raise ParagraphEvaluationError("semantic_pair_record_sha_mismatch")
            if pair.get("status") != "judged":
                continue  # unavailable labels remain absent/unknown, never no_match
            text = declared_source.get(key)
            if text is None or any(not isinstance(row.get("quote"), str) or not row["quote"] or
                                   row["quote"] not in text for row in (p, g)):
                raise ParagraphEvaluationError("semantic_judged_quote_not_literal")
            payload = {"question": artifact["scope"]["question"],
                       "source_paragraph": {"doc_id": key[0], "para_id": key[1], "text": text},
                       "submitted": project_fact(p), "reference": project_fact(g)}
            payload_sha = fingerprint(payload)
            cache_key = fingerprint({"prompt_version": SEMANTIC_PROMPT, "prompt_sha256": SEMANTIC_PROMPT_SHA,
                                     "judge": report["judge"], "payload_sha256": payload_sha})
            if pair.get("payload_sha256") != payload_sha or pair.get("cache_key") != cache_key:
                raise ParagraphEvaluationError("semantic_pair_payload_or_cache_binding_invalid")
            verdict = _validated_verdict(pair.get("verdict"))
            index.setdefault((tid, fingerprint(project_fact(p)), fingerprint(project_fact(g))), []).append(
                {"question_id": qid, "verdict": verdict, "reason": pair["verdict"]["reason"],
                 "payload_sha256": payload_sha, "cache_key": cache_key})
        audits.append({"topic_id": tid, "question_id": qid, "frozen_input_sha256": fingerprint(artifact),
                       "report_sha256": fingerprint(report), "pairs_recorded": len(seen),
                       "judge": deepcopy(report["judge"]), "created_at": report.get("created_at")})
    return index, audits


def _relation_rows(relations, evidence, *, submitted, issues, context):
    by_id = defaultdict(list)
    for row in evidence:
        if isinstance(row, dict) and row.get("fact_id"):
            by_id[row["fact_id"]].append(row)
    result = []
    for i, ref in enumerate(_list(relations, "relations")):
        row = None
        if isinstance(ref, dict) and "evidence_index" in ref:
            ix = ref["evidence_index"]
            if not submitted or type(ix) is not int or not 0 <= ix < len(evidence):
                raise ParagraphEvaluationError("submitted_relation_evidence_index_invalid")
            row = evidence[ix]
        elif isinstance(ref, dict) and ref.get("fact_id"):
            matches = by_id[ref["fact_id"]]
            if len(matches) == 1:
                row = matches[0]
                for name in ("doc_id", "para_id"):
                    if name in ref and ref[name] != row.get(name):
                        issues.append({"code": "relation_explicit_source_disagrees_with_fact_id",
                                       "context": context, "relation_index": i, "field": name})
            else:
                issues.append({"code": "missing_or_ambiguous_relation_fact_id", "context": context,
                               "relation_index": i, "fact_id": ref["fact_id"], "resolved_rows": len(matches)})
        elif isinstance(ref, dict) and location(ref) is not None:
            row = ref
        if row is None:
            issues.append({"code": "unresolved_relation_counted_in_raw_denominator", "context": context,
                           "relation_index": i})
            row = {"_unresolved_relation": deepcopy(ref)}
        result.append(deepcopy(row))
    return result


def _units(rows, unique):
    if not unique:
        return [{"key": location(row), "rows": [row], "original_indices": [i]} for i, row in enumerate(rows)]
    result, by_key = [], {}
    for i, row in enumerate(rows):
        key = location(row)
        if key is not None and key in by_key:
            unit = result[by_key[key]]
            unit["rows"].append(row); unit["original_indices"].append(i)
        else:
            if key is not None:
                by_key[key] = len(result)
            result.append({"key": key, "rows": [row], "original_indices": [i]})
    return result


def _duplicate_audit(rows):
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        if location(row) is not None:
            grouped[location(row)].append(i)
    return {"raw_records": len(rows), "unique_located_paragraphs": len(grouped),
            "unlocated_records_retained": sum(location(row) is None for row in rows),
            "duplicate_paragraph_groups": [{"doc_id": key[0], "para_id": key[1], "row_indices": ids}
                                            for key, ids in sorted(grouped.items()) if len(ids) > 1]}


def _literal_audit(rows, source):
    """Diagnose original anomalies without repairing or deleting any row."""
    texts = {location(r): r["text"] for r in source}
    issues = []
    for i, row in enumerate(rows):
        key = location(row); quote = row.get("quote")
        if key not in texts or not isinstance(quote, str) or not quote or quote not in texts[key]:
            issues.append({"row_index": i, "doc_id": row.get("doc_id"), "para_id": row.get("para_id"),
                           "reason": "quote_not_literal_in_declared_original_paragraph_or_unlocated",
                           "quote_sha256": hashlib.sha256(quote.encode()).hexdigest() if isinstance(quote, str) else None})
    return {"raw_records": len(rows), "literal_in_declared_paragraph": len(rows) - len(issues),
            "anomalies": issues, "rows_removed_or_repaired": 0}


def compare_rows(topic_id, submitted, reference, *, semantic_index=None, question_id=None):
    """Public small API for T2: standard rows, namespaced topic, no disk/network."""
    _text(topic_id, "topic_id")
    submitted, reference = _list(submitted, "submitted"), _list(reference, "reference")
    if any(not isinstance(row, dict) for row in [*submitted, *reference]):
        raise ParagraphEvaluationError("evidence_rows_must_be_objects")
    if semantic_index is None:
        semantic_index = _ClosedSemanticIndex()
    if not isinstance(semantic_index, _ClosedSemanticIndex):
        raise ParagraphEvaluationError("bare_semantic_labels_not_accepted_use_bound_bundle")
    results = {}
    for name, unique in (("raw_records", False), ("unique_paragraphs", True)):
        left, right = _units(submitted, unique), _units(reference, unique)
        a, confirmed, possible, unknown, rejected = [], [], [], [], []
        for li, lu in enumerate(left):
            for ri, ru in enumerate(right):
                if lu["key"] is None or lu["key"] != ru["key"]:
                    continue
                a.append((li, ri)); labels, reasons = [], []
                for p in lu["rows"]:
                    for g in ru["rows"]:
                        entries = semantic_index.get((topic_id, fingerprint(project_fact(p)),
                                                      fingerprint(project_fact(g))), [])
                        entries = [e for e in entries if question_id is None or e["question_id"] == question_id]
                        values = {e["verdict"] for e in entries}
                        labels.append(next(iter(values)) if len(values) == 1 else "uncertain")
                        reasons.extend(entries)
                label = "match" if "match" in labels else "no_match" if labels and set(labels) == {"no_match"} else "uncertain"
                if label == "match":
                    confirmed.append((li, ri)); possible.append((li, ri))
                elif label == "uncertain":
                    possible.append((li, ri)); unknown.append({"submitted_index": li, "reference_index": ri,
                        "doc_id": lu["key"][0], "para_id": lu["key"][1],
                        "reason": "no_exact_closed_judgment_or_uncertain_or_disagreeing_labels", "known_records": reasons})
                else:
                    rejected.append({"submitted_index": li, "reference_index": ri,
                        "doc_id": lu["key"][0], "para_id": lu["key"][1], "known_records": reasons})
        counts = len(left), len(right)
        am = maximum_matching(*counts, a); cm = maximum_matching(*counts, confirmed); pm = maximum_matching(*counts, possible)
        results[name] = {"A_id_only": {"metrics": metrics(len(am), *counts), "matching": am,
            "not_semantic_success": True,
            "unmatched_submitted_indices": [i for i in range(counts[0]) if i not in {m["submitted_index"] for m in am}],
            "unmatched_reference_indices": [i for i in range(counts[1]) if i not in {m["reference_index"] for m in am}]},
            "B_existing_semantics": {"status": "partial_unknown" if unknown else "closed_known",
                "confirmed_lower_bound": metrics(len(cm), *counts), "possible_upper_bound": metrics(len(pm), *counts),
                "point_metrics": None if unknown else metrics(len(cm), *counts),
                "confirmed_matching": cm, "possible_matching": pm, "unknown_pairs": unknown,
                "known_rejected_pairs": rejected, "new_model_calls": 0}}
    return {**results, "denominator_audit": {"submitted": _duplicate_audit(submitted),
                                             "reference": _duplicate_audit(reference)}}


def _aggregate(comparisons):
    output = {}
    for mode in ("raw_records", "unique_paragraphs"):
        parts = []
        for label, field in (("A_id_only", "metrics"), ("B_lower", "confirmed_lower_bound"), ("B_upper", "possible_upper_bound")):
            rows = [item[mode]["A_id_only" if label == "A_id_only" else "B_existing_semantics"][field] for item in comparisons]
            micro = metrics(sum(r["matches"] for r in rows), sum(r["submitted_records"] for r in rows), sum(r["reference_records"] for r in rows))
            weight = sum(r["reference_records"] for r in rows)
            weighted = {"weight_total": weight, "weight": "reference_records_in_this_raw_or_unique_policy", "all_case_denominator": len(rows)}
            for key in ("precision", "recall", "f1"):
                missing = [i for i, r in enumerate(rows) if r["reference_records"] and r[key] is None]
                weighted[key] = (None if not weight or missing else sum((r[key] or 0) * r["reference_records"] for r in rows) / weight)
                weighted[key + "_undefined_positive_weight_indices"] = missing
                weighted[key + "_zero_filled_operational"] = (sum((r[key] or 0) * r["reference_records"] for r in rows) / weight if weight else None)
            parts.append((label, {"micro": micro, "reference_scale_weighted": weighted}))
        output[mode] = dict(parts)
        output[mode]["B_point_metrics"] = (None if any(c[mode]["B_existing_semantics"]["unknown_pairs"] for c in comparisons)
                                           else output[mode]["B_lower"])
    return output


def evaluate_bundle(bundle):
    """Reference topics + submissions + source-bound closed semantic artifacts.

    See module CLI and tests for a small fixture. This function never writes,
    imports app code, opens SQLite, creates Settings, or calls an external model.
    """
    before = fingerprint(bundle)
    topics_list = _list(bundle.get("topics"), "topics")
    topics = {}
    for topic in topics_list:
        tid = _text(topic.get("topic_id"), "topic_id")
        if tid in topics:
            raise ParagraphEvaluationError("duplicate_topic_id")
        _list(topic.get("evidence"), "topic_evidence"); qs = _list(topic.get("conclusions"), "topic_conclusions")
        source = _list(topic.get("source_paragraphs"), "source_paragraphs")
        if any(not isinstance(row, dict) for row in [*topic["evidence"], *qs, *source]):
            raise ParagraphEvaluationError("topic_rows_must_be_objects")
        qids = [_text(q.get("question_id"), "question_id") for q in qs]
        if len(qids) != len(set(qids)):
            raise ParagraphEvaluationError("duplicate_reference_question_id")
        for q in qs:
            _text(q.get("question"), "question")
            for side in SIDES:
                _list(q.get(side), side)
        if any(location(r) is None or not isinstance(r.get("text"), str) for r in source) or len({location(r) for r in source}) != len(source):
            raise ParagraphEvaluationError("source_paragraphs_invalid_or_duplicate")
        topics[tid] = topic
    submissions = {}
    for sub in _list(bundle.get("submissions"), "submissions"):
        tid = _text(sub.get("topic_id"), "submission_topic_id")
        if tid not in topics or tid in submissions:
            raise ParagraphEvaluationError("unknown_or_duplicate_submission_topic")
        if sub.get("status") not in {"succeeded", "failed"}:
            raise ParagraphEvaluationError("submission_must_be_terminal")
        _list(sub.get("evidence"), "submitted_evidence"); _list(sub.get("conclusions"), "submitted_conclusions")
        qids = _list(sub.get("evaluated_question_ids"), "evaluated_question_ids")
        known = {q["question_id"]: q for q in topics[tid]["conclusions"]}
        if len(qids) != len(set(qids)) or any(q not in known for q in qids):
            raise ParagraphEvaluationError("evaluated_question_ids_invalid")
        seen = set()
        for q in sub["conclusions"]:
            qid = q.get("question_id")
            if qid not in qids or qid in seen or q.get("question") != known[qid]["question"]:
                raise ParagraphEvaluationError("submitted_question_binding_invalid")
            seen.add(qid)
            for side in SIDES:
                _list(q.get(side), side)
        if sub["status"] == "failed" and (sub["evidence"] or sub["conclusions"]):
            raise ParagraphEvaluationError("failed_submission_must_not_publish_rows")
        scope = _list(sub.get("source_scope"), "source_scope")
        sourcekeys = {location(r) for r in topics[tid]["source_paragraphs"]}
        if any(location(r) not in sourcekeys for r in scope) or len({location(r) for r in scope}) != len(scope):
            raise ParagraphEvaluationError("source_scope_invalid_or_duplicate")
        submissions[tid] = sub
    index, semantic_audit = _semantic_index(_list(bundle.get("semantic_artifacts", []), "semantic_artifacts"), topics)
    topic_reports, question_reports, issues = [], [], []
    fp_examples, fn_examples, rejected_examples = [], [], []
    for tid, topic in topics.items():
        sub = submissions.get(tid, {"evidence": [], "conclusions": [], "source_scope": [], "evaluated_question_ids": [], "status": "not_executed"})
        selected = {location(r) for r in sub["source_scope"]}
        scoped = lambda rows: [r for r in rows if location(r) in selected]
        full = compare_rows(tid, sub["evidence"], topic["evidence"], semantic_index=index)
        limited = compare_rows(tid, sub["evidence"], scoped(topic["evidence"]), semantic_index=index)
        topic_reports.append({"topic_id": tid, "status": sub["status"], "source_scope_paragraphs": len(selected),
                              "full_source_paragraphs": len(topic["source_paragraphs"]),
                              "literal_quote_audit": {"reference": _literal_audit(topic["evidence"], topic["source_paragraphs"]),
                                                      "submitted": _literal_audit(sub["evidence"], topic["source_paragraphs"])},
                              "full_topic_reference": full, "within_source_scope_reference": limited})
        goldkeys = {location(r) for r in topic["evidence"]}
        predkeys = {location(r) for r in sub["evidence"]}
        for i in full["raw_records"]["A_id_only"]["unmatched_submitted_indices"]:
            row = sub["evidence"][i]
            fp_examples.append({"topic_id": tid, "submitted_index": i, "doc_id": row.get("doc_id"), "para_id": row.get("para_id"),
                "fact_des_excerpt": row.get("fact_des", "")[:180],
                "reason": "paragraph_not_in_original_topic_evidence_keys" if location(row) not in goldkeys else "same_paragraph_extra_row_has_no_remaining_1_to_1_reference_capacity",
                "semantic_truth_not_adjudicated": True})
        for i in full["raw_records"]["A_id_only"]["unmatched_reference_indices"]:
            row = topic["evidence"][i]; key = location(row)
            reason = ("reference_outside_declared_source_scope" if key not in selected else
                      "selected_source_paragraph_has_no_formal_submitted_evidence" if key not in predkeys else
                      "same_paragraph_reference_row_has_no_remaining_1_to_1_submission_capacity")
            fn_examples.append({"topic_id": tid, "reference_index": i, "doc_id": row.get("doc_id"), "para_id": row.get("para_id"),
                "fact_des_excerpt": row.get("fact_des", "")[:180], "reason": reason})
        for pair in full["raw_records"]["B_existing_semantics"]["known_rejected_pairs"]:
            rejected_examples.append({"topic_id": tid, **deepcopy(pair)})
        provided = {q["question_id"]: q for q in sub["conclusions"]}
        for q in topic["conclusions"]:
            qid = q["question_id"]; sq = provided.get(qid)
            qr = {"topic_id": tid, "question_id": qid, "question": q["question"],
                  "executed_question": qid in sub["evaluated_question_ids"], "submitted_conclusion_present": sq is not None}
            for side in SIDES:
                gold = _relation_rows(q[side], topic["evidence"], submitted=False, issues=issues, context=[tid, qid, side, "reference"])
                predicted = _relation_rows(sq[side] if sq else [], sub["evidence"], submitted=True, issues=issues, context=[tid, qid, side, "submitted"])
                qr[side] = {"full_question_reference": compare_rows(tid, predicted, gold, semantic_index=index, question_id=qid),
                    "within_source_scope_reference": compare_rows(tid, predicted, scoped(gold), semantic_index=index, question_id=qid)}
            question_reports.append(qr)
    aggregate_questions = {}
    for name, qs in (("all_reference_questions", question_reports), ("executed_questions_only", [q for q in question_reports if q["executed_question"]])):
        aggregate_questions[name] = {side: {scope: _aggregate([q[side][scope] for q in qs])
            for scope in ("full_question_reference", "within_source_scope_reference")} for side in SIDES}
        aggregate_questions[name]["question_denominator"] = len(qs)
    def diverse_examples(rows, limit=5):
        groups = defaultdict(list)
        for row in rows:
            groups[row["topic_id"]].append(row)
        result = []
        while len(result) < limit and any(groups.values()):
            for values in groups.values():
                if values and len(result) < limit:
                    result.append(values.pop(0))
        return result

    result = {"format": FORMAT, "official_score": False, "new_model_calls": 0, "input_sha256": before,
        "topic_reports": topic_reports, "question_reports": question_reports,
        "evidence_aggregates": {scope: _aggregate([t[scope] for t in topic_reports])
                                for scope in ("full_topic_reference", "within_source_scope_reference")},
        "question_aggregates": aggregate_questions, "semantic_input_audit": semantic_audit, "issues": issues,
        "error_examples": {"A_id_only_false_positive_candidates": {"available_rows": len(fp_examples), "examples": diverse_examples(fp_examples)},
            "A_id_only_false_negative_candidates": {"available_rows": len(fn_examples), "examples": diverse_examples(fn_examples)},
            "B_closed_semantically_rejected_pairs": {"available_pairs": len(rejected_examples), "examples": diverse_examples(rejected_examples)},
            "limits": "A errors are deterministic reference-key/1:1 diagnostics, not independent proof that a submitted real-world claim is false. Missing B judgments remain unknown."},
        "coverage": {"topics": len(topics), "full_reference_questions": len(question_reports),
            "executed_questions": sum(q["executed_question"] for q in question_reports),
            "selected_source_paragraphs": sum(t["source_scope_paragraphs"] for t in topic_reports),
            "full_source_paragraphs": sum(t["full_source_paragraphs"] for t in topic_reports)},
        "policy": {"A": "ID-only necessary filter; never official evidence matching or semantic success.",
            "B": "Only exact source-bound closed four-dimension v2 labels; missing, uncertain or conflicting labels remain unknown, not zero or newly judged.",
            "matching": "Deterministic maximum-cardinality 1:1; official duplicate policy unconfirmed. Raw rows and alternate unique-paragraph results are both retained.",
            "weight": "Engineering interpretation of standard-answer scale: per-question reference-side raw row count, or unique paragraph count in the explicitly alternate unique policy. Weighted precision differs from micro precision.",
            "zero_denominator": "Undefined precision/recall stay null; optional zero-filled operational values are explicitly separate.",
            "conflicts": "Single-side conflicting_facts diagnostics only; original examples lack explicit both-side conflict-pair mapping, so this is not an official conflict-pair score.",
            "subsets": "Complete-topic and complete-question reference denominators stay visible alongside selected-source-scope denominators; missing questions do not disappear.",
            "anomalies": "No silent quote/paragraph/gold repairs or removal of unresolved rows; unlocated rows remain in raw denominators and are audited."}}
    if fingerprint(bundle) != before:
        raise ParagraphEvaluationError("evaluation_input_mutated")
    return result


def _read_bound(root, descriptor, expected_name, guards):
    if (not isinstance(descriptor, dict) or not isinstance(descriptor.get("sha256"), str) or
            len(descriptor["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in descriptor["sha256"])):
        raise ParagraphEvaluationError("closed_artifact_sha256_required")
    path = (Path(root) / expected_name).resolve()
    if Path(descriptor.get("file", "")).resolve() != path:
        raise ParagraphEvaluationError("closed_artifact_path_mismatch")
    guards.append(file_descriptor(path, descriptor.get("sha256")))
    return load(path)


def make_closed_pilot_bundle(comparison_dir, pilot_plan, sample_root, run_label):
    """Read only explicit closed pilot artifacts; no live database or journals."""
    comparison_dir, pilot_plan, sample_root = map(lambda p: Path(p).resolve(), (comparison_dir, pilot_plan, sample_root))
    guards = [file_descriptor(comparison_dir / "comparison-report.json"), file_descriptor(comparison_dir / "comparison-plan.json"), file_descriptor(pilot_plan)]
    report, scoreplan, plan = (load(Path(d["file"])) for d in guards)
    if report.get("status") != "completed" or report.get("measured") is not True:
        raise ParagraphEvaluationError("comparison_not_closed_and_measured")
    if scoreplan.get("plan_sha256") != guards[2]["sha256"] or report.get("plan_sha256") != guards[2]["sha256"]:
        raise ParagraphEvaluationError("pilot_plan_binding_invalid")
    identities = [r for r in scoreplan["runs"] if r["label"] == run_label]
    if len(identities) != 1:
        raise ParagraphEvaluationError("closed_run_label_unknown")
    run = Path(identities[0]["path"]).resolve()
    finalpath = run / "final-receipt.json"
    if not finalpath.is_file():
        raise ParagraphEvaluationError("run_final_receipt_required_before_case_reads")
    guards.append(file_descriptor(finalpath)); final = load(finalpath)
    if final.get("plan_sha256") != guards[2]["sha256"] or final.get("app_sources_unchanged") is not True or final.get("no_gold_read") is not True:
        raise ParagraphEvaluationError("closed_run_boundary_invalid")
    terminalmap = {r["case_id"]: r for r in final["cases"]}
    cases = [r for r in report["cases"] if r["run_label"] == run_label]
    if {c["case_id"] for c in cases} != {c["case_id"] for c in plan["cases"]}:
        raise ParagraphEvaluationError("closed_cohort_mismatch")
    topics, submissions, semantics = [], [], []
    for case in plan["cases"]:
        cid, title = case["case_id"], case["title"]
        sample = (sample_root / title).resolve()
        if sample.parent != sample_root:
            raise ParagraphEvaluationError("sample_topic_path_escape")
        files = [sample / name for name in ("evidence_samples.json", "conclusions.json", "raw_documents.json")]
        guards.extend(file_descriptor(f) for f in files)
        evidence, conclusions, raw = map(load, files)
        source = [{"doc_id": d["doc_id"], "para_id": p["para_id"], "text": d["content"][p["char_start"]:p["char_end"]]}
                  for d in raw for p in d["paragraphs"]]
        inputpath = (pilot_plan.parent / case["input_file"]).resolve()
        if not inputpath.is_relative_to(pilot_plan.parent):
            raise ParagraphEvaluationError("pilot_input_path_escape")
        guards.append(file_descriptor(inputpath, case["input_sha256"])); payload = load(inputpath)
        planned = next(row for row in scoreplan["cases"] if row["comparison_id"] == run_label + "-" + cid)
        scored = next(row for row in cases if row["case_id"] == cid)
        artifactpath = comparison_dir / "frozen-evaluation-inputs" / (run_label + "-" + cid + ".json")
        guards.append(file_descriptor(artifactpath)); artifact = load(artifactpath)
        if artifact != planned["artifact"] or fingerprint(artifact) != scored["report"]["frozen_input_sha256"]:
            raise ParagraphEvaluationError("scored_input_not_same_frozen_artifact")
        for f in files:
            file_descriptor(f, artifact["reference"]["source_files_sha256"][f.name])
        folder = run / cid; closed = terminalmap[cid]
        terminal = _read_bound(folder, closed["terminal"], "terminal.json", guards)
        if terminal["status"] != closed["status"] or terminal["input_sha256"] != case["input_sha256"] or terminal["status"] not in {"succeeded", "failed"}:
            raise ParagraphEvaluationError("closed_terminal_binding_invalid")
        standard, question_rows = [], []
        if terminal["status"] == "succeeded":
            result = _read_bound(folder, terminal["artifacts"]["result"], "result.json", guards)
            standard = _read_bound(folder, terminal["artifacts"]["standard_evidence"], "standard_evidence.json", guards)
            if standard != artifact["submission"]["evidence"] or len(standard) != len(result["evidence"]):
                raise ParagraphEvaluationError("terminal_standard_not_scored_projection")
            sourcebyid = {p["id"]: p for p in payload["paragraphs"]}; mapping = {}
            for i, (row, standardrow) in enumerate(zip(result["evidence"], standard)):
                p = sourcebyid[row["paragraph_id"]]
                if row["id"] in mapping or row["raw_doc_id"] != p["raw_doc_id"] or row["raw_para_id"] != p["raw_para_id"] or p["text"][row["start_cp"]:row["end_cp"]] != row["quote"]:
                    raise ParagraphEvaluationError("runtime_evidence_source_binding_invalid")
                projected = {"doc_id": p["raw_doc_id"], "para_id": p["raw_para_id"], "quote": row["quote"], "fact_des": row["fact_des"],
                    "argument": {k: row.get("argument", {}).get(k) or "" for k in ("时间", "主体", "地点", "内容", "影响", "数值口径")},
                    "entities": list(dict.fromkeys(row["entities"]))}
                if projected != standardrow:
                    raise ParagraphEvaluationError("runtime_standard_projection_changed")
                mapping[row["id"]] = i
            finalrow = result["final"]
            question_rows = [{"question_id": case["question_id"], "question": payload["question"]["question"],
                **{side: [{"evidence_index": mapping[identity]} for identity in finalrow["supporting_evidence_ids" if side == "supporting_facts" else "conflicting_evidence_ids"]]
                   for side in SIDES}}]
        elif artifact["submission"]["evidence"]:
            raise ParagraphEvaluationError("failed_run_has_scored_submission")
        topics.append({"topic_id": title, "evidence": evidence, "conclusions": conclusions, "source_paragraphs": source})
        submissions.append({"topic_id": title, "evidence": standard, "conclusions": question_rows, "status": terminal["status"],
            "evaluated_question_ids": [case["question_id"]], "source_scope": [{"doc_id": p["raw_doc_id"], "para_id": p["raw_para_id"]} for p in payload["paragraphs"]]})
        semantics.append({"topic_id": title, "artifact": artifact, "report": scored["report"]})
    if any(file_descriptor(g["file"])["sha256"] != g["sha256"] for g in guards):
        raise ParagraphEvaluationError("closed_input_changed_while_reading")
    return {"topics": topics, "submissions": submissions, "semantic_artifacts": semantics}, guards


def comparison_markdown(reports):
    lines = ["# T1：闭合样例的确定性段落对照", "", "纯离线，0新模型调用。A是ID来源覆盖诊断；B是已有四维语义标签的区间，不是官方评分或完整智能体表现。原行/唯一段落分别展示，重复条目未静默消失。", "",
             "| 版本/范围 | 口径 | A 命中/提交/参考 | A P / R / F1 | B 确认—可能命中 |", "|---|---|---|---|---|"]
    fmt = lambda value: "未定义" if value is None else f"{value*100:.2f}%"
    for label, report in reports.items():
        for scope, scope_label in (("full_topic_reference", "完整专题"), ("within_source_scope_reference", "选段范围")):
            for mode, mode_label in (("raw_records", "原条目"), ("unique_paragraphs", "唯一段落")):
                row = report["evidence_aggregates"][scope][mode]; a = row["A_id_only"]["micro"]
                lines.append(f"| {label}/{scope_label} | {mode_label} | {a['matches']}/{a['submitted_records']}/{a['reference_records']} | {fmt(a['precision'])} / {fmt(a['recall'])} / {fmt(a['f1'])} | {row['B_lower']['micro']['matches']}—{row['B_upper']['micro']['matches']} |")
    lines += ["", "## 每题关系与分母", "", "完整专题参考包含全部原始证据；全部原题关系分母与已执行题分母分别保留。支撑证据按每题标准支撑规模加权，另列micro；冲突同时列micro与规模加权，但仅是样例可表达的单侧清单，不能冒称冲突双方对评分。零分母为null，B缺判定的point为null。", ""]
    lines += ["| 版本/已执行题范围 | 关系 | A micro P/R | A 按标准规模加权 P/R | 参考条数 |", "|---|---|---|---|---:|"]
    for label, report in reports.items():
        for scope, scope_label in (("full_question_reference", "原问题完整参考"), ("within_source_scope_reference", "选段内参考")):
            for side in SIDES:
                row = report["question_aggregates"]["executed_questions_only"][side][scope]["raw_records"]["A_id_only"]
                micro, weighted = row["micro"], row["reference_scale_weighted"]
                lines.append(f"| {label}/{scope_label} | {'支撑' if side == 'supporting_facts' else '冲突'} | {fmt(micro['precision'])}/{fmt(micro['recall'])} | {fmt(weighted['precision'])}/{fmt(weighted['recall'])} | {micro['reference_records']} |")
    for label, report in reports.items():
        lines += [f"### {label}", "", "| 问题 | 关系 | 原标准行/选段内 | A选段命中/提交 | B选段确认—可能 |", "|---|---|---:|---:|---:|"]
        for q in report["question_reports"]:
            if not q["executed_question"]:
                continue
            for side in SIDES:
                full = q[side]["full_question_reference"]["raw_records"]; local = q[side]["within_source_scope_reference"]["raw_records"]
                a = local["A_id_only"]["metrics"]; b = local["B_existing_semantics"]
                lines.append(f"| {q['topic_id']} / {q['question_id']} | {'支撑' if side == 'supporting_facts' else '冲突'} | {full['A_id_only']['metrics']['reference_records']}/{a['reference_records']} | {a['matches']}/{a['submitted_records']} | {b['confirmed_lower_bound']['matches']}—{b['possible_upper_bound']['matches']} |")
    lines += ["", "## 至少5条误报与5条漏报的定位诊断", "", "以下是A口径的参考清单差异，不是对现实事实真假的独立判定；B的已拒绝事实对及原裁判短理由见JSON，缺失判定不作为语义错误。", ""]
    for label, report in reports.items():
        for key, title in (("A_id_only_false_positive_candidates", "误报候选"), ("A_id_only_false_negative_candidates", "漏报候选")):
            examples = report["error_examples"][key]
            lines.append(f"**{label} {title}**（可用{examples['available_rows']}条，展示{len(examples['examples'])}条；不足5条时不补造）：")
            lines.append("")
            for row in examples["examples"]:
                lines.append(f"- {row['topic_id']} · `{row.get('doc_id')}` / `{row.get('para_id')}`：{row['reason']}。{row['fact_des_excerpt']}")
            lines.append("")
    lines += ["## 实际评测范围", "", "下表来自本次闭合输入；已声明来源范围不等于模型实际阅读量。", "",
              "| 版本 | 专题 | 已执行/参考问题 | 声明来源段/全部来源段 |", "|---|---:|---:|---:|"]
    for label, report in reports.items():
        scope = report["coverage"]
        lines.append(f"| {label} | {scope['topics']} | {scope['executed_questions']}/{scope['full_reference_questions']} | {scope['selected_source_paragraphs']}/{scope['full_source_paragraphs']} |")
    lines += ["", "## 限制", "", "A同段不代表同一事实；B只复用精确绑定的已闭合判定。未匹配不等于事实已被证伪。完整专题/全问题分母可以计算，但不代表已执行全部问题或阅读全部来源；实际范围以上表为准，不能把局部对照写成隐藏集或官方分数。官方文档未明确重复条目、零分母以及冲突双方清单与附录单侧列表的差异，本报告不擅自补齐。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--pilot-plan", type=Path)
    parser.add_argument("--sample-root", type=Path)
    parser.add_argument("--rubric-docx", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ParagraphEvaluationError("new_output_directory_required")
    destination = args.output.resolve()
    protected = [p.resolve() for p in (args.comparison_dir, args.sample_root, args.pilot_plan.parent if args.pilot_plan else None) if p]
    if any(destination == root or destination.is_relative_to(root) or root.is_relative_to(destination) for root in protected):
        raise ParagraphEvaluationError("output_overlaps_protected_input_directory")
    if args.bundle:
        guards = [file_descriptor(args.bundle)]
        reports = {"submitted": evaluate_bundle(load(args.bundle))}
    else:
        if not all((args.comparison_dir, args.pilot_plan, args.sample_root)):
            parser.error("--bundle or all three closed-comparison paths are required")
        reports, guards = {}, []
        for label in ("v17", "v18"):
            bundle, bound = make_closed_pilot_bundle(args.comparison_dir, args.pilot_plan, args.sample_root, label)
            reports[label] = evaluate_bundle(bundle); guards.extend(bound)
    rubric = args.rubric_docx
    if rubric is None and args.comparison_dir:
        rubric = Path(__file__).resolve().parents[2] / "S4-多智能体可信研判与证据链溯源系统_评分标准.docx"
    rubric_source = file_descriptor(rubric) if rubric else None
    if rubric_source:
        guards.append(rubric_source)
    evaluator_source = file_descriptor(Path(__file__))
    guards.append(evaluator_source)
    if any(file_descriptor(g["file"])["sha256"] != g["sha256"] for g in guards):
        raise ParagraphEvaluationError("input_changed_during_evaluation")
    args.output.mkdir(parents=True, exist_ok=False)
    output = {"format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(), "new_model_calls": 0,
              "input_guards": guards, "input_guards_unchanged": True, "rubric_source": rubric_source,
              "evaluator_source": evaluator_source, "reports": reports}
    (args.output / "report.json").write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    markdown = comparison_markdown(reports)
    if rubric_source:
        markdown += ("\n评分口径来源：原始《S4评分标准》2.1/2.2/2.4/2.5，"
                     f"SHA256 `{rubric_source['sha256']}`。2.4“按标准答案规模加权”在本诊断显式解释为每题标准关系条目数；不是用全局micro替代加权precision。\n")
    (args.output / "report.md").write_text(markdown, encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output.resolve()), "new_model_calls": 0, "labels": list(reports)}))


if __name__ == "__main__":
    main()
