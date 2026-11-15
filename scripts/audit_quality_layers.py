"""Read-only audit of candidate -> selected -> final-linked evidence windows.

No gold/sample answers, model clients, runtime providers or workflows are loaded.
The revision must be explicitly pinned and ready. Missing historical candidate
ledgers remain unknown. CLI prints JSON unless --output names a new file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@contextmanager
def ro_connect(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        yield connection
    finally:
        connection.close()


def source_digest(connection):
    return digest({name: [dict(row) for row in connection.execute("SELECT * FROM " + name + " ORDER BY id")]
                   for name in ("documents", "paragraphs")})


def _ids(rows):
    return {row["id"] for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)}


def _additional(row):
    views = row.get("additional_interpretations", [])
    return views if isinstance(views, list) else []


def _location(row):
    if isinstance(row.get("paragraph_id"), str) and row["paragraph_id"]:
        return "internal", row["paragraph_id"]
    doc, para = row.get("raw_doc_id"), row.get("raw_para_id")
    if isinstance(doc, str) and isinstance(para, str) and doc and para:
        return "raw", doc, para
    return None


def candidate_anchor_audit(candidates, selected):
    """A derived fact ID is legitimate only when its parent geometry agrees.

    This establishes source lineage, not semantic approval of the annotation.
    Missing historical candidate ledgers stay unknown.
    """
    if candidates is None:
        return {"available": False, "derived_fact_records": None,
                "selected_without_valid_candidate_anchor": None}
    before = {row["id"]: row for row in candidates if isinstance(row, dict) and isinstance(row.get("id"), str)}
    derived, missing, invalid = [], [], []
    fields = ("paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote")
    for row in selected:
        if not isinstance(row, dict):
            continue
        identity = row.get("id")
        parent_id = row.get("source_evidence_id") or identity
        parent = before.get(parent_id)
        if parent is None:
            missing.append(identity)
        elif any(row.get(field) != parent.get(field) for field in fields):
            invalid.append(identity)
        elif identity != parent_id:
            derived.append({"evidence_id": identity, "source_evidence_id": parent_id,
                            "fact_view_id": row.get("fact_view_id")})
    return {"available": True, "derived_fact_records": len(derived), "derived_fact_manifest": derived,
            "selected_without_valid_candidate_anchor": sorted(missing + invalid),
            "selected_parent_geometry_mismatch": sorted(invalid),
            "meaning": "Parent source identity and exact geometry only; this does not establish fact entailment or target relevance."}


def _union_length(intervals):
    if not intervals:
        return 0
    rows = sorted(intervals)
    start, end = rows[0]
    covered = 0
    for left, right in rows[1:]:
        if left <= end:
            end = max(end, right)
        else:
            covered += end - start
            start, end = left, right
    return covered + end - start


def layer_stats(rows, *, paragraphs=None, details_limit=200):
    """All counts are complete; only optional pair-detail display is bounded."""
    if rows is None:
        return {"available": False, "window_records": None,
                "note": "This fixed question result has no persisted candidate ledger; not reconstructed from selected evidence."}
    if not isinstance(rows, list):
        raise ValueError("layer_must_be_list")
    grouped, invalid, manifest = defaultdict(list), [], []
    additional_windows = additional_records = malformed_annotations = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid.append({"row_index": index, "check": "record_type"})
            continue
        views = _additional(row)
        additional_windows += bool(views)
        additional_records += len(views)
        malformed_annotations += not isinstance(row.get("additional_interpretations", []), list)
        quote = row.get("quote")
        manifest.append({"row_index": index, "evidence_id": row.get("id"),
            "paragraph_id": row.get("paragraph_id"), "doc_id": row.get("raw_doc_id"), "para_id": row.get("raw_para_id"),
            "start_cp": row.get("start_cp"), "end_cp": row.get("end_cp"),
            "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest() if isinstance(quote, str) else None,
            "fact_des_sha256": digest(row.get("fact_des")),
            "additional_interpretation_records": len(views), "interpretation_origin": row.get("interpretation_origin")})
        start, end, location = row.get("start_cp"), row.get("end_cp"), _location(row)
        if (location is None or type(start) is not int or type(end) is not int or not 0 <= start < end
                or not isinstance(quote, str) or len(quote) != end - start):
            invalid.append({"row_index": index, "evidence_id": row.get("id"), "check": "interval_shape"})
            continue
        if paragraphs is not None:
            source = paragraphs.get(row.get("paragraph_id"))
            if source is None or end > len(source) or source[start:end] != quote:
                invalid.append({"row_index": index, "evidence_id": row.get("id"), "check": "source_span"})
                continue
        grouped[location].append((index, row))
    overlap_pairs, nested_pairs, identical_pairs = 0, 0, 0
    involved, pair_details, paragraph_details = set(), [], []
    total_length = covered_length = 0
    for location, items in sorted(grouped.items()):
        local_pairs, local_involved = 0, set()
        intervals = [(row["start_cp"], row["end_cp"]) for _, row in items]
        total_length += sum(end - start for start, end in intervals)
        covered = _union_length(intervals)
        covered_length += covered
        for left_index, (row_index, left) in enumerate(items):
            for other_index, right in items[left_index + 1:]:
                overlap = min(left["end_cp"], right["end_cp"]) - max(left["start_cp"], right["start_cp"])
                if overlap <= 0:
                    continue
                overlap_pairs += 1
                local_pairs += 1
                involved.update((row_index, other_index))
                local_involved.update((row_index, other_index))
                nested = ((left["start_cp"] <= right["start_cp"] and left["end_cp"] >= right["end_cp"])
                          or (right["start_cp"] <= left["start_cp"] and right["end_cp"] >= left["end_cp"]))
                identical = (left["start_cp"], left["end_cp"]) == (right["start_cp"], right["end_cp"])
                nested_pairs += nested
                identical_pairs += identical
                if len(pair_details) < details_limit:
                    pair_details.append({"left_row_index": row_index, "right_row_index": other_index,
                        "left_evidence_id": left.get("id"), "right_evidence_id": right.get("id"),
                        "doc_id": left.get("raw_doc_id"), "para_id": left.get("raw_para_id"),
                        "overlap_codepoints": overlap, "nested": nested, "identical_span": identical})
        if len(items) > 1:
            paragraph_details.append({"paragraph_identity": list(location), "doc_id": items[0][1].get("raw_doc_id"),
                "para_id": items[0][1].get("raw_para_id"), "window_records": len(items),
                "overlapping_window_pairs": local_pairs, "windows_in_overlap": len(local_involved),
                "union_covered_codepoints": covered})
    return {"available": True, "window_records": len(rows), "unique_window_ids": len(_ids(rows)),
        "valid_source_window_records": sum(map(len, grouped.values())), "invalid_window_records": invalid,
        "paragraphs": len(grouped), "paragraphs_with_multiple_windows": sum(len(items) > 1 for items in grouped.values()),
        "same_paragraph_overlapping_window_pairs": overlap_pairs, "windows_in_same_paragraph_overlap": len(involved),
        "nested_window_pairs": nested_pairs, "identical_span_pairs": identical_pairs,
        "sum_window_codepoints": total_length, "union_source_codepoints": covered_length,
        "overlapping_duplicate_codepoints": total_length - covered_length,
        "windows_with_additional_interpretations": additional_windows, "additional_interpretation_records": additional_records,
        "malformed_additional_interpretations_fields": malformed_annotations,
        "human_interpretation_windows": sum(isinstance(row, dict) and row.get("interpretation_origin") == "human" for row in rows),
        "window_manifest": manifest, "overlap_pair_details": pair_details,
        "overlap_pair_details_truncated": overlap_pairs > len(pair_details), "multi_window_paragraphs": paragraph_details}


def final_reference_layers(final):
    if not isinstance(final, dict):
        raise ValueError("question_final_missing")
    direct, judgment, conflict = set(), set(), set()
    for field in ("supporting_evidence_ids", "conflicting_evidence_ids"):
        direct.update(value for value in final.get(field, []) if isinstance(value, str))
    for row in final.get("judgments", []):
        if isinstance(row, dict):
            judgment.update(value for value in row.get("evidence_ids", []) if isinstance(value, str))
    for row in final.get("conflicts", []):
        if isinstance(row, dict):
            for field in ("supporting_evidence_ids", "conflicting_evidence_ids"):
                conflict.update(value for value in row.get(field, []) if isinstance(value, str))
    return direct, direct | judgment | conflict, {"direct_conclusion": sorted(direct),
        "judgment": sorted(judgment), "conflict_record": sorted(conflict)}


def additional_retention(candidates, selected, direct_ids, any_ids):
    if candidates is None:
        return {"available": False, "note": "Historical candidate annotations are unavailable."}
    before = {row["id"]: row for row in candidates if isinstance(row, dict) and isinstance(row.get("id"), str) and _additional(row)}
    after = {row["id"]: row for row in selected if isinstance(row, dict) and isinstance(row.get("id"), str)}
    lineage = candidate_anchor_audit(candidates, selected)
    valid_derived = {row["evidence_id"]: row["source_evidence_id"] for row in lineage["derived_fact_manifest"]}
    invalid_anchors = set(lineage["selected_without_valid_candidate_anchor"])
    by_parent = defaultdict(list)
    for identity, row in after.items():
        if identity in invalid_anchors:
            continue
        if identity in before:
            by_parent[identity].append(row)
        elif identity in valid_derived:
            by_parent[valid_derived[identity]].append(row)
    def annotation_digest(view):
        argument = view.get("argument") if isinstance(view.get("argument"), dict) else {}
        return digest({"fact_des": view.get("fact_des"),
                       "argument": {key: argument.get(key) for key in ("时间", "主体", "地点", "内容", "影响", "数值口径")},
                       "entities": sorted(set(view.get("entities", [])))})
    missing_views, kept_views = [], 0
    for identity, row in before.items():
        # A verified extra annotation may now be the primary annotation of its
        # own canonical fact row. Never infer this from a parent ID alone.
        available = {annotation_digest(view) for child in by_parent[identity]
                     for view in [child, *_additional(child)]}
        for view in _additional(row):
            if annotation_digest(view) in available:
                kept_views += 1
            else:
                missing_views.append({"evidence_id": identity, "interpretation_sha256": digest(view),
                                      "window_selected": bool(by_parent[identity])})
    selected_parents = {identity for identity, rows in by_parent.items() if rows}
    direct_parents = {valid_derived.get(identity, identity) for identity in direct_ids}
    any_parents = {valid_derived.get(identity, identity) for identity in any_ids}
    return {"available": True, "candidate_windows_with_additional": len(before),
        "candidate_additional_windows_still_selected": len(before.keys() & selected_parents),
        "candidate_additional_windows_direct_final_linked": len(before.keys() & direct_parents),
        "candidate_additional_windows_any_final_linked": len(before.keys() & any_parents),
        "candidate_additional_views_still_present_in_selected": kept_views,
        "candidate_additional_views_not_present_in_selected": missing_views,
        "meaning": "Candidate annotations share immutable source geometry. Independently verified versions may be promoted to separate canonical fact rows; unchecked annotations are not assumed exported."}


def audit_revision(database, *, topic_id, revision_id, question_id=None):
    with ro_connect(database) as connection:
        connection.execute("BEGIN")
        source_before = source_digest(connection)
        revision = connection.execute("SELECT topic_id,state FROM graph_revisions WHERE id=?", (revision_id,)).fetchone()
        if revision is None or revision["topic_id"] != topic_id or revision["state"] != "ready":
            raise ValueError("audit_requires_fixed_ready_revision_for_topic")
        artifact = connection.execute("SELECT payload_json FROM revision_artifacts WHERE revision_id=?", (revision_id,)).fetchone()
        if artifact is None:
            raise ValueError("audit_revision_artifact_missing")
        snapshot = json.loads(artifact["payload_json"])
        questions = {row["id"]: dict(row) for row in connection.execute("SELECT id,raw_question_id,question FROM questions WHERE topic_id=?", (topic_id,))}
        paragraphs = {row["id"]: row["text"] for row in connection.execute("SELECT id,text FROM paragraphs")}
        reports = []
        for internal_id, result in sorted(snapshot.get("question_results", {}).items()):
            meta = questions.get(internal_id, {})
            raw_id = meta.get("raw_question_id")
            if question_id is not None and question_id not in {internal_id, raw_id}:
                continue
            if not isinstance(result, dict) or not isinstance(result.get("evidence"), list):
                raise ValueError("audit_question_evidence_missing")
            selected = result["evidence"]
            candidate = result.get("candidate_evidence")
            if candidate is not None and not isinstance(candidate, list):
                raise ValueError("audit_candidate_ledger_type")
            direct_ids, any_ids, references = final_reference_layers(result.get("final"))
            selected_ids = _ids(selected)
            direct_rows = [row for row in selected if isinstance(row, dict) and row.get("id") in direct_ids]
            all_rows = [row for row in selected if isinstance(row, dict) and row.get("id") in any_ids]
            origin = result.get("origin_run_id")
            run = connection.execute("SELECT model_config_json,status,created_at,completed_at FROM runs WHERE id=?", (origin,)).fetchone() if origin else None
            config = json.loads(run["model_config_json"]) if run else {}
            safe_config = {key: config.get(key) for key in ("prompt_version", "prompt_sha256", "code_version", "model_name", "model_provider",
                "source_strategy", "source_batch_codepoints", "source_batch_spans")}
            layers = {"candidate": layer_stats(candidate, paragraphs=paragraphs),
                      "selected": layer_stats(selected, paragraphs=paragraphs),
                      "final_direct_linked": layer_stats(direct_rows, paragraphs=paragraphs),
                      "final_any_arbitration_linked": layer_stats(all_rows, paragraphs=paragraphs)}
            reports.append({"question_id": raw_id, "internal_question_id": internal_id,
                "question": meta.get("question"), "origin_run_id": origin,
                "origin_run_status": run["status"] if run else None,
                "origin_created_at": run["created_at"] if run else None,
                "origin_completed_at": run["completed_at"] if run else None, "origin_version": safe_config,
                "layers": layers,
                "selected_without_direct_final_reference_count": len(selected_ids - direct_ids),
                "selected_without_any_arbitration_reference_count": len(selected_ids - any_ids),
                "selected_without_direct_final_reference_ids": sorted(selected_ids - direct_ids),
                "selected_without_any_arbitration_reference_ids": sorted(selected_ids - any_ids),
                "direct_final_references_absent_from_selected": sorted(direct_ids - selected_ids),
                "any_arbitration_references_absent_from_selected": sorted(any_ids - selected_ids),
                "selected_ids_absent_from_candidate_ledger": sorted(selected_ids - _ids(candidate)) if candidate is not None else None,
                "candidate_anchor_lineage": candidate_anchor_audit(candidate, selected),
                "final_reference_layers": references,
                "additional_interpretation_retention": additional_retention(candidate, selected, direct_ids, any_ids),
                "selection_counts": {key: result.get("evidence_selection", {}).get(key) for key in (
                    "applied", "candidate_count", "selected_count", "rejected_count", "manual_preserved_count")},
                "source_scan": {key: result.get("source_coverage", {}).get(key) for key in (
                    "strategy", "batch_count", "completed_batches", "total_paragraphs", "scanned_paragraphs", "complete")}})
        if not reports:
            raise ValueError("audit_question_not_present_in_revision")
        source_after = source_digest(connection)
        if source_after != source_before:
            raise ValueError("audit_source_changed")
        return {"format": "evidence-quality-layer-audit-v2", "created_at": datetime.now(timezone.utc).isoformat(),
            "topic_id": topic_id, "fixed_revision_id": revision_id, "revision_state": revision["state"],
            "artifact_payload_sha256": hashlib.sha256(artifact["payload_json"].encode("utf-8")).hexdigest(),
            "source_fingerprint_sha256": source_before, "source_unchanged": True,
            "gold_read": False, "model_called": False, "database_mode": "ro+query_only",
            "notes": ["Counts describe source windows and reference plumbing, not semantic precision or recall.",
                "Overlapping same-paragraph windows can contain different facts; overlap alone does not prove redundant evidence.",
                "Final direct links are the final conclusion's support/conflict lists; any arbitration links additionally include judgments and conflict records.",
                "Selected fact rows can have new IDs after independent annotation verification; parent geometry is audited separately from literal candidate ID membership.",
                "Questions reused from older origins retain their actual prompt version; missing historical candidate ledgers remain unknown."],
            "questions": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / ".runtime/business.sqlite3")
    parser.add_argument("--topic-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--question-id", help="Optional raw or internal question ID; otherwise audit every completed question in the fixed revision.")
    parser.add_argument("--output", type=Path, help="New JSON file, normally under docs/. Existing files are never overwritten; omit to print only.")
    args = parser.parse_args()
    try:
        if args.output and (args.output.exists() or args.output.resolve() == args.database.resolve()
                            or args.output.resolve().is_relative_to((ROOT / ".runtime").resolve())):
            raise ValueError("audit_output_must_be_new_outside_runtime")
        report = audit_revision(args.database, topic_id=args.topic_id, revision_id=args.revision, question_id=args.question_id)
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            print(json.dumps({"report": str(args.output.resolve()), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "questions": [{"question_id": row["question_id"], "origin_run_id": row["origin_run_id"],
                    "prompt_version": row["origin_version"]["prompt_version"],
                    "counts": {layer: details["window_records"] for layer, details in row["layers"].items()}}
                    for row in report["questions"]]}, ensure_ascii=False))
        else:
            print(text, end="")
    except Exception:
        # Arbitrary DB/source/configuration text is never exposed in errors.
        print(json.dumps({"error": "quality_layer_audit_failed", "model_called": False}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
