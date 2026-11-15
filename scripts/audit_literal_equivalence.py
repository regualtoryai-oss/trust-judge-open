"""Offline shared-literal audit; never a semantic judge or an official score.

Reads a closed export plus original samples. No application/provider imports,
network calls, runtime writes, transliteration, case folding, or fuzzy spans.
CLI outputs are restricted to docs/reports. Similarity is a review hint only.
"""
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import re


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def read_bound(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return json.loads(data.decode("utf-8-sig")), {
        "path": str(path), "sha256": sha256(data).hexdigest(), "bytes": len(data)}


def occurrences(text, quote):
    if not quote:
        return []
    result, start = [], 0
    while (start := text.find(quote, start)) >= 0:
        result.append([start, start + len(quote)])
        start += 1
    return result


def exact_fragments(left, right, minimum=24):
    """All maximal exact matches >= minimum, including repeated occurrences.

    Coordinates are Unicode codepoint half-open spans in the ORIGINAL strings.
    A seed is extended only if its preceding characters differ or one starts at
    zero. Thus no normalized text or SequenceMatcher block becomes a span.
    """
    if minimum < 1:
        raise ValueError("positive_fragment_minimum_required")
    index = defaultdict(list)
    for j in range(len(right) - minimum + 1):
        index[right[j:j + minimum]].append(j)
    result = []
    for i in range(len(left) - minimum + 1):
        for j in index.get(left[i:i + minimum], ()):
            if i and j and left[i - 1] == right[j - 1]:
                continue
            length = minimum
            while i + length < len(left) and j + length < len(right) and left[i + length] == right[j + length]:
                length += 1
            result.append({"left_span_cp": [i, i + length], "right_span_cp": [j, j + length],
                           "length_cp": length, "literal": left[i:i + length]})
    return sorted(result, key=lambda x: (-x["length_cp"], x["left_span_cp"], x["right_span_cp"]))


def sentence_units(text):
    # Diagnostic sentence/line units only; source paragraph boundaries stay intact.
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^。！？\n]+[。！？]?|\n", text)]


def exact_sentence_matches(left, right, minimum):
    result = []
    for a, b, literal in sentence_units(left):
        if len(literal) >= minimum:
            for c, d in occurrences(right, literal):
                result.append({"left_span_cp": [a, b], "right_span_cp": [c, d], "literal": literal})
    return result


def compare_literals(left, right, minimum=24):
    equal = bool(left) and left == right
    fragments = exact_fragments(left, right, minimum)
    # Hints are intentionally separate and cannot become certified match edges.
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    longest = matcher.find_longest_match()
    return {
        "whole_quote_equal": equal,
        "right_whole_quote_in_left": occurrences(left, right),
        "left_whole_quote_in_right": occurrences(right, left),
        "exact_fragments": fragments,
        "exact_fragment_occurrences": len(fragments),
        "unique_exact_fragment_literals": len({x["literal"] for x in fragments}),
        "exact_left_sentence_units_in_right": exact_sentence_matches(left, right, minimum),
        "semantic_equivalence": "unknown",
        "semantic_reason": "literal_identity_does_not_validate_fact_annotation" if equal else "overlap_or_similarity_cannot_establish_same_fact_time_value_or_negation",
        "review_hints_only": {
            "whitespace_removed_equal": bool(left) and re.sub(r"\s+", "", left) == re.sub(r"\s+", "", right),
            "sequence_similarity_ratio": matcher.ratio(),
            "longest_exact_fragment_below_threshold_included_as_hint": {
                "left_span_cp": [longest.a, longest.a + longest.size],
                "right_span_cp": [longest.b, longest.b + longest.size],
                "literal": left[longest.a:longest.a + longest.size], "length_cp": longest.size},
            "accepted_for_certificate": False}}


def one_to_one(left_count, right_count, edges):
    """Maximum cardinality matching; every raw record contributes at most once."""
    adjacency = defaultdict(list)
    for left, right in sorted(set(edges)):
        adjacency[left].append(right)
    chosen = {}

    def visit(left, seen):
        for right in adjacency[left]:
            if right in seen:
                continue
            seen.add(right)
            if right not in chosen or visit(chosen[right], seen):
                chosen[right] = left
                return True
        return False

    for left in range(left_count):
        visit(left, set())
    pairs = sorted([[left, right] for right, left in chosen.items()])
    return {"matches": len(pairs), "left_records": left_count, "right_records": right_count,
            "matched_left_fraction": len(pairs) / left_count if left_count else 0,
            "matched_right_fraction": len(pairs) / right_count if right_count else 0,
            "one_to_one_pairs": pairs,
            "unmatched_left_indices": sorted(set(range(left_count)) - {p[0] for p in pairs}),
            "unmatched_right_indices": sorted(set(range(right_count)) - {p[1] for p in pairs}),
            "eligible_edges": len(set(edges)), "many_to_one_edge_count_is_not_match_count": True}


def group_equal(rows, field):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row[field]].append(i)
    return [{"literal_sha256": digest(text), "length_cp": len(text), "row_indices": ids,
             "sources": [{"doc_id": rows[i]["doc_id"], "para_id": rows[i]["para_id"]} for i in ids]}
            for text, ids in groups.items()]


def record_context(row, doc, paragraphs, *, submitted):
    quote = row.get("quote", "")
    para_id = row.get("raw_para_id", row.get("para_id"))
    paragraph = paragraphs.get((doc["doc_id"], para_id))
    start, end = row.get("start_cp"), row.get("end_cp")
    if submitted:
        valid = bool(paragraph) and isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(paragraph["text"]) and paragraph["text"][start:end] == quote
        document_start = paragraph["start_cp"] + start if valid else None
        document_end = paragraph["start_cp"] + end if valid else None
    else:
        document_start, document_end = row.get("char_start"), row.get("char_end")
        valid = isinstance(document_start, int) and isinstance(document_end, int) and 0 <= document_start <= document_end <= len(doc["content"]) and doc["content"][document_start:document_end] == quote
    overlaps = [p["para_id"] for p in doc["paragraphs"] if valid and p["char_start"] < document_end and p["char_end"] > document_start]
    return {"id": row.get("id", row.get("fact_id")), "doc_id": doc["doc_id"], "para_id": para_id,
            "title": doc.get("title"), "content_url": doc.get("content_url"), "publish_time": doc.get("publish_time"),
            "quote": quote, "quote_sha256": digest(quote), "quote_length_cp": len(quote),
            "fact_des": row.get("fact_des"), "argument": row.get("argument"), "entities": row.get("entities"),
            "application_source_group_metadata": {"id": row.get("source_group_id"), "basis": row.get("source_group_basis"),
                "independent_source_certified_here": False} if submitted else None,
            "literal_coordinate_verified": valid,
            "document_span_cp": [document_start, document_end] if valid else None,
            "original_declared_span": [start, end] if submitted else [row.get("char_start"), row.get("char_end")],
            "overlapping_original_paragraph_ids": overlaps,
            "quote_crosses_original_paragraph_boundary": len(overlaps) > 1,
            "anchor_paragraph": paragraph,
            "context_before": doc["content"][max(0, document_start - 120):document_start] if valid else None,
            "context_after": doc["content"][document_end:document_end + 120] if valid else None}


def audit(source_path, result_path, topic_path, score_path, question_id, minimum=24):
    topic = Path(topic_path)
    source, source_d = read_bound(source_path)
    result, result_d = read_bound(result_path)
    documents, docs_d = read_bound(topic / "raw_documents.json")
    facts, facts_d = read_bound(topic / "evidence_samples.json")
    conclusions, conclusions_d = read_bound(topic / "conclusions.json")
    score, score_d = read_bound(score_path)
    bindings = {"source_paragraphs": source_d, "question_result": result_d, "raw_documents": docs_d,
                "reference_evidence": facts_d, "reference_conclusions": conclusions_d, "existing_score": score_d}
    question = next(x for x in conclusions if x["question_id"] == question_id)
    if score["question"]["question_id"] != question_id or score["question"]["question"] != question["question"]:
        raise ValueError("score_question_mismatch")
    docs = {x["doc_id"]: x for x in documents}
    if len(docs) != len(documents):
        raise ValueError("duplicate_original_document_id")
    paragraphs = {}
    for doc in documents:
        for p in doc["paragraphs"]:
            if (doc["doc_id"], p["para_id"]) in paragraphs:
                raise ValueError("duplicate_original_paragraph_id")
            paragraphs[doc["doc_id"], p["para_id"]] = {"doc_id": doc["doc_id"], "para_id": p["para_id"],
                "start_cp": p["char_start"], "end_cp": p["char_end"], "text": doc["content"][p["char_start"]:p["char_end"]]}
    source_checks = []
    for row in source:
        p = paragraphs.get((row["raw_doc_id"], row["raw_para_id"]))
        source_checks.append(bool(p) and p["text"] == row["text"] and p["start_cp"] == row["document_start_cp"] and p["end_cp"] == row["document_end_cp"])
    if not all(source_checks) or len(source) != len(paragraphs) or len({(r["raw_doc_id"], r["raw_para_id"]) for r in source}) != len(source):
        raise ValueError("complete_archive_original_paragraph_binding_failed")
    by_fact = {r["fact_id"]: r for r in facts}
    if len(by_fact) != len(facts):
        raise ValueError("duplicate_reference_fact_id")
    reference_links = [(role, r["fact_id"]) for role in ("supporting_facts", "conflicting_facts") for r in question[role]]
    ids = list(dict.fromkeys(fact_id for _, fact_id in reference_links))
    reference = [record_context(by_fact[i], docs[by_fact[i]["doc_id"]], paragraphs, submitted=False) for i in ids]
    submitted = [record_context(row, docs[row["raw_doc_id"]], paragraphs, submitted=True) for row in result["evidence"]]
    pairs = []
    for i, left in enumerate(submitted):
        for j, right in enumerate(reference):
            comparison = compare_literals(left["quote"], right["quote"], minimum)
            for fragment in comparison["exact_fragments"]:
                for side, record in (("left", left), ("right", right)):
                    base = record["document_span_cp"][0] if record["literal_coordinate_verified"] else None
                    fragment[f"{side}_document_span_cp"] = [base + x for x in fragment[f"{side}_span_cp"]] if base is not None else None
            pairs.append({"pair_id": f"S{i + 1}-R{j + 1}", "submitted_index": i, "reference_index": j,
                "submitted_source": [left["doc_id"], left["para_id"]], "reference_source": [right["doc_id"], right["para_id"]],
                "cross_document": left["doc_id"] != right["doc_id"],
                "both_original_quote_spans_verified": left["literal_coordinate_verified"] and right["literal_coordinate_verified"],
                **comparison})
    def eligible(p):
        return p["both_original_quote_spans_verified"]
    policies = {
        "whole_quote_equality": lambda p: p["whole_quote_equal"],
        "whole_quote_containment_either_direction": lambda p: bool(p["left_whole_quote_in_right"] or p["right_whole_quote_in_left"]),
        "exact_fragment_overlap_only_not_fact_match": lambda p: bool(p["exact_fragments"])}
    coverage = {name: one_to_one(len(submitted), len(reference), [(p["submitted_index"], p["reference_index"]) for p in pairs if eligible(p) and policy(p)])
                for name, policy in policies.items()}
    left_groups, right_groups = group_equal(submitted, "quote"), group_equal(reference, "quote")
    left_group_of = {i: g for g, row in enumerate(left_groups) for i in row["row_indices"]}
    right_group_of = {i: g for g, row in enumerate(right_groups) for i in row["row_indices"]}
    unique_coverage = {name: one_to_one(len(left_groups), len(right_groups),
        [(left_group_of[p["submitted_index"]], right_group_of[p["reference_index"]]) for p in pairs if eligible(p) and policy(p)])
        for name, policy in policies.items()}
    source_rows = list(paragraphs.values())
    relevant_sources = {(r["doc_id"], r["para_id"]) for r in submitted + reference}
    mirror_groups = [g for g in group_equal(source_rows, "text") if len(g["row_indices"]) > 1 and
                     any((r["doc_id"], r["para_id"]) in relevant_sources for r in g["sources"])]
    parent_containment = []
    for j, row in enumerate(reference):
        target = row["anchor_paragraph"]["text"]
        for p in source_rows:
            if p["text"] and (target in p["text"] or p["text"] in target):
                parent_containment.append({"reference_index": j, "reference_anchor": row["para_id"],
                    "source_doc_id": p["doc_id"], "source_para_id": p["para_id"], "whole_parent_equal": p["text"] == target,
                    "reference_parent_in_source": occurrences(p["text"], target), "source_parent_in_reference": occurrences(target, p["text"]),
                    "source_selected_final_indices": [i for i, s in enumerate(submitted) if (s["doc_id"], s["para_id"]) == (p["doc_id"], p["para_id"])],
                    "semantic_equivalence": "unknown"})
    # A complete matrix is preserved, including empty-overlap pairs. Ranking is
    # solely a human review queue; no threshold turns similarity into truth.
    review = sorted(pairs, key=lambda p: (-p["review_hints_only"]["sequence_similarity_ratio"], p["pair_id"]))
    submitted_pairs = [{"submitted_indices": [i, j], "sources": [left["para_id"], right["para_id"]],
        **compare_literals(left["quote"], right["quote"], minimum)}
        for i, left in enumerate(submitted) for j, right in enumerate(submitted) if i < j]
    output = {"format": "offline-exact-shared-literal-audit-v1", "question_id": question_id, "question": question["question"],
        "input_bindings": bindings, "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs_unchanged_after_read": all(sha256(Path(d["path"]).read_bytes()).hexdigest() == d["sha256"] for d in bindings.values()),
        "source_inventory": {"archived_paragraph_count": len(source), "original_paragraph_count": len(paragraphs), "all_original_boundaries_and_text_equal": all(source_checks)},
        "policy": {"minimum_exact_fragment_codepoints": minimum, "normalization_for_certificates": "none", "coordinate_system": "Unicode_codepoints_half_open_original_source",
            "index_system": "zero_based_record_and_group_indices_one_based_S_R_display_labels",
            "same_fact_decisions": "all_unknown_no_semantic_judge", "exact_fragments_may_be_boilerplate": True,
            "reference_use": "offline_only", "reference_written_to_runtime": False, "new_model_calls": 0,
            "new_paid_tokens": 0, "official_score": False, "semantic_B_point_score": None,
            "certificate_scope": "shared_literal_coverage_only_not_fact_precision_recall"},
        "submitted": submitted, "reference": reference,
        "reference_relation_links": [{"role": role, "fact_id": fid} for role, fid in reference_links],
        "record_accounting": {"submitted_raw_records": len(submitted), "reference_union_records": len(reference),
            "reference_relation_links": len(reference_links), "submitted_exact_quote_groups": left_groups,
            "reference_exact_quote_groups": right_groups, "complete_pair_count": len(pairs)},
        "one_to_one_literal_coverage": coverage,
        "one_to_one_unique_quote_group_literal_coverage": unique_coverage,
        "relevant_exact_parent_mirror_groups": mirror_groups,
        "reference_parent_containment_in_complete_archive": parent_containment,
        "complete_pair_matrix": pairs,
        "complete_within_submitted_pair_matrix_for_multiplicity_review": submitted_pairs,
        "human_review_queue_only": [{"pair_id": p["pair_id"], "submitted_index": p["submitted_index"], "reference_index": p["reference_index"],
            "submitted_source": p["submitted_source"], "reference_source": p["reference_source"], "semantic_equivalence": "unknown",
            "similarity_hint": p["review_hints_only"]["sequence_similarity_ratio"],
            "maximum_exact_fragment_cp": p["review_hints_only"]["longest_exact_fragment_below_threshold_included_as_hint"]["length_cp"]} for p in review],
        "existing_A_id_proxy_unchanged": score["after"]["question_selection"]["A_id_only_proxy"]}
    if not output["inputs_unchanged_after_read"]:
        raise ValueError("audit_input_changed_during_read")
    return output


def markdown(report):
    lines = ["# Exact shared-literal audit (offline)", "", f"Question: {report['question_id']} — {report['question']}", "",
        "This certificate measures shared ORIGINAL literal text, not semantic fact equivalence, official B, or final evidence precision/recall. All semantic pairs remain unknown. No paid calls, runtime writes, transliteration, or fuzzy span correction occurred.", "",
        f"Complete source inventory: {report['source_inventory']['archived_paragraph_count']} original-bound paragraphs. All input files are SHA-256 bound in audit.json and unchanged. Complete pair matrix: {len(report['complete_pair_matrix'])} pairs.", "",
        "| Literal-only criterion | Maximum one-to-one | Submitted denominator | Reference denominator |", "|---|---:|---:|---:|"]
    for name, score in report["one_to_one_literal_coverage"].items():
        lines.append(f"| {name} | {score['matches']} | {score['left_records']} | {score['right_records']} |")
    lines += ["", "Exact fragment overlap may be article footer text; its matching count is not a count of assertions. A reference record can match at most one submitted record in each criterion. Duplicate quotes and parent mirrors retain all source IDs in the JSON.", "", "Reference quote boundaries:", ""]
    for i, row in enumerate(report["reference"]):
        lines.append(f"- R{i+1} {row['id']} {row['para_id']}: document {row['document_span_cp']}, verified={row['literal_coordinate_verified']}, overlaps {', '.join(row['overlapping_original_paragraph_ids'])}.")
    lines += ["", "Review queue (similarity hints only; full original quote/annotation/context and source URL are in audit.json):", "", "| Pair | Submitted source | Reference source | Similarity hint | Longest exact fragment CP |", "|---|---|---|---:|---:|"]
    for p in report["human_review_queue_only"][:18]:
        lines.append(f"| {p['pair_id']} | {p['submitted_source'][1]} | {p['reference_source'][1]} | {p['similarity_hint']:.4f} | {p['maximum_exact_fragment_cp']} |")
    lines += ["", "Wrong-time, wrong-value, negated-context and incidental overlaps are not promoted to same-fact matches. Simplified/traditional script differences are untouched. The existing A-ID proxy is copied for provenance only and no current score is changed.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-paragraphs", required=True)
    parser.add_argument("--question-result", required=True)
    parser.add_argument("--sample-topic", required=True)
    parser.add_argument("--score-report", required=True)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-fragment-length", type=int, default=24)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    allowed = Path(__file__).resolve().parents[1] / "docs" / "reports"
    if not out.is_relative_to(allowed):
        parser.error("offline outputs must be under repository docs/reports")
    report = audit(args.source_paragraphs, args.question_result, args.sample_topic, args.score_report, args.question_id, args.min_fragment_length)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "submitted_records": len(report["submitted"]), "reference_records": len(report["reference"]),
        "one_to_one_literal_matches": {k: v["matches"] for k, v in report["one_to_one_literal_coverage"].items()}, "new_paid_tokens": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
