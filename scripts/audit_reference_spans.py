"""Offline source-contract audit. Never a runtime selector or semantic judge."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


def literal_occurrences(content, quote):
    """Enumerate all exact codepoint occurrences, including overlapping ones.

    Search is confined to the declared source document. Empty/non-string quotes
    are not searched; whitespace/script variants are never normalized.
    """
    if not isinstance(quote, str) or not quote:
        return []
    positions, begin = [], 0
    while (begin := content.find(quote, begin)) >= 0:
        positions.append([begin, begin + len(quote)])
        begin += 1
    return positions


def span_diagnostic(doc, paragraphs, reference, start, end):
    overlap = [p for p in paragraphs if p["char_start"] < end and p["char_end"] > start]
    containing = [p for p in paragraphs if p["char_start"] <= start < end <= p["char_end"]]
    anchor = next((p for p in paragraphs if p["para_id"] == reference.get("para_id")), None)
    return {"document_span_cp": [start, end], "quote_length_cp": end - start,
        "overlapped_original_paragraphs": [p["para_id"] for p in overlap],
        "overlapped_original_paragraph_count": len(overlap),
        "overlaps_multiple_original_paragraphs": len(overlap) > 1,
        # The original overlap-only count is retained explicitly for v1 totals.
        "v1_requires_cross_paragraph_source_span_overlap_indicator": len(overlap) > 1,
        "requires_cross_paragraph_source_span": len(overlap) > 1 and not containing,
        "wholly_containing_original_paragraphs": [p["para_id"] for p in containing],
        "quote_wholly_inside_any_original_paragraph": bool(containing),
        "crosses_paragraphs_without_single_parent_containment": len(overlap) > 1 and not containing,
        "original_quote_starts_at_claimed_paragraph_start": anchor is not None and start == anchor["char_start"],
        "quote_wholly_inside_claimed_paragraph": anchor is not None and anchor["char_start"] <= start < end <= anchor["char_end"],
        "anchor_paragraph_exists": anchor is not None,
        "quote_start_inside_claimed_anchor": anchor is not None and anchor["char_start"] <= start < anchor["char_end"],
        "context_before": doc["content"][max(0, start - 100):start],
        "context_after": doc["content"][end:end + 100]}


def audit_topic(documents, references):
    by_doc, overlapping_intervals = {}, []
    for document in documents:
        identity = document["doc_id"]
        if identity in by_doc:
            raise ValueError("duplicate_source_document")
        content = document["content"]
        paragraphs = sorted(document["paragraphs"], key=lambda p: (p["char_start"], p["char_end"]))
        seen, previous = set(), 0
        for paragraph in paragraphs:
            start, end = paragraph["char_start"], paragraph["char_end"]
            if (paragraph["para_id"] in seen or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(content)
                    or paragraph.get("text", content[start:end]) != content[start:end]):
                raise ValueError("invalid_source_paragraph_manifest")
            if start < previous:
                overlapping_intervals.append({"doc_id": identity, "para_id": paragraph["para_id"],
                    "start_cp": start, "end_cp": end, "previous_max_end_cp": previous,
                    "interpretation": "valid original intervals can overlap; not repaired or collapsed"})
            previous = max(previous, end)
            seen.add(paragraph["para_id"])
        by_doc[identity] = (document, paragraphs)
    results = []
    for reference in references:
        item = {"fact_id": reference.get("fact_id"), "doc_id": reference.get("doc_id"),
                "para_id": reference.get("para_id"), "issues": [],
                "declared_document_span_cp": [reference.get("char_start"), reference.get("char_end")],
                "original_quote": reference.get("quote"), "reference_offsets_changed": False}
        source = by_doc.get(reference.get("doc_id"))
        if source is None:
            item["issues"].append("unknown_document")
            item.update(literal_document_span_verified=False,
                same_document_literal_search={"status": "unknown_document", "occurrence_count": None,
                    "occurrences": [], "chosen_occurrence": None, "semantic_equivalence": "unknown"},
                scope_diagnostic_basis="unavailable_unknown_document", scope_diagnostic=None)
        else:
            doc, paragraphs = source
            start, end = reference.get("char_start"), reference.get("char_end")
            in_bounds = type(start) is int and type(end) is int and 0 <= start <= end <= len(doc["content"])
            item.update(source_title=doc.get("title"), content_url=doc.get("content_url"),
                declared_source_slice=doc["content"][start:end] if in_bounds else None,
                quote_sha256=sha256(reference["quote"].encode("utf-8")).hexdigest() if isinstance(reference.get("quote"), str) else None)
            valid = (type(start) is int and type(end) is int and 0 <= start < end <= len(doc["content"])
                     and isinstance(reference.get("quote"), str) and doc["content"][start:end] == reference["quote"])
            item["literal_document_span_verified"] = valid
            if not valid:
                item["issues"].append("quote_document_span_mismatch")
            else:
                item.update(span_diagnostic(doc, paragraphs, reference, start, end))
                if not item["anchor_paragraph_exists"]:
                    item["issues"].append("unknown_anchor_paragraph")
                elif not item["quote_start_inside_claimed_anchor"]:
                    item["issues"].append("quote_start_outside_claimed_anchor")
                if not item["overlapped_original_paragraphs"]:
                    item["issues"].append("quote_outside_source_paragraphs")
            occurrences = []
            for left, right in literal_occurrences(doc["content"], reference.get("quote")):
                occurrences.append({**span_diagnostic(doc, paragraphs, reference, left, right),
                    "matches_declared_coordinates": [left, right] == item["declared_document_span_cp"]})
            status = ("invalid_quote" if not isinstance(reference.get("quote"), str) or not reference["quote"]
                      else "not_found" if not occurrences else "unique" if len(occurrences) == 1 else "multiple")
            item["same_document_literal_search"] = {"status": status, "occurrence_count": len(occurrences),
                "occurrences": occurrences, "chosen_occurrence": None,
                "semantic_equivalence": "unknown", "normalization": "none",
                "unique_location_is_diagnostic_only": status == "unique", "reference_offsets_changed": False}
            if valid:
                item.update(scope_diagnostic_basis="declared_verified_span", scope_diagnostic=span_diagnostic(doc, paragraphs, reference, start, end))
            elif status == "unique":
                item.update(scope_diagnostic_basis="unique_literal_diagnostic_only_offsets_unapplied", scope_diagnostic=occurrences[0])
            else:
                item.update(scope_diagnostic_basis=f"unavailable_{status}", scope_diagnostic=None)
        results.append(item)
    valid = [r for r in results if r.get("literal_document_span_verified")]
    diagnostics = [r for r in results if r["scope_diagnostic"] is not None]
    return {"documents": len(documents), "source_paragraphs": sum(len(d["paragraphs"]) for d in documents),
            "overlapping_original_intervals": overlapping_intervals,
            "reference_records": len(results), "literal_document_span_verified": len(valid),
            "cross_paragraph_quotes": sum(r.get("v1_requires_cross_paragraph_source_span_overlap_indicator", False) for r in valid),
            "inside_claimed_single_paragraph": sum(r.get("quote_wholly_inside_claimed_paragraph", False) for r in valid),
            "overlapped_paragraph_count_histogram": dict(sorted(Counter(r["overlapped_original_paragraph_count"] for r in valid).items())),
            "records_with_issues": sum(bool(r["issues"]) for r in results),
            "declared_span_denominators": {"all_reference_records": len(results), "verified_declared_spans": len(valid),
                "unverified_declared_spans": len(results) - len(valid),
                "cross_paragraph_count_applies_only_to_verified_declared_spans": True},
            "literal_search_status_counts": dict(sorted(Counter(r["same_document_literal_search"]["status"] for r in results).items())),
            "unverified_declared_span_literal_status_counts": dict(sorted(Counter(r["same_document_literal_search"]["status"] for r in results if not r["literal_document_span_verified"]).items())),
            "scope_diagnostic_totals": {"all_reference_records": len(results), "available_records": len(diagnostics),
                "unknown_records": len(results) - len(diagnostics),
                "basis_counts": dict(sorted(Counter(r["scope_diagnostic_basis"] for r in results).items())),
                "overlaps_multiple_original_paragraphs": sum(r["scope_diagnostic"]["overlaps_multiple_original_paragraphs"] for r in diagnostics),
                "crosses_paragraphs_without_single_parent_containment": sum(r["scope_diagnostic"]["crosses_paragraphs_without_single_parent_containment"] for r in diagnostics),
                "wholly_inside_claimed_single_paragraph": sum(r["scope_diagnostic"]["quote_wholly_inside_claimed_paragraph"] for r in diagnostics),
                "wholly_inside_any_original_paragraph": sum(r["scope_diagnostic"]["quote_wholly_inside_any_original_paragraph"] for r in diagnostics),
                "quote_start_outside_claimed_anchor": sum(not r["scope_diagnostic"]["quote_start_inside_claimed_anchor"] for r in diagnostics),
                "not_a_repaired_reference_or_recall_ceiling": True},
            "records": results}


DECLARED_TOTAL_FIELDS = (
    "documents", "source_paragraphs", "reference_records", "literal_document_span_verified",
    "cross_paragraph_quotes", "inside_claimed_single_paragraph", "records_with_issues",
)


def prior_input_pairs(path):
    """Use only the already-enumerated prior report files, with SHA guards."""
    data = Path(path).read_bytes()
    prior = json.loads(data.decode("utf-8-sig"))
    grouped = {}
    for guard in prior["input_guards"]:
        file = Path(guard["file"])
        if file.name not in {"raw_documents.json", "evidence_samples.json"}:
            raise ValueError("unexpected_prior_input_file")
        files = grouped.setdefault(file.parent, {})
        if file.name in files:
            raise ValueError("duplicate_prior_input_file")
        files[file.name] = (file, guard["sha256"])
    pairs = []
    for folder, files in sorted(grouped.items(), key=lambda x: str(x[0])):
        if set(files) != {"raw_documents.json", "evidence_samples.json"}:
            raise ValueError("incomplete_prior_input_pair")
        pairs.append((folder.name, files["raw_documents.json"], files["evidence_samples.json"]))
    return pairs, prior, {"file": str(Path(path).resolve()), "sha256": sha256(data).hexdigest()}


def build_report(input_pairs, *, prior=None, prior_guard=None):
    topics, guards = [], []
    for title, raw, evidence in input_pairs:
        values = []
        for path, expected_sha in (raw, evidence):
            data = Path(path).read_bytes()
            checksum = sha256(data).hexdigest()
            if expected_sha is not None and checksum != expected_sha:
                raise ValueError("prior_input_sha256_changed")
            guards.append({"file": str(Path(path).resolve()), "sha256": checksum})
            values.append(json.loads(data.decode("utf-8-sig")))
        topics.append({"topic": title, **audit_topic(*values)})
    totals = {key: sum(topic[key] for topic in topics) for key in DECLARED_TOTAL_FIELDS}
    if prior is not None and any(prior["totals"][key] != totals[key] for key in DECLARED_TOTAL_FIELDS):
        raise ValueError("prior_declared_totals_changed")
    def summed_counter(key):
        counts = Counter()
        for topic in topics:
            counts.update(topic[key])
        return dict(sorted(counts.items()))
    basis_counts = Counter()
    for topic in topics:
        basis_counts.update(topic["scope_diagnostic_totals"]["basis_counts"])
    diagnostic_keys = ("all_reference_records", "available_records", "unknown_records", "overlaps_multiple_original_paragraphs",
        "crosses_paragraphs_without_single_parent_containment", "wholly_inside_claimed_single_paragraph", "wholly_inside_any_original_paragraph", "quote_start_outside_claimed_anchor")
    diagnostic_totals = {key: sum(topic["scope_diagnostic_totals"][key] for topic in topics) for key in diagnostic_keys}
    diagnostic_totals.update(basis_counts=dict(sorted(basis_counts.items())), not_a_repaired_reference_or_recall_ceiling=True)
    anomalies = [{"topic": topic["topic"], **record} for topic in topics for record in topic["records"]
                 if record["issues"] or record["same_document_literal_search"]["status"] != "unique"]
    result = {"format": "offline-reference-source-span-contract-v2", "offline_only": True,
        "new_provider_calls": 0, "new_paid_tokens": 0, "references_written_to_runtime": False,
        "reference_files_modified": False, "reference_offset_repair_applied": False,
        "semantic_accuracy_measured": False, "official_score": False,
        "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "topics": topics, "input_guards": guards, "totals": totals,
        "declared_span_denominators": {"all_reference_records": totals["reference_records"],
            "verified_declared_spans": totals["literal_document_span_verified"],
            "unverified_declared_spans": totals["reference_records"] - totals["literal_document_span_verified"],
            "cross_paragraph_count_applies_only_to_verified_declared_spans": True},
        "literal_search_status_counts": summed_counter("literal_search_status_counts"),
        "unverified_declared_span_literal_status_counts": summed_counter("unverified_declared_span_literal_status_counts"),
        "scope_diagnostic_totals": diagnostic_totals,
        "anomalies": anomalies,
        "declared_span_anomalies": [r for r in anomalies if not r["literal_document_span_verified"]],
        "prior_report": prior_guard,
        "anomaly_explanations": {
            "unknown_document": "The declared source document ID is absent. No alternate document is searched and scope remains unknown.",
            "not_found": "The complete original quote has no exact occurrence in its declared source document. This does not establish that the factual assertion is false or quantify retrieval recall.",
            "multiple": "Every same-document exact occurrence is reported. A valid explicit declared span remains usable for source-scope diagnostics; an invalid declared span is not resolved by choosing an occurrence.",
            "unique_with_invalid_declared_span": "The unique literal location may describe paragraph/anchor scope only. It does not replace, repair, or validate the original declared coordinates.",
            "invalid_quote": "Empty or non-string quote cannot be searched or located as a nonempty literal source span."},
        "prior_declared_totals_preserved": prior is not None,
        "interpretation": "Original declared-span totals are retained and their denominator is explicit. Unique exact same-document locations are diagnostics only; multiple or absent locations remain unresolved. No offset is repaired. Neither representability, unknown locations, nor cross-paragraph counts measure fact accuracy or an evidence recall ceiling.",
        "method": {"search_scope": "declared_document_only", "normalization": "none", "coordinate_system": "original_Unicode_codepoints_half_open",
            "occurrences": "all_exact_occurrences_including_overlaps", "multiple_occurrences": "all_reported_none_chosen_for_invalid_declared_span",
            "overlap_warning": "The preserved v1 cross_paragraph_quotes total and explicitly named v1 overlap indicator count multiple overlapped original intervals. The v2 per-record requires_cross_paragraph_source_span and crosses_paragraphs_without_single_parent_containment are true only when multiple original intervals overlap the quote and no single original parent contains it. Claimed-parent and any-parent containment are reported separately.",
            "unique_location": "Used only to describe paragraph scope and anchor differences, never to rewrite or validate the declared reference offset."}}
    if prior_guard is not None:
        guards_to_check = guards + [prior_guard]
    else:
        guards_to_check = guards
    result["inputs_unchanged_after_read"] = all(sha256(Path(g["file"]).read_bytes()).hexdigest() == g["sha256"] for g in guards_to_check)
    if not result["inputs_unchanged_after_read"]:
        raise ValueError("audit_inputs_changed_during_read")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-root", type=Path)
    source.add_argument("--prior-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prior_report is not None:
        pairs, prior, prior_guard = prior_input_pairs(args.prior_report)
        result = build_report(pairs, prior=prior, prior_guard=prior_guard)
    else:
        pairs = [(folder.name, (folder / "raw_documents.json", None), (folder / "evidence_samples.json", None))
                 for folder in sorted(args.sample_root.iterdir())
                 if (folder / "raw_documents.json").is_file() and (folder / "evidence_samples.json").is_file()]
        result = build_report(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"totals": result["totals"], "literal_search_status_counts": result["literal_search_status_counts"],
        "unverified_declared_span_literal_status_counts": result["unverified_declared_span_literal_status_counts"],
        "scope_diagnostic_totals": result["scope_diagnostic_totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
