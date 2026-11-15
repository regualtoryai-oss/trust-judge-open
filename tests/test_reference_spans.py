from copy import deepcopy
from hashlib import sha256
import json

import pytest

from scripts.audit_reference_spans import audit_topic, build_report, literal_occurrences, prior_input_pairs


def inputs():
    text = "😀前提\r\n\t结论。"
    return ([{"doc_id": "doc", "content": text, "paragraphs": [
        {"para_id": "opaque-a", "char_start": 0, "char_end": 3},
        {"para_id": "opaque-b", "char_start": 6, "char_end": 9}]}],
        [{"fact_id": "f", "doc_id": "doc", "para_id": "opaque-a", "quote": text,
          "char_start": 0, "char_end": len(text)}])


def test_cross_paragraph_reference_is_literal_contiguous_doc_slice_with_unchanged_inputs():
    documents, refs = inputs(); original = deepcopy((documents, refs))
    audit = audit_topic(documents, refs)
    assert audit["cross_paragraph_quotes"] == 1 and audit["inside_claimed_single_paragraph"] == 0
    assert audit["records"][0]["overlapped_original_paragraphs"] == ["opaque-a", "opaque-b"]
    assert (documents, refs) == original


def test_unique_literal_diagnostic_never_repairs_reference_coordinates():
    docs, refs = inputs(); refs[0].update(quote="前提", char_start=0, char_end=2)
    original = deepcopy(refs)
    result = audit_topic(docs, refs)
    assert result["literal_document_span_verified"] == 0
    record = result["records"][0]
    assert record["issues"] == ["quote_document_span_mismatch"]
    assert record["declared_document_span_cp"] == [0, 2]
    assert "document_span_cp" not in record
    assert record["same_document_literal_search"]["status"] == "unique"
    assert record["same_document_literal_search"]["occurrences"][0]["document_span_cp"] == [1, 3]
    assert record["scope_diagnostic_basis"] == "unique_literal_diagnostic_only_offsets_unapplied"
    assert record["scope_diagnostic"]["document_span_cp"] == [1, 3]
    assert not record["reference_offsets_changed"]
    assert refs == original


def test_wrong_anchor_is_reported_separately_from_verified_quote():
    docs, refs = inputs(); refs[0]["para_id"] = "opaque-b"
    result = audit_topic(docs, refs)
    assert result["literal_document_span_verified"] == 1 and result["records_with_issues"] == 1
    assert result["records"][0]["issues"] == ["quote_start_outside_claimed_anchor"]


def test_single_paragraph_quote_is_not_called_cross_paragraph():
    docs, refs = inputs(); refs[0].update(quote="结论。", char_start=6, char_end=9, para_id="opaque-b")
    result = audit_topic(docs, refs)
    assert result["inside_claimed_single_paragraph"] == 1 and result["cross_paragraph_quotes"] == 0


def test_overlapping_original_intervals_are_disclosed_without_changing_raw_definitions():
    docs, refs = inputs(); docs[0]["paragraphs"][1]["char_start"] = 2
    result = audit_topic(docs, refs)
    assert len(result["overlapping_original_intervals"]) == 1
    assert docs[0]["paragraphs"][1]["char_start"] == 2


@pytest.mark.parametrize("mutation", ["duplicate", "wrong-text", "wrong-type"])
def test_invalid_source_manifest_is_never_used_for_counts(mutation):
    docs, refs = inputs()
    if mutation == "duplicate": docs[0]["paragraphs"][1]["para_id"] = "opaque-a"
    if mutation == "wrong-text": docs[0]["paragraphs"][0]["text"] = "模型改写"
    if mutation == "wrong-type": docs[0]["paragraphs"][0]["char_start"] = False
    with pytest.raises(ValueError): audit_topic(docs, refs)


def test_literal_enumeration_includes_overlapping_occurrences_and_unicode_codepoints():
    assert literal_occurrences("😀甲甲甲", "甲甲") == [[1, 3], [2, 4]]
    assert literal_occurrences("😀甲甲甲", "") == []
    assert literal_occurrences("😀甲甲甲", None) == []


def repeated_inputs(start=0, end=2):
    text = "😀甲😀甲"
    return ([{"doc_id": "doc", "content": text, "paragraphs": [
        {"para_id": "opaque-first", "char_start": 0, "char_end": 2},
        {"para_id": "opaque-second", "char_start": 2, "char_end": 4}]}],
        [{"fact_id": "f", "doc_id": "doc", "para_id": "opaque-second", "quote": "😀甲", "char_start": start, "char_end": end}])


def test_multiple_occurrences_do_not_block_a_valid_explicit_declared_span():
    docs, refs = repeated_inputs(2, 4)
    result = audit_topic(docs, refs)
    record = result["records"][0]
    assert result["literal_search_status_counts"] == {"multiple": 1}
    assert result["literal_document_span_verified"] == 1
    assert record["same_document_literal_search"]["occurrence_count"] == 2
    assert [x["matches_declared_coordinates"] for x in record["same_document_literal_search"]["occurrences"]] == [False, True]
    assert record["scope_diagnostic_basis"] == "declared_verified_span"
    assert record["scope_diagnostic"]["document_span_cp"] == [2, 4]
    assert record["same_document_literal_search"]["chosen_occurrence"] is None


def test_invalid_declared_span_with_multiple_occurrences_remains_unknown():
    docs, refs = repeated_inputs(1, 3)
    result = audit_topic(docs, refs)
    record = result["records"][0]
    assert record["same_document_literal_search"]["status"] == "multiple"
    assert record["scope_diagnostic"] is None
    assert record["scope_diagnostic_basis"] == "unavailable_multiple"
    assert result["scope_diagnostic_totals"]["unknown_records"] == 1
    assert result["literal_document_span_verified"] == 0


def test_unique_cross_paragraph_location_is_diagnostic_not_declared_verified_count():
    docs, refs = inputs()
    refs[0]["char_start"] = 1
    result = audit_topic(docs, refs)
    assert result["cross_paragraph_quotes"] == 0
    assert result["literal_document_span_verified"] == 0
    assert result["declared_span_denominators"] == {"all_reference_records": 1, "verified_declared_spans": 0,
        "unverified_declared_spans": 1, "cross_paragraph_count_applies_only_to_verified_declared_spans": True}
    assert result["scope_diagnostic_totals"]["overlaps_multiple_original_paragraphs"] == 1
    assert result["scope_diagnostic_totals"]["basis_counts"] == {"unique_literal_diagnostic_only_offsets_unapplied": 1}


def test_unique_location_reports_wrong_anchor_without_changing_original_issue_count():
    docs, refs = inputs()
    refs[0].update(quote="前提", char_start=0, char_end=2, para_id="opaque-b")
    result = audit_topic(docs, refs)
    record = result["records"][0]
    assert record["issues"] == ["quote_document_span_mismatch"]
    assert not record["scope_diagnostic"]["quote_start_inside_claimed_anchor"]
    assert result["scope_diagnostic_totals"]["quote_start_outside_claimed_anchor"] == 1


def test_literal_absence_is_not_repaired_by_whitespace_or_cross_document_search():
    docs, refs = inputs()
    refs[0]["quote"] = "😀前提结论。"  # spaces/newlines are immutable
    docs.append({"doc_id": "another", "content": refs[0]["quote"], "paragraphs": [
        {"para_id": "somewhere-else", "char_start": 0, "char_end": len(refs[0]["quote"])}]})
    result = audit_topic(docs, refs)
    record = result["records"][0]
    assert record["same_document_literal_search"]["status"] == "not_found"
    assert record["same_document_literal_search"]["occurrences"] == []
    assert record["scope_diagnostic"] is None
    assert result["unverified_declared_span_literal_status_counts"] == {"not_found": 1}


def test_missing_document_is_distinct_from_literal_absence():
    docs, refs = inputs(); refs[0]["doc_id"] = "missing"
    result = audit_topic(docs, refs)
    record = result["records"][0]
    assert record["issues"] == ["unknown_document"]
    assert record["same_document_literal_search"]["status"] == "unknown_document"
    assert record["same_document_literal_search"]["occurrence_count"] is None
    assert record["scope_diagnostic"] is None


def test_overlapping_original_intervals_are_not_automatically_multiple_parent_requirement():
    docs, refs = inputs()
    docs[0]["paragraphs"] = [
        {"para_id": "opaque-a", "char_start": 0, "char_end": len(docs[0]["content"])},
        {"para_id": "nested", "char_start": 1, "char_end": 3}]
    result = audit_topic(docs, refs)
    assert result["cross_paragraph_quotes"] == 1  # preserved v1 overlap indicator
    assert result["scope_diagnostic_totals"]["crosses_paragraphs_without_single_parent_containment"] == 0
    assert result["records"][0]["wholly_containing_original_paragraphs"] == ["opaque-a"]
    assert result["records"][0]["quote_wholly_inside_any_original_paragraph"]
    assert not result["records"][0]["requires_cross_paragraph_source_span"]
    assert result["records"][0]["v1_requires_cross_paragraph_source_span_overlap_indicator"]


def test_prior_guarded_inputs_preserve_original_declared_totals_and_source_files(tmp_path):
    topic = tmp_path / "opaque-topic"
    topic.mkdir()
    docs, refs = inputs()
    guards = []
    for filename, value in (("raw_documents.json", docs), ("evidence_samples.json", refs)):
        path = topic / filename
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        guards.append({"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest()})
    original = audit_topic(docs, refs)
    keys = ("documents", "source_paragraphs", "reference_records", "literal_document_span_verified", "cross_paragraph_quotes", "inside_claimed_single_paragraph", "records_with_issues")
    prior = {"input_guards": guards, "totals": {key: original[key] for key in keys}}
    prior_path = tmp_path / "v1.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    pairs, loaded, guard = prior_input_pairs(prior_path)
    report = build_report(pairs, prior=loaded, prior_guard=guard)
    assert report["format"] == "offline-reference-source-span-contract-v2"
    assert report["totals"] == prior["totals"]
    assert report["inputs_unchanged_after_read"] and report["prior_declared_totals_preserved"]
    assert report["new_provider_calls"] == 0 and not report["reference_offset_repair_applied"]
    with (topic / "evidence_samples.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="prior_input_sha256_changed"):
        build_report(pairs, prior=loaded, prior_guard=guard)
