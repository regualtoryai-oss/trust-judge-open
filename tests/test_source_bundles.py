from copy import deepcopy
from hashlib import sha256
from uuid import UUID

import pytest

from app.domain.source_bundles import (
    SOURCE_BUNDLE_CORE_FIELDS, assemble_source_bundle, build_source_bundles,
    paragraph_manifest_sha256, validate_source_bundle,
)


def text_sha(text):
    return sha256(text.encode("utf-8")).hexdigest()


def corpus(pieces, *, gaps=None, doc_id="doc-uuid", raw_doc_id="原始文档", prefix="前缀|", suffix="|尾部"):
    gaps = gaps if gaps is not None else ["\n"] * max(0, len(pieces) - 1)
    content, rows, manifest = prefix, [], []
    for i, text in enumerate(pieces):
        start = len(content)
        content += text
        end = len(content)
        # Deliberately opaque names: no ordinal inference is possible.
        pid, raw_pid = f"{doc_id}-uuid-{i}", f"原段-{chr(937 - i)}-保留"
        rows.append({"id": pid, "document_id": doc_id, "raw_doc_id": raw_doc_id, "raw_para_id": raw_pid,
                     "text": text, "document_start_cp": start, "document_end_cp": end})
        manifest.append({"id": pid, "raw_para_id": raw_pid, "start_cp": start, "end_cp": end, "text_sha256": text_sha(text)})
        if i < len(gaps):
            content += gaps[i]
    content += suffix
    doc = {"id": doc_id, "raw_doc_id": raw_doc_id, "content": content, "content_sha256": text_sha(content),
           "paragraph_manifest": manifest, "paragraph_manifest_sha256": paragraph_manifest_sha256(manifest),
           "paragraph_ids": [p["id"] for p in rows]}
    return [doc], rows


def seal_manifest(doc):
    doc["paragraph_manifest_sha256"] = paragraph_manifest_sha256(doc["paragraph_manifest"])
    doc["paragraph_ids"] = [p["id"] for p in doc["paragraph_manifest"]]


def test_document_codepoints_preserve_emoji_and_non_newline_gaps():
    docs, rows = corpus(["😀第一段", "同一句。", "第三段🚀"], gaps=["\r\n<原分隔符>", "|不是换行|"])
    before = deepcopy((docs, rows))
    result = assemble_source_bundle(docs, rows, rows[0]["id"], rows[2]["id"])
    assert set(result) == set(SOURCE_BUNDLE_CORE_FIELDS)
    assert result["span_scope"] == "document"
    assert result["start_cp"] == len("前缀|")
    assert result["end_cp"] == rows[2]["document_end_cp"]
    assert result["quote"] == "😀第一段\r\n<原分隔符>同一句。|不是换行|第三段🚀"
    assert result["quote"] != "\n".join(r["text"] for r in rows)
    assert result["source_paragraph_ids"] == [r["id"] for r in rows]
    assert result["document_content_sha256"] == text_sha(docs[0]["content"])
    assert UUID(result["bundle_id"]).version == 5
    assert (docs, rows) == before


def test_same_repeated_quotes_keep_explicit_original_locations():
    docs, rows = corpus(["相同引文😀", "相同引文😀"], gaps=["--"])
    first = assemble_source_bundle(docs, rows, rows[0]["id"], rows[0]["id"])
    second = assemble_source_bundle(docs, rows, rows[1]["id"], rows[1]["id"])
    assert first["quote"] == second["quote"]
    assert first["start_cp"] != second["start_cp"]
    assert first["bundle_id"] != second["bundle_id"]
    assert validate_source_bundle(second, docs, rows) == second


def test_read_context_and_final_citation_use_different_real_first_parent():
    docs, rows = corpus([f"段落{i}。" for i in range(7)])
    context = build_source_bundles(docs, rows, [rows[2]["id"]])[0]
    assert context["source_paragraph_ids"] == [p["id"] for p in rows[1:6]]
    assert context["bundle_purpose"] == "read_context"
    assert context["anchor_paragraph_ids"] == [rows[2]["id"]]
    citation = assemble_source_bundle(docs, rows, rows[3]["id"], rows[4]["id"])
    assert citation["paragraph_id"] == rows[3]["id"]
    assert citation["raw_para_id"] == rows[3]["raw_para_id"]
    assert citation["start_cp"] == rows[3]["document_start_cp"]
    assert set(citation["source_paragraph_ids"]) <= set(context["source_paragraph_ids"])
    assert "bundle_purpose" not in citation
    assert citation["bundle_id"] != context["bundle_id"]


def test_context_edges_deduplication_and_anchor_order():
    docs, rows = corpus(["甲", "乙"])
    bundles = build_source_bundles(docs, rows, [rows[1]["id"], rows[0]["id"], rows[1]["id"]], before=1, after=1, max_paragraphs=3)
    assert len(bundles) == 1
    assert bundles[0]["source_paragraph_ids"] == [r["id"] for r in rows]
    assert bundles[0]["anchor_paragraph_ids"] == [rows[1]["id"], rows[0]["id"]]
    assert build_source_bundles(docs, rows, []) == []


def test_empty_original_paragraphs_and_same_cp_manifest_order_are_preserved():
    docs, rows = corpus(["甲", "", "", "乙"], gaps=["", "", ""])
    result = assemble_source_bundle(docs, rows, rows[0]["id"], rows[-1]["id"])
    assert result["quote"] == "甲乙"
    assert result["source_paragraph_ids"] == [r["id"] for r in rows]
    empty = assemble_source_bundle(docs, rows, rows[1]["id"], rows[2]["id"])
    assert empty["start_cp"] == empty["end_cp"]
    assert empty["quote"] == ""
    assert empty["source_paragraph_ids"] == [rows[1]["id"], rows[2]["id"]]
    assert validate_source_bundle(empty, docs, rows) == empty


def test_stable_id_ignores_collection_order_and_extra_metadata():
    docs, rows = corpus(["甲", "乙", "丙"])
    other_docs, other_rows = corpus(["外文"], doc_id="other-uuid", raw_doc_id="其他")
    bundle = assemble_source_bundle(docs + other_docs, rows + other_rows, rows[0]["id"], rows[-1]["id"])
    docs[0]["title"] = "Non-source annotation metadata"
    repeated = assemble_source_bundle(other_docs + docs, list(reversed(rows + other_rows)), rows[0]["id"], rows[-1]["id"])
    assert repeated == bundle
    annotated = {**bundle, "fact_des": "Unverified annotation", "argument": {"时间": "未确认"}, "entities": ["任意实体"]}
    copied = validate_source_bundle(annotated, docs + other_docs, rows + other_rows)
    assert copied == annotated and copied is not annotated
    copied["entities"].append("Other")
    assert annotated["entities"] == ["任意实体"]


def test_missing_middle_opaque_id_rejected_even_when_text_gap_looks_legitimate():
    docs, rows = corpus(["甲", "被遗漏的真实原段", "丙"], gaps=[":::分隔:::全保留", "<br>"])
    with pytest.raises(ValueError, match="complete_paragraph_rows"):
        assemble_source_bundle(docs, [rows[0], rows[2]], rows[0]["id"], rows[2]["id"])
    with pytest.raises(ValueError, match="complete_paragraph_rows"):
        build_source_bundles(docs, [rows[0], rows[2]], [rows[0]["id"]])


def test_manifest_and_document_sha256_tampering_rejected():
    docs, rows = corpus(["甲", "乙"])
    bad_doc = deepcopy(docs)
    bad_doc[0]["content"] += "篡改"
    with pytest.raises(ValueError, match="document_sha256"):
        assemble_source_bundle(bad_doc, rows, rows[0]["id"], rows[1]["id"])
    bad_doc = deepcopy(docs)
    bad_doc[0]["paragraph_manifest"].pop(0)
    with pytest.raises(ValueError, match="manifest_sha256"):
        assemble_source_bundle(bad_doc, rows, rows[0]["id"], rows[1]["id"])
    bad_doc = deepcopy(docs)
    bad_doc[0]["paragraph_manifest"][0]["text_sha256"] = "0" * 64
    seal_manifest(bad_doc[0])
    with pytest.raises(ValueError, match="manifest_text_sha256"):
        assemble_source_bundle(bad_doc, rows, rows[0]["id"], rows[1]["id"])


def test_trusted_manifest_is_required_no_guessing_from_numeric_ids():
    docs, rows = corpus(["甲", "乙"])
    del docs[0]["paragraph_manifest"]
    with pytest.raises(ValueError, match="complete_paragraph_manifest_required"):
        assemble_source_bundle(docs, rows, rows[0]["id"], rows[1]["id"])


@pytest.mark.parametrize("mutation", ["text", "bounds", "raw_id", "document_id", "raw_doc_id", "doc_hash", "text_hash"])
def test_every_original_row_is_bound_to_trusted_manifest(mutation):
    docs, rows = corpus(["甲", "乙"])
    if mutation == "text":
        rows[1]["text"] = "伪造"
    elif mutation == "bounds":
        rows[1]["document_start_cp"] += 1
    elif mutation == "raw_id":
        rows[1]["raw_para_id"] = "另一个ID"
    elif mutation == "document_id":
        rows[1]["document_id"] = "other"
    elif mutation == "raw_doc_id":
        rows[1]["raw_doc_id"] = "其他"
    elif mutation == "doc_hash":
        rows[1]["content_sha256"] = "0" * 64
    else:
        rows[1]["text_sha256"] = "0" * 64
    # Validate even an unselected original paragraph; no partial manifest proof.
    with pytest.raises(ValueError, match="source_bundle_paragraph_"):
        assemble_source_bundle(docs, rows, rows[0]["id"], rows[0]["id"])


@pytest.mark.parametrize("mutation", ["reverse", "decreasing_start", "out_of_bounds", "boolean_coordinate", "paragraph_ids"])
def test_original_manifest_order_boundaries_and_redundant_ids(mutation):
    docs, rows = corpus(["甲", "乙"])
    manifest = docs[0]["paragraph_manifest"]
    if mutation == "reverse":
        manifest.reverse()
    elif mutation == "decreasing_start":
        manifest[1]["start_cp"] = manifest[0]["start_cp"] - 1
    elif mutation == "out_of_bounds":
        manifest[1]["end_cp"] = len(docs[0]["content"]) + 1
    elif mutation == "boolean_coordinate":
        manifest[0]["start_cp"] = True
    else:
        docs[0]["paragraph_ids"] = list(reversed(docs[0]["paragraph_ids"]))
    if mutation != "paragraph_ids":
        seal_manifest(docs[0])
    with pytest.raises(ValueError, match="source_bundle_"):
        assemble_source_bundle(docs, rows, rows[0]["id"], rows[-1]["id"])


def test_unknown_cross_document_reversed_and_duplicate_source_inputs():
    docs, rows = corpus(["甲", "乙"])
    docs2, rows2 = corpus(["丙"], doc_id="other-uuid", raw_doc_id="其他")
    with pytest.raises(ValueError, match="unknown_endpoint"):
        assemble_source_bundle(docs, rows, "unknown", rows[0]["id"])
    with pytest.raises(ValueError, match="unknown_anchor"):
        build_source_bundles(docs, rows, ["unknown"])
    with pytest.raises(ValueError, match="cross_document"):
        assemble_source_bundle(docs + docs2, rows + rows2, rows[0]["id"], rows2[0]["id"])
    with pytest.raises(ValueError, match="reverse_range"):
        assemble_source_bundle(docs, rows, rows[-1]["id"], rows[0]["id"])
    with pytest.raises(ValueError, match="duplicate_supplied"):
        assemble_source_bundle(docs, rows + [rows[0]], rows[0]["id"], rows[-1]["id"])
    with pytest.raises(ValueError, match="unknown_supplied"):
        assemble_source_bundle(docs, rows + rows2, rows[0]["id"], rows[-1]["id"])


def test_real_range_cap_and_invalid_window_configuration():
    docs, rows = corpus([str(i) for i in range(6)])
    with pytest.raises(ValueError, match="cap_exceeded"):
        assemble_source_bundle(docs, rows, rows[0]["id"], rows[-1]["id"])
    with pytest.raises(ValueError, match="neighbor_window_exceeds_cap"):
        build_source_bundles(docs, rows, [rows[0]["id"]], before=2, after=3)
    for cap in (0, 6, True, 1.0):
        with pytest.raises(ValueError, match="max_paragraphs"):
            assemble_source_bundle(docs, rows, rows[0]["id"], rows[0]["id"], max_paragraphs=cap)
    with pytest.raises(ValueError, match="invalid_neighbor_window"):
        build_source_bundles(docs, rows, [rows[0]["id"]], before=True)


@pytest.mark.parametrize("field", SOURCE_BUNDLE_CORE_FIELDS)
def test_core_span_fields_cannot_be_replaced_by_annotations(field):
    docs, rows = corpus(["甲", "乙", "丙"])
    bundle = assemble_source_bundle(docs, rows, rows[0]["id"], rows[-1]["id"])
    if field == "source_paragraph_ids":
        bundle[field] = [rows[0]["id"], rows[-1]["id"]]  # cannot skip the middle
    elif field in {"start_cp", "end_cp"}:
        bundle[field] += 1  # no partial original paragraph
    else:
        bundle[field] = "fabricated"
    with pytest.raises(ValueError, match="core_identity_mismatch"):
        validate_source_bundle(bundle, docs, rows)


def test_wrong_context_anchor_and_local_offsets_never_validate_as_citation():
    docs, rows = corpus(["甲", "乙", "丙"])
    bundle = assemble_source_bundle(docs, rows, rows[1]["id"], rows[-1]["id"])
    bundle["paragraph_id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="core_identity_mismatch"):
        validate_source_bundle(bundle, docs, rows)
    bundle = assemble_source_bundle(docs, rows, rows[1]["id"], rows[-1]["id"])
    bundle["start_cp"], bundle["end_cp"] = 0, len(bundle["quote"])
    with pytest.raises(ValueError, match="core_identity_mismatch"):
        validate_source_bundle(bundle, docs, rows)


def overlapping_corpus():
    docs, rows = corpus(["abcdefghij"], prefix="前缀|", suffix="|尾部")
    content = docs[0]["content"]
    manifest, actual = [], []
    # A long first parent contains two distinct-ID identical intervals; the
    # following parent partially overlaps both the previous and long parent.
    for i, (start, end) in enumerate([(3, 13), (5, 7), (5, 7), (6, 11)]):
        pid, raw_pid, text = f"opaque-{i}", f"原ID-{chr(1024+i)}", content[start:end]
        manifest.append({"id": pid, "raw_para_id": raw_pid, "start_cp": start, "end_cp": end, "text_sha256": text_sha(text)})
        actual.append({"id": pid, "document_id": docs[0]["id"], "raw_doc_id": docs[0]["raw_doc_id"],
            "raw_para_id": raw_pid, "text": text, "document_start_cp": start, "document_end_cp": end})
    docs[0]["paragraph_manifest"] = manifest
    seal_manifest(docs[0])
    return docs, actual


def test_original_nested_overlap_uses_full_envelope_without_truncating_long_parent():
    docs, rows = overlapping_corpus()
    citation = assemble_source_bundle(docs, rows, rows[0]["id"], rows[1]["id"])
    assert citation["start_cp"] == 3 and citation["end_cp"] == 13
    assert citation["end_cp"] > rows[1]["document_end_cp"]
    assert citation["quote"] == "abcdefghij"
    assert citation["source_paragraph_ids"] == [rows[0]["id"], rows[1]["id"]]
    assert validate_source_bundle(citation, docs, rows) == citation
    overlap = assemble_source_bundle(docs, rows, rows[1]["id"], rows[3]["id"])
    assert overlap["quote"] == docs[0]["content"][5:11]
    assert overlap["source_paragraph_ids"] == [p["id"] for p in rows[1:4]]


def test_identical_original_intervals_never_mix_ids_and_missing_middle_still_rejected():
    docs, rows = overlapping_corpus()
    first = assemble_source_bundle(docs, rows, rows[1]["id"], rows[1]["id"])
    second = assemble_source_bundle(docs, rows, rows[2]["id"], rows[2]["id"])
    assert first["quote"] == second["quote"]
    assert first["start_cp"] == second["start_cp"] and first["end_cp"] == second["end_cp"]
    assert first["source_paragraph_ids"] != second["source_paragraph_ids"]
    assert first["bundle_id"] != second["bundle_id"]
    with pytest.raises(ValueError, match="complete_paragraph_rows"):
        assemble_source_bundle(docs, [rows[0], rows[1], rows[3]], rows[0]["id"], rows[3]["id"])
    complete = assemble_source_bundle(docs, rows, rows[0]["id"], rows[3]["id"])
    complete["source_paragraph_ids"].remove(rows[2]["id"])
    with pytest.raises(ValueError, match="core_identity_mismatch"):
        validate_source_bundle(complete, docs, rows)
