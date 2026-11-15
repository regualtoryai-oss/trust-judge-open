from copy import deepcopy
from hashlib import sha256
from uuid import UUID

import pytest

from app.domain.source_bundles import (
    assemble_source_bundle, paragraph_manifest_sha256, validate_source_bundle,
)
from app.domain.source_passages import (
    SOURCE_GEOMETRY, SOURCE_PASSAGE_CORE_FIELDS, assemble_source_passage,
    build_sentence_manifest, validate_source_passage,
)


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def corpus(texts, *, prefix="文前😀\r\n", gaps=None, doc_id="document-internal", raw_doc_id="raw文"):
    gaps = gaps if gaps is not None else ["\r\n<原始间隔>\x00"] * max(0, len(texts) - 1)
    content, rows, manifest = prefix, [], []
    for index, text in enumerate(texts):
        start = len(content)
        content += text
        pid, raw_pid = f"{doc_id}-opaque-{index}", f"原段-{99-index}-opaque"
        rows.append({"id": pid, "document_id": doc_id, "raw_doc_id": raw_doc_id, "raw_para_id": raw_pid,
                     "text": text, "document_start_cp": start, "document_end_cp": len(content)})
        manifest.append({"id": pid, "raw_para_id": raw_pid, "start_cp": start, "end_cp": len(content),
                         "text_sha256": digest(text)})
        if index < len(gaps):
            content += gaps[index]
    content += "\r\n文后🚀"
    document = {"id": doc_id, "raw_doc_id": raw_doc_id, "content": content, "content_sha256": digest(content),
        "paragraph_manifest": manifest, "paragraph_manifest_sha256": paragraph_manifest_sha256(manifest),
        "paragraph_ids": [p["id"] for p in rows]}
    return [document], rows


def test_manifest_is_exhaustive_byte_exact_and_coordinates_are_original_codepoints():
    text = 'Dr. Lee in the U.S. counted 3.14.  “Really?”\r\n甲😀。\x00乙！\n\r\n尾部仍未证实。'
    docs, parents = corpus([text, "其他原段。"])
    original = deepcopy((docs, parents))
    result = build_sentence_manifest(docs, parents)
    assert result["source_geometry"] == SOURCE_GEOMETRY
    parent = result["paragraphs"][0]
    assert parent["paragraph_id"] == parents[0]["id"]
    assert parent["raw_para_id"] == parents[0]["raw_para_id"]
    sentences = parent["sentences"]
    assert [s["index"] for s in sentences] == list(range(len(sentences)))
    assert "".join(s["text"] for s in sentences).encode() == text.encode()
    offset = 0
    for sentence in sentences:
        assert sentence["local_start_cp"] == offset
        assert sentence["local_end_cp"] == offset + len(sentence["text"])
        assert sentence["start_cp"] == parents[0]["document_start_cp"] + offset
        assert docs[0]["content"][sentence["start_cp"]:sentence["end_cp"]] == sentence["text"]
        assert sentence["text_sha256"] == digest(sentence["text"])
        offset = sentence["local_end_cp"]
    assert offset == len(text)
    assert (docs, parents) == original


def test_abbreviations_decimals_quotes_crlf_and_nul_are_not_corrupted_or_mid_cut():
    text = 'Dr. Lee visited the U.S. at 3.14 p.m. He left.  “Yes!”\r\n字面\\u0000与真\x00。\x00尾句。'
    docs, rows = corpus([text])
    sentences = build_sentence_manifest(docs, rows)["paragraphs"][0]["sentences"]
    assert [s["text"] for s in sentences] == [
        'Dr. Lee visited the U.S. at 3.14 p.m. He left.  ', '“Yes!”\r\n',
        '字面\\u0000与真\x00。\x00', '尾句。']
    second = assemble_source_passage(docs, rows, rows[0]["id"], 1, 2)
    assert second["quote"] == '“Yes!”\r\n'
    assert second["local_start_cp"] == len(sentences[0]["text"])
    assert second["start_cp"] == rows[0]["document_start_cp"] + second["local_start_cp"]
    assert validate_source_passage(second, docs, rows) == second


def test_long_sentence_and_tail_qualifications_are_all_available_without_prefix_cutting():
    long_sentence = "长句😀" * 7000 + "暂未获得官方确认。"
    text = "早期报道。" + long_sentence + "不是最终结论！" + "尾部新增情况仍未证实。"
    docs, rows = corpus([text])
    sentences = build_sentence_manifest(docs, rows)["paragraphs"][0]["sentences"]
    assert [s["text"] for s in sentences] == ["早期报道。", long_sentence, "不是最终结论！", "尾部新增情况仍未证实。"]
    selected = assemble_source_passage(docs, rows, rows[0]["id"], 1, 4)
    assert selected["quote"] == long_sentence + "不是最终结论！尾部新增情况仍未证实。"
    assert selected["end_cp"] == rows[0]["document_end_cp"]
    assert len(sentences[1]["text"]) > 20000
    # Every single group and every adjacent group is a real exact source slice.
    for begin in range(len(sentences)):
        for end in range(begin + 1, len(sentences) + 1):
            source = assemble_source_passage(docs, rows, rows[0]["id"], begin, end)
            assert source["quote"] == "".join(s["text"] for s in sentences[begin:end])
            assert source["quote"] == docs[0]["content"][source["start_cp"]:source["end_cp"]]


def test_default_whole_parent_retains_original_identity_but_not_old_bundle_protocol():
    docs, rows = corpus(["短原段😀。\r\n", "下一个原段。"])
    source = assemble_source_passage(docs, rows, rows[0]["id"])
    assert set(source) == set(SOURCE_PASSAGE_CORE_FIELDS)
    assert source["quote"] == rows[0]["text"]
    assert source["source_paragraph_ids"] == [rows[0]["id"]]
    assert source["raw_para_id"] == rows[0]["raw_para_id"]
    assert source["parent_start_cp"] == source["start_cp"] == rows[0]["document_start_cp"]
    assert source["parent_end_cp"] == source["end_cp"] == rows[0]["document_end_cp"]
    assert source["local_start_cp"] == 0 and source["local_end_cp"] == len(rows[0]["text"])
    assert source["passage_id"] == source["bundle_id"]
    assert UUID(source["passage_id"]).version == 5
    old = assemble_source_bundle(docs, rows, rows[0]["id"], rows[0]["id"])
    assert old["bundle_id"] != source["bundle_id"]
    with pytest.raises(ValueError):
        validate_source_bundle(source, docs, rows)
    with pytest.raises(ValueError, match="geometry_core_required"):
        validate_source_passage(old, docs, rows)


def test_empty_and_whitespace_original_parents_are_retained_as_source_only():
    docs, rows = corpus(["", " \t\r\n\n\r  ", "尾段。"])
    manifest = build_sentence_manifest(docs, rows)
    assert manifest["paragraphs"][0]["sentences"] == []
    empty = assemble_source_passage(docs, rows, rows[0]["id"])
    assert empty["quote"] == "" and empty["start_cp"] == empty["end_cp"]
    assert empty["start_sentence_index"] == empty["end_sentence_index"] == 0
    assert validate_source_passage(empty, docs, rows) == empty
    assert assemble_source_passage(docs, rows, rows[0]["id"], 0, 0) == empty
    white = assemble_source_passage(docs, rows, rows[1]["id"])
    assert white["quote"].encode() == rows[1]["text"].encode()


def test_repeated_identical_text_keeps_distinct_original_locations_and_source_ids():
    docs, rows = corpus(["同句。后句。", "同句。后句。"])
    left = assemble_source_passage(docs, rows, rows[0]["id"], 0, 1)
    right = assemble_source_passage(docs, rows, rows[1]["id"], 0, 1)
    assert left["quote"] == right["quote"]
    assert left["start_cp"] != right["start_cp"]
    assert left["paragraph_id"] != right["paragraph_id"]
    assert left["passage_id"] != right["passage_id"]
    assert left["sentence_manifest_sha256"] != right["sentence_manifest_sha256"]


@pytest.mark.parametrize("begin,end", [(None, 1), (0, None), (True, 1), (0, True),
    (0.0, 1), (0, 1.0), ("0", 1), (-1, 1), (0, 99), (1, 1), (2, 1), (0, 0)])
def test_invalid_or_partial_ranges_fail_without_clamping_or_repairs(begin, end):
    docs, rows = corpus(["首句。末句。"])
    with pytest.raises(ValueError, match="source_passage_"):
        assemble_source_passage(docs, rows, rows[0]["id"], begin, end)


@pytest.mark.parametrize("field,value", [
    ("quote", "首句"), ("start_cp", 0), ("end_cp", 999), ("local_start_cp", 1),
    ("local_end_cp", True), ("parent_start_cp", 0), ("parent_end_cp", 999),
    ("parent_text_sha256", "0" * 64), ("sentence_manifest_sha256", "0" * 64),
    ("start_sentence_index", True), ("end_sentence_index", 2),
    ("source_geometry", "document-source-bundle-v1"), ("span_scope", "paragraph"),
    ("paragraph_id", "unknown"), ("raw_doc_id", "changed"), ("raw_para_id", "changed"),
    ("source_paragraph_ids", ["other"]), ("document_id", "other"),
    ("document_content_sha256", "0" * 64), ("paragraph_manifest_sha256", "0" * 64),
    ("passage_id", "changed"), ("bundle_id", "changed"),
])
def test_each_geometry_core_cannot_be_replaced_by_model_coordinates_or_free_quote(field, value):
    docs, rows = corpus(["首句。末句。"])
    source = assemble_source_passage(docs, rows, rows[0]["id"], 0, 1)
    source[field] = value
    with pytest.raises(ValueError):
        validate_source_passage(source, docs, rows)


def test_no_cross_parent_or_neighbor_gap_can_be_smuggled_into_single_parent_passage():
    docs, rows = corpus(["甲。", "乙。"])
    source = assemble_source_passage(docs, rows, rows[0]["id"])
    source.update(end_cp=rows[1]["document_end_cp"], source_paragraph_ids=[r["id"] for r in rows],
                  quote=docs[0]["content"][source["start_cp"]:rows[1]["document_end_cp"]])
    with pytest.raises(ValueError, match="core_identity_mismatch"):
        validate_source_passage(source, docs, rows)


def test_all_sources_are_required_and_original_manifest_is_revalidated():
    docs, rows = corpus(["甲。", "乙。"])
    with pytest.raises(ValueError, match="complete_paragraph_rows_required"):
        assemble_source_passage(docs, rows[:1], rows[0]["id"])
    altered = deepcopy(rows)
    altered[0]["text"] += "新增"
    with pytest.raises(ValueError, match="paragraph_manifest_mismatch"):
        build_sentence_manifest(docs, altered)
    with pytest.raises(ValueError, match="unknown_parent"):
        assemble_source_passage(docs, rows, rows[0]["raw_para_id"])


def test_annotations_are_detached_unverified_and_collection_order_does_not_change_passage():
    docs, rows = corpus(["甲。乙。", "丙。"])
    other_docs, other_rows = corpus(["其他文。"], doc_id="other", raw_doc_id="其他raw")
    source = assemble_source_passage(docs + other_docs, rows + other_rows, rows[0]["id"], 1, 2)
    docs[0]["title"] = "外部metadata"
    repeat = assemble_source_passage(other_docs + docs, list(reversed(rows + other_rows)), rows[0]["id"], 1, 2)
    assert repeat == source
    annotated = {**source, "fact_des": "任意未验证文字", "read": False,
                 "argument": {"时间": "未检验时间"}, "entities": ["未检验实体"]}
    copied = validate_source_passage(annotated, docs + other_docs, rows + other_rows)
    assert copied == annotated and copied is not annotated
    copied["entities"].append("改动")
    assert annotated["entities"] == ["未检验实体"]


def test_changes_outside_parent_still_change_document_binding_and_reject_old_passage():
    docs, rows = corpus(["源句。"])
    source = assemble_source_passage(docs, rows, rows[0]["id"])
    changed = deepcopy(docs)
    changed[0]["content"] += "外部追加"
    changed[0]["content_sha256"] = digest(changed[0]["content"])
    with pytest.raises(ValueError, match="core_identity_mismatch"):
        validate_source_passage(source, changed, rows)


def test_overlapping_original_parents_remain_separate_and_no_text_join_is_invented():
    docs, rows = corpus(["甲。乙。", "unused"])
    docs[0]["paragraph_manifest"][1].update(start_cp=rows[0]["document_start_cp"] + 2,
        end_cp=rows[0]["document_end_cp"], text_sha256=digest("乙。"))
    docs[0]["paragraph_manifest_sha256"] = paragraph_manifest_sha256(docs[0]["paragraph_manifest"])
    rows[1].update(document_start_cp=rows[0]["document_start_cp"] + 2,
                   document_end_cp=rows[0]["document_end_cp"], text="乙。")
    outer = assemble_source_passage(docs, rows, rows[0]["id"], 1, 2)
    inner = assemble_source_passage(docs, rows, rows[1]["id"])
    assert outer["quote"] == inner["quote"] == "乙。"
    assert outer["start_cp"] == inner["start_cp"] and outer["end_cp"] == inner["end_cp"]
    assert outer["paragraph_id"] != inner["paragraph_id"]
    assert outer["passage_id"] != inner["passage_id"]
