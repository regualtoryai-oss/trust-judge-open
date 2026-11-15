from copy import deepcopy
from hashlib import sha256
import json

import pytest

from app.domain.lossless_text_book import (
    FORMAT, native_text_book, pack_texts, restore_texts, validate_text_book,
)


def originals():
    return [
        {"id": "source-a#p1", "doc_id": "a", "quote": 'Dr. Lee in the U.S. counted 3.14.  “Really?”\r\n原文😀。\x00后续！\n',
         "fact_des": "自由文字不压缩。", "argument": {"内容": "原始内容。", "时间": ""},
         "metadata": {"quote": "嵌套quote不改", "order": [None, False, 1, 2.5]}},
        {"id": "source-b#p8", "doc_id": "b", "quote": "原文😀。\x00后续！\n"},
        {"id": "empty", "quote": ""},
        {"id": "spaces", "quote": " \t\r\n\n\r  "},
    ]


def test_roundtrip_preserves_every_text_byte_metadata_row_and_occurrence():
    rows = originals()
    frozen = deepcopy(rows)
    book = pack_texts(rows)
    assert rows == frozen
    restored = restore_texts(book)
    assert restored == rows
    assert [r["id"] for r in restored] == [r["id"] for r in rows]
    for expected, actual in zip(rows, restored, strict=True):
        assert actual["quote"].encode("utf-8") == expected["quote"].encode("utf-8")
        assert {k: v for k, v in actual.items() if k != "quote"} == {
            k: v for k, v in expected.items() if k != "quote"}
    assert book["records"][2]["quote"] == {"text_ref": []}
    assert book["audit"]["records"][2]["sha256"] == sha256(b"").hexdigest()
    assert validate_text_book(rows, book)["source_texts_equal"] is True
    view = native_text_book(book, trusted_original=rows)
    assert view["format"] == FORMAT and "audit" not in view
    assert view["records"][0]["argument"] == rows[0]["argument"]
    view["records"][0]["metadata"]["order"].append("changed")
    assert book["records"][0]["metadata"]["order"] == rows[0]["metadata"]["order"]


def test_exact_block_dedup_preserves_duplicate_citations_and_all_occurrences():
    rows = [{"id": "a", "quote": "同一句😀。同一句😀。\n"},
            {"id": "b", "quote": "同一句😀。"},
            {"id": "c", "quote": "同一句😀。 "}]
    book = pack_texts(rows)
    literals = [b["text"] for b in book["text_blocks"]]
    assert literals == ["同一句😀。", "同一句😀。\n", "同一句😀。 "]
    assert book["records"][0]["quote"] == {"text_ref": ["B1", "B2"]}
    assert book["records"][1]["quote"] == {"text_ref": ["B1"]}
    assert len(book["records"]) == 3
    assert restore_texts(book) == rows
    spans = book["audit"]["records"][0]["refs"]
    assert spans[0] == {"block_id": "B1", "start_cp": 0, "end_cp": 5,
                        "start_utf8": 0, "end_utf8": 16}
    assert spans[1]["start_cp"] == 5 and spans[1]["start_utf8"] == 16
    assert spans[1]["end_utf8"] == len(rows[0]["quote"].encode())


def test_no_fuzzy_unicode_or_whitespace_dedup_and_no_fixed_long_text_cut():
    texts = ["é。", "e\u0301。", "A\x00B", "a" * 20000, "句子。", "句子。 "]
    rows = [{"id": str(i), "quote": text} for i, text in enumerate(texts)]
    book = pack_texts(rows)
    assert [r["text"] for r in book["text_blocks"]] == texts
    assert restore_texts(book) == rows


def test_boundaries_keep_english_abbreviations_decimals_closers_and_crlf():
    text = 'Dr. Lee visited the U.S. at 3.14 p.m. Then he left.  “Yes!”\r\nNext\n'
    book = pack_texts([{"id": "x", "quote": text}])
    assert [b["text"] for b in book["text_blocks"]] == [
        'Dr. Lee visited the U.S. at 3.14 p.m. Then he left.  ', '“Yes!”\r\n', 'Next\n']
    assert "".join(b["text"] for b in book["text_blocks"]) == text


def test_empty_input_and_custom_fields_remain_reversible():
    assert restore_texts(pack_texts([])) == []
    rows = [{"original_id": "d#p8", "body": "原文\r\n末句。", "quote": "自由metadata原样"}]
    book = pack_texts(rows, text_key="body", id_key="original_id")
    assert restore_texts(book) == rows
    assert book["records"][0]["quote"] == rows[0]["quote"]


def test_line_endings_nul_literals_and_partial_sentence_sources_never_blend():
    texts = ["甲\r\n", "甲\n", "甲\r\n\r\n", "甲", "真\x00值", r"真\u0000值",
             "同一完整句的前", "同一完整句的前半句。"]
    rows = [{"id": str(i), "quote": text, "source_identity": {"id": str(i)}}
            for i, text in enumerate(texts)]
    book = pack_texts(rows)
    assert restore_texts(book) == rows
    refs = [row["quote"]["text_ref"] for row in book["records"]]
    assert len({tuple(ref) for ref in refs}) == len(rows)
    assert [row["source_identity"] for row in book["records"]] == [r["source_identity"] for r in rows]
    assert "同一完整句的前" in [row["text"] for row in book["text_blocks"]]


def test_changing_only_source_identity_fails_trusted_validation():
    rows = [{"id": "official#p1", "quote": "同一原文。"}, {"id": "mirror#p8", "quote": "同一原文。"}]
    changed = deepcopy(rows)
    changed[1]["id"] = "invented#p8"
    book = pack_texts(changed)
    assert len(book["text_blocks"]) == 1 and len(book["records"]) == 2
    with pytest.raises(ValueError, match="trusted_original"):
        validate_text_book(rows, book)


@pytest.mark.parametrize("change", [
    lambda b: b["records"][0]["quote"]["text_ref"].append("B999"),
    lambda b: b["records"][0]["quote"]["text_ref"].pop(),
    lambda b: b["records"][0]["quote"]["text_ref"].append("B1"),
    lambda b: b["text_blocks"].append(deepcopy(b["text_blocks"][0])),
    lambda b: b["text_blocks"].append({"id": "B" + str(len(b["text_blocks"]) + 1), "text": "未引用"}),
    lambda b: b["text_blocks"].reverse(),
    lambda b: b["records"].reverse(),
    lambda b: b["records"][0]["quote"]["text_ref"].reverse(),
    lambda b: b["text_blocks"][0].update(text="篡改。"),
    lambda b: b["records"][0].update(fact_des="被改的自由metadata"),
    lambda b: b["audit"]["records"][0]["refs"][0].update(start_cp=1),
    lambda b: b["audit"]["records"][0]["refs"][0].update(end_utf8=999),
    lambda b: b["audit"]["records"][0]["refs"][0].update(start_cp=False),
    lambda b: b["audit"].update(original_records_sha256="0" * 64),
    lambda b: b["audit"].update(native_view_sha256="0" * 64),
    lambda b: b["audit"].update(semantic_verified=True),
    lambda b: b.update(unrecognized="extra"),
])
def test_tampering_missing_extra_unknown_reordered_and_modified_data_fail_closed(change):
    book = pack_texts(originals())
    change(book)
    with pytest.raises(ValueError, match="lossless_text_book_"):
        restore_texts(book)


def test_self_consistent_recomputed_hashes_are_not_a_trusted_source():
    rows = originals()
    original_book = pack_texts(rows)
    changed = deepcopy(rows)
    changed[0]["quote"] = "完全不同但自洽的内容。"
    forged = pack_texts(changed)
    assert restore_texts(forged) == changed  # Internal consistency is not identity.
    with pytest.raises(ValueError, match="trusted_original"):
        validate_text_book(rows, forged)
    with pytest.raises(ValueError, match="trusted_original"):
        restore_texts(forged, expected_original_sha256=original_book["audit"]["original_records_sha256"])
    with pytest.raises(ValueError, match="trusted_original"):
        native_text_book(forged, trusted_original=rows)


@pytest.mark.parametrize("rows", [
    [{"id": "a", "quote": "x"}, {"id": "a", "quote": "x"}],
    [{"id": True, "quote": "x"}], [{"id": "", "quote": "x"}],
    [{"id": "a", "quote": {"text_ref": ["B1"]}}],
    [{"id": "a", "quote": "x", "value": float("nan")}],
    [{"id": "a", "quote": "x", "value": (1, 2)}],
    [{"id": "a", "quote": "x", "value": {1: "silently stringified"}}],
    [{"id": "a", "quote": "\ud800"}],
])
def test_invalid_original_input_is_not_silently_coerced(rows):
    with pytest.raises(ValueError, match="lossless_text_book_"):
        pack_texts(rows)


def test_native_dictionary_has_enough_literals_for_full_reconstruction_without_local_audit():
    rows = originals()
    view = native_text_book(pack_texts(rows), trusted_original=rows)
    blocks = {row["id"]: row["text"] for row in view["text_blocks"]}
    for expected, actual in zip(rows, view["records"], strict=True):
        assert "".join(blocks[ref] for ref in actual["quote"]["text_ref"]) == expected["quote"]
    assert json.loads(json.dumps(view, ensure_ascii=False)) == view
