"""Pure provenance/coverage contracts for the zero-provider offline audit."""
from copy import deepcopy

import pytest

from scripts.audit_cross_topic_candidate_delivery import (
    coverage, groups_for_order, parents, windows,
)
from scripts.audit_cost_aware_candidate_pack import pack


def source(text="同一原文😀。"):
    return {"title": "专题", "questions": [{"question_id": "q1", "question": "哪方？"}],
        "documents": [{"doc_id": "d1", "title": "标题", "content": text,
            "paragraphs": [{"para_id": "p1", "char_start": 0,
                "char_end": len(text), "text": text}]}]}


def test_immutable_unicode_codepoints_are_copied_from_original():
    raw = source()
    before = deepcopy(raw)
    result = parents(raw)
    assert result[0]["text"] == "同一原文😀。"
    assert result[0]["end_cp"] == len("同一原文😀。")
    assert raw == before


@pytest.mark.parametrize("edit", ["text", "range", "identity"])
def test_changed_source_text_range_or_identity_fails_closed(edit):
    raw = source()
    row = raw["documents"][0]["paragraphs"][0]
    if edit == "text":
        row["text"] += "changed"
    elif edit == "range":
        row["char_end"] += 1
    else:
        raw["documents"][0]["paragraphs"].append(deepcopy(row))
    with pytest.raises(ValueError):
        parents(raw)


def test_exact_mirror_group_retains_unranked_alias_but_does_not_create_fact():
    rows = [{"text_sha256": "exact", "text": "完整原段"},
            {"text_sha256": "exact", "text": "完整原段"},
            {"text_sha256": "different", "text": "完整原段。"}]
    result = groups_for_order(rows, [1.0, 0.0, 0.0])
    assert len(result) == 1
    assert result[0]["member_indices"] == [0, 1]
    assert result[0]["source_member_count"] == 2
    assert "fact_id" not in result[0]
    assert "supporting_facts" not in result[0]


def test_zero_rank_scores_are_not_fabricated_as_positive_candidates():
    rows = [{"text_sha256": "a", "text": "a"}]
    assert groups_for_order(rows, [0.0]) == []


@pytest.mark.parametrize("length", [1, 319, 320, 321, 561, 1001])
def test_navigation_windows_cover_all_original_characters_including_tail(length):
    text = "".join(chr(0x3400 + i) for i in range(length))
    pieces = windows(text)
    assert all(piece in text and len(piece) <= 320 for piece in pieces)
    assert pieces[0].startswith(text[0])
    assert pieces[-1].endswith(text[-1])
    positions = [set(range(text.index(piece), text.index(piece) + len(piece))) for piece in pieces]
    assert set.union(*positions) == set(range(length))


def test_source_id_presence_does_not_claim_cross_paragraph_quote_was_read():
    raw = source("前段。后段。")
    raw["documents"][0]["paragraphs"] = [
        {"para_id": "p1", "char_start": 0, "char_end": 3, "text": "前段。"},
        {"para_id": "p2", "char_start": 3, "char_end": 6, "text": "后段。"}]
    rows = parents(raw)
    metrics = coverage([0], rows, [{"doc_id": "d1", "para_id": "p1", "quote": "前段。后段。"}],
        {"d1": raw["documents"][0]})
    assert metrics["source_id_hits"] == 1
    assert metrics["full_parent_reference_quote_deliverable"] == 0
    assert metrics["reference_issues"][0]["found_in_declared_document"] is True


def test_quote_check_never_fuzzily_repairs_source_characters():
    raw = source("前段。"); rows = parents(raw)
    metrics = coverage([0], rows, [{"doc_id": "d1", "para_id": "p1", "quote": "前段."}],
        {"d1": raw["documents"][0]})
    assert metrics["source_id_hits"] == 1
    assert metrics["full_parent_reference_quote_deliverable"] == 0


def test_cost_pack_keeps_long_unprovided_unknown_and_fills_later_whole_rows():
    rows = [{"paragraph_index": 1, "payload_tokens": 20},
        {"paragraph_index": 2, "payload_tokens": 5}, {"paragraph_index": 3, "payload_tokens": 5}]
    result = pack(rows, budget=10)
    assert result["selected_indices"] == [2, 3]
    assert result["content_payload_proxy_tokens"] == 10
    assert result["unprovided"] == [{"paragraph_index": 1, "reason": "unprovided_budget"}]


def test_old_prefix_stops_at_first_whole_row_overflow_without_claiming_irrelevance():
    rows = [{"paragraph_index": 1, "payload_tokens": 20},
        {"paragraph_index": 2, "payload_tokens": 5}]
    result = pack(rows, budget=10, stop_at_first_overflow=True)
    assert result["selected_indices"] == []
    assert [row["reason"] for row in result["unprovided"]] == [
        "unprovided_budget", "unprovided_after_prefix_budget_stop"]


def test_pack_candidate_limit_does_not_publish_remaining_as_negatives():
    rows = [{"paragraph_index": ix, "payload_tokens": 1} for ix in range(70)]
    result = pack(rows, budget=15000, limit=60)
    assert len(result["selected_indices"]) == 60
    assert result["content_payload_proxy_tokens"] == 60
    assert len(result["unprovided"]) == 10
    assert all(row["reason"] == "unprovided_candidate_limit" for row in result["unprovided"])


def test_empty_pack_is_not_a_fabricated_successful_evidence_set():
    assert pack([], budget=15000) == {
        "selected_indices": [], "unprovided": [], "content_payload_proxy_tokens": 0}
