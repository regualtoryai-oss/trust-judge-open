"""Synthetic text only; never a proxy for competition P/R."""
from copy import deepcopy
import pytest

from app.domain.query_context import WINDOW_CHARS, build_query_context, verify_context


def parent(text, **extra):
    return {"id": "t/d/p", "raw_doc_id": "d", "raw_para_id": "p", "title": "原标题", "text": text, **extra}


def test_tail_query_cue_moves_context_preserves_literal_unicode_offsets_and_parent():
    text = "背景🙂\r\n" * 40 + "开始。申请已经送达，但尚未生效。后续仍待核验。" + "末尾。" * 30
    row = parent(text, document_start_cp=71); before = deepcopy(row)
    plan = build_query_context("申请是否已经生效？", [row]); context = plan["contexts"][0]
    assert context["start_cp"] > 80 and context["excerpt_text"] == text[context["start_cp"]:context["end_cp"]]
    assert "尚未生效" in context["excerpt_text"] and len(context["excerpt_text"]) == WINDOW_CHARS
    assert context["omitted_before"] and context["omitted_after"] and context["omission_means_unknown"]
    assert context["document_start_cp"] == 71 + context["start_cp"] and verify_context(row, context)
    assert row == before and not context["field_or_support_conflict_verified"]


def test_unmatched_parent_keeps_exact_prefix_and_unknown_instead_of_dropping():
    text = "完全不同的原始内容。" * 30
    plan = build_query_context("机场灯光状态？", [parent(text)])
    context = plan["contexts"][0]
    assert context["start_cp"] == 0 and context["excerpt_text"] == text[:80]
    assert context["selection_reason"] == "no_literal_body_hit_prefix_fallback"
    assert context["lexical_feature_score"] == 0 and context["omitted_after"]
    assert plan["parent_count"] == 1 and not plan["evidence_or_role_filter_applied"]


def test_short_whitespace_and_repeated_substrings_are_not_normalized_or_discarded():
    rows = [parent(" \r\n🙂 保留原文 "), parent(" " * 99, id="p2", raw_para_id="p2")]
    out = build_query_context("无命中检索？", rows)["contexts"]
    assert out[0]["excerpt_text"] == rows[0]["text"] and not out[0]["omitted_before"] and not out[0]["omitted_after"]
    assert out[1]["excerpt_text"] == " " * 80 and out[1]["omitted_after"]
    repeated = parent("前缀。" * 40 + "目标记录已保留。" * 2 + "后文。" * 40)
    view = build_query_context("目标记录？", [repeated])["contexts"][0]
    assert repeated["text"][view["start_cp"]:view["end_cp"]] == view["excerpt_text"]


def test_provided_query_can_locate_context_but_never_claims_semantic_verification():
    row = parent("无关前言。" * 30 + "现场日志显示闸门尚未闭合。" + "无关后文。" * 30)
    queries = [{"group": "official", "terms": ["闸门 现场日志"]}]
    result = build_query_context("处置状态仍待判断？", [row], query_groups=queries)
    assert "现场日志" in result["contexts"][0]["excerpt_text"]
    assert not result["provided_expansions_semantically_verified"] and result["model_calls"] == 0


def test_literal_latin_and_numeric_features_do_not_hit_inside_other_tokens():
    row = parent("chair 2027 " * 30)
    context = build_query_context("AI 2026？", [row])["contexts"][0]
    assert context["lexical_feature_score"] == 0 and context["start_cp"] == 0


@pytest.mark.parametrize("field,value", [("start_cp", 1), ("end_cp", 79), ("excerpt_text", "编造"), ("source_text_sha256", "0"*64), ("raw_para_id", "other"), ("omitted_after", False), ("document_end_cp", 999), ("omission_means_unknown", False), ("field_or_support_conflict_verified", True)])
def test_changed_spans_source_ids_or_omission_flags_rejected(field, value):
    row = parent("真实原文。" * 50, document_start_cp=10)
    view = build_query_context("不命中问题？", [row])["contexts"][0];view[field] = value
    with pytest.raises(ValueError): verify_context(row, view)


def test_reference_fields_and_duplicate_parent_ids_rejected_without_mutation():
    with pytest.raises(ValueError):build_query_context({"question":"原题？", "conclusion":"禁止"}, [parent("原文")])
    with pytest.raises(ValueError):build_query_context("原题？", [parent("原文", gold="禁止")])
    with pytest.raises(ValueError):build_query_context("原题？", [parent("原文"), parent("另一原文")])


def test_unique_feature_scoring_and_earliest_sentence_tie_are_deterministic():
    text = "目标记录。" * 60
    row = parent(text);one = build_query_context("目标记录？", [row]);two = build_query_context("目标记录？", [row])
    assert one == two and one["contexts"][0]["start_cp"] == 0
    assert one["contexts"][0]["lexical_feature_score"] < 100  # no term-frequency bonus


def test_selected_score_is_global_maximum_for_contiguous_windows():
    from app.domain.query_context import _literal, _query_weights
    text = "开场。" * 30 + "日志已有记录，但批准尚未生效。" + "普通背景。" * 17 + "批准状态和日志记录再次出现。"
    result = build_query_context("日志是否记录批准生效？", [parent(text)])
    weights = _query_weights(result["query_plan"])
    occurrences = [(w, [(m.start(1),m.end(1))for m in _literal(term).finditer(text)])for term,w in weights]
    brute = max(sum(w for w,hits in occurrences if any(s<=a and b<=s+80 for a,b in hits))for s in range(len(text)-79))
    assert result["contexts"][0]["lexical_feature_score"] == brute


@pytest.mark.parametrize("identifier,filler,query", [("xAI",76,"AI"),("12026",74,"2026")])
def test_crop_never_creates_latin_or_numeric_token_boundary_match(identifier,filler,query):
    text = "甲"*100 + identifier + "乙"*filler + "记录" + "丙"*100
    context = build_query_context(query + " 记录？", [parent(text)])["contexts"][0]
    assert context["start_cp"] == 101 and context["excerpt_text"].startswith(query)
    assert query.lower() not in context["matched_query_features"]
    assert context["lexical_feature_score"] == 8


def test_document_geometry_cannot_be_invented_without_original_document_offset():
    row = parent("原文。"*40);view = build_query_context("原文？",[row])["contexts"][0]
    view["document_start_cp"] = 50
    with pytest.raises(ValueError):verify_context(row,view)
