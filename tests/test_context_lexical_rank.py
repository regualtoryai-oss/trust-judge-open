"""Synthetic local navigation only; no gold, provider, ONNX or runtime files."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from app.domain.context_candidates import HYBRID_POLICY, POLICY, build_context_plan, validate_context_plan
from app.domain.context_lexical_rank import literal_queries, rank_lexical_windows
from app.domain.local_parent_rank import fingerprint
from test_lc_context_case import raw_case, windows


def fixture(question="申请已获批准还是仍待核验？", texts=None):
    raw = raw_case(texts or ["申请已获受理，批准结果尚未确认。", "仍待核验，不能据此确认。", "无关的天气记录。"])
    raw["question"]["question"] = question
    scores = windows(raw)
    return raw, scores


def build(raw, scores, **kwargs):
    return build_context_plan(raw, scores, full_window_scores_digest=fingerprint(scores), **kwargs)


@pytest.mark.parametrize("question", [
    "截至某时点，统计是否包括失联人数？口径指什么？",
    "原话中的 discouraged 指什么？是教学环境导致的吗？",
    "“尚未确认”的原意是什么；该结论是否成立？",
    "方案由甲方或者乙方执行，抑或仍未确定？",
    "甲方观点是什么",
])
def test_facets_are_exact_literal_spans_and_every_query_keeps_all_conditions(question):
    queries = literal_queries(question)
    assert queries[0]["text"] == question and queries[0]["focus"] is None
    assert all(q["whole_question"] == question and q["text"].startswith(question) for q in queries)
    for q in queries[1:]:
        focus = q["focus"]
        assert question[focus["start_cp"]:focus["end_cp"]] == focus["text"]
        assert q["text"] == question + "\n重点核验：" + focus["text"]
    if "discouraged" in question:
        assert any(q["focus"] and q["focus"]["text"] == "discouraged" for q in queries)
    if "“" in question:
        assert any(q["focus"] and "quoted_span" in q["focus"]["kinds"] for q in queries)


def test_generic_derivation_has_no_source_metadata_or_answer_insertions():
    question = "“new-term”原意指什么？发布者甲对此是否确认？"
    queries = literal_queries(question)
    assert len({q["focus"]["text"] for q in queries if q["focus"]}) == len(queries) - 1
    assert all(q["focus"] is None or q["focus"]["text"] in question for q in queries)
    # Names, dates or source IDs play no role: only original spans are emitted.
    altered = question.replace("发布者甲", "任意主体乙")
    assert [q["focus"]["kinds"] for q in queries[1:]] == [q["focus"]["kinds"] for q in literal_queries(altered)[1:]]


def test_hybrid_has_exact_equal_family_weight_despite_different_facet_counts():
    raw, scores = fixture("原话term意指什么？批准结果是否确定？")
    assert len(scores["queries"]) == 1  # No binary delimiter in the old BGE query policy.
    plan = build(raw, scores, hybrid_retrieval=True)
    assert plan["policy"] == HYBRID_POLICY
    lexical = plan["audit"]["hybrid_retrieval"]
    assert len(lexical["queries"]) > 1
    by_window = {(r["paragraph_id"], r["start_cp"], r["end_cp"]): r for r in lexical["windows"]}
    for candidate in plan["candidates"]:
        for origin in candidate["window_origins"]:
            key = (origin["paragraph_id"], origin["start_cp"], origin["end_cp"])
            bge = sum(r["rrf_contribution"] for r in origin["facet_scores"]) / len(scores["queries"])
            lex = by_window[key]
            assert lex["mean_rrf_score"] == pytest.approx(sum(r["rrf_contribution"] for r in lex["facet_scores"]) / len(lexical["queries"]))
            assert origin["rrf_score"] == pytest.approx((bge + lex["mean_rrf_score"]) / 2)
            assert origin["family_contributions"]["bge_weighted"] == pytest.approx(bge / 2)
            assert origin["family_contributions"]["lexical_weighted"] == pytest.approx(lex["mean_rrf_score"] / 2)
    assert plan["provider_calls"] == plan["new_local_inference_calls"] == 0


def test_positive_rescaling_bge_logits_per_facet_does_not_change_hybrid_ranking():
    raw, scores = fixture()
    before = build(raw, scores, hybrid_retrieval=True)
    changed = deepcopy(scores)
    for row in changed["records"]:
        factor = {"F0": 1000, "F1": 0.01, "F2": 7}[row["query_facet"]]
        row["maxlogit"] = factor * row["maxlogit"] + factor
    after = build(raw, changed, hybrid_retrieval=True)
    assert [(r["context_id"], r["rrf_score"]) for r in before["candidates"]] == [(r["context_id"], r["rrf_score"]) for r in after["candidates"]]


def test_false_default_is_identical_and_true_requires_explicit_external_mode():
    raw, scores = fixture()
    default = build(raw, scores)
    explicit = build(raw, scores, hybrid_retrieval=False)
    assert json.dumps(default, ensure_ascii=False, sort_keys=True) == json.dumps(explicit, ensure_ascii=False, sort_keys=True)
    assert default["policy"] == POLICY and "hybrid_retrieval" not in default["audit"]
    hybrid = build(raw, scores, hybrid_retrieval=True)
    with pytest.raises(ValueError, match="plan_changed"):
        validate_context_plan(raw, scores, hybrid, full_window_scores_digest=fingerprint(scores))
    assert validate_context_plan(raw, scores, hybrid, full_window_scores_digest=fingerprint(scores), hybrid_retrieval=True) == hybrid
    with pytest.raises(ValueError, match="plan_changed"):
        validate_context_plan(raw, scores, default, full_window_scores_digest=fingerprint(scores), hybrid_retrieval=True)
    with pytest.raises(ValueError, match="explicit_hybrid"):
        build(raw, scores, hybrid_retrieval=1)


@pytest.mark.parametrize("change", ["query", "rank", "contribution", "policy", "source"])
def test_replay_recomputes_lexical_derivation_ranks_contributions_and_source(change):
    raw, scores = fixture()
    plan = build(raw, scores, hybrid_retrieval=True)
    changed = deepcopy(plan)
    if change == "query": changed["audit"]["hybrid_retrieval"]["queries"][0]["text"] += "注入答案"
    elif change == "rank": changed["audit"]["hybrid_retrieval"]["windows"][0]["facet_scores"][0]["rank"] = 999
    elif change == "contribution": changed["candidates"][0]["window_origins"][0]["family_contributions"]["lexical_weighted"] = 1
    elif change == "policy": changed["policy"]["hybrid_retrieval"]["family_weights"]["lexical"] = 1
    else: changed["candidates"][0]["source_passage"]["quote"] += "不在原文"
    with pytest.raises(ValueError, match="plan_changed"):
        validate_context_plan(raw, scores, changed, full_window_scores_digest=fingerprint(scores), hybrid_retrieval=True)


def test_invalid_or_incomplete_bge_input_is_rejected_before_lexical_processing(monkeypatch):
    import app.domain.context_lexical_rank as lexical
    raw, scores = fixture()
    monkeypatch.setattr(lexical, "rank_lexical_windows", lambda *a: pytest.fail("unverified BGE input reached lexical ranker"))
    bad = deepcopy(scores)
    bad["records"].pop()
    with pytest.raises(ValueError, match="complete_window_facet_coverage"):
        build(raw, bad, hybrid_retrieval=True)


def test_mirrored_contexts_keep_both_original_parent_ids_and_source_bytes():
    text = "相同原文仍待进一步核验。"
    raw, scores = fixture(texts=[text, text, "另一说法。"])
    frozen = deepcopy(raw)
    plan = build(raw, scores, hybrid_retrieval=True)
    mirrored = [r["source_passage"] for r in plan["candidates"] if r["source_passage"]["quote"] == text]
    assert len(mirrored) == 2 and len({r["raw_para_id"] for r in mirrored}) == 2
    assert len({r["passage_id"] for r in mirrored}) == 2
    assert raw == frozen


def test_bm25_title_and_complete_body_are_bound_and_zero_matches_have_no_rank():
    raw = {"question": {"question": "alpha是否批准？"}, "documents": [
        {"id": "d1", "title": "alpha"}, {"id": "d2", "title": "unrelated"}],
        "paragraphs": [{"id": "p1", "document_id": "d1", "text": "完整body😀\x00尾巴。"},
                       {"id": "p2", "document_id": "d2", "text": "weather only"}]}
    windows_ = [{"window_id": p["id"], "paragraph_id": p["id"], "start_cp": 0, "end_cp": len(p["text"]),
                 "window_text_sha256": sha256(p["text"].encode()).hexdigest()} for p in raw["paragraphs"]]
    frozen = deepcopy(raw)
    result = rank_lexical_windows(raw, windows_)
    assert all(r["rank"] is not None for r in result["windows"][0]["facet_scores"])
    assert all(r["rank"] is None and r["rrf_contribution"] == 0 for r in result["windows"][1]["facet_scores"])
    assert result["windows"][0]["body_sha256"] == sha256(raw["paragraphs"][0]["text"].encode()).hexdigest()
    assert result["windows"][0]["title_sha256"] == sha256(b"alpha").hexdigest()
    assert raw == frozen
    invalid = deepcopy(windows_); invalid[0]["end_cp"] -= 1
    with pytest.raises(ValueError, match="body_changed"):
        rank_lexical_windows(raw, invalid)


def test_replay_does_not_silently_switch_bm25_library_versions(monkeypatch):
    import app.domain.context_lexical_rank as lexical
    raw, scores = fixture()
    monkeypatch.setattr(lexical, "package_version", lambda _name: "different-version")
    with pytest.raises(ValueError, match="bm25_library_version_changed"):
        build(raw, scores, hybrid_retrieval=True)
