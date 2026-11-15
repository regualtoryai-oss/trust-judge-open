"""Offline synthetic range scoring: no provider, reference files or old answers."""
from copy import deepcopy
import json

import pytest

from scripts import score_real_quality_episode as scorer
from scripts import score_bundle_workbench_real as bundle_scorer
from test_score_real_quality_episode import Q, fact, normalized, reference


def context(doc, parent, quote):
    start = parent.index(quote)
    return {"raw_doc_id": doc, "raw_para_id": "p1", "local_start_cp": start,
            "local_end_cp": start + len(quote), "quote": quote}


def score(rows, refs, contexts, ranges, *, candidates=None, read_keys=None):
    candidates = set(contexts) if candidates is None else candidates
    return scorer.score_normalized(Q, normalized(rows, [r["fact_id"] for r in rows]), refs,
        reference([r["fact_id"] for r in refs], []), contexts, candidates, read_keys or set(),
        delivered_context_ranges=ranges)


def test_verified_partial_literal_scores_same_ids_without_fabricating_full_reads():
    parent, quote = "未投递前文😀。\r\n关键原文尚待核验。未投递尾文。", "关键原文尚待核验。"
    contexts = {("a", "p1"): parent}
    rows, refs = [fact("f", "a", text=quote)], [fact("g", "a", text=parent)]
    result = score(rows, refs, contexts, [context("a", parent, quote)])
    full = scorer.score_normalized(Q, normalized(rows, ["f"]), refs, reference(["g"], []),
        contexts, set(contexts), set(contexts))
    assert result["question_selection"] == full["question_selection"]
    assert result["supporting"] == full["supporting"]
    assert result["question_selection"]["A_id_only_proxy"]["precision"] == 1
    assert result["question_selection"]["semantic_B_point_score"] is None
    assert result["literal_quote_issues"] == []
    pipeline = result["pipeline"]
    assert pipeline["completely_delivered_parents"] == 0
    assert pipeline["partial_context_parents"] == pipeline["explicit_delivered_context_ranges"] == 1
    assert pipeline["reference_union_completely_delivered"] == 0
    assert pipeline["reference_union_parent_ids_with_some_delivered_context"] == 1
    assert pipeline["reference_quote_delivery_or_entailment_unknown"] is True
    assert pipeline["context_ranges_are_complete_parent_reads"] is False


@pytest.mark.parametrize("change", [
    {"local_start_cp": -1}, {"local_start_cp": True}, {"local_end_cp": 99},
    {"local_end_cp": 3.0}, {"quote": "模型改写"}, {"raw_para_id": "p2"},
    {"doc_id": "other", "para_id": "p1"},
])
def test_context_coordinates_quotes_and_parent_id_are_strict(change):
    parent, quote = "前😀。原文。后。", "原文。"
    delivered = {**context("a", parent, quote), **change}
    with pytest.raises(ValueError, match="delivered_context_range"):
        score([fact("f", "a", text=quote)], [], {("a", "p1"): parent}, [delivered])


@pytest.mark.parametrize("cut", [(4, 10), (4, 4)])
def test_two_ranges_cannot_be_spliced_or_joined_to_authorize_a_larger_quote(cut):
    parent = "甲方事实。未提供限定。乙方事实。"
    left_end, right_start = cut
    delivered = [context("a", parent, parent[:left_end]), context("a", parent, parent[right_start:])]
    with pytest.raises(ValueError, match="single_actually_delivered"):
        score([fact("f", "a", text=parent)], [], {("a", "p1"): parent}, delivered)


def test_context_requires_candidate_and_does_not_borrow_identical_other_parent():
    parent, quote = "前文。字面相同。尾文。", "字面相同。"
    contexts = {(doc, "p1"): parent for doc in ("a", "b")}
    with pytest.raises(ValueError, match="candidate_original_parent"):
        score([], [], contexts, [context("a", parent, quote)], candidates=set())
    with pytest.raises(ValueError, match="single_actually_delivered"):
        score([fact("f", "b", text=quote)], [], contexts, [context("a", parent, quote)])


def test_old_default_still_requires_full_parent_delivery():
    parent, quote = "前文。局部事实。尾文。", "局部事实。"
    with pytest.raises(ValueError, match="completely_delivered"):
        scorer.score_normalized(Q, normalized([fact("f", "a", text=quote)]), [], reference([], []),
            {("a", "p1"): parent}, {("a", "p1")}, set())


def test_partial_not_selected_is_unknown_range_not_a_claim_of_full_parent_failure():
    contexts = {("a", "p1"): "前文。已给事实。尾文。", ("b", "p1"): "已给背景。未给限定。"}
    ranges = [context("a", contexts[("a", "p1")], "已给事实。"),
              context("b", contexts[("b", "p1")], "已给背景。")]
    result = score([fact("f", "a", text="已给事实。")],
        [fact("ga", "a", text="已给事实。"), fact("gb", "b", text="未给限定。")], contexts, ranges)
    missed = result["question_selection_errors"]["missed_reference_sources"]
    assert len(missed) == 1 and missed[0]["doc_id"] == "b"
    assert missed[0]["reason"] == "partial_context_delivered_not_selected_outside_ranges_unknown"
    assert missed[0]["reference_quote_within_single_delivered_range"] is False
    assert missed[0]["outside_delivered_context_semantics"] == "unknown"
    assert result["pipeline"]["reference_union_parent_ids_with_partial_context_only"] == 2
    assert result["pipeline"]["completely_delivered_parents"] == 0


def test_full_reads_and_partial_contexts_keep_separate_counts_and_relation_ranges():
    contexts = {("a", "p1"): "完整原段。", ("b", "p1"): "前文。局部事实。尾文。"}
    rows = [fact("fa", "a", text=contexts[("a", "p1")]), fact("fb", "b", text="局部事实。")]
    ranges = [context("b", contexts[("b", "p1")], "局部事实。")]
    result = score(rows, [], contexts, ranges, read_keys={("a", "p1")})
    assert result["pipeline"]["completely_delivered_parents"] == 1
    assert result["pipeline"]["partial_context_parents"] == 1
    submitted = normalized(rows, ["fb"])
    submitted["supporting_facts"][0]["quote"] = contexts[("b", "p1")]
    with pytest.raises(ValueError, match="single_actually_delivered"):
        scorer.score_normalized(Q, submitted, [], reference([], []), contexts, set(contexts), {("a", "p1")},
            delivered_context_ranges=ranges)


def test_bundle_caller_passes_only_after_closed_partial_scope_to_scorer(tmp_path, monkeypatch):
    parent, quote = "前文😀。局部事实。尾文。", "局部事实。"
    row = fact("f", "a", text=quote)
    submitted = normalized([row], ["f"])
    prepared = {"failed": False, "guards": [], "question": Q, "normalized": submitted,
        "baseline": deepcopy(submitted), "contexts": {("a", "p1"): parent},
        "candidates": {("a", "p1")}, "read_keys": set(),
        "delivered_context_ranges": [context("a", parent, quote)],
        "source_scope": {"native_full_read_paragraphs": 0, "native_partial_context_paragraphs": 1},
        "tail": {}, "front": {"report": {}}, "baseline_usage": {}, "browser_download": {}}
    monkeypatch.setattr(bundle_scorer, "prepare_closed", lambda *a: prepared)
    evidence, conclusions = tmp_path / "synthetic-evidence.json", tmp_path / "synthetic-conclusions.json"
    evidence.write_text(json.dumps([{**row, "fact_id": "g"}], ensure_ascii=False), encoding="utf8")
    conclusions.write_text(json.dumps([reference(["g"], [])], ensure_ascii=False), encoding="utf8")
    result = bundle_scorer.score_episode("synthetic-closed", "unused", evidence, conclusions,
        browser_binding="synthetic", browser_sha256="unused")
    assert result["after"]["pipeline"]["completely_delivered_parents"] == 0
    assert result["after"]["pipeline"]["partial_context_parents"] == 1
    assert result["before"]["pipeline"]["completely_delivered_parents"] == 1
    assert result["after"]["question_selection"]["A_id_only_proxy"] == result["before"]["question_selection"]["A_id_only_proxy"]
    assert result["new_model_calls"] == 0 and result["semantic_B_point_score"] is None
