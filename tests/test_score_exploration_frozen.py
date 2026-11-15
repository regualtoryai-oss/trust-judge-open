"""Synthetic offline coverage only, including the evaluator's matching-dict API."""
import pytest
from scripts import score_exploration_frozen as scorer


def fixture():
    case = {"topic_id": "t", "question_id": "q", "topic_title": "独立合成", "question": {"question": "原始问题？"},
        "catalog": [{"paragraph_ref": "P1", "doc_id": "d1", "para_id": "p"}, {"paragraph_ref": "P2", "doc_id": "d2", "para_id": "p"}]}
    full = {"catalog": [{"paragraph_ref": f"t/d{i}/p", "doc_id": f"d{i}", "para_id": "p"} for i in (1,2,3)]}
    fixed = {"cases": [{"schedules": {
        "frozen_widepool": {"bootstrap5": ["P1"], "original20": ["P1"], "original40": ["P1","P2"], "multi_facet20": ["P2"], "multi_facet40": ["P1","P2"]},
        "raw_all_paragraphs": {"bootstrap5": ["t/d1/p"], "original20": ["t/d1/p","t/d2/p"], "original40": ["t/d1/p","t/d2/p"],
            "multi_facet20": ["t/d2/p","t/d3/p"], "multi_facet40": ["t/d1/p","t/d2/p","t/d3/p"]}}}]}
    return fixed, [(None,case,full)], []


def reference(case):
    return [{"fact_id":"f1","doc_id":"d1","para_id":"p"}, {"fact_id":"f2","doc_id":"d1","para_id":"p"},
        {"fact_id":"f3","doc_id":"d3","para_id":"p"}], {"question_id":"q","question":"原始问题？",
        "supporting_facts":[{"fact_id":"f1"},{"fact_id":"f2"}],"conflicting_facts":[{"fact_id":"f3"}]}, []


def test_real_matching_dicts_and_duplicate_reference_source_mapping(monkeypatch):
    monkeypatch.setattr(scorer,"validate_freeze",lambda *a:fixture())
    result=scorer.score_views("synthetic","unused",None,reference_loader=reference)
    wide=result["topic_and_total"]["all"]["frozen_widepool"]
    assert wide["domain_all_sources"]["matches"]==1 and wide["domain_all_sources"]["reference_records"]==2
    assert wide["multi_facet20"]["matches"]==0 and wide["original20"]["matches"]==1
    assert result["cases"][0]["same_budget_fullraw_vs_widepool"]["multi_facet20"]["gained_IDs"]==[("d3","p")]
    assert result["cases"][0]["domains"]["raw_all_paragraphs"]["same_budget_scheduler_vs_original"]["20"]=={
        "gained_IDs":[("d3","p")],"lost_IDs":[("d1","p")]}
    assert not result["candidate_coverage_is_final_precision_or_recall"] and not result["model_reading_executed"]


def test_closure_gate_failure_happens_before_any_reference(monkeypatch):
    calls=[]
    def fail(*a):raise ValueError("source_changed")
    monkeypatch.setattr(scorer,"validate_freeze",fail)
    with pytest.raises(ValueError):scorer.score_views("synthetic","unused",None,reference_loader=lambda case:calls.append(case))
    assert calls==[]


def test_unknown_reference_identity_kept_in_denominator_not_dropped(monkeypatch):
    monkeypatch.setattr(scorer,"validate_freeze",lambda *a:fixture())
    def unresolved(case):
        evidence,conclusion,guards=reference(case)
        conclusion["conflicting_facts"].append({"fact_id":"missing-reference-fact"})
        return evidence,conclusion,guards
    result=scorer.score_views("synthetic","unused",None,reference_loader=unresolved)
    assert result["topic_and_total"]["all"]["raw_all_paragraphs"]["domain_all_sources"]["matches"]==2
    assert result["topic_and_total"]["all"]["raw_all_paragraphs"]["domain_all_sources"]["reference_records"]==3
    assert len(result["cases"][0]["reference_issues"])==1
