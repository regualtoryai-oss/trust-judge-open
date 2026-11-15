"""Synthetic-only deterministic allocation tests: no model, reference or API."""
from copy import deepcopy

import pytest

from app.domain.evidence_balance import balance_selection


def inputs(specs):
    """Each spec is (document, score, stance[, original text])."""
    catalog, decisions = [], []
    for number, spec in enumerate(specs, 1):
        doc, score, stance, *text = spec
        ref = f"P{number}"
        catalog.append({"paragraph_ref": ref, "doc_id": doc, "para_id": f"p{number}",
            "title": f"合成文档{doc}", "text": text[0] if text else f"完整合成原段{number}。",
            "paragraph_complete": True, "parent_complete": True})
        decisions.append({"p": ref, "relevance": score, "stance": stance, "hypothesis_id": "H1"})
    return ({"question": {"question_id": "synthetic", "question": "是否已完成程序？"}, "catalog": catalog,
             "source_metadata": {row["paragraph_ref"]: {"source_credibility": .9} for row in catalog}},
            {"hypotheses": [{"id": "H1", "claim": "程序已完成。"}], "decisions": decisions})


S, O, Q = "supports_hypothesis", "opposes_hypothesis", "qualification"


def test_long_document_is_capped_without_excluding_other_qualified_documents():
    raw, selection = inputs([("long", 3, S)] * 8 + [("other-a", 2, S), ("other-b", 2, S)])
    result = balance_selection(raw, selection, target=6)
    assert result["retained_refs"] == ["P1", "P2", "P9", "P10"]
    assert result["audit"]["truncated_refs"] == [f"P{i}" for i in range(3, 9)]
    assert result["audit"]["cap_relaxations"] == []
    assert result["audit"]["source_counts"]["retained_document_ids"] == 3


def test_opposition_and_qualification_survive_many_high_score_supports():
    raw, selection = inputs([(f"support-{i}", 3, S) for i in range(10)] + [("counter", 2, O), ("limit", 2, Q)])
    result = balance_selection(raw, selection, target=5)
    assert result["retained_refs"] == ["P1", "P2", "P3", "P11", "P12"]
    assert all(result["audit"]["lanes"][stance]["retained_refs"] for stance in (S, O, Q))


def test_no_synthetic_missing_stance_or_low_score_padding():
    raw, selection = inputs([("a", 3, S), ("b", 1, O), ("c", 0, Q)])
    result = balance_selection(raw, selection)
    assert result["retained_refs"] == ["P1"]
    assert result["audit"]["lanes"][O]["retained_refs"] == []
    assert result["audit"]["lanes"][Q]["retained_refs"] == []
    assert result["audit"]["below_threshold_refs"] == ["P2", "P3"]


def test_cross_document_mirrors_are_kept_without_claiming_independent_votes():
    raw, selection = inputs([("original", 3, S, "完全相同的镜像段。"), ("mirror", 3, S, "完全相同的镜像段。")])
    result = balance_selection(raw, selection)
    assert result["retained_refs"] == ["P1", "P2"]
    assert result["policy"]["mirrors_deduplicated"] is False
    assert result["policy"]["independent_source_votes_computed"] is False
    assert result["audit"]["source_counts"]["basis"] == "document_ids_not_independent_source_votes"


def test_inputs_scores_roles_metadata_and_original_text_are_immutable():
    raw, selection = inputs([("a", 3, S, "\n😀完整段。\r\n"), ("b", 2, O), ("c", 2, Q)])
    before = deepcopy((raw, selection))
    result = balance_selection(raw, selection)
    assert (raw, selection) == before
    result["audit"]["lanes"][S]["qualified_refs"].clear()
    assert (raw, selection) == before


def test_same_score_and_document_coverage_ties_use_original_retrieval_order():
    raw, selection = inputs([(f"d{i}", 3, S) for i in range(6)])
    selection["decisions"].reverse()
    assert balance_selection(raw, selection, target=3)["retained_refs"] == ["P1", "P2", "P3"]


def test_score_three_precedes_two_when_stance_and_cap_do_not_require_exception():
    raw, selection = inputs([("first", 2, S), ("second", 3, S), ("third", 3, S), ("fourth", 2, S)])
    result = balance_selection(raw, selection, target=2)
    assert result["retained_refs"] == ["P2", "P3"]


def test_same_score_fill_prefers_new_document_before_second_from_existing_document():
    raw, selection = inputs([("a", 3, S), ("a", 3, S), ("b", 3, S), ("c", 3, S)])
    result = balance_selection(raw, selection, target=3)
    assert result["retained_refs"] == ["P1", "P3", "P4"]


def test_cap_relaxes_only_for_necessary_existing_stance_coverage():
    raw, selection = inputs([("only", 3, S), ("only", 3, O), ("only", 2, Q), ("only", 3, S)])
    result = balance_selection(raw, selection, target=10)
    assert result["retained_refs"] == ["P1", "P2", "P3"]
    assert result["audit"]["cap_relaxations"] == [{"p":"P3", "doc_id":"only", "stance":Q,
        "previous_allocation":2, "reason":"required_existing_stance_coverage",
        "no_full_lane_cover_with_lower_total_cap_excess":True}]


def test_coverage_does_not_relax_cap_when_alternative_role_assignment_exists():
    raw, selection = inputs([("a", 3, S), ("a", 3, O), ("a", 2, Q), ("b", 3, S), ("c", 3, O)])
    result = balance_selection(raw, selection, target=3)
    assert "P3" in result["retained_refs"] and len(result["retained_refs"]) == 3
    assert result["audit"]["cap_relaxations"] == []
    assert all(row["allocation"] <= 2 for row in result["audit"]["document_allocations"])


def test_target_smaller_than_present_lanes_is_explicitly_raised():
    raw, selection = inputs([("a", 2, S), ("b", 2, O), ("c", 2, Q)])
    result = balance_selection(raw, selection, target=1)
    assert result["retained_refs"] == ["P1", "P2", "P3"]
    assert result["audit"]["effective_target"] == 3 and result["audit"]["target_raised_for_lane_coverage"]


def test_no_qualified_evidence_returns_empty_and_undetermined():
    raw, selection = inputs([("a", 1, S), ("b", 0, O)])
    result = balance_selection(raw, selection)
    assert result["retained_refs"] == [] and result["audit"]["decision"] == "undetermined"
    assert result["audit"]["semantic_quality_verified"] is False
    assert not result["audit"]["truncated_refs"]


def test_empty_catalog_is_an_explicit_undetermined_not_fabricated_evidence():
    raw, selection = inputs([])
    result = balance_selection(raw, selection)
    assert result["retained_refs"] == [] and result["audit"]["decision"] == "undetermined"


def test_policy_and_audit_are_reproducible_and_do_not_include_quality_metric_claims():
    raw, selection = inputs([("a", 3, S), ("b", 2, O), ("c", 2, Q)])
    result = balance_selection(raw, selection)
    assert result == balance_selection(deepcopy(raw), deepcopy(selection))
    assert len(result["audit"]["raw_case_sha256"]) == len(result["audit"]["canonical_selection_sha256"]) == 64
    assert not {"precision", "recall", "f1", "accuracy"}.intersection(result["audit"])


@pytest.mark.parametrize("target,cap", [(0,2), (True,2), (3,0), (3,False), (2.5,2), (3,1.5)])
def test_policy_parameters_are_explicit_positive_integers(target, cap):
    raw, selection = inputs([("a", 3, S)])
    with pytest.raises(ValueError): balance_selection(raw, selection, target=target, per_doc_soft_cap=cap)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "invalid_score", "invalid_stance", "same_original_identity"])
def test_invalid_input_identity_or_incomplete_model_decisions_cannot_be_silently_filtered(change):
    raw, selection = inputs([("a", 3, S), ("b", 2, O)])
    if change == "missing": selection["decisions"].pop()
    elif change == "duplicate": selection["decisions"][1]["p"] = "P1"
    elif change == "unknown": selection["decisions"][1]["p"] = "P99"
    elif change == "invalid_score": selection["decisions"][0]["relevance"] = True
    elif change == "invalid_stance": selection["decisions"][0]["stance"] = "invented"
    elif change == "same_original_identity": raw["catalog"][1].update(doc_id="a", para_id="p1")
    with pytest.raises(ValueError): balance_selection(raw, selection)
