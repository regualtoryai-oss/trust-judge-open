from copy import deepcopy
from dataclasses import replace
import pytest

from app.domain.evidence_portfolio import (
    Contribution, PortfolioCandidate, PortfolioPolicy, legacy_contributions, plan_portfolio,
)


def candidate(ref, *, score=2, role="premise", stance="neutral", aspect="A", group=None,
              credibility=.7, text=None, doc=None):
    text = text or "Original source " + ref
    return PortfolioCandidate(ref, doc or ref, ref, text, score,
                              (Contribution(aspect, role, stance, text),), credibility, group)


def test_question_coverage_precedes_repeated_high_score_claim():
    items = [candidate("claim1", score=3, role="direct_answer", stance="supports_X", group="one"),
             candidate("claim2", score=3, role="direct_answer", stance="supports_X", group="one"),
             candidate("official", role="direct_answer", stance="official")]
    plan = plan_portfolio(items, aspect_ids=["A"], policy=PortfolioPolicy(maximum_paragraphs=2))
    assert plan["selected_refs"] == ["claim1", "official"]
    assert plan["independent_source_votes_created"] == 0


def test_complex_questions_have_no_uniform_ten_paragraph_cutoff():
    items = [candidate(str(i), aspect="A" + str(i)) for i in range(21)]
    plan = plan_portfolio(items, aspect_ids=["A" + str(i) for i in range(21)])
    assert len(plan["selected_refs"]) == 21
    assert plan["unrepresented_aspect_ids"] == []


def test_same_origin_still_admitted_for_a_distinct_contribution():
    items = [candidate("one", aspect="A", group="syndicated"),
             candidate("two", aspect="B", group="syndicated"),
             candidate("mirror", aspect="A", group="syndicated")]
    plan = plan_portfolio(items, aspect_ids=["A", "B"])
    assert len(plan["selected_refs"]) == 2
    assert "two" in plan["selected_refs"]
    assert len(plan["decisions"]) == 3  # Catalogue is never deleted.


def test_corroboration_does_not_erase_mirrors_or_create_votes():
    items = [candidate("original", group="original", credibility=.9),
             candidate("mirror", group="mirror", credibility=.7)]
    plan = plan_portfolio(items, aspect_ids=["A"])
    assert set(plan["selected_refs"]) == {"original", "mirror"}
    assert plan["independent_source_votes_created"] == 0


def test_rank_qualified_exact_mirror_locations_survive_the_same_group():
    items = [candidate("original", group="one", text="same original paragraph"),
             candidate("mirror", group="one", text="same original paragraph")]
    plan = plan_portfolio(items, aspect_ids=["A"])
    assert len(plan["analysis_core_refs"]) == 1
    assert set(plan["selected_refs"]) == {"original", "mirror"}
    assert plan["selection_trace"][-1]["reason"] == "preserve_rank_qualified_exact_mirror_source_location"
    assert plan["independent_source_votes_created"] == 0


def test_distinct_procedural_claims_survive_the_same_facet_and_source():
    items = [replace(candidate(str(i), group="same_official_record"), contributions=(
        Contribution("A", "official_record", "official", "Original source " + str(i), "procedure_" + str(i)),
    )) for i in range(21)]
    plan = plan_portfolio(items, aspect_ids=["A"])
    assert len(plan["selected_refs"]) == 21


def test_credibility_is_source_prior_not_a_truth_measure():
    items = [candidate("low", credibility=.6), candidate("high", credibility=.9)]
    plan = plan_portfolio(items, aspect_ids=["A"], policy=PortfolioPolicy(maximum_paragraphs=1))
    assert plan["selected_refs"] == ["high"]
    assert plan["semantic_coverage_verified"] is False


def test_literal_anchor_and_question_binding_are_required():
    item = candidate("P1")
    for contributions in [(Contribution("A", "premise", "neutral", "invented"),),
                          (Contribution("unknown", "premise", "neutral", item.source_text),)]:
        with pytest.raises(ValueError, match="source_bound"):
            plan_portfolio([replace(item, contributions=contributions)], aspect_ids=["A"])


def test_duplicate_labels_cannot_multiply_gain():
    item = candidate("P1")
    duplicated = replace(item, contributions=item.contributions * 30)
    assert plan_portfolio([item], aspect_ids=["A"]) == plan_portfolio([duplicated], aspect_ids=["A"])


def test_low_rank_and_context_are_explicitly_rejected():
    items = [candidate("low", score=1), candidate("context", role="context")]
    plan = plan_portfolio(items, aspect_ids=["A"])
    assert plan["selected_refs"] == []
    assert [r["reason"] for r in plan["decisions"]] == [
        "rank_below_evidence_threshold", "no_nonbackground_question_contribution"]


def test_literal_character_budget_reports_uncovered_aspect():
    items = [candidate("short", text="12", aspect="A"), candidate("long", text="12345", aspect="B")]
    plan = plan_portfolio(items, aspect_ids=["A", "B"],
                          policy=PortfolioPolicy(maximum_quote_characters=4))
    assert plan["selected_refs"] == ["short"]
    assert plan["unrepresented_aspect_ids"] == ["B"]
    assert plan["decisions"][1]["reason"] == "deferred_source_character_budget"
    assert plan["character_count_is_not_tokens"]


def test_permutation_is_deterministic_and_source_unchanged():
    items = [candidate("P2"), candidate("P1"), candidate("P3", aspect="B")]
    before = deepcopy(items)
    forward = plan_portfolio(items, aspect_ids=["A", "B"])
    backward = plan_portfolio(items[::-1], aspect_ids=["A", "B"])
    assert forward["selected_refs"] == backward["selected_refs"]
    assert forward["selection_trace"] == backward["selection_trace"]
    assert items == before


@pytest.mark.parametrize("credibility", [True, float("nan"), float("inf"), -.1, 1.1])
def test_invalid_source_priors_are_not_silently_accepted(credibility):
    with pytest.raises(ValueError, match="credibility"):
        plan_portfolio([candidate("P1", credibility=credibility)], aspect_ids=["A"])


def test_duplicate_source_keys_and_reference_aliases_rejected():
    item = candidate("P1")
    for other in [item, replace(item, paragraph_ref="P2")]:
        with pytest.raises(ValueError, match="candidate_invalid"):
            plan_portfolio([item, other], aspect_ids=["A"])


def test_legacy_adapter_does_not_infer_semantics_from_reason():
    label = legacy_contributions(source_text="original", score=2, stance="official")
    assert label == (Contribution("whole_question", "premise", "official", "original"),)
    qualifier = legacy_contributions(source_text="original", score=2, stance="neutral", critical_qualifier=True)
    assert qualifier[0].role == "qualifier"


@pytest.mark.parametrize("policy", [PortfolioPolicy(maximum_paragraphs=0),
                                    PortfolioPolicy(maximum_quote_characters=True),
                                    PortfolioPolicy(corroborating_groups_per_claim=0)])
def test_policy_needs_positive_integer_safety_limits(policy):
    with pytest.raises(ValueError, match="policy_invalid"):
        plan_portfolio([], aspect_ids=["A"], policy=policy)
