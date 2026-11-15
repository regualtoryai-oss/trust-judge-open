from copy import deepcopy

import pytest

from app.domain.evidence_selection import validate_evidence_selection


POOL = [{"id": name, "quote": text} for name, text in (
    ("a", "截至周三，泵站仍未恢复。"), ("b", "媒体称周三已恢复。"),
    ("c", "截至周三，泵站仍未恢复。"), ("d", "附近正在开展演练。"))]


def selection():
    return {"selected_evidence_ids": ["a", "b"], "assessments": [
        {"evidence_id": "a", "relation": "support", "reason": "直接说明截至时点状态。", "equivalent_to": None},
        {"evidence_id": "b", "relation": "conflict", "reason": "相同时点状态有相反表述。", "equivalent_to": None},
        {"evidence_id": "c", "relation": "duplicate", "reason": "相同状态和时点已完整保留。", "equivalent_to": "a"},
        {"evidence_id": "d", "relation": "background", "reason": "演练不证明恢复状态。", "equivalent_to": None},
    ]}


def test_review_retains_opposing_side_and_audits_every_rejection_without_mutation():
    before = deepcopy(POOL)
    assert validate_evidence_selection(selection(), POOL) == selection()
    assert POOL == before


@pytest.mark.parametrize("mutate", [
    lambda x: x["selected_evidence_ids"].append("unknown"),
    lambda x: x["selected_evidence_ids"].append("a"),
    lambda x: x["selected_evidence_ids"].remove("b"),
    lambda x: x["assessments"].pop(),
    lambda x: x["assessments"].append(deepcopy(x["assessments"][0])),
    lambda x: x["assessments"][2].update(equivalent_to="d"),
    lambda x: x["assessments"][2].update(equivalent_to="c"),
    lambda x: x["assessments"][0].update(equivalent_to="b"),
])
def test_review_cannot_silently_drop_candidates_invent_references_or_discard_both_duplicates(mutate):
    value = selection()
    mutate(value)
    with pytest.raises(ValueError):
        validate_evidence_selection(value, POOL)


def test_empty_review_is_valid_without_fabricated_support():
    assert validate_evidence_selection({"selected_evidence_ids": [], "assessments": []}, []) == {
        "selected_evidence_ids": [], "assessments": []}
