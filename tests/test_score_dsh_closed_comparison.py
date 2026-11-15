import pytest

from scripts import score_dsh_closed_comparison as scorer
from scripts.offline_reference_tools import load_reference_tools


def test_invalid_closure_cannot_reach_reference_or_baseline_loader(monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("closed_bytes_changed")
    def forbidden(*args, **kwargs):
        raise AssertionError("loader_ran_before_closure")
    monkeypatch.setattr(scorer, "audit_closed_episode", reject)
    monkeypatch.setattr(scorer, "read", forbidden)
    monkeypatch.setattr(scorer, "audit_legacy", forbidden)
    with pytest.raises(ValueError, match="closed_bytes"):
        scorer.score_closed("unused", "a" * 64, "unused", baseline_root="unused",
                            case_id="unused", raw_input="unused", sample_root="unused")


def test_budget_blocked_no_final_has_no_quality_score(monkeypatch):
    monkeypatch.setattr(scorer, "audit_closed_episode", lambda *a, **k: {"input_guards": [], "has_final_artifact": False})
    monkeypatch.setattr(scorer, "read", lambda *a, **k: {"status": "budget_blocked"})
    monkeypatch.setattr(scorer, "audit_legacy", lambda *a, **k: pytest.fail("baseline_or_reference_read_on_unfinished_run"))
    with pytest.raises(ValueError, match="complete_proposed_submission"):
        scorer.score_closed("unused", "a" * 64, "unused", baseline_root="unused",
                            case_id="unused", raw_input="unused", sample_root="unused")


def test_exact_parent_mirror_is_unknown_not_semantic_match_with_actual_evaluator():
    _, engine, _ = load_reference_tools()
    a, b = ("candidate", "p2"), ("reference", "p1")
    submitted, reference = [dict(zip(("doc_id", "para_id"), a))], [dict(zip(("doc_id", "para_id"), b))]
    contexts = {a: "登记不等于生效。", b: "登记不等于生效。"}
    result = scorer.examples(engine, submitted, reference, {a}, contexts)
    assert result["all_nonreference_source_ids"][0]["reason"] == "exact_parent_mirror_of_reference_semantics_unknown"
    assert result["all_nonreference_source_ids"][0]["semantic_false_fact_asserted"] is False
    missed = result["all_missed_reference_source_ids"][0]
    assert missed["reason"] == "outside_frozen_24_source_pool"
    assert missed["cross_id_semantic_match"] == "unknown"


def test_unlocated_reference_is_retained_in_examples_and_unique_denominator():
    _, engine, _ = load_reference_tools()
    result = scorer.examples(engine, [], [{"_unresolved_reference": "absent"}], set(), {})
    assert len(result["all_missed_reference_source_ids"]) == 1
    assert result["all_missed_reference_source_ids"][0]["reason"] == "unlocated_reference_retained"
