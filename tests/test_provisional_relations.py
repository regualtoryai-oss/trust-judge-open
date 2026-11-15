"""Generic A/B metadata isolation; no real-provider or reference fixtures."""
from copy import deepcopy

import pytest

from app.agents.provider import ModelOutputInvalid
from app.domain.analysis_wire_projection import project_analyses, restore_analyses
from app.domain.provisional_relations import quarantine_provisional_relations, validate_relation_quality
from app.workflows.analysis import _validate_analysis


ROLES = ("authority_analyst", "cross_validation_analyst")
EVIDENCE = [{"id": key, "quote": "保持原文。"} for key in ("e1", "e2", "e3")]


def output(decision="undetermined"):
    return {"decision": decision, "conclusion": "现有记录有支持与相反说法，范围仍待核实。",
        "judgments": [{"id": "J1", "statement": "原判断、限定与引用全部保留。",
            "evidence_ids": ["e1", "e2", "e3"], "prerequisite_judgment_ids": []}],
        "supporting_evidence_ids": ["e1", "e2"], "conflicting_evidence_ids": ["e1", "e3"]}


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("decision", ["asserted", "undetermined"])
def test_provisional_overlap_is_isolated_without_directional_choice_or_response_mutation(role, decision):
    raw = output(decision); before = deepcopy(raw)
    projected, quality = quarantine_provisional_relations(raw, role)
    assert raw == before
    assert projected["supporting_evidence_ids"] == ["e2"]
    assert projected["conflicting_evidence_ids"] == ["e3"]
    assert projected["judgments"] == raw["judgments"] and projected["conclusion"] == raw["conclusion"]
    assert projected["decision"] == decision and "relation_quality" not in projected
    assert quality["quarantined_evidence_ids"] == ["e1"] and quality["quarantined_count"] == 1
    assert quality["raw_model_response_modified"] is quality["evidence_removed"] is False
    validated = _validate_analysis(raw, EVIDENCE, "q", role)
    assert validated["relation_quality"] == quality
    assert validate_relation_quality(validated, role=role) == quality
    assert validated["judgments"][0]["statement"] == raw["judgments"][0]["statement"]
    assert validated["judgments"][0]["evidence_ids"] == raw["judgments"][0]["evidence_ids"]


def test_asserted_without_remaining_support_is_rejected_without_decision_downgrade():
    raw = output("asserted"); raw["supporting_evidence_ids"] = ["e1"]
    before = deepcopy(raw)
    with pytest.raises(ModelOutputInvalid) as error:
        _validate_analysis(raw, EVIDENCE, "q", ROLES[0])
    assert error.value.details["check"] == "assertion_without_support" and raw == before


def test_undetermined_can_retain_all_evidence_and_judgments_with_empty_provisional_labels():
    raw = output(); raw["conflicting_evidence_ids"] = list(raw["supporting_evidence_ids"])
    value = _validate_analysis(raw, EVIDENCE, "q", ROLES[1])
    assert value["supporting_evidence_ids"] == value["conflicting_evidence_ids"] == []
    assert value["judgments"][0]["evidence_ids"] == ["e1", "e2", "e3"]
    assert value["relation_quality"]["quarantined_count"] == 2


@pytest.mark.parametrize("change", ["unknown", "duplicate", "ungrounded", "injected_audit"])
def test_quarantine_does_not_hide_invalid_references_dag_or_model_injected_metadata(change):
    raw = output()
    if change == "unknown":
        raw["supporting_evidence_ids"].append("unknown")
        raw["conflicting_evidence_ids"].append("unknown")
    elif change == "duplicate": raw["supporting_evidence_ids"].append("e1")
    elif change == "ungrounded": raw["judgments"][0]["evidence_ids"] = []
    else: raw["relation_quality"] = {"format": "model_cannot_write_program_audit"}
    with pytest.raises(ModelOutputInvalid): _validate_analysis(raw, EVIDENCE, "q", ROLES[0])


def test_final_arbitration_and_non_ab_outputs_remain_strictly_disjoint():
    for role, arbitration in (("arbitrate", True), ("baseline", False)):
        raw = output()
        if arbitration: raw["conflicts"] = []
        with pytest.raises(ModelOutputInvalid) as error:
            _validate_analysis(raw, EVIDENCE, "q", role, arbitration=arbitration)
        assert error.value.details["check"] == "overlapping_evidence_relations"


def test_no_overlap_keeps_the_old_view_exactly_and_adds_no_audit():
    raw = output(); raw["conflicting_evidence_ids"] = ["e3"]
    projected, quality = quarantine_provisional_relations(raw, ROLES[0])
    assert projected == raw and quality is None
    assert "relation_quality" not in _validate_analysis(raw, EVIDENCE, "q", ROLES[0])


@pytest.mark.parametrize("change", ["original", "hash", "quarantined", "projected", "flag", "role"])
def test_relation_quality_replay_rejects_changed_provenance_or_projection(change):
    value = _validate_analysis(output(), EVIDENCE, "q", ROLES[0])
    if change == "original": value["relation_quality"]["original_relations"]["supporting_evidence_ids"].reverse()
    elif change == "hash": value["relation_quality"]["original_relations_sha256"] = "0" * 64
    elif change == "quarantined": value["relation_quality"]["quarantined_evidence_ids"] = []
    elif change == "projected": value["conflicting_evidence_ids"] = []
    elif change == "flag": value["relation_quality"]["semantic_verified"] = True
    else: value["relation_quality"]["role"] = ROLES[1]
    with pytest.raises(ValueError, match="provisional_relation_quality_contract"):
        validate_relation_quality(value, role=ROLES[0])


def test_compact_analyst_projection_preserves_complete_optional_quarantine_and_dag():
    rows = []
    for role in ROLES:
        row = _validate_analysis(output(), EVIDENCE, "q", role)
        row.update(role=role, evidence_snapshot_id="same-frozen-snapshot")
        rows.append(row)
    before = deepcopy(rows)
    compact = project_analyses(rows)
    assert restore_analyses(**compact, trusted_original=rows) == before
    assert rows == before
    for original, projected in zip(rows, compact["projected"]):
        assert projected["relation_quality"] == original["relation_quality"]
        assert projected["judgments"][0]["evidence_ids"] == ["e1", "e2", "e3"]
    compact["projected"][0]["relation_quality"]["quarantined_count"] = 0
    with pytest.raises(ValueError): restore_analyses(**compact, trusted_original=rows)
