"""Validate a model's visible relevance decisions without manufacturing facts."""

from app.agents.schemas import EvidenceSelectionOutput


def validate_evidence_selection(output: dict, evidence: list[dict]) -> dict:
    """Every candidate needs one auditable disposition; deletion is never implicit.

    This checks completeness and reference integrity, not semantic truth. A
    model's relevance assessment remains a measured source of error.
    """
    value = EvidenceSelectionOutput.model_validate(output).model_dump()
    available = {item["id"] for item in evidence}
    if len(available) != len(evidence):
        raise ValueError("duplicate_selection_candidate")
    selected = value["selected_evidence_ids"]
    if len(selected) != len(set(selected)) or not set(selected) <= available:
        raise ValueError("invalid_selected_evidence_ids")
    rows = value["assessments"]
    ids = [row["evidence_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != available:
        raise ValueError("incomplete_evidence_assessments")
    retained = {row["evidence_id"] for row in rows if row["relation"] in {"support", "conflict", "qualifier"}}
    if set(selected) != retained:
        raise ValueError("evidence_selection_disposition_mismatch")
    for row in rows:
        target = row["equivalent_to"]
        if row["relation"] == "duplicate":
            if target not in retained or target == row["evidence_id"]:
                raise ValueError("duplicate_evidence_requires_retained_representative")
        elif target is not None:
            raise ValueError("unexpected_evidence_equivalence")
    return value
