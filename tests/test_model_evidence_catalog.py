"""Compact analyst boundary; pure offline tests, never a provider account."""
from copy import deepcopy
import json

import pytest

from app.agents.schemas import AnalysisOutput, ArbitrationOutput
from app.domain.model_evidence_catalog import (
    ModelEvidenceCatalogInvalid, build_evidence_catalog, compact_schema,
)


QUESTION = {"id": "private-question", "question": "甲方说法是否有依据？", "time_scope": "截至原文时点"}


def row(identity="private-evidence", group="private-group"):
    return {"id": identity, "paragraph_id": "private-para", "raw_doc_id": "doc_001",
            "raw_para_id": "doc_001#p1", "quote": "甲方称尚未确认。E1不是额外ID。",
            "fact_des": "甲方称尚未确认。", "argument": {"时间": None, "主体": "甲方", "地点": None,
            "内容": "尚未确认。", "影响": None, "数值口径": None}, "entities": ["甲方"],
            "source_level": "unknown", "grade_score": 0.85,
            "source_group_id": group, "source_group_basis": "linked_not_proven_independent",
            "interpretation_origin": "human", "stance": "待核验人工立场",
            "quote_verified": True, "paragraph_fields_verified": True,
            "start_cp": 0, "end_cp": 16, "internal_hash": "do-not-forward"}


def output(eid="E1"):
    return {"decision": "undetermined", "conclusion": "E1原文保留限定。", "judgments": [
            {"id": "J1", "statement": "保留甲方指称。", "evidence_ids": [eid], "prerequisite_judgment_ids": []},
            {"id": "J2", "statement": "J1作为前提。", "evidence_ids": [], "prerequisite_judgment_ids": ["J1"]}],
            "supporting_evidence_ids": [eid], "conflicting_evidence_ids": []}


def test_catalog_preserves_all_source_facts_and_known_group_equivalence():
    originals = [row(), row("private-2"), row("private-3", "other-group")]
    before = deepcopy(originals)
    catalog = build_evidence_catalog(QUESTION, originals, "frozen-snapshot")
    assert originals == before
    projected = catalog.evidence
    assert [r["id"] for r in projected] == ["E1", "E2", "E3"]
    assert [r["source_group_id"] for r in projected] == ["G1", "G1", "G2"]
    for actual, original in zip(projected, originals):
        for key in ("quote", "fact_des", "argument", "entities", "source_level", "grade_score", "source_group_basis", "stance", "interpretation_origin"):
            assert actual[key] == original[key]
    wire = json.dumps(projected, ensure_ascii=False)
    assert all(value not in wire for value in ("private-para", "doc_001", "do-not-forward", "quote_verified", "start_cp"))
    projected[0]["quote"] = "mutated"
    mapping = catalog.aliases
    mapping["E1"] = "mutated"
    assert catalog.evidence[0]["quote"] == originals[0]["quote"]
    assert catalog.aliases["E1"] == originals[0]["id"]


def test_resolve_only_declared_evidence_reference_slots():
    catalog = build_evidence_catalog(QUESTION, [row()], "s")
    original = output()
    resolved = catalog.resolve(original, question=QUESTION, snapshot_id="s")
    assert AnalysisOutput.model_validate(resolved)
    assert resolved["supporting_evidence_ids"] == ["private-evidence"]
    assert resolved["judgments"][0]["evidence_ids"] == ["private-evidence"]
    assert resolved["conclusion"] == original["conclusion"]
    assert resolved["judgments"][1]["prerequisite_judgment_ids"] == ["J1"]
    assert original == output()


def test_arbitration_conflict_ids_roundtrip_without_prose_changes():
    catalog = build_evidence_catalog(QUESTION, [row(), row("private-2", "other-group")], "s")
    original = {**output(), "conflicting_evidence_ids": ["E2"], "conflicts": [{
        "id": "C1", "statement": "存在不同说法。", "status": "unresolved",
        "supporting_evidence_ids": ["E1"], "conflicting_evidence_ids": ["E2"],
        "resolution_type": "Other", "basis": "E1为待核验指称。", "accepted_reason": "未采纳。",
        "rejected_reason": "未排除。", "minority_opinion": "两方保留。", "reopen_conditions": "取得原件。"}]}
    resolved = catalog.resolve(original, question=QUESTION, snapshot_id="s")
    assert ArbitrationOutput.model_validate(resolved)
    assert resolved["conflicts"][0]["conflicting_evidence_ids"] == ["private-2"]
    assert resolved["conflicts"][0]["basis"] == "E1为待核验指称。"
    visible = catalog.project_analyses([{**resolved, "role": "authority_analyst", "evidence_snapshot_id": "s", "runtime_secret": "drop"}], question=QUESTION, snapshot_id="s")
    assert visible == [{**original, "role": "authority_analyst"}]


@pytest.mark.parametrize("field,value", [("supporting_evidence_ids", ["E99"]), ("conflicting_evidence_ids", ["private-evidence"]), ("supporting_evidence_ids", ["E1", "E1"])])
def test_unknown_raw_or_duplicate_model_reference_fails_closed(field, value):
    catalog = build_evidence_catalog(QUESTION, [row()], "s")
    value_output = output()
    value_output[field] = value
    with pytest.raises(ModelEvidenceCatalogInvalid, match="unknown_or_duplicate"):
        catalog.resolve(value_output, question=QUESTION, snapshot_id="s")


def test_scope_or_content_drift_never_reuses_private_mapping():
    catalog = build_evidence_catalog(QUESTION, [row()], "s")
    for question, snapshot in [({**QUESTION, "time_scope": "另一个时点"}, "s"), (QUESTION, "different-snapshot")]:
        with pytest.raises(ModelEvidenceCatalogInvalid, match="scope_mismatch"):
            catalog.resolve(output(), question=question, snapshot_id=snapshot)
    changed = row()
    changed["quote"] += " 新文字。"
    assert build_evidence_catalog(QUESTION, [changed], "s").binding_sha256 != catalog.binding_sha256
    with pytest.raises(ModelEvidenceCatalogInvalid, match="analysis_scope_mismatch"):
        catalog.project_analyses([{**output("private-evidence"), "evidence_snapshot_id": "other"}], question=QUESTION, snapshot_id="s")


@pytest.mark.parametrize("extra", [{"source_lineage": {}}, {"source_counts": {}}, {"source_inventory_basis": {}}, {"source_group_members": []}, {"quote_verified": False}, {"paragraph_fields_verified": False}])
def test_unsupported_lineage_inventory_or_unverified_content_does_not_silently_shrink(extra):
    with pytest.raises(ModelEvidenceCatalogInvalid):
        build_evidence_catalog(QUESTION, [{**row(), **extra}], "s")


def test_duplicate_identity_rejected_and_empty_evidence_is_legal():
    with pytest.raises(ModelEvidenceCatalogInvalid):
        build_evidence_catalog(QUESTION, [row(), row()], "s")
    catalog = build_evidence_catalog(QUESTION, [], "s")
    unresolved = {"decision": "undetermined", "conclusion": "未取得资料。", "judgments": [], "supporting_evidence_ids": [], "conflicting_evidence_ids": []}
    assert catalog.resolve(unresolved, question=QUESTION, snapshot_id="s") == unresolved


def test_schema_compaction_keeps_constraints_descriptions_and_title_named_field():
    schema = AnalysisOutput.model_json_schema()
    compact = compact_schema(schema)
    assert "title" not in compact
    assert compact["properties"]["conclusion"]["description"] == schema["properties"]["conclusion"]["description"]
    assert compact["allOf"] == schema["allOf"]
    assert compact["$defs"]["JudgmentOutput"]["anyOf"] == schema["$defs"]["JudgmentOutput"]["anyOf"]
    assert compact_schema({"type": "object", "title": "Display", "properties": {"title": {"type": "string", "title": "Field display", "minLength": 1}}}) == {
        "type": "object", "properties": {"title": {"type": "string", "minLength": 1}}}
    literal = {"title": "literal-data", "nested": {"title": "still literal"}}
    for keyword in ("default", "const", "examples", "enum"):
        assert compact_schema({keyword: literal, "title": "Display"}) == {keyword: literal}
