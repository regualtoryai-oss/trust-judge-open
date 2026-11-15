"""Generic final-conclusion relation contracts; no API, topic gold or replay gains."""
from copy import deepcopy
from hashlib import sha256
import json

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError
import pytest

from app.agents.schemas import ArbitrationOutput
from app.agents.scoped_arbitration import (
    CONCLUSION_FRAME_VERSION, CONCLUSION_ASSESSMENT_CODEBOOK, ISSUE_FRAME_VERSION, ScopedArbitrationError,
    canonical_scoped_arbitration, prepare_scoped_arbitration, validate_scoped_result,
)


QUESTION = {"id": "synthetic-only", "question": "该设施用于试验，还是仅提供培训服务？"}
IDS = ["E1", "E2", "E3", "E4"]
INVALID = (ScopedArbitrationError, ValidationError)


def wire():
    return {"decision": "undetermined", "conclusion": "现有记录显示设施同时开展试验与培训，尚不能据此认定已完成安全验收。",
        "issue_frame": {"question_anchor": QUESTION["question"], "claim_kind": "fact", "scope": "仅限当前设施功能与验收记录，时点不明。",
            "alternatives": {"X": "设施开展试验。", "Y": "设施提供培训服务。"}, "alternatives_relation": "compatible"},
        "judgments": [{"id": "J1", "statement": "原文分别记载试验和培训活动。", "evidence_ids": ["E1", "E2"],
            "prerequisite_judgment_ids": []}], "conflicts": [],
        "relations": {"E1": {"assessment": "S_fact", "side": "X"},
            "E2": {"assessment": "S_fact", "side": "Y"},
            "E3": {"assessment": "QS_procedure", "side": "both"},
            "E4": {"assessment": "U_gap", "side": "neither"}}}


def decode(value=None, ids=IDS):
    return canonical_scoped_arbitration(wire() if value is None else value, ids, question=QUESTION, issue_frame=True)


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_default_issue_frame_has_one_authoritative_conclusion_and_one_short_assessment_per_evidence():
    schema, instructions = prepare_scoped_arbitration(ArbitrationOutput, IDS, question=QUESTION, issue_frame=True)
    tool = convert_to_openai_tool(schema, strict=True)
    props = tool["function"]["parameters"]["properties"]
    assert tool["function"]["name"] == "ScopedArbitrationWireOutput"
    assert "proposition" not in props["issue_frame"]["properties"]
    assert props["relations"]["additionalProperties"] is False
    assert set(props["relations"]["properties"]["E1"]["properties"]) == {"assessment", "side"}
    assert set(props["relations"]["properties"]["E1"]["properties"]["assessment"]["enum"]) == set(CONCLUSION_ASSESSMENT_CODEBOOK)
    assert schema.model_validate(wire()).model_dump() == wire()
    assert "唯一关系目标" in instructions and "不自动等于反证" in instructions
    output, audit = decode()
    assert audit["format"] == CONCLUSION_FRAME_VERSION
    assert audit["principal_claim"] == output["conclusion"] == wire()["conclusion"]
    assert audit["conclusion_sha256"] == digest(output["conclusion"])
    assert audit["target_binding_verified"] is True and audit["semantic_target_verified"] is False


def test_compatible_alternatives_do_not_become_opposition_or_automatic_conflicts():
    before = wire()
    output, audit = decode(before)
    assert output["supporting_evidence_ids"] == ["E1", "E2", "E3"]
    assert output["conflicting_evidence_ids"] == [] and output["conflicts"] == []
    assert audit["evidence_assessments"]["E1"]["side"] == "X"
    assert audit["evidence_assessments"]["E2"]["side"] == "Y"
    assert audit["alternative_side_used_for_mapping"] is False
    assert before == wire()


def test_undetermined_preserves_substantive_facts_and_qualifiers_without_supporting_a_generic_unknown_target():
    output, audit = decode()
    assert output["decision"] == "undetermined"
    assert audit["unused_for_principal_claim_evidence_ids"] == ["E4"]
    assert audit["qualification_support_evidence_ids"] == ["E3"]
    assert audit["qualification_oppose_evidence_ids"] == []


@pytest.mark.parametrize("code", ["S_background", "O_gap", "QO_fact", "S_direct_contradiction"])
def test_invalid_role_basis_combinations_are_rejected_by_the_wire_schema_itself(code):
    value = wire(); value["relations"]["E4"]["assessment"] = code
    schema, _ = prepare_scoped_arbitration(ArbitrationOutput, IDS, question=QUESTION, issue_frame=True)
    with pytest.raises(ValidationError): schema.model_validate(value)
    with pytest.raises(ValidationError): decode(value)


def test_questionable_exclusive_counterclaim_is_visible_without_silently_changing_model_direction():
    value = wire(); value["relations"]["E2"]["assessment"] = "O_exclusive_alternative"
    output, audit = decode(value)
    assert output["conflicting_evidence_ids"] == ["E2"]
    assessment = audit["evidence_assessments"]["E2"]
    assert assessment["assessment"] == "O_exclusive_alternative" and assessment["role"] == "O"
    assert assessment["semantic_warnings"] == ["exclusive_alternative_not_established_by_frame_and_side"]
    value["relations"]["E2"]["assessment"] = "O_direct_contradiction"
    assert decode(value)[1]["evidence_assessments"]["E2"]["semantic_warnings"] == []


def test_attribution_layer_disagreement_is_audited_without_silently_rewriting_or_failing_the_result():
    value = wire(); value["issue_frame"]["claim_kind"] = "causal"
    value["relations"]["E1"]["assessment"] = "S_attribution_only"
    output, audit = decode(value)
    assert "E1" in output["supporting_evidence_ids"]
    assert audit["evidence_assessments"]["E1"]["semantic_warnings"] == ["attribution_only_direct_support_of_other_claim_layer"]
    value["relations"]["E1"]["assessment"] = "QS_attribution_only"
    assert decode(value)[1]["evidence_assessments"]["E1"]["semantic_warnings"] == []
    value["relations"]["E1"]["assessment"] = "S_attribution_only"
    value["issue_frame"]["claim_kind"] = "attribution"
    assert decode(value)[1]["evidence_assessments"]["E1"]["semantic_warnings"] == []


@pytest.mark.parametrize("code", list(CONCLUSION_ASSESSMENT_CODEBOOK))
@pytest.mark.parametrize("relation", ["exclusive", "compatible", "unknown", "not_binary"])
def test_every_schema_assessment_decodes_in_every_frame_without_cross_field_semantic_failure(code, relation):
    value = wire()
    value["relations"]["E1"]["assessment"] = code
    value["issue_frame"]["alternatives_relation"] = relation
    schema, _ = prepare_scoped_arbitration(ArbitrationOutput, IDS, question=QUESTION, issue_frame=True)
    typed = schema.model_validate(value)
    output, audit = decode(typed)
    assessment = audit["evidence_assessments"]["E1"]
    assert assessment["assessment"] == code
    assert {key: assessment[key] for key in ("role", "basis")} == CONCLUSION_ASSESSMENT_CODEBOOK[code]
    assert ("E1" in output["supporting_evidence_ids"]) == (assessment["role"] in {"S", "QS"})
    assert ("E1" in output["conflicting_evidence_ids"]) == (assessment["role"] in {"O", "QO"})
    assert validate_scoped_result(audit, output, IDS, question=QUESTION) == audit


def test_frame_and_side_warnings_preserve_original_values_and_are_bound_on_replay():
    value = wire(); value["issue_frame"]["alternatives_relation"] = "not_binary"
    output, audit = decode(value)
    assert audit["issue_frame"] == value["issue_frame"]
    assert audit["issue_frame_semantic_warnings"] == ["nonbinary_alternatives_not_empty"]
    assert audit["evidence_assessments"]["E1"]["semantic_warnings"] == ["nonbinary_alternative_side_not_neither"]
    assert audit["evidence_assessments"]["E1"]["side"] == "X"
    audit["evidence_assessments"]["E1"]["semantic_warnings"] = []
    with pytest.raises(ScopedArbitrationError, match="expanded_assessment_changed"):
        validate_scoped_result(audit, output, IDS, question=QUESTION)


def test_literal_question_anchor_is_still_a_hard_contract():
    value = wire(); value["issue_frame"]["question_anchor"] = "改写的原题"
    with pytest.raises(ScopedArbitrationError, match="question_anchor_not_literal"):
        decode(value)


@pytest.mark.parametrize("change", ["conclusion", "role", "side", "target", "question", "order", "extra"])
def test_replay_binds_final_text_roles_sides_question_and_model_order_without_semantic_repair(change):
    output, audit = decode()
    if change == "conclusion": output["conclusion"] = "相反结论。"
    elif change == "role": audit["evidence_roles"]["E1"] = "O"
    elif change == "side": audit["evidence_assessments"]["E1"]["side"] = "Y"
    elif change == "target": audit["principal_claim"] = "不属于最终结论的另一个主张。"
    elif change == "question": audit["question_sha256"] = "0" * 64
    elif change == "order": audit["evidence_order"].reverse()
    elif change == "extra": audit["inferred_relation"] = True
    if change == "side":
        # The evidence-side statement is model metadata, not a semantic fact:
        # changing it coherently is not detectable by contract replay alone.
        assert validate_scoped_result(audit, output, IDS, question=QUESTION)["semantic_target_verified"] is False
        return
    with pytest.raises(INVALID): validate_scoped_result(audit, output, IDS, question=QUESTION)


def test_reference_remap_and_public_sort_preserve_complete_v3_audit():
    output, audit = decode()
    mapping = {ref: "actual-" + ref for ref in IDS}
    audit["evidence_roles"] = {mapping[key]: val for key, val in audit["evidence_roles"].items()}
    audit["evidence_assessments"] = {mapping[key]: val for key, val in audit["evidence_assessments"].items()}
    for key in ("evidence_order", "qualification_evidence_ids", "qualification_support_evidence_ids",
                "qualification_oppose_evidence_ids", "unused_for_principal_claim_evidence_ids"):
        audit[key] = [mapping[ref] for ref in audit[key]]
    audit["evidence_order_sha256"] = digest(audit["evidence_order"])
    for key in ("supporting_evidence_ids", "conflicting_evidence_ids"):
        output[key] = [mapping[ref] for ref in output[key]]
    output["judgments"][0]["evidence_ids"] = [mapping[ref] for ref in output["judgments"][0]["evidence_ids"]]
    assert validate_scoped_result(audit, output, list(reversed(list(mapping.values()))), question=QUESTION) == audit


def test_explicit_v2_generation_and_replay_remain_available():
    from test_issue_frame_arbitration import wire as legacy_wire, QUESTION as legacy_question, ALIASES
    schema, _ = prepare_scoped_arbitration(ArbitrationOutput, ALIASES, question=legacy_question,
        issue_frame=True, conclusion_aligned=False)
    typed = schema.model_validate(legacy_wire())
    output, audit = canonical_scoped_arbitration(typed, ALIASES, question=legacy_question, issue_frame=True)
    assert audit["format"] == ISSUE_FRAME_VERSION
    assert validate_scoped_result(audit, output, ALIASES, question=legacy_question) == audit
