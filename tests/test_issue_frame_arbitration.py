"""Synthetic directional issue frames; no provider, gold, or historical output."""
from copy import deepcopy
from hashlib import sha256
import itertools
import json

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, ValidationError
import pytest

from app.agents.schemas import ArbitrationOutput
from app.agents.scoped_arbitration import (
    ISSUE_CODEBOOK, ISSUE_FRAME_VERSION, ScopedArbitrationError,
    canonical_scoped_arbitration, prepare_scoped_arbitration, validate_scoped_result,
)


QUESTION = {"id": "synthetic-question", "question": "企业是否承担事故责任，还是仅缺少官方因果认定？"}
ALIASES = ["E1", "E2", "E3", "E4", "E5"]
CODES = ["S", "O", "QS", "QO", "U"]
INVALID = (ValidationError, ScopedArbitrationError)


def wire():
    return {"decision": "undetermined", "conclusion": "现有材料支持责任审查，但尚未证实官方因果认定，两个层级不能互换。",
        "issue_frame": {"question_anchor": "企业是否承担事故责任", "proposition": "事故责任应包括企业在其经营范围内的责任。",
            "claim_kind": "normative", "scope": "仅针对企业经营责任，不替代官方事故因果认定或政府监管责任。"},
        "judgments": [{"id": "J1", "statement": "企业经营责任与官方因果认定属于不同层级，原文限定应保留。",
            "evidence_ids": ["E1"], "prerequisite_judgment_ids": []},
            {"id": "J2", "statement": "据该层级区别，不能从未完成认定推断企业责任已经被否认。",
                "evidence_ids": ["E3"], "prerequisite_judgment_ids": ["J1"]}],
        "conflicts": [], "relations": dict(zip(ALIASES, CODES))}


def conflict(status="unresolved"):
    return {"id": "C1", "statement": "企业责任的范围存在两种明确说法。", "status": status,
        "supporting_evidence_ids": ["E1"], "conflicting_evidence_ids": ["E2"], "resolution_type": "Other",
        "basis": "两方对于同一企业责任范围存在分歧。", "accepted_reason": "保留已经有依据的经营责任范围。",
        "rejected_reason": "未决材料不足以排除该范围。", "minority_opinion": "保留企业责任不成立的反向说法。",
        "reopen_conditions": "出现可核实的责任认定或新的经营事实。"}


def decode(value=None, aliases=ALIASES, question=QUESTION):
    return canonical_scoped_arbitration(wire() if value is None else value, aliases,
        question=question, issue_frame=True, conclusion_aligned=False)


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def test_dynamic_schema_keeps_tool_name_full_base_fields_and_closed_short_code_map():
    schema, instructions = prepare_scoped_arbitration(ArbitrationOutput, ALIASES,
        question=QUESTION, issue_frame=True, conclusion_aligned=False)
    tool = convert_to_openai_tool(schema, strict=True)
    assert tool["function"]["name"] == "ScopedArbitrationWireOutput"
    properties = tool["function"]["parameters"]["properties"]
    assert set(properties) == {"decision", "conclusion", "judgments", "conflicts", "issue_frame", "relations"}
    assert set(properties["relations"]["required"]) == set(ALIASES)
    assert properties["relations"]["additionalProperties"] is False
    assert properties["relations"]["properties"]["E1"]["enum"] == CODES
    assert set(properties["issue_frame"]["required"]) == {"question_anchor", "proposition", "claim_kind", "scope"}
    assert properties["issue_frame"]["additionalProperties"] is False
    for field in ("judgments", "conflicts"):
        assert schema.model_fields[field].annotation == ArbitrationOutput.model_fields[field].annotation
    assert schema.model_validate(wire()).model_dump() == wire()
    assert "不设支持或冲突数量配额" in instructions and "不自动互斥" in instructions
    assert "未确认某原因不等于已证实该原因未发生" in instructions


@pytest.mark.parametrize("first,second", list(itertools.product(CODES, repeat=2)))
def test_complete_directional_matrix_is_literal_model_mapping_not_decision_inference(first, second):
    value = wire(); value["judgments"] = []
    value["relations"] = {"E1": first, "E2": second}
    output, scope = decode(value, ["E1", "E2"])
    assert output["decision"] == "undetermined"
    assert output["supporting_evidence_ids"] == [ref for ref, code in value["relations"].items() if code in {"S", "QS"}]
    assert output["conflicting_evidence_ids"] == [ref for ref, code in value["relations"].items() if code in {"O", "QO"}]
    assert scope["qualification_support_evidence_ids"] == [ref for ref, code in value["relations"].items() if code == "QS"]
    assert scope["qualification_oppose_evidence_ids"] == [ref for ref, code in value["relations"].items() if code == "QO"]
    assert scope["evidence_roles"] == value["relations"]


def test_frame_conclusion_full_statements_dag_and_conflicts_are_not_rewritten():
    value = wire(); value["conflicts"] = [conflict("resolved")]
    before = deepcopy(value)
    output, scope = decode(value)
    for field in ("decision", "conclusion", "judgments", "conflicts"):
        assert output[field] == value[field]
    assert output["supporting_evidence_ids"] == ["E1", "E3"]
    assert output["conflicting_evidence_ids"] == ["E2", "E4"]
    assert scope["principal_claim"] == value["issue_frame"]["proposition"]
    assert scope["issue_frame"] == value["issue_frame"]
    assert scope["issue_frame_sha256"] == digest(value["issue_frame"])
    assert scope["question_sha256"] == sha256(QUESTION["question"].encode("utf8")).hexdigest()
    assert scope["semantic_target_verified"] is False
    assert scope["missing_fields_fabricated"] is False
    assert value == before


@pytest.mark.parametrize("kind", ["fact", "official_finding", "causal", "normative", "attribution"])
def test_claim_kind_does_not_override_model_direction_or_make_unknown_a_counterfact(kind):
    value = wire(); value["issue_frame"]["claim_kind"] = kind
    value["issue_frame"]["proposition"] = "现有材料尚未证实官方已认定事故因果关系。"
    value["relations"] = dict.fromkeys(ALIASES, "U"); value["judgments"] = []
    output, scope = decode(value)
    assert output["supporting_evidence_ids"] == output["conflicting_evidence_ids"] == []
    assert scope["issue_frame"]["claim_kind"] == kind


@pytest.mark.parametrize("aliases", [[], ["E1", "E2"]])
def test_no_usable_evidence_all_unused_undetermined_is_valid_without_fabrication(aliases):
    value = wire(); value["judgments"] = []; value["relations"] = dict.fromkeys(aliases, "U")
    output, scope = decode(value, aliases)
    assert output["judgments"] == output["conflicts"] == []
    assert output["supporting_evidence_ids"] == output["conflicting_evidence_ids"] == []
    assert scope["unused_for_principal_claim_evidence_ids"] == aliases


@pytest.mark.parametrize("code,valid", [("S", True), ("QS", True), ("O", False), ("QO", False), ("U", False)])
def test_asserted_still_requires_support_without_forcing_opposition(code, valid):
    value = wire(); value["decision"] = "asserted"; value["judgments"] = []
    value["relations"] = {"E1": code}
    if valid:
        output, _ = decode(value, ["E1"])
        assert output["supporting_evidence_ids"] == ["E1"] and not output["conflicting_evidence_ids"]
    else:
        with pytest.raises(ScopedArbitrationError, match="assertion_without_support"):
            decode(value, ["E1"])


def test_selection_stance_h1_and_use_type_are_not_final_role_authority():
    evidence = [{"id": ref, "stance": "opposes_hypothesis", "use_type": "background", "hypothesis_id": "H1"}
        for ref in ALIASES]
    before = deepcopy(evidence)
    output, scope = decode(aliases=evidence)
    assert output["supporting_evidence_ids"] == ["E1", "E3"] and output["conflicting_evidence_ids"] == ["E2", "E4"]
    assert scope["selection_stances_used_for_mapping"] is False and evidence == before


@pytest.mark.parametrize("change", ["missing_frame", "old_claim", "extra_frame", "missing_scope", "unknown_kind",
    "empty_anchor", "punctuation_anchor", "not_literal_anchor", "empty_proposition", "bare_unknown", "empty_scope",
    "missing_E", "extra_E", "old_qualification", "unknown_code", "integer_code", "old_arrays", "missing_judgments", "missing_conflicts"])
def test_invalid_or_incomplete_wire_is_rejected_without_repairs(change):
    value = wire(); frame = value["issue_frame"]
    if change == "missing_frame": value.pop("issue_frame")
    elif change == "old_claim": value["principal_claim"] = frame["proposition"]
    elif change == "extra_frame": frame["verified"] = True
    elif change == "missing_scope": frame.pop("scope")
    elif change == "unknown_kind": frame["claim_kind"] = "true"
    elif change == "empty_anchor": frame["question_anchor"] = " "
    elif change == "punctuation_anchor": frame["question_anchor"] = "？"
    elif change == "not_literal_anchor": frame["question_anchor"] = "企业承担事故的责任"
    elif change == "empty_proposition": frame["proposition"] = "😀；。"
    elif change == "bare_unknown": frame["proposition"] = "不可判定。"
    elif change == "empty_scope": frame["scope"] = "  \r\n"
    elif change == "missing_E": value["relations"].pop("E5")
    elif change == "extra_E": value["relations"]["E6"] = "U"
    elif change == "old_qualification": value["relations"]["E3"] = "qualification"
    elif change == "unknown_code": value["relations"]["E3"] = "A1"
    elif change == "integer_code": value["relations"]["E3"] = 1
    elif change == "old_arrays": value["conflicting_evidence_ids"] = ["E2"]
    elif change == "missing_judgments": value.pop("judgments")
    elif change == "missing_conflicts": value.pop("conflicts")
    before = deepcopy(value)
    with pytest.raises(INVALID): decode(value)
    assert value == before


def test_literal_question_anchor_preserves_unicode_and_crlf_without_whitespace_normalization():
    question = "😀企业\r\n事故责任，还是监管责任？"
    value = wire(); value["issue_frame"]["question_anchor"] = "企业\r\n事故责任"
    _, scope = decode(value, question=question)
    assert scope["question_sha256"] == sha256(question.encode("utf8")).hexdigest()
    value["issue_frame"]["question_anchor"] = "企业\n事故责任"
    with pytest.raises(ScopedArbitrationError, match="anchor_not_literal"): decode(value, question=question)


@pytest.mark.parametrize("question", [None, "", " \n", {}, {"question": 7}, {"question": ""}])
def test_v2_requires_original_question_for_unbound_preparation_and_dictionary_decode(question):
    with pytest.raises(ScopedArbitrationError, match="original_question_required"):
        prepare_scoped_arbitration(ArbitrationOutput, ALIASES, question=question, issue_frame=True, conclusion_aligned=False)
    with pytest.raises(ScopedArbitrationError, match="original_question_required"):
        decode(question=question)


def test_typed_v2_binds_question_order_and_mode_and_preserves_custom_base_fields():
    class Extended(ArbitrationOutput):
        audit_note: str = Field(min_length=3)
    schema, _ = prepare_scoped_arbitration(Extended, ALIASES, question=QUESTION, issue_frame=True, conclusion_aligned=False)
    value = wire(); value["audit_note"] = "原有自定义字段必须保留。"
    typed = schema.model_validate(value)
    output, scope = canonical_scoped_arbitration(typed, ALIASES)
    assert output["audit_note"] == value["audit_note"] and scope["format"] == ISSUE_FRAME_VERSION
    assert canonical_scoped_arbitration(typed, ALIASES, question={**QUESTION, "id": "different-internal-id"}) == (output, scope)
    with pytest.raises(ScopedArbitrationError, match="typed_question_binding_changed"):
        canonical_scoped_arbitration(typed, ALIASES, question=QUESTION["question"] + " ")
    with pytest.raises(ScopedArbitrationError, match="typed_evidence_order_changed"):
        canonical_scoped_arbitration(typed, list(reversed(ALIASES)))
    value.pop("audit_note")
    with pytest.raises(ValidationError): schema.model_validate(value)


def test_v1_typed_value_cannot_be_reinterpreted_as_v2_and_v2_dictionary_does_not_autoupgrade():
    legacy = wire(); legacy["principal_claim"] = legacy.pop("issue_frame")["proposition"]
    legacy["relations"] = dict.fromkeys(ALIASES, "unused")
    schema, _ = prepare_scoped_arbitration(ArbitrationOutput, ALIASES)
    typed = schema.model_validate(legacy)
    with pytest.raises(ScopedArbitrationError, match="typed_protocol_binding_changed"):
        canonical_scoped_arbitration(typed, ALIASES, question=QUESTION, issue_frame=True, conclusion_aligned=False)
    with pytest.raises(ValidationError): canonical_scoped_arbitration(wire(), ALIASES, question=QUESTION)


def test_wire_relation_key_order_and_published_row_order_do_not_change_bound_model_ledger():
    value = wire(); value["relations"] = dict(reversed(list(value["relations"].items())))
    assert decode(value) == decode()
    output, scope = decode()
    assert validate_scoped_result(scope, output, list(reversed(ALIASES)), question=QUESTION) == scope
    scope["evidence_order"].reverse()
    with pytest.raises(ScopedArbitrationError, match="evidence_order_changed"):
        validate_scoped_result(scope, output, ALIASES, question=QUESTION)


@pytest.mark.parametrize("change", ["extra_key", "missing_key", "frame", "principal_claim", "frame_hash", "question_hash",
    "codebook", "codebook_version", "support_codes", "conflict_codes", "target", "order", "order_hash", "missing_role", "extra_role",
    "role", "qualification", "qualification_support", "qualification_oppose", "unused", "h1_origin", "selection_origin",
    "role_origin", "semantic_verified", "fabricated", "output_support", "output_conflict", "output_duplicate", "output_missing"])
def test_publish_scope_replay_rejects_tampered_frame_codebook_inventory_and_canonical_relations(change):
    output, scope = decode()
    if change == "extra_key": scope["unverified"] = True
    elif change == "missing_key": scope.pop("evidence_order")
    elif change == "frame": scope["issue_frame"]["scope"] += "改变"
    elif change == "principal_claim": scope["principal_claim"] += "改变"
    elif change == "frame_hash": scope["issue_frame_sha256"] = "0" * 64
    elif change == "question_hash": scope["question_sha256"] = "0" * 64
    elif change == "codebook": scope["codebook"]["QO"] = "support_target_proposition"
    elif change == "codebook_version": scope["codebook_version"] = "v0"
    elif change == "support_codes": scope["support_role_codes"] = ["S", "QS", "QO"]
    elif change == "conflict_codes": scope["conflict_role_codes"] = ["O"]
    elif change == "target": scope["target"] = "all_evidence_supports_unknown"
    elif change == "order": scope["evidence_order"].reverse()
    elif change == "order_hash": scope["evidence_order_sha256"] = "0" * 64
    elif change == "missing_role": scope["evidence_roles"].pop("E3")
    elif change == "extra_role": scope["evidence_roles"]["E6"] = "U"
    elif change == "role": scope["evidence_roles"]["E4"] = "qualification"
    elif change == "qualification": scope["qualification_evidence_ids"] = ["E3"]
    elif change == "qualification_support": scope["qualification_support_evidence_ids"] = ["E4"]
    elif change == "qualification_oppose": scope["qualification_oppose_evidence_ids"] = ["E3"]
    elif change == "unused": scope["unused_for_principal_claim_evidence_ids"] = []
    elif change == "h1_origin": scope["h1_role"] = "final_truth"
    elif change == "selection_origin": scope["selection_stances_used_for_mapping"] = True
    elif change == "role_origin": scope["role_origin"] = "analyst_A"
    elif change == "semantic_verified": scope["semantic_target_verified"] = True
    elif change == "fabricated": scope["missing_fields_fabricated"] = True
    elif change == "output_support": output["supporting_evidence_ids"] = ["E1", "E3", "E4"]
    elif change == "output_conflict": output["conflicting_evidence_ids"] = ["E2"]
    elif change == "output_duplicate": output["supporting_evidence_ids"].append("E1")
    elif change == "output_missing": output.pop("judgments")
    with pytest.raises(INVALID): validate_scoped_result(scope, output, ALIASES, question=QUESTION)


def test_question_replay_checks_literal_anchor_even_when_attacker_rehashes_changed_frame():
    output, scope = decode()
    scope["issue_frame"]["question_anchor"] = "企业已经完全免责"
    scope["issue_frame_sha256"] = digest(scope["issue_frame"])
    with pytest.raises(ScopedArbitrationError, match="anchor_not_literal"):
        validate_scoped_result(scope, output, ALIASES, question=QUESTION)
    output, scope = decode()
    with pytest.raises(ScopedArbitrationError, match="question_binding_changed"):
        validate_scoped_result(scope, output, ALIASES, question=QUESTION["question"] + " ")


@pytest.mark.parametrize("change", ["foreign_A", "foreign_B", "analyst_evidence", "cycle", "unknown_E", "duplicate_E",
    "duplicate_judgment", "ungrounded", "conflict_overlap", "conflict_foreign", "resolved_both_supported"])
def test_independent_role_and_grounding_boundaries_survive_v2(change):
    value = wire(); value["conflicts"] = [conflict()]
    if change == "foreign_A": value["judgments"][1]["prerequisite_judgment_ids"] = ["A1"]
    elif change == "foreign_B": value["judgments"][1]["prerequisite_judgment_ids"] = ["B1"]
    elif change == "analyst_evidence": value["judgments"][0]["evidence_ids"] = ["A1"]
    elif change == "cycle": value["judgments"][0]["prerequisite_judgment_ids"] = ["J2"]
    elif change == "unknown_E": value["judgments"][0]["evidence_ids"] = ["E99"]
    elif change == "duplicate_E": value["judgments"][0]["evidence_ids"] = ["E1", "E1"]
    elif change == "duplicate_judgment": value["judgments"].append(deepcopy(value["judgments"][0]))
    elif change == "ungrounded": value["judgments"][0]["evidence_ids"] = []
    elif change == "conflict_overlap": value["conflicts"][0]["conflicting_evidence_ids"] = ["E1"]
    elif change == "conflict_foreign": value["conflicts"][0]["conflicting_evidence_ids"] = ["B1"]
    elif change == "resolved_both_supported":
        value["conflicts"][0]["status"] = "resolved"; value["relations"]["E2"] = "QS"
    with pytest.raises(INVALID): decode(value)


def test_final_scope_replay_rechecks_dag_and_accepts_real_service_ids_and_unrelated_metadata():
    output, scope = decode()
    mapping = {ref: f"evidence-service-uuid-{i}" for i, ref in enumerate(ALIASES)}
    def remap(value):
        if isinstance(value, str): return mapping.get(value, value)
        if isinstance(value, list): return [remap(row) for row in value]
        if isinstance(value, dict): return {mapping.get(key, key): remap(row) for key, row in value.items()}
        return value
    output, scope = remap(output), remap(scope)
    scope["evidence_order_sha256"] = digest(scope["evidence_order"])
    output["judgments"][0]["id"] = "service-judgment-1"
    output["judgments"][1]["prerequisite_judgment_ids"] = ["service-judgment-1"]
    output["snapshot_id"] = "service-snapshot"
    before = deepcopy((output, scope))
    assert validate_scoped_result(scope, output, list(mapping.values()), question=QUESTION) == scope
    assert (output, scope) == before
    output["judgments"][1]["prerequisite_judgment_ids"] = ["A1"]
    with pytest.raises(ScopedArbitrationError, match="unknown_or_duplicate_prerequisite"):
        validate_scoped_result(scope, output, list(mapping.values()), question=QUESTION)


def test_v2_codebook_is_complete_and_self_declared_question_digest_cannot_authorize_replay():
    output, scope = decode()
    assert scope["codebook"] == ISSUE_CODEBOOK
    assert scope["qualification_evidence_ids"] == ["E3", "E4"]
    assert scope["evidence_order"] == ALIASES
    assert scope["evidence_order_sha256"] == digest(ALIASES)
    with pytest.raises(ScopedArbitrationError, match="original_question_required"):
        validate_scoped_result(scope, output, ALIASES)
    scope["issue_frame"]["question_anchor"] = "完全不在原题中的主张"
    scope["issue_frame_sha256"] = digest(scope["issue_frame"])
    scope["question_sha256"] = "0" * 64
    with pytest.raises(ScopedArbitrationError, match="original_question_required"):
        validate_scoped_result(scope, output, ALIASES)


@pytest.mark.parametrize("value", [1, "true", None])
def test_protocol_optin_is_an_explicit_boolean(value):
    with pytest.raises(ScopedArbitrationError, match="boolean_required"):
        prepare_scoped_arbitration(ArbitrationOutput, ALIASES, question=QUESTION, issue_frame=value)
    with pytest.raises(ScopedArbitrationError, match="boolean_required"):
        canonical_scoped_arbitration(wire(), ALIASES, question=QUESTION, issue_frame=value)
