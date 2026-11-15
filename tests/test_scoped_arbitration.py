"""Synthetic strict wire/claim relation tests, with no model API or references."""
from copy import deepcopy

from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, ValidationError
import pytest

from app.agents.schemas import ArbitrationOutput
from app.agents.scoped_arbitration import (
    ScopedArbitrationError, canonical_scoped_arbitration, prepare_scoped_arbitration, validate_scoped_result,
)


def wire():
    return {"decision": "undetermined", "conclusion": "公开说法仍有分歧；正式验收尚未获程序记录证实。",
        "principal_claim": "公开材料尚不能证实正式验收已经完成，已知试开放不等同完成验收。",
        "judgments": [{"id": "J1", "statement": "试开放记录没有表明正式验收已完成。",
                       "evidence_ids": ["E1"], "prerequisite_judgment_ids": []}],
        "conflicts": [], "relations": {"E1": "support_principal_claim", "E2": "oppose_principal_claim", "E3": "qualification"}}


def conflict(status="unresolved"):
    return {"id":"C1", "statement":"是否完成正式验收存在两种说法。", "status":status,
        "supporting_evidence_ids":["E1"], "conflicting_evidence_ids":["E2"], "resolution_type":"Other",
        "basis":"一方仅称试开放，另一方声称已验收但未提供记录。", "accepted_reason":"保留程序状态限定。",
        "rejected_reason":"对方说法尚无对应程序记录。", "minority_opinion":"保留已经完成验收的说法。",
        "reopen_conditions":"出现可核实的验收记录。"}


def test_strict_dynamic_tool_requires_every_alias_and_retains_original_nested_fields():
    schema, instructions = prepare_scoped_arbitration(ArbitrationOutput, ["E1", "E2", "E3"])
    tool = convert_to_openai_tool(schema, strict=True)
    properties = tool["function"]["parameters"]["properties"]
    assert "supporting_evidence_ids" not in properties and "conflicting_evidence_ids" not in properties
    assert {"principal_claim", "relations", "decision", "conclusion", "judgments", "conflicts"} == set(properties)
    assert set(properties["relations"]["required"]) == {"E1", "E2", "E3"}
    assert properties["relations"]["additionalProperties"] is False
    assert "前提链" in instructions and "H1" in instructions and "undetermined" in instructions
    assert "不输出这两个列表" in instructions and "conflicts内部" in instructions
    assert "conclusion应自然说明同一实质主张" in instructions and "不要求强行选边" in instructions
    assert schema.model_validate(wire()).model_dump() == wire()


def test_undetermined_preserves_opposition_to_substantive_claim_instead_of_supporting_unknown():
    value = wire(); value["conflicts"] = [conflict()]
    output, scope = canonical_scoped_arbitration(value, ["E1", "E2", "E3"])
    assert output["decision"] == "undetermined"
    assert output["supporting_evidence_ids"] == ["E1", "E3"]
    assert output["conflicting_evidence_ids"] == ["E2"]
    assert output["conflicts"] == value["conflicts"]
    assert scope["principal_claim"] == value["principal_claim"]
    assert scope["qualification_evidence_ids"] == ["E3"]
    assert scope["semantic_target_verified"] is False


def test_h1_opposition_cannot_be_copied_to_final_conflict_role_when_claim_direction_reverses():
    evidence = [{"id":"E1", "stance":"opposes_hypothesis", "hypothesis_id":"H1"},
                {"id":"E2", "stance":"supports_hypothesis", "hypothesis_id":"H1"}, {"id":"E3"}]
    value = wire(); value["principal_claim"] = "原题假设的正式验收未完成，目前只有试开放记录。"
    before = deepcopy(evidence)
    output, scope = canonical_scoped_arbitration(value, evidence)
    assert output["supporting_evidence_ids"] == ["E1", "E3"] and output["conflicting_evidence_ids"] == ["E2"]
    assert scope["selection_stances_used_for_mapping"] is False and evidence == before


def test_qualification_supports_claim_boundary_without_inventing_counterevidence():
    value = wire(); value["relations"]["E2"] = "unused"
    output, scope = canonical_scoped_arbitration(value, ["E1", "E2", "E3"])
    assert output["conflicting_evidence_ids"] == []
    assert output["supporting_evidence_ids"] == ["E1", "E3"]
    assert scope["unused_for_principal_claim_evidence_ids"] == ["E2"]


@pytest.mark.parametrize("aliases", [[], ["E1", "E2"]])
def test_undetermined_all_unused_or_no_evidence_is_allowed_with_specific_gap_claim(aliases):
    value = {"decision":"undetermined", "conclusion":"缺少正式验收记录，不能核实是否已完成该程序。",
        "principal_claim":"现有材料未提供可核实的正式验收记录，因此程序状态仍未获证实。",
        "judgments":[], "conflicts":[], "relations":{alias:"unused" for alias in aliases}}
    output, scope = canonical_scoped_arbitration(value, aliases)
    assert output["supporting_evidence_ids"] == output["conflicting_evidence_ids"] == []
    assert scope["evidence_roles"] == value["relations"]


@pytest.mark.parametrize("change", ["missing_key", "unknown_key", "invalid_role", "missing_claim", "empty_claim", "bare_unknown", "missing_conflicts", "missing_judgments", "old_arrays", "extra_field"])
def test_wire_does_not_invent_missing_fields_or_silently_filter_unknowns(change):
    value = wire()
    if change == "missing_key": value["relations"].pop("E2")
    elif change == "unknown_key": value["relations"]["E4"] = "unused"
    elif change == "invalid_role": value["relations"]["E1"] = "supports_hypothesis"
    elif change == "missing_claim": value.pop("principal_claim")
    elif change == "empty_claim": value["principal_claim"] = "   "
    elif change == "bare_unknown": value["principal_claim"] = "不可判定。"
    elif change == "missing_conflicts": value.pop("conflicts")
    elif change == "missing_judgments": value.pop("judgments")
    elif change == "old_arrays": value["supporting_evidence_ids"] = ["E1"]
    elif change == "extra_field": value["invented"] = "not allowed"
    with pytest.raises((ValidationError, ScopedArbitrationError)):
        canonical_scoped_arbitration(value, ["E1", "E2", "E3"])


@pytest.mark.parametrize("change", ["ungrounded", "unknown_evidence", "duplicate_evidence", "unknown_prerequisite", "cycle", "duplicate_judgment", "missing_prerequisite_field", "asserted_unused"])
def test_original_reference_and_grounding_invariants_remain_enforced(change):
    value = wire()
    if change == "ungrounded": value["judgments"][0]["evidence_ids"] = []
    elif change == "unknown_evidence": value["judgments"][0]["evidence_ids"] = ["E99"]
    elif change == "duplicate_evidence": value["judgments"][0]["evidence_ids"] = ["E1", "E1"]
    elif change == "unknown_prerequisite": value["judgments"][0]["prerequisite_judgment_ids"] = ["J99"]
    elif change == "cycle": value["judgments"][0]["prerequisite_judgment_ids"] = ["J1"]
    elif change == "duplicate_judgment": value["judgments"].append(deepcopy(value["judgments"][0]))
    elif change == "missing_prerequisite_field": value["judgments"][0].pop("prerequisite_judgment_ids")
    elif change == "asserted_unused":
        value["decision"] = "asserted"; value["relations"] = {alias:"unused" for alias in value["relations"]}
    with pytest.raises((ValidationError, ScopedArbitrationError)):
        canonical_scoped_arbitration(value, ["E1", "E2", "E3"])


def test_valid_grounded_dependency_chain_and_resolved_conflict_remain_unchanged():
    value = wire(); value["decision"] = "asserted"; value["conflicts"] = [conflict("resolved")]
    value["judgments"].append({"id":"J2", "statement":"该程序的完成状态不能由试开放替代证明。",
                                "evidence_ids":[], "prerequisite_judgment_ids":["J1"]})
    original = deepcopy(value)
    output, _ = canonical_scoped_arbitration(value, ["E1", "E2", "E3"])
    assert output["judgments"] == original["judgments"] and output["conflicts"] == original["conflicts"]
    assert value == original


@pytest.mark.parametrize("change", ["overlap", "unknown", "duplicate", "missing_side", "ambiguous_resolved", "duplicate_conflict"])
def test_original_conflict_schema_and_side_checks_are_preserved(change):
    value = wire(); value["conflicts"] = [conflict()]
    row = value["conflicts"][0]
    if change == "overlap": row["conflicting_evidence_ids"] = ["E1"]
    elif change == "unknown": row["conflicting_evidence_ids"] = ["E99"]
    elif change == "duplicate": row["supporting_evidence_ids"] = ["E1", "E1"]
    elif change == "missing_side": row.pop("conflicting_evidence_ids")
    elif change == "ambiguous_resolved": row["status"] = "resolved"; value["relations"]["E2"] = "support_principal_claim"
    elif change == "duplicate_conflict": value["conflicts"].append(deepcopy(row))
    with pytest.raises((ValidationError, ScopedArbitrationError)):
        canonical_scoped_arbitration(value, ["E1", "E2", "E3"])


@pytest.mark.parametrize("aliases", [["E1", "E1"], ["doc_1"], ["E0"], ["E01"], [{"id":"E1"}, {}], {"E1":"uuid"}])
def test_alias_identity_is_explicit_unique_and_not_inferred(aliases):
    with pytest.raises(ScopedArbitrationError): prepare_scoped_arbitration(ArbitrationOutput, aliases)


def test_original_base_field_constraints_are_preserved_for_typed_custom_wire():
    class CustomArbitration(ArbitrationOutput):
        audit_note: str = Field(min_length=3)
    schema, _ = prepare_scoped_arbitration(CustomArbitration, ["E1", "E2", "E3"])
    value = wire(); value["audit_note"] = "原有必需字段。"
    typed = schema.model_validate(value)
    output, _ = canonical_scoped_arbitration(typed, ["E1", "E2", "E3"])
    assert output["audit_note"] == value["audit_note"]
    value.pop("audit_note")
    with pytest.raises(ValidationError): schema.model_validate(value)


def test_scope_replay_accepts_actual_evidence_ids_and_mapped_judgment_ids_without_mutation():
    output, scope = canonical_scoped_arbitration(wire(), ["E1", "E2", "E3"])
    mapping = {"E1":"real-evidence-1", "E2":"real-evidence-2", "E3":"real-evidence-3"}
    scope["evidence_roles"] = {mapping[key]:value for key,value in scope["evidence_roles"].items()}
    for field in ("qualification_evidence_ids", "unused_for_principal_claim_evidence_ids"):
        scope[field] = [mapping[ref] for ref in scope[field]]
    for field in ("supporting_evidence_ids", "conflicting_evidence_ids"):
        output[field] = [mapping[ref] for ref in output[field]]
    output["judgments"][0]["id"] = "service-allocated-judgment-uuid"
    original = deepcopy((scope, output))
    assert validate_scoped_result(scope, output, list(mapping.values())) == scope
    assert (scope, output) == original


@pytest.mark.parametrize("change", ["version", "missing_claim", "empty_claim", "target", "missing_role", "extra_role", "unknown_role", "missing_s", "duplicate_s", "unknown_c", "overlap", "role_disagrees_s", "qualifier_inventory", "unused_inventory"])
def test_publish_scope_replay_refuses_scope_or_final_relation_drift(change):
    output, scope = canonical_scoped_arbitration(wire(), ["E1", "E2", "E3"])
    if change == "version": scope["format"] = "other-v0"
    elif change == "missing_claim": scope.pop("principal_claim")
    elif change == "empty_claim": scope["principal_claim"] = " "
    elif change == "target": scope["target"] = "undetermined_global_label"
    elif change == "missing_role": scope["evidence_roles"].pop("E2")
    elif change == "extra_role": scope["evidence_roles"]["E99"] = "unused"
    elif change == "unknown_role": scope["evidence_roles"]["E1"] = "selection_oppose"
    elif change == "missing_s": output.pop("supporting_evidence_ids")
    elif change == "duplicate_s": output["supporting_evidence_ids"].append("E1")
    elif change == "unknown_c": output["conflicting_evidence_ids"] = ["E99"]
    elif change == "overlap": output["conflicting_evidence_ids"] = ["E1"]
    elif change == "role_disagrees_s": output["supporting_evidence_ids"] = ["E1"]
    elif change == "qualifier_inventory": scope["qualification_evidence_ids"] = []
    elif change == "unused_inventory": scope["unused_for_principal_claim_evidence_ids"] = ["E2"]
    with pytest.raises(ScopedArbitrationError):
        validate_scoped_result(scope, output, ["E1", "E2", "E3"])


def test_scope_replay_cannot_turn_valid_all_unused_undetermined_into_unsupported_assertion():
    value = wire(); value["judgments"] = []; value["relations"] = {alias:"unused" for alias in value["relations"]}
    output, scope = canonical_scoped_arbitration(value, ["E1", "E2", "E3"])
    output["decision"] = "asserted"
    with pytest.raises(ScopedArbitrationError, match="assertion_without_support"):
        validate_scoped_result(scope, output, ["E1", "E2", "E3"])


async def test_provider_glue_strict_wire_and_scope_aliases_roundtrip_through_http_mock(tmp_path, monkeypatch):
    """Run installed LC+provider glue with an explicit mock, never provider I/O."""
    import json
    import httpx
    from app.agents import strict_transport as transport
    from app.config import Settings
    requests = []
    def handle(request):
        body = json.loads(request.content); requests.append(body)
        name = body["tools"][0]["function"]["name"]
        return httpx.Response(200, json={"id":"mock-scoped", "object":"chat.completion", "created":1,
            "model":"deepseek-v4-pro", "choices":[{"index":0, "finish_reason":"tool_calls",
                "message":{"role":"assistant", "content":"", "tool_calls":[{"id":"mock-call", "type":"function",
                    "function":{"name":name, "arguments":json.dumps(wire(), ensure_ascii=False)}}]}}],
            "usage":{"prompt_tokens":100, "completion_tokens":20, "total_tokens":120}})
    monkeypatch.setattr(transport.httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handle))
    class ExplicitMockAudit(transport.AuditedTransport, httpx.MockTransport):
        pass
    monkeypatch.setattr(transport, "AuditedTransport", ExplicitMockAudit)
    monkeypatch.setattr(transport, "count_wire", lambda body, directory, ordinal:
        (100, transport.freeze(directory / f"request-{ordinal:02d}-wire.json", body)))
    adapter = transport.StrictWorkbenchProvider(Settings(_env_file=None, data_dir=tmp_path,
        model_name="deepseek-v4-pro", model_provider="deepseek", model_api_key="offline-mock-only",
        model_strict_evidence_enabled=True, strict_evidence_balanced_enabled=True))
    adapter.selection_hypotheses = [{"id":"H1", "claim":"正式验收已完成。"}]
    evidence = [{"id":f"real-evidence-{i}", "quote":"完整合成原段。", "fact_des":"程序状态的合成说明。",
                 "raw_doc_id":f"raw-doc-{i}", "raw_para_id":"p1", "source_group_id":f"group-{i}"}
                for i in range(1,4)]
    result = await adapter.arbitrate({"question":"正式验收已完成还是仅试开放？"}, evidence, [])
    assert len(requests) == 1
    tool = requests[0]["tools"][0]["function"]
    assert tool["name"] == "ScopedArbitrationWireOutput" and tool["strict"] is True
    assert set(tool["parameters"]["properties"]["relations"]["required"]) == {"E1","E2","E3"}
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert [row["id"] for row in payload["evidence"]] == ["E1","E2","E3"]
    assert result["supporting_evidence_ids"] == ["real-evidence-1", "real-evidence-3"]
    assert result["conflicting_evidence_ids"] == ["real-evidence-2"]
    scope = adapter.arbitration_scope
    assert set(scope["evidence_roles"]) == {row["id"] for row in evidence}
    assert scope["qualification_evidence_ids"] == ["real-evidence-3"]
    validate_scoped_result(scope, result, evidence)
    audit = json.loads((adapter.directory / "request-01-relation-projection.json").read_text("utf-8"))
    assert audit["scope"] == scope and audit["new_model_calls"] == 0
