"""Synthetic-only strict schema and fixed-graph tests; no real network/model."""
from copy import deepcopy
import json
import socket

import httpx
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from typing import Literal

from app.agents.strict_evidence import (
    StrictAnnotationOutput, StrictEvidenceError, StrictSelectionOutput,
    build_proposal_artifact, build_strict_graph, prepare_annotation_request,
    prepare_selection_request, validate_annotation, validate_proposal, validate_raw_case, validate_selection,
    canonical_selection_wire, canonical_annotation_wire,
    annotation_time_projection,
)


def raw():
    return {"question": {"question_id": "q_synthetic", "question": "场馆已开放，还是仍在验收？"}, "catalog": [
        {"paragraph_ref": "P1", "doc_id": "d1", "para_id": "p1", "title": "原始记录 {{literal}}", "text": "\n😀周五试开放，正式验收尚未完成。\r\n", "paragraph_complete": True, "parent_complete": True},
        {"paragraph_ref": "P2", "doc_id": "d2", "para_id": "p3", "title": "竞争说法", "text": "有人声称全部手续已完成，有关部门未确认此说法。", "paragraph_complete": True, "parent_complete": True},
        {"paragraph_ref": "P3", "doc_id": "d3", "para_id": "p2", "title": "背景", "text": "场馆外墙蓝色。", "paragraph_complete": True, "parent_complete": True},
    ], "source_metadata": {"P1": {"source_credibility": .9, "parent_start_cp": 0, "parent_end_cp": len("\n😀周五试开放，正式验收尚未完成。\r\n")}}}


def selection():
    return {"hypotheses": [{"id": "H1", "claim": "场馆正式验收已完成。"}], "decisions": [
        {"p": "P1", "relevance": 3, "stance": "opposes_hypothesis", "hypothesis_id": "H1"},
        {"p": "P2", "relevance": 3, "stance": "supports_hypothesis", "hypothesis_id": "H1"},
        {"p": "P3", "relevance": 1, "stance": "qualification", "hypothesis_id": "H1"},
    ]}


def annotation():
    return {"annotations": [
        {"p": "P1", "fact_des": "场馆周五试开放，正式验收尚未完成。", "argument": {"时间": "周五", "主体": "场馆", "地点": "", "内容": "试开放不代表完成正式验收。", "影响": "", "数值口径": ""}, "entities": ["场馆"]},
        {"p": "P2", "fact_des": "有人声称手续全部完成，但有关部门未确认。", "argument": {"时间": "", "主体": "有关部门", "地点": "", "内容": "有人声称全部手续已完成，有关部门未确认此说法。", "影响": "", "数值口径": ""}, "entities": ["有关部门"]},
    ], "conclusion": "公开信息存在未确认的相反说法，不能认定正式验收已完成。", "verdict_label": "undetermined",
        "supporting": [], "conflicting": [{"p": "P1", "resolution_type": "Other", "hypothesis_id": "H1"}], "basis": "对H1，试开放与正式验收是不同程序，另一说法尚未确认。"}


def metadata():
    return {"annotations": annotation()["annotations"]}


def wire_selection():
    return {"hypothesis": selection()["hypotheses"][0]["claim"],
            "decisions": {"P1": "3_oppose", "P2": "3_support", "P3": "1"}}


def wire_metadata():
    return {"annotations": {row["p"]: {key: value for key, value in row.items() if key != "p"} for row in metadata()["annotations"]}}


def test_both_stages_keep_complete_original_sources_and_do_not_leak_unselected_paragraphs():
    value = raw(); schema, messages = prepare_selection_request(value)
    payload = json.loads(messages[1].content)
    assert set(schema.model_fields) == {"hypothesis", "decisions"}
    assert payload["question"] == value["question"]["question"]
    assert [p["text"] for p in payload["paragraphs"]] == [p["text"] for p in value["catalog"]]
    assert payload["paragraphs"][0]["title"] == value["catalog"][0]["title"]
    schema, messages = prepare_annotation_request(value, selection())
    payload = json.loads(messages[1].content)
    assert schema.__name__ == "StrictAnnotationWireOutput"
    assert [p["p"] for p in payload["paragraphs"]] == ["P1", "P2"]
    assert [p["text"] for p in payload["paragraphs"]] == [p["text"] for p in value["catalog"][:2]]
    assert "场馆外墙蓝色。" not in messages[1].content
    assert "未确认" in messages[0].content and "独立研判及仲裁" in messages[0].content
    assert set(schema.model_fields) == {"annotations"}


@pytest.mark.parametrize("change,code", [
    (lambda value: value["catalog"][0].update(parent_complete=False), "complete_original_parent_required"),
    (lambda value: value["catalog"][1].update(paragraph_ref="P1"), "unique_original_source_required"),
    (lambda value: value["source_metadata"]["P1"].update(source_text_sha256_utf8="wrong"), "source_text_sha256_mismatch"),
    (lambda value: value["source_metadata"]["P1"].update(parent_end_cp=1), "source_parent_coordinates_mismatch"),
])
def test_original_identity_and_binding_validation(change, code):
    value = raw(); change(value)
    with pytest.raises(StrictEvidenceError, match=code): validate_raw_case(value)


@pytest.mark.parametrize("change,code", [
    (lambda value: value["decisions"].pop(), "selection_must_cover_each_candidate"),
    (lambda value: value["decisions"][1].update(p="P1"), "unknown_or_duplicate_selection_ref"),
    (lambda value: value["decisions"][0].update(p="P99"), "unknown_or_duplicate_selection_ref"),
    (lambda value: value["decisions"][0].update(hypothesis_id="H99"), "unknown_hypothesis_ref"),
    (lambda value: value.update(hypotheses=[]), "explicit_hypothesis_required"),
])
def test_selection_rejects_missing_duplicate_or_unknown_sources_and_targets(change, code):
    value = selection(); change(value)
    with pytest.raises(StrictEvidenceError, match=code): validate_selection(raw(), value)


@pytest.mark.parametrize("change,code", [
    (lambda value: value["annotations"].pop(), "metadata_must_cover_all_selected_paragraphs"),
    (lambda value: value["annotations"][1].update(p="P3"), "unknown_or_duplicate_annotation_ref"),
    (lambda value: value["annotations"][0].update(fact_des=" "), "nonempty_fact_des_required"),
    (lambda value: value["annotations"][0]["argument"].update(内容=""), "nonempty_argument_content_required"),
    (lambda value: value["annotations"][0]["argument"].update(时间="明年"), "time_must_be_literal_or_empty"),
    (lambda value: value["annotations"][0]["entities"].append("99人"), "annotation_number_not_in_source"),
])
def test_annotation_is_complete_source_local_and_relation_bound(change, code):
    value = metadata(); change(value)
    with pytest.raises(StrictEvidenceError, match=code): validate_annotation(raw(), selection(), value)


@pytest.mark.parametrize("change,code", [
    (lambda value: value.update(supporting=["P3"]), "unknown_or_duplicate_relation_ref"),
    (lambda value: value["conflicting"][0].update(hypothesis_id="H99"), "unknown_conflict_hypothesis_ref"),
    (lambda value: value.update(basis=""), "conflict_basis_required"),
    (lambda value: value.update(verdict_label="full_support"), "asserted_conclusion_requires_support"),
    (lambda value: value.update(conclusion="有99人到场。"), "conclusion_number_not_in_cited_sources"),
])
def test_isolated_proposal_validation_is_separate_from_production_metadata(change, code):
    value = annotation(); change(value)
    with pytest.raises(StrictEvidenceError, match=code): validate_proposal(raw(), selection(), value)


def test_program_owned_quote_codepoints_relations_and_undetermined_are_preserved():
    source = raw(); original = deepcopy(source); proposed = annotation()
    result = build_proposal_artifact(source, selection(), proposed)
    assert result["format"] == "strict-evidence-v1"
    assert result["question_id"] == "q_synthetic"
    assert source == original
    assert result["evidence"][0]["quote"] == source["catalog"][0]["text"]
    assert result["evidence"][0]["parent_end_cp"] == len(source["catalog"][0]["text"])
    assert result["conclusion"]["question_id"] == "q_synthetic"
    assert result["conclusion"]["supporting_facts"] == []
    assert result["conclusion"]["conflicting_facts"][0]["hypothesis_id"] == "H1"
    assert result["quality"]["metadata_complete"] is True
    assert result["quality"]["semantic_verified"] is False
    assert result["quality"]["publishable"] is False
    assert result["scope"]["source_context_delivery_verified"] is False


def test_fixed_graph_calls_each_stage_once_and_does_not_retry_failures():
    calls = []
    def invoke(stage, schema, messages):
        calls.append(stage)
        return schema.model_validate(wire_selection() if stage == "selection" else wire_metadata())
    result = build_strict_graph(invoke).invoke({"raw_case": raw()})
    assert calls == ["selection", "annotation"]
    assert result["annotation"] == metadata()
    assert "artifact" not in result and "conclusion" not in result["annotation"]
    calls.clear()
    def invalid(stage, _schema, _messages):
        calls.append(stage); return {"hypotheses": [], "decisions": []}
    with pytest.raises((StrictEvidenceError, ValidationError)): build_strict_graph(invalid).invoke({"raw_case": raw()})
    assert calls == ["selection"]


def test_all_wire_objects_have_required_fields_no_array_limits_and_no_model_owned_quote():
    def check(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                assert value["additionalProperties"] is False
                assert set(value["required"]) == set(value["properties"])
            assert not set(value) & {"minItems", "maxItems"}
            for child in value.values(): check(child)
        elif isinstance(value, list):
            for child in value: check(child)
    for schema in (StrictSelectionOutput, StrictAnnotationOutput):
        value = schema.model_json_schema(); check(value)
        assert '"quote"' not in json.dumps(value)
    missing = metadata(); del missing["annotations"][0]["argument"]["地点"]
    with pytest.raises(ValidationError): StrictAnnotationOutput.model_validate(missing)
    invalid_score = selection(); invalid_score["decisions"][0]["relevance"] = True
    with pytest.raises(ValidationError): StrictSelectionOutput.model_validate(invalid_score)


def test_required_key_wire_converts_only_declared_codes_and_preserves_all_metadata_without_repairs():
    assert canonical_selection_wire(raw(), wire_selection()).model_dump() == selection()
    assert canonical_annotation_wire(raw(), selection(), wire_metadata()).model_dump() == metadata()
    original = deepcopy(wire_metadata()); canonical_annotation_wire(raw(), selection(), original)
    assert original == wire_metadata()
    for mutation in [
        lambda value: value["decisions"].pop("P3"),
        lambda value: value["decisions"].update(P99="3_support"),
        lambda value: value["decisions"].update(P1="2"),
        lambda value: value["decisions"].update(P1=3),
    ]:
        value = wire_selection(); mutation(value)
        with pytest.raises(ValidationError): canonical_selection_wire(raw(), value)
    for mutation in [lambda value: value["annotations"].pop("P2"),
                     lambda value: value["annotations"].update(P3=value["annotations"]["P1"])]:
        value = wire_metadata(); mutation(value)
        with pytest.raises(ValidationError): canonical_annotation_wire(raw(), selection(), value)


def test_optional_time_projection_retains_literal_time_and_original_model_response():
    original = wire_metadata()
    projected, changes = annotation_time_projection(raw(), original)
    assert projected == original and projected is not original and changes == []
    assert projected["annotations"]["P1"]["argument"]["时间"] == "周五"
    schema, _messages = prepare_annotation_request(raw(), selection())
    parsed = schema.model_validate(original)
    projected, changes = annotation_time_projection(raw(), parsed)
    assert projected == original and changes == [] and parsed.model_dump() == original


@pytest.mark.parametrize("time", ["2026年周五", "试开放周五", "明年"])
def test_nonliteral_optional_time_alone_becomes_unknown_with_independent_audit(time):
    value = wire_metadata(); value["annotations"]["P1"]["argument"]["时间"] = time
    original = deepcopy(value)
    projected, changes = annotation_time_projection(raw(), value)
    expected = deepcopy(original); expected["annotations"]["P1"]["argument"]["时间"] = ""
    assert projected == expected and value == original
    assert changes == [{"p": "P1", "field": "argument.时间", "before": time, "after": "", "reason": "time_must_be_literal_or_empty"}]
    accepted = canonical_annotation_wire(raw(), selection(), value).model_dump()
    assert accepted["annotations"][0]["argument"]["时间"] == ""
    assert len(accepted["annotations"]) == 2
    assert accepted["annotations"][0]["fact_des"] == original["annotations"]["P1"]["fact_des"]
    assert accepted["annotations"][0]["argument"]["内容"] == original["annotations"]["P1"]["argument"]["内容"]
    assert value == original


def test_time_projection_does_not_hide_other_numeric_failures_or_missing_required_keys():
    value = wire_metadata(); value["annotations"]["P1"]["argument"]["时间"] = "2026年周五"
    value["annotations"]["P1"]["entities"].append("99人")
    projected, changes = annotation_time_projection(raw(), value)
    assert changes and projected["annotations"]["P1"]["entities"][-1] == "99人"
    with pytest.raises(StrictEvidenceError, match="annotation_number_not_in_source"):
        canonical_annotation_wire(raw(), selection(), value)
    value = wire_metadata(); value["annotations"]["P1"]["argument"]["时间"] = "明年"; value["annotations"].pop("P2")
    projected, _ = annotation_time_projection(raw(), value)
    assert set(projected["annotations"]) == {"P1"}
    with pytest.raises(ValidationError): canonical_annotation_wire(raw(), selection(), projected)
    value = wire_metadata(); value["annotations"]["P1"]["argument"]["时间"] = 2026
    projected, changes = annotation_time_projection(raw(), value)
    assert projected == value and changes == []
    with pytest.raises(ValidationError): canonical_annotation_wire(raw(), selection(), projected)


def test_dynamic_schema_has_every_source_required_and_compact_codes_reduce_expanded_tool_schema():
    value = raw(); value["source_metadata"] = {}
    value["catalog"] = [{**raw()["catalog"][0], "paragraph_ref": f"P{i}", "doc_id": f"d{i}", "para_id": f"p{i}"} for i in range(1, 61)]
    schema, _ = prepare_selection_request(value)
    tool = convert_to_openai_tool(schema, strict=True)
    decisions = tool["function"]["parameters"]["properties"]["decisions"]
    assert set(decisions["required"]) == {f"P{i}" for i in range(1, 61)}
    assert decisions["additionalProperties"] is False
    assert all(row["type"] == "string" and len(row["enum"]) == 8 for row in decisions["properties"].values())
    class VerboseValue(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        relevance: Literal[0, 1, 2, 3]
        stance: Literal["supports_hypothesis", "opposes_hypothesis", "qualification"]
        hypothesis_id: str
    verbose_map = create_model("VerboseMap", __config__=ConfigDict(extra="forbid", strict=True), **{f"P{i}": (VerboseValue, ...) for i in range(1, 61)})
    verbose = create_model("VerboseWire", __config__=ConfigDict(extra="forbid", strict=True), hypothesis=(str, ...), decisions=(verbose_map, ...))
    compact_chars = len(json.dumps(tool, ensure_ascii=False, separators=(",", ":")))
    verbose_chars = len(json.dumps(convert_to_openai_tool(verbose, strict=True), ensure_ascii=False, separators=(",", ":")))
    assert compact_chars < .7 * verbose_chars
    annotation_schema, _ = prepare_annotation_request(raw(), selection())
    annotation_tool = convert_to_openai_tool(annotation_schema, strict=True)["function"]["parameters"]["properties"]["annotations"]
    assert set(annotation_tool["required"]) == {"P1", "P2"} and annotation_tool["additionalProperties"] is False


def test_installed_langchain_dynamic_required_wire_rejects_missing_source_before_business(monkeypatch):
    def blocked(*_args, **_kwargs): pytest.fail("fixture must not open sockets")
    monkeypatch.setattr(socket.socket, "connect", blocked); monkeypatch.setattr(socket, "create_connection", blocked)
    schema, messages = prepare_selection_request(raw()); requests = []
    incomplete = wire_selection(); incomplete["decisions"].pop("P3")
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0, "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "mock-call", "type": "function", "function": {"name": schema.__name__, "arguments": json.dumps(incomplete)}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(model="deepseek-v4-pro", api_key="fixture-only", base_url="https://api.deepseek.com/beta", http_client=client,
                           max_retries=0, temperature=0, extra_body={"thinking": {"type": "disabled"}}, http_socket_options=())
        result = model.with_structured_output(schema, method="function_calling", strict=True, include_raw=True).invoke(messages)
    assert len(requests) == 1
    required = requests[0]["tools"][0]["function"]["parameters"]["properties"]["decisions"]["required"]
    assert set(required) == {"P1", "P2", "P3"}
    assert result["parsed"] is None and isinstance(result["parsing_error"], ValidationError)
    assert result["raw"].usage_metadata["total_tokens"] == 15


def test_installed_langchain_emits_provider_strict_schema_and_rejects_tuple_without_retry(monkeypatch):
    def blocked(*_args, **_kwargs): pytest.fail("fixture must not open sockets")
    monkeypatch.setattr(socket.socket, "connect", blocked); monkeypatch.setattr(socket, "create_connection", blocked)
    requests = []
    def respond(request):
        payload = json.loads(request.content); requests.append(payload)
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0, "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "mock-call", "type": "function", "function": {
                "name": "StrictSelectionOutput", "arguments": json.dumps({"hypotheses": [["H1", "项目完成"]], "decisions": []})}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(model="deepseek-v4-pro", api_key="fixture-only", base_url="https://api.deepseek.com/beta", http_client=client,
                           max_retries=0, temperature=0, extra_body={"thinking": {"type": "disabled"}}, http_socket_options=())
        result = model.with_structured_output(StrictSelectionOutput, method="function_calling", strict=True, include_raw=True).invoke("Return JSON.")
    assert len(requests) == 1
    assert requests[0]["tools"][0]["function"]["strict"] is True
    assert requests[0]["tool_choice"]["function"]["name"] == "StrictSelectionOutput"
    assert requests[0]["thinking"]["type"] == "disabled"
    assert result["parsed"] is None and isinstance(result["parsing_error"], ValidationError)
    assert result["raw"].usage_metadata["total_tokens"] == 15  # explicitly synthetic usage
