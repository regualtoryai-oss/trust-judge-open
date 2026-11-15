"""Actual installed LC serialization with HTTP MockTransport; no account use."""
import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.agents.provider import ModelOutputInvalid, capture_usage
from app.agents import strict_transport as transport
from app.config import Settings


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1)


def provider(tmp_path, monkeypatch, arguments, finish="tool_calls", *, tool_count=1, encoded_arguments=None, balanced=False):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        function_name = requests[-1]["tools"][0]["function"]["name"]
        return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 1,
            "model": "deepseek-v4-pro", "choices": [{"index": 0, "finish_reason": finish,
                "message": {"role": "assistant", "content": "", "tool_calls": [{"id": "call1", "type": "function",
                    "function": {"name": function_name, "arguments": json.dumps(arguments) if encoded_arguments is None else encoded_arguments}}] * tool_count}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
    monkeypatch.setattr(transport.httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handle))
    class ExplicitMockAudit(transport.AuditedTransport, httpx.MockTransport):
        pass
    monkeypatch.setattr(transport, "AuditedTransport", ExplicitMockAudit)
    monkeypatch.setattr(transport, "count_wire", lambda body, directory, ordinal:
        (100, transport.freeze(directory / f"request-{ordinal:02d}-wire.json", body)))
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="deepseek-v4-pro",
        model_provider="deepseek", model_api_key="test-only-placeholder", strict_evidence_balanced_enabled=balanced)
    return transport.StrictWorkbenchProvider(settings), requests


async def test_balanced_transport_keeps_complete_keys_and_integer_wire(tmp_path, monkeypatch):
    from test_compact_selection import raw
    payload = {"hypothesis":"甲设备已确认", "decisions":{f"P{i+1}":i for i in range(8)}}
    adapter, requests = provider(tmp_path, monkeypatch, payload, balanced=True)
    result = await adapter.strict_selection(raw())
    decisions = requests[0]["tools"][0]["function"]["parameters"]["properties"]["decisions"]
    assert decisions["required"] == [f"P{i+1}" for i in range(8)]
    assert all(item["type"] == "integer" for item in decisions["properties"].values())
    assert result["decisions"][6]["stance"] == "opposes_hypothesis"
    episode = json.loads((adapter.directory / "episode.json").read_text("utf8"))
    assert episode["balanced_scoped_enabled"] is True
    from pathlib import Path
    from hashlib import sha256
    for key, descriptor in episode["source_archive"].items():
        assert Path(descriptor["file"]).is_relative_to(adapter.directory)
        assert sha256(Path(descriptor["file"]).read_bytes()).hexdigest() == episode["sources"][key] == descriptor["sha256"]


async def test_scoped_transport_preserves_selection_context_and_remaps_role_keys(tmp_path, monkeypatch):
    from app.agents.schemas import ArbitrationOutput
    payload = {"principal_claim":"主管部门表示尚未确认。", "relations":{"E1":"oppose_principal_claim","E2":"qualification"},
        "decision":"undetermined", "conclusion":"主管部门表示尚未确认，相反说法待核。", "judgments":[], "conflicts":[]}
    adapter, requests = provider(tmp_path, monkeypatch, payload, balanced=True)
    adapter.selection_hypotheses = [{"id":"H1","claim":"设备已确认"}]
    evidence = [{"id":"real-one","quote":"已确认。","selection_relevance":3,
        "selection_stance":"supports_hypothesis","selection_hypothesis_id":"H1","selection_semantics_verified":False},
        {"id":"real-two","quote":"待确认。","selection_relevance":2,"selection_stance":"qualification"}]
    result = await adapter._request("arbitrate", "仲裁。", {"evidence":evidence,"question":{"question":"是否确认？"}}, ArbitrationOutput)
    assert result["conflicting_evidence_ids"] == ["real-one"] and result["supporting_evidence_ids"] == ["real-two"]
    assert adapter.arbitration_scope["evidence_roles"] == {"real-one":"oppose_principal_claim","real-two":"qualification"}
    body = requests[0]
    projected = json.loads(body["messages"][-1]["content"])
    assert projected["selection_hypotheses"] == adapter.selection_hypotheses
    assert projected["evidence"][0]["selection_semantics_verified"] is False
    assert projected["evidence"][0]["selection_stance"] == "supports_hypothesis"
    wire_props = body["tools"][0]["function"]["parameters"]["properties"]
    assert "relations" in wire_props and "supporting_evidence_ids" not in wire_props
    assert (adapter.directory / "request-01-relation-projection.json").is_file()


async def test_strict_actual_wire_has_named_tool_and_accounts_usage(tmp_path, monkeypatch):
    adapter, requests = provider(tmp_path, monkeypatch, {"text": "完整结果"})
    with capture_usage() as usage:
        result = await adapter.invoke_strict("strict_selection", Example, [{"role": "user", "content": "raw"}])
    assert result == {"text": "完整结果"}
    assert len(requests) == 1
    body = requests[0]
    assert body["tools"][0]["function"]["strict"] is True
    assert body["tool_choice"]["function"]["name"] == "Example"
    assert body["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert body["tools"][0]["function"]["parameters"]["required"] == ["text"]
    assert "minLength" not in json.dumps(body["tools"])
    assert body["thinking"] == {"type": "disabled"} and body["max_tokens"] == 4096
    assert usage[0]["total_tokens"] == adapter.guard.report()["known_total_tokens"] == 120
    assert adapter.guard.report()["unknown_usage_requests"] == 0
    from hashlib import sha256
    preflight = json.loads((adapter.directory / "request-01-preflight.json").read_text("utf8"))
    actual = (adapter.directory / "request-01-actual-wire.json").read_bytes()
    assert json.loads(actual) == body
    assert preflight["actual_wire"]["sha256"] == sha256(actual).hexdigest()
    assert not actual.endswith(b"\n")  # Preserve HTTP bytes, rather than the canonical audit JSON.
    assert "test-only-placeholder" not in "".join(p.read_text("utf8") for p in adapter.directory.glob("*.json"))


async def test_invalid_tuple_remains_failure_with_usage_and_no_hidden_retry(tmp_path, monkeypatch):
    adapter, requests = provider(tmp_path, monkeypatch, {"text": ["wrong shape"]})
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid, match="业务字段"):
            await adapter.invoke_strict("strict_selection", Example, [{"role": "user", "content": "raw"}])
    assert len(requests) == 1 and usage[0]["total_tokens"] == 120
    assert adapter.guard.report()["records"][0]["outcome"] == "invalid_output"


async def test_length_has_real_usage_and_no_automatic_capacity_retry(tmp_path, monkeypatch):
    adapter, requests = provider(tmp_path, monkeypatch, {"text": "不完整"}, finish="length")
    with pytest.raises(ModelOutputInvalid) as error:
        await adapter.invoke_strict("strict_selection", Example, [{"role": "user", "content": "raw"}])
    assert error.value.code == "model_output_truncated"
    assert len(requests) == 1 and adapter.guard.report()["known_total_tokens"] == 120


async def test_complete_malformed_quote_is_locally_recovered_with_original_wire_and_usage_preserved(tmp_path, monkeypatch):
    original = '{"text": "原文有"引用"内容。"}'
    adapter, requests = provider(tmp_path, monkeypatch, {}, encoded_arguments=original)
    result = await adapter.invoke_strict("authority_analyst", Example, [{"role":"user","content":"原始资料。"}])
    assert result == {"text": '原文有"引用"内容。'}
    assert len(requests) == 1 and adapter.guard.report()["known_total_tokens"] == 120
    saved = json.loads((adapter.directory / "request-01-response.json").read_text("utf8"))
    assert saved["tool_calls"] == []
    assert saved["tool_calls_wire"][0]["function"]["arguments"] == original
    audit = json.loads((adapter.directory / "request-01-syntax-recovery.json").read_text("utf8"))
    assert len(audit["insertions"]) == 2 and audit["new_model_calls"] == 0
    assert audit["original_response_preserved"] is True


async def test_truncated_field_cannot_be_filled_by_local_syntax_recovery(tmp_path, monkeypatch):
    adapter, requests = provider(tmp_path, monkeypatch, {}, encoded_arguments='{"text": "原文')
    with pytest.raises(ModelOutputInvalid):
        await adapter.invoke_strict("authority_analyst", Example, [{"role":"user","content":"原始资料。"}])
    assert len(requests) == 1 and adapter.guard.report()["known_total_tokens"] == 120
    assert not list(adapter.directory.glob("*syntax-recovery.json"))


async def test_preflight_stop_causes_no_http_and_no_billing(tmp_path, monkeypatch):
    adapter, requests = provider(tmp_path, monkeypatch, {"text": "never called"})
    adapter.threshold = 100
    with pytest.raises(Exception):
        await adapter.invoke_strict("strict_selection", Example, [{"role": "user", "content": "raw"}])
    assert not requests and adapter.guard.report()["actual_requests_started"] == 0


def test_wire_schema_slimming_does_not_weaken_local_schema():
    tool = transport.strict_tool(Example)
    assert "minLength" not in json.dumps(tool)
    with pytest.raises(Exception):
        Example.model_validate({"text": ""})


def test_native_analyst_schema_accepts_a_complete_grounded_judgment():
    from app.agents.schemas import AnalysisOutput, ArbitrationOutput
    from jsonschema import Draft202012Validator
    value = {"decision":"asserted", "conclusion":"原文尚未确认。", "judgments":[{
        "id":"J1", "statement":"原文表示尚未确认。", "evidence_ids":["E1"], "prerequisite_judgment_ids":[]}],
        "supporting_evidence_ids":["E1"], "conflicting_evidence_ids":[]}
    for schema, payload in [(AnalysisOutput,value),(ArbitrationOutput,{**value,"conflicts":[]})]:
        native = transport.strict_tool(schema)["function"]["parameters"]
        assert not list(Draft202012Validator(native).iter_errors(payload))
        assert schema.model_validate(payload)
    invalid = {**value,"judgments":[{**value["judgments"][0],"evidence_ids":[]}]}
    from app.workflows.analysis import _validate_analysis
    with pytest.raises(ModelOutputInvalid) as error:
        _validate_analysis(invalid, [{"id":"E1"}], "q", "authority_analyst")
    assert error.value.details["check"] == "ungrounded_judgment"


async def test_resume_prior_usage_reduces_remaining_budget_not_reset(tmp_path, monkeypatch):
    adapter, requests = provider(tmp_path, monkeypatch, {"text": "never called"})
    adapter.seed_prior_usage([{"usage_source":"provider", "input_tokens":35000,"output_tokens":1000,"total_tokens":36000}])
    with pytest.raises(ModelOutputInvalid) as error:
        await adapter.invoke_strict("strict_selection", Example, [{"role":"user","content":"raw"}])
    assert error.value.code == "experiment_usage_stop" and not requests
    assert adapter.prior_known_tokens == 36000


def test_resume_unknown_usage_blocks_before_model(tmp_path, monkeypatch):
    adapter, requests = provider(tmp_path, monkeypatch, {"text":"never called"})
    with pytest.raises(ModelOutputInvalid) as error:
        adapter.seed_prior_usage([{"usage_source":"unavailable","input_tokens":None,"output_tokens":None,"total_tokens":None}])
    assert error.value.code == "experiment_usage_unknown" and not requests


@pytest.mark.parametrize("kwargs", [{"tool_count":2}, {"encoded_arguments":'{"text":"first","text":"last"}'}])
async def test_multiple_tools_and_duplicate_json_keys_cannot_be_first_only_success(tmp_path, monkeypatch, kwargs):
    adapter, requests = provider(tmp_path, monkeypatch, {"text":"valid"}, **kwargs)
    with pytest.raises(ModelOutputInvalid) as error:
        await adapter.invoke_strict("strict_selection", Example, [{"role":"user","content":"raw"}])
    assert error.value.code == "model_output_invalid" and len(requests) == 1
    assert adapter.guard.report()["known_total_tokens"] == 120


async def test_settlement_audit_failure_stops_queued_later_calls(tmp_path, monkeypatch):
    adapter, requests = provider(tmp_path, monkeypatch, {"text":"valid"})
    original = transport.freeze
    def fail_settlement(path, value):
        if str(path).endswith("settlement.json"):
            raise OSError("test disk failure")
        return original(path, value)
    monkeypatch.setattr(transport, "freeze", fail_settlement)
    with pytest.raises(ModelOutputInvalid) as error:
        await adapter.invoke_strict("strict_selection", Example, [{"role":"user","content":"raw"}])
    assert error.value.code == "model_audit_write_failed"
    assert adapter.guard.report()["stop_reason"] == "cost_guard_settlement_journal_failed"
    with pytest.raises(ModelOutputInvalid):
        await adapter.invoke_strict("strict_annotation", Example, [{"role":"user","content":"next"}])
    assert len(requests) == 1


async def test_real_provider_adapter_decodes_required_source_keys_without_filling_missing(tmp_path, monkeypatch):
    args = {"hypothesis":"工程已完成", "decisions":{"P1":"3_oppose","P2":"1"}}
    adapter, requests = provider(tmp_path, monkeypatch, args)
    raw = {"question":{"question_id":"q","question":"工程是否完成？"},"catalog":[
        {"paragraph_ref":f"P{i}","doc_id":"doc","para_id":f"p{i}","text":"尚未完成。","title":"标题","paragraph_complete":True,"parent_complete":True}
        for i in (1,2)]}
    result = await adapter.strict_selection(raw)
    assert [row["p"] for row in result["decisions"]] == ["P1","P2"]
    assert result["decisions"][0]["relevance"] == 3
    assert result["decisions"][0]["stance"] == "opposes_hypothesis"
    assert requests[0]["tools"][0]["function"]["parameters"]["properties"]["decisions"]["required"] == ["P1","P2"]


@pytest.mark.parametrize("fact,bad_number", [("工程尚未完成。",False),("工程已有9人完成。",True)])
async def test_annotation_time_projection_preserves_raw_and_other_invalid_fields_fail_at_actual_stage(tmp_path, monkeypatch, fact, bad_number):
    args={"annotations":{"P1":{"fact_des":fact,"argument":{"时间":"昨日","主体":"","地点":"","内容":fact,"影响":"","数值口径":""},"entities":[]}}}
    adapter, requests=provider(tmp_path,monkeypatch,args)
    raw={"question":{"question_id":"q","question":"工程是否完成？"},"catalog":[{"paragraph_ref":"P1","doc_id":"doc","para_id":"p1","text":"工程尚未完成。","title":"标题","paragraph_complete":True,"parent_complete":True}]}
    selection={"hypotheses":[{"id":"H1","claim":"工程已完成"}],"decisions":[{"p":"P1","relevance":3,"stance":"opposes_hypothesis","hypothesis_id":"H1"}]}
    if bad_number:
        with pytest.raises(ModelOutputInvalid) as error:
            await adapter.strict_annotation(raw,selection)
        assert error.value.stage=="strict_annotation" and "数字" in str(error.value)
    else:
        result=await adapter.strict_annotation(raw,selection)
        assert result["annotations"][0]["argument"]["时间"]==""
        visible=json.loads((adapter.directory/'request-01-response.json').read_text('utf8'))
        assert visible['tool_calls'][0]['args']['annotations']['P1']['argument']['时间']=='昨日'
        audit=json.loads((adapter.directory/'request-01-field-projection.json').read_text('utf8'))
        assert len(audit['changes'])==1 and audit['changes'][0]['before']=='昨日'
    assert len(requests)==1 and adapter.guard.report()['known_total_tokens']==120
