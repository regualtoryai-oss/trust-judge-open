"""Shared planner/executor input and native SDK serialization, zero account use."""
from copy import deepcopy
from hashlib import sha256
import json

import httpx
import pytest

from app.agents import strict_transport as transport
from app.agents.provider import ModelOutputInvalid
from app.agents.schemas import AnalysisOutput, ArbitrationOutput
from app.config import Settings
from app.domain.analysis_wire_projection import restore_analyses, validate_projection
from test_analysis_wire_projection import analyses, uid


def input_payload(count=2):
    original = analyses(count)
    refs = list(dict.fromkeys(ref for row in original for key in ("supporting_evidence_ids", "conflicting_evidence_ids") for ref in row[key]))
    evidence = [{"id": identifier, "quote": f"完整原段{index}😀。须核实归属和时点。\r\n实际全文不得截断。",
        "fact_des": "原文保留完整限定。", "raw_doc_id": f"doc_{index}", "raw_para_id": "p_original",
        "grade_score": 0, "source_group_id": f"source-group-{index % 2}",
        "selection_relevance": 3, "selection_stance": "qualification", "selection_hypothesis_id": "H1",
        "selection_semantics_verified": False, "api_key": "synthetic-private-field-never-on-wire"}
        for index, identifier in enumerate(refs)]
    return {"question": {"id": uid("question"), "question": "项目原始情况？"}, "evidence": evidence, "analyses": original}


def adapter(tmp_path, monkeypatch, *, compact, balanced):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        body = requests[-1]
        name = body["tools"][0]["function"]["name"]
        output = {"decision": "undetermined", "conclusion": "原文表示尚待核实。", "judgments": []}
        if name == "ScopedArbitrationWireOutput":
            evidence = json.loads(body["messages"][-1]["content"])["evidence"]
            output.update(principal_claim="原文表示项目尚待核实。", relations={row["id"]: "qualification" for row in evidence}, conflicts=[])
        else:
            output.update(supporting_evidence_ids=[], conflicting_evidence_ids=[])
            if name == "ArbitrationOutput":
                output["conflicts"] = []
        return httpx.Response(200, json={"id": "explicit-offline-http-fixture", "object": "chat.completion", "created": 1,
            "model": "deepseek-v4-pro", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "tool_calls": [{"id": "fixture-call", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(output, ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
    monkeypatch.setattr(transport.httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handle))
    class ExplicitMockAudit(transport.AuditedTransport, httpx.MockTransport):
        pass
    monkeypatch.setattr(transport, "AuditedTransport", ExplicitMockAudit)
    monkeypatch.setattr(transport, "count_wire", lambda body, directory, ordinal:
        (100, transport.freeze(directory / f"request-{ordinal:02d}-wire.json", body)))
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="deepseek-v4-pro", model_provider="deepseek",
        model_api_key="synthetic-never-return-this-key", model_source_bundle_enabled=True, source_bundle_front_framework="langchain",
        strict_evidence_max_requests=10, strict_evidence_balanced_enabled=balanced,
        source_bundle_compact_analyses_enabled=compact)
    return transport.StrictWorkbenchProvider(settings), requests


@pytest.mark.parametrize("operation,balanced,compact", [
    ("authority_analyst", False, True), ("cross_validation_analyst", True, True),
    ("arbitrate", False, False), ("arbitrate", False, True), ("arbitrate", True, True),
])
async def test_planner_native_json_equals_actual_installed_lc_http_json(tmp_path, monkeypatch, operation, balanced, compact):
    provider, requests = adapter(tmp_path, monkeypatch, compact=compact, balanced=balanced)
    payload = input_payload()
    if operation != "arbitrate":
        del payload["analyses"]
        payload["evidence_snapshot_id"] = uid("frozen-snapshot")
    untouched = deepcopy(payload)
    schema = ArbitrationOutput if operation == "arbitrate" else AnalysisOutput
    prepared = transport.prepare_strict_request(operation, "保留完整限定。", payload, schema,
        balanced_scoped_enabled=balanced, selection_hypotheses=provider.selection_hypotheses,
        compact_analyses_enabled=compact)
    planned = transport.build_strict_request_payload(prepared, max_output_tokens=provider.output_limits[operation])
    await provider._request(operation, "保留完整限定。", payload, schema)
    assert payload == untouched and len(requests) == 1
    assert planned["body"] == requests[0]
    assert json.loads(planned["sdk_preview_body_json"]) == requests[0]
    assert planned["canonical_wire_json"] == transport.canonical(requests[0]) + "\n"
    assert planned["canonical_wire_sha256"] == sha256(planned["canonical_wire_json"].encode()).hexdigest()
    assert "synthetic-never-return-this-key" not in planned["canonical_wire_json"]
    assert "synthetic-private-field-never-on-wire" not in planned["canonical_wire_json"]
    paths = list(provider.directory.glob("*-analysis-input-projection.json"))
    assert bool(paths) is (operation == "arbitrate" and compact)
    if paths:
        audit = json.loads(paths[0].read_text("utf8"))
        validate_projection(audit["trusted_original"], audit["projected"], audit["audit"])
        assert restore_analyses(audit["projected"], audit["audit"], trusted_original=audit["trusted_original"]) == payload["analyses"]
        wire_analyses = json.loads(requests[0]["messages"][-1]["content"])["analyses"]
        assert audit["wire_analyses"] == wire_analyses
        assert transport._remap(audit["projected"], audit["evidence_aliases"]) == wire_analyses
        assert wire_analyses[0]["judgments"][1]["prerequisite_judgment_ids"] == ["A1"]
        assert wire_analyses[1]["judgments"][1]["prerequisite_judgment_ids"] == ["B1"]
        assert all("evidence_snapshot_id" not in row for row in wire_analyses)


def test_high_fanout_projection_keeps_full_roles_dag_and_every_quote_for_planning():
    payload = input_payload(20)
    prepared = transport.prepare_strict_request("arbitrate", "仲裁。", payload, ArbitrationOutput,
        balanced_scoped_enabled=True, compact_analyses_enabled=True)
    wire = transport.build_strict_request_payload(prepared, max_output_tokens=8192)
    view = json.loads(wire["body"]["messages"][-1]["content"])
    assert sum(len(row["judgments"]) for row in view["analyses"]) == 40
    assert sum(len(j["prerequisite_judgment_ids"]) for row in view["analyses"] for j in row["judgments"]) == 380
    assert [row["quote"] for row in view["evidence"]] == [row["quote"] for row in payload["evidence"]]
    assert [row["conclusion"] for row in view["analyses"]] == [row["conclusion"] for row in payload["analyses"]]
    assert [row["summary"] for row in view["analyses"]] == [row["summary"] for row in payload["analyses"]]
    assert prepared["analysis_projection"]["trusted_original"] == payload["analyses"]


async def test_actual_serializer_drift_stops_before_provider_or_usage(tmp_path, monkeypatch):
    provider, requests = adapter(tmp_path, monkeypatch, compact=False, balanced=False)
    original = transport._native_request_payload
    def changed(*args, **kwargs):
        wire = original(*args, **kwargs)
        wire["body"]["temperature"] = 0.75
        return wire
    monkeypatch.setattr(transport, "_native_request_payload", changed)
    with pytest.raises(ModelOutputInvalid) as failure:
        await provider._request("authority_analyst", "分析。", {"evidence": []}, AnalysisOutput)
    assert failure.value.code == "strict_request_plan_mismatch"
    assert requests == [] and provider.guard.report()["actual_requests_started"] == 0
    assert provider.guard.report()["known_total_tokens"] == 0


@pytest.mark.parametrize("graded,ids,expected", [(False, False, 4096), (False, True, 1024), (True, True, 2048)])
def test_base_capacity_is_frozen_before_lc_subclass_can_override(tmp_path, graded, ids, expected):
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="deepseek-v4-pro", model_provider="deepseek",
        model_api_key="synthetic-only", model_source_bundle_enabled=True, source_bundle_front_framework="langchain",
        strict_evidence_max_requests=10, source_bundle_selection_only=ids, source_bundle_graded_selection=graded)
    provider = transport.StrictWorkbenchProvider(settings)
    assert provider.output_limits["strict_selection"] == expected
    assert provider.capacity_policy("strict_selection")["initial_tokens"] == expected
