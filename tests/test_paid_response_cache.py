"""Synthetic closed paid artifacts and official LC cache; never API or gold."""
from copy import deepcopy
from hashlib import sha256
import json

from langchain_core.caches import InMemoryCache
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict
import pytest

from app.agents.paid_response_cache import PaidResponseCacheError, prime_paid_response


class CacheShape(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: str


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode())
    return {"file": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}


def read(desc):
    return json.loads(__import__("pathlib").Path(desc["file"]).read_text("utf-8"))


def fixture(tmp_path, operation="strict_selection"):
    cache = InMemoryCache(maxsize=2)
    model = ChatOpenAI(model="deepseek-v4-pro", api_key="explicit-offline-fixture-only", base_url="https://api.deepseek.com/beta",
        max_retries=0, temperature=0, use_responses_api=False, cache=cache,
        extra_body={"thinking": {"type": "disabled"}, "max_tokens": 4096})
    structured = model.with_structured_output(convert_to_openai_tool(CacheShape, strict=True), method="function_calling", strict=True, include_raw=True)
    messages = [SystemMessage(content="只返回指定完整结构。"), HumanMessage(content="显式合成原文：尚未确认。")]
    kwargs = {k: v for k, v in structured.steps[0].steps__["raw"].kwargs.items() if k != "ls_structured_output_format"}
    wire = model._get_request_payload(messages, stop=None, **kwargs); wire.update(wire.pop("extra_body"))
    usage = {"input_tokens": 17, "output_tokens": 7, "total_tokens": 24}
    raw_call = {"id": "fixture-call", "type": "function", "function": {"name": "CacheShape", "arguments": json.dumps({"value": "已付完整工具响应。"}, ensure_ascii=False)}}
    response = {"tool_calls_wire": [raw_call], "tool_calls": [{"name": "CacheShape", "args": {"value": "已付完整工具响应。"}, "id": "fixture-call", "type": "tool_call"}],
                "usage": deepcopy(usage), "finish_reason": "tool_calls"}
    native_dir = tmp_path / "parent-native"
    row = {"wire": save(native_dir / "request-01-wire.json", wire)}
    row["response"] = save(native_dir / "request-01-response.json", response)
    row["preflight"] = save(native_dir / "request-01-preflight.json", {"operation": operation, "admitted": True, "wire": row["wire"], "reserved_output_tokens": 4096})
    row["finished"] = save(native_dir / "request-01-finished.json", {"operation": operation, "request_started": True, "unknown_usage_requests": 0, "outcome": "success"})
    row["settlement"] = save(native_dir / "request-01-settlement.json", {"operation": operation, **usage,
        "actual_request_started": True, "usage_known": True, "is_usage": True, "provider_http_status": 200,
        "usage_total_basis": "provider_reported_consistent_full_total", "outcome": "success"})
    episode = save(native_dir / "episode.json", {"strict": True, "method": "function_calling", "model": "deepseek-v4-pro", "thinking": "disabled"})
    run = {"id": "synthetic-parent", "status": "failed", "strict_evidence_enabled": True, "question_ids": ["q-internal"],
           "usage": {"q-internal": [{"operation": operation, "model": "deepseek-v4-pro", "usage_source": "provider", **usage}]}}
    native = {"closed": True, "fixture": False, "real_provider": True, "run_id": run["id"], "episode": episode,
              "requests": [row], "known_usage": usage, "costparents": []}
    closure = {"closed": True, "status": "failed", "references_read": False,
               "artifacts": {"run": save(tmp_path / "parent-run.json", run),
                             "native_episode_manifest": save(tmp_path / "native-episode-manifest.json", native),
                             "activity": save(tmp_path / "activity.json", {"explicit_synthetic": True})}}
    closure_path = tmp_path / "failed-closure.json"; save(closure_path, closure)
    return {"cache": cache, "model": model, "structured": structured, "messages": messages,
            "operation": operation, "closure_path": closure_path, "allowed_parent_run_ids": [run["id"]],
            "output_dir": tmp_path / "new-episode", "ordinal": 1}


def mutate_bound_parent(kwargs, section, change):
    closure = json.loads(kwargs["closure_path"].read_text("utf-8"))
    native_desc = closure["artifacts"]["native_episode_manifest"]
    native = read(native_desc)
    if section in {"run", "activity"}:
        desc = closure["artifacts"][section]; value = read(desc); change(value)
        closure["artifacts"][section] = save(__import__("pathlib").Path(desc["file"]), value)
    else:
        desc = native["requests"][0][section]; value = read(desc); change(value)
        native["requests"][0][section] = save(__import__("pathlib").Path(desc["file"]), value)
        closure["artifacts"]["native_episode_manifest"] = save(__import__("pathlib").Path(native_desc["file"]), native)
    save(kwargs["closure_path"], closure)


@pytest.mark.parametrize("operation", ["strict_selection", "strict_annotation", "authority_analyst", "cross_validation_analyst", "arbitrate"])
async def test_existing_lc_cache_hit_avoids_generation_and_preserves_recorded_values(tmp_path, monkeypatch, operation):
    kwargs = fixture(tmp_path, operation)
    result = prime_paid_response(**kwargs)
    async def forbidden(*args, **kw):
        raise AssertionError("a cache hit must never invoke model generation")
    monkeypatch.setattr(kwargs["model"], "_agenerate", forbidden)
    output = await kwargs["structured"].ainvoke(kwargs["messages"])
    assert output["parsed"] == {"value": "已付完整工具响应。"} and output["parsing_error"] is None
    assert output["raw"].additional_kwargs["tool_calls"] == result["source_response"]["tool_calls_wire"]
    assert output["raw"].usage_metadata["total_tokens"] == 24  # Historical, not new billing.
    assert output["raw"].usage_metadata["total_cost"] == 0
    assert result["parent_usage"] == {"input_tokens": 17, "output_tokens": 7, "total_tokens": 24}
    hit = read(result["cache_hit_file"])
    assert hit["exact_request_JSON_values_equal"] and hit["no_provider_request_started"]
    assert hit["parent_run_id"] == "synthetic-parent" and hit["parent_request_ordinal"] == 1
    assert hit["new_provider_tokens"] == 0 and hit["structured_invoke_confirmation_owned_by_caller"]
    for name in ("wire_file", "response_file", "cache_hit_file"):
        assert sha256(__import__("pathlib").Path(result[name]["file"]).read_bytes()).hexdigest() == result[name]["sha256"]


async def test_serialized_message_dicts_use_the_same_lc_cache_input_conversion(tmp_path, monkeypatch):
    kwargs = fixture(tmp_path, "authority_analyst")
    closure = json.loads(kwargs["closure_path"].read_text("utf-8"))
    native = read(closure["artifacts"]["native_episode_manifest"])
    kwargs["messages"] = read(native["requests"][0]["wire"])["messages"]
    prime_paid_response(**kwargs)
    async def forbidden(*args, **kw):
        raise AssertionError("serialized inputs must hit cache without generation")
    monkeypatch.setattr(kwargs["model"], "_agenerate", forbidden)
    output = await kwargs["structured"].ainvoke(kwargs["messages"])
    assert output["parsed"] == {"value": "已付完整工具响应。"}


@pytest.mark.parametrize("change", ["messages", "capacity", "thinking", "base", "parent", "stage", "cache_instance"])
def test_request_mismatch_or_unauthorized_scope_fails_without_prime_or_output(tmp_path, change):
    kwargs = fixture(tmp_path)
    if change == "messages": kwargs["messages"][1] = HumanMessage(content="不同完整原文。")
    elif change == "capacity": kwargs["model"].extra_body["max_tokens"] = 8192
    elif change == "thinking": kwargs["model"].extra_body["thinking"] = {"type": "enabled"}
    elif change == "base": kwargs["model"].openai_api_base = "https://example.invalid/beta"
    elif change == "parent": kwargs["allowed_parent_run_ids"] = ["other-parent"]
    elif change == "stage": kwargs["operation"] = "arbitrate"
    elif change == "cache_instance": kwargs["model"].cache = False
    with pytest.raises(PaidResponseCacheError): prime_paid_response(**kwargs)
    assert kwargs["cache"]._cache == {} and not kwargs["output_dir"].exists()


@pytest.mark.parametrize("change", ["partial", "http", "not_started", "length", "usage", "ledger", "tool_values", "duplicate_tool_key"])
def test_unproved_paid_response_is_rejected_even_after_rebinding_file_hashes(tmp_path, change):
    kwargs = fixture(tmp_path)
    if change == "partial": mutate_bound_parent(kwargs, "settlement", lambda v: v.update(usage_known=False))
    elif change == "http": mutate_bound_parent(kwargs, "settlement", lambda v: v.update(provider_http_status=402))
    elif change == "not_started": mutate_bound_parent(kwargs, "finished", lambda v: v.update(request_started=False))
    elif change == "length": mutate_bound_parent(kwargs, "response", lambda v: v.update(finish_reason="length"))
    elif change == "usage": mutate_bound_parent(kwargs, "response", lambda v: v["usage"].update(total_tokens=23))
    elif change == "ledger": mutate_bound_parent(kwargs, "run", lambda v: v["usage"]["q-internal"][0].update(operation="strict_annotation"))
    elif change == "tool_values": mutate_bound_parent(kwargs, "response", lambda v: v["tool_calls"][0].update(args={"value": "改写值"}))
    elif change == "duplicate_tool_key": mutate_bound_parent(kwargs, "response", lambda v: v["tool_calls_wire"][0]["function"].update(arguments='{"value":"甲","value":"乙"}'))
    with pytest.raises(PaidResponseCacheError): prime_paid_response(**kwargs)
    assert kwargs["cache"]._cache == {} and not kwargs["output_dir"].exists()


def test_all_descriptor_hashes_including_nonselected_artifacts_are_verified(tmp_path):
    kwargs = fixture(tmp_path)
    (tmp_path / "activity.json").write_bytes(b'{"changed":true}')
    with pytest.raises(PaidResponseCacheError, match="descriptor_sha_mismatch"):
        prime_paid_response(**kwargs)
    assert kwargs["cache"]._cache == {}


def test_nested_cost_parent_native_manifest_descriptors_are_checked(tmp_path):
    kwargs = fixture(tmp_path); closure = json.loads(kwargs["closure_path"].read_text("utf-8"))
    desc = closure["artifacts"]["native_episode_manifest"]; native = read(desc)
    deep_desc = save(tmp_path / "ancestor-native.json", {"closed": True})
    native["costparents"] = [{"native_manifest": deep_desc}]
    closure["artifacts"]["native_episode_manifest"] = save(__import__("pathlib").Path(desc["file"]), native)
    save(kwargs["closure_path"], closure); (tmp_path / "ancestor-native.json").write_bytes(b"{}")
    with pytest.raises(PaidResponseCacheError, match="descriptor_sha_mismatch"): prime_paid_response(**kwargs)


def test_cache_files_are_exclusive_and_do_not_silently_replace_previous_audit(tmp_path):
    kwargs = fixture(tmp_path); first = prime_paid_response(**kwargs)
    with pytest.raises(FileExistsError): prime_paid_response(**kwargs)
    assert read(first["cache_hit_file"])["parent_run_id"] == "synthetic-parent"


def test_durable_sqlite_run_may_omit_api_computed_strict_flag(tmp_path):
    kwargs = fixture(tmp_path)
    mutate_bound_parent(kwargs, "run", lambda v: v.pop("strict_evidence_enabled"))
    assert prime_paid_response(**kwargs)["parent_run_id"] == "synthetic-parent"


@pytest.mark.parametrize("flag", [False, None, 0, "true"])
def test_explicit_nontrue_strict_flag_is_rejected(tmp_path, flag):
    kwargs = fixture(tmp_path)
    mutate_bound_parent(kwargs, "run", lambda v: v.update(strict_evidence_enabled=flag))
    with pytest.raises(PaidResponseCacheError, match="parent_binding_invalid"):
        prime_paid_response(**kwargs)
    assert kwargs["cache"]._cache == {} and not kwargs["output_dir"].exists()


def malformed_analyst_fixture(tmp_path, *, operation="cross_validation_analyst"):
    kwargs = fixture(tmp_path, operation)
    mutate_bound_parent(kwargs, "settlement", lambda v: v.update(outcome="invalid_output"))
    mutate_bound_parent(kwargs, "finished", lambda v: v.update(outcome="invalid_output", safe_error_code="model_output_invalid"))
    def malformed(value):
        value["tool_calls"] = []
        value["tool_calls_wire"][0]["function"]["arguments"] = '{"value": "原样"内嵌引号"句子"}'
    mutate_bound_parent(kwargs, "response", malformed)
    return kwargs


@pytest.mark.parametrize("operation", ["authority_analyst", "cross_validation_analyst"])
async def test_quote_recovery_cache_retains_original_malformed_visible_response(tmp_path, monkeypatch, operation):
    kwargs = malformed_analyst_fixture(tmp_path, operation=operation)
    closure = json.loads(kwargs["closure_path"].read_text("utf-8"))
    parent_response = read(closure["artifacts"]["native_episode_manifest"])["requests"][0]["response"]
    result = prime_paid_response(**kwargs)
    async def forbidden(*args, **kw):
        raise AssertionError("malformed cache reuse must never invoke generation")
    monkeypatch.setattr(kwargs["model"], "_agenerate", forbidden)
    output = await kwargs["structured"].ainvoke(kwargs["messages"])
    original = result["source_response"]
    assert original["tool_calls"] == [] and read(result["response_file"]) == original
    assert output["raw"].tool_calls == [] and output["raw"].invalid_tool_calls
    assert output["raw"].additional_kwargs["tool_calls"] == original["tool_calls_wire"]
    assert output["parsed"] is None and output["parsing_error"] is not None
    hit = read(result["cache_hit_file"])
    assert hit["syntax_recovery_expected"] and result["syntax_recovery_expected"]
    assert hit["scope"] == "exact_prior_visible_args_requires_local_quote_syntax_recovery"
    assert hit["parent_usage"]["total_tokens"] == 24 and hit["new_provider_tokens"] == 0
    assert hit["parent_response_sha256"] == parent_response["sha256"]


@pytest.mark.parametrize("change", ["truncated", "missing_field", "nonanalyst", "unknown_outcome", "wrong_tool", "valid_but_empty_normalized", "normalized_present"])
def test_only_complete_analyst_quote_syntax_failure_is_reusable(tmp_path, change):
    kwargs = malformed_analyst_fixture(tmp_path, operation="strict_annotation" if change == "nonanalyst" else "cross_validation_analyst")
    if change == "truncated": mutate_bound_parent(kwargs, "response", lambda v: v.update(finish_reason="length"))
    elif change == "missing_field": mutate_bound_parent(kwargs, "response", lambda v: v["tool_calls_wire"][0]["function"].update(arguments='{"value": "缺闭合"'))
    elif change == "unknown_outcome": mutate_bound_parent(kwargs, "settlement", lambda v: v.update(outcome="transport_failure"))
    elif change == "wrong_tool": mutate_bound_parent(kwargs, "response", lambda v: v["tool_calls_wire"][0]["function"].update(name="OtherShape"))
    elif change == "valid_but_empty_normalized": mutate_bound_parent(kwargs, "response", lambda v: v["tool_calls_wire"][0]["function"].update(arguments='{"value": "本来有效"}'))
    elif change == "normalized_present": mutate_bound_parent(kwargs, "response", lambda v: v.update(tool_calls=[{"name":"CacheShape", "id":"fixture-call", "type":"tool_call", "args":{"value":"不能替换"}}]))
    with pytest.raises((PaidResponseCacheError, ValueError)):
        prime_paid_response(**kwargs)
    assert kwargs["cache"]._cache == {} and not kwargs["output_dir"].exists()


def native_bundle_parent_fixture(tmp_path):
    """A real source-codec proof plus a synthetic already-paid tail request."""
    from test_lc_bundle_workbench import proof_fixture
    kwargs = fixture(tmp_path, "authority_analyst")
    question, _case, response, _ = proof_fixture(tmp_path / "front", parent=0)
    closure = json.loads(kwargs["closure_path"].read_text("utf8"))
    run = read(closure["artifacts"]["run"])
    native = read(closure["artifacts"]["native_episode_manifest"])
    proof = response["proof"]
    front = {"native_requests": [], "langchain_requests": proof["front_requests"],
        "provider-response.json": save(tmp_path / "front" / "provider-response.json", response),
        **{name: proof[key] for name, key in (("closed-receipt.json", "closed_receipt"),
            ("raw-case.json", "raw_case"), ("artifact.json", "artifact"),
            ("documents.json", "documents"), ("paragraphs.json", "paragraphs"))}}
    run.update(source_bundle_enabled=True, source_bundle_front_framework="langchain")
    front_paid = read(proof["closed_receipt"])["usage_records"]
    run["usage"][run["question_ids"][0]] = front_paid + run["usage"][run["question_ids"][0]]
    closure["artifacts"]["run"] = save(tmp_path / "parent-run.json", run)
    closure.update(raw_question_id=question["question_id"], question_internal_id=run["question_ids"][0])
    bundle = {"format": "bundle-front-and-strict-tail-files-v1", "front": front,
        "tail": {"requests": native["requests"], "episode.json": native["episode"]},
        "episode_directory": str((tmp_path / "parent-native").resolve())}
    closure["artifacts"]["native_episode_manifest"] = save(tmp_path / "native-episode-manifest.json", bundle)
    save(kwargs["closure_path"], closure)
    return kwargs


async def test_native_paid_front_parent_replays_source_and_counts_source_plus_tail_once(tmp_path, monkeypatch):
    kwargs = native_bundle_parent_fixture(tmp_path)
    result = prime_paid_response(**kwargs)
    assert result["parent_usage"]["total_tokens"] == 24
    assert result["parent_new_run_usage"]["total_tokens"] == 174  # source150 + tail24
    async def forbidden(*args, **kw):
        raise AssertionError("recorded tail reuse cannot send HTTP")
    monkeypatch.setattr(kwargs["model"], "_agenerate", forbidden)
    output = await kwargs["structured"].ainvoke(kwargs["messages"])
    assert output["parsed"] == {"value": "已付完整工具响应。"}


@pytest.mark.parametrize("mutation", ["drop_source_cost", "double_source_cost", "front_hash"])
def test_native_front_cannot_disappear_duplicate_cost_or_change_source_bytes(tmp_path, mutation):
    kwargs = native_bundle_parent_fixture(tmp_path)
    if mutation in {"drop_source_cost", "double_source_cost"}:
        def alter(run):
            rows = run["usage"]["q-internal"]
            if mutation == "drop_source_cost": rows.pop(0)
            else: rows.insert(0, deepcopy(rows[0]))
        mutate_bound_parent(kwargs, "run", alter)
    else:
        closure = json.loads(kwargs["closure_path"].read_text("utf8"))
        native = read(closure["artifacts"]["native_episode_manifest"])
        __import__("pathlib").Path(native["front"]["documents.json"]["file"]).write_bytes(b"[]")
    with pytest.raises(PaidResponseCacheError): prime_paid_response(**kwargs)
    assert kwargs["cache"]._cache == {} and not kwargs["output_dir"].exists()
