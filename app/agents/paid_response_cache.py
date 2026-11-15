"""Prime the installed LangChain cache from a bound, already-paid tool response.

No network, business-result replay, schema repair, agent loop or fallback. The
caller must reject outbound requests for these stages after requesting reuse.
Only the recorded visible structured tool result is reconstructed; unrecorded
assistant text and hidden reasoning are never reconstructed or read.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from langchain_core.caches import InMemoryCache
from langchain_core.language_models.chat_models import _normalize_messages
from langchain_core.load import dumps as lc_dumps
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


class PaidResponseCacheError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _fail(code):
    raise PaidResponseCacheError(code)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail("paid_cache_duplicate_json_key")
        value[key] = item
    return value


def _json(path):
    return json.loads(Path(path).read_text("utf-8-sig"), object_pairs_hook=_unique)


def _descriptor(value):
    if (not isinstance(value, dict) or not isinstance(value.get("file"), str)
            or not isinstance(value.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", value["sha256"])):
        _fail("paid_cache_descriptor_required")
    path = Path(value["file"])
    if path.name.lower() in {".env", "evidence_samples.json", "conclusions.json"}:
        _fail("paid_cache_reference_or_credential_forbidden")
    if not path.is_absolute() or not path.is_file() or _sha(path) != value["sha256"]:
        _fail("paid_cache_descriptor_sha_mismatch")
    return path


def _verify_descriptors(value, *, expand_native=False, visited=None):
    visited = set() if visited is None else visited
    if isinstance(value, dict):
        if "file" in value or "sha256" in value:
            path = _descriptor(value)
            identity = str(path.resolve()), value["sha256"]
            if expand_native and identity not in visited:
                visited.add(identity)
                _verify_descriptors(_json(path), visited=visited)
            return
        for key, item in value.items():
            _verify_descriptors(item, expand_native=key in {"native_episode_manifest", "native_manifest"}, visited=visited)
    elif isinstance(value, list):
        for item in value:
            _verify_descriptors(item, visited=visited)


def _usage(value):
    keys = ("input_tokens", "output_tokens", "total_tokens")
    if (not isinstance(value, dict) or any(type(value.get(key)) is not int or value[key] < 0 for key in keys)
            or value["input_tokens"] + value["output_tokens"] != value["total_tokens"]):
        _fail("paid_cache_usage_unknown_or_inconsistent")
    return {key: value[key] for key in keys}


def _wire_and_key(cache, model, structured, messages):
    if not isinstance(cache, InMemoryCache) or model.cache is not cache:
        _fail("paid_cache_instance_not_attached")
    endpoint = urlsplit(str(model.openai_api_base or ""))
    if (endpoint.scheme != "https" or endpoint.hostname != "api.deepseek.com"
            or endpoint.path.rstrip("/") != "/beta" or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment or endpoint.port not in {None, 443}):
        _fail("paid_cache_official_endpoint_required")
    try:
        binding = structured.steps[0].steps__["raw"]
        if binding.bound is not model:
            _fail("paid_cache_model_binding_changed")
        kwargs = {key: value for key, value in binding.kwargs.items()
                  if key not in {"ls_structured_output_format", "structured_output_format"}}
        stop = kwargs.pop("stop", None)
        payload = model._get_request_payload(messages, stop=stop, **kwargs)
        extra = payload.pop("extra_body", {})
        if not isinstance(extra, dict):
            _fail("paid_cache_native_body_invalid")
        body = {**payload, **extra}  # Installed OpenAI SDK extra_body uses this shallow override.
        # ainvoke accepts LC messages and serialized role/content dictionaries;
        # mirror its existing input conversion before constructing the cache key.
        normalized = _normalize_messages(model._convert_input(messages).to_messages())
        prompt = lc_dumps([m.model_copy(update={"id": None}) if m.id is not None else m for m in normalized])
        key = model._get_llm_string(stop=stop, **kwargs)
    except PaidResponseCacheError:
        raise
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        _fail("paid_cache_binding_shape_changed")
    tools = body.get("tools")
    if (body.get("model") != model.model_name or body.get("thinking") != {"type": "disabled"}
            or body.get("stream") is not False or body.get("temperature") != 0
            or type(body.get("max_tokens")) is not int or body["max_tokens"] < 1
            or not isinstance(tools, list) or len(tools) != 1
            or tools[0].get("function", {}).get("strict") is not True
            or body.get("tool_choice", {}).get("function", {}).get("name") != tools[0]["function"].get("name")
            or {"api_key", "authorization", "headers", "extra_headers"}.intersection(body)):
        _fail("paid_cache_native_contract_invalid")
    return body, prompt, key


def _save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(value) + "\n")
    return {"file": str(path.resolve()), "sha256": _sha(path)}


def _paid_bundle_front(native, run, closure):
    """Replay a normal, paid LC front instead of inventing a recovery receipt.

    This branch accounts for the parent's own source call exactly once. The
    existing sealed-front branch has zero new source calls and stays separate.
    """
    front = native["front"]
    requests = front.get("langchain_requests")
    if not isinstance(requests, list) or len(requests) != 1:
        _fail("paid_cache_bundle_native_front_required")
    response = _json(_descriptor(front.get("provider-response.json")))
    proof = response.get("proof", {})
    if (proof.get("format") != "langchain-source-bundle-workbench-proof-v1"
            or proof.get("front_requests") != requests
            or any(proof.get(key) != front.get(name) for key, name in (
                ("closed_receipt", "closed-receipt.json"), ("raw_case", "raw-case.json"),
                ("artifact", "artifact.json"), ("documents", "documents.json"), ("paragraphs", "paragraphs.json")))):
        _fail("paid_cache_bundle_native_front_binding")
    _verify_descriptors(response)
    raw = _json(_descriptor(proof["raw_case"]))
    original_question = raw.get("question", {})
    if (not isinstance(original_question, dict)
            or not isinstance(original_question.get("question_id"), str)
            or not isinstance(original_question.get("question"), str)
            or closure.get("raw_question_id", original_question["question_id"]) != original_question["question_id"]
            or closure.get("question_internal_id", run["question_ids"][0]) != run["question_ids"][0]):
        _fail("paid_cache_bundle_native_front_question")
    question = {"id": run["question_ids"][0], "raw_question_id": original_question["question_id"],
        "question": original_question["question"]}
    # This is the existing isolated, source-archived native replay, with no
    # network or model use. It verifies complete source bytes, model output,
    # closed receipt, request/response and provider-settled source usage.
    from app.workflows.lc_bundle_workbench import replay
    try:
        replay(question, raw["paragraphs"], raw["documents"], response, proof["excluded_spans"])
    except (KeyError, TypeError, ValueError):
        _fail("paid_cache_bundle_native_front_unproved")
    entry = requests[0]
    settlement = _json(_descriptor(entry.get("settlement")))
    if settlement.get("operation") != "strict_selection":
        _fail("paid_cache_bundle_native_front_usage")
    return [(entry, {"operation": "strict_selection", **_usage(settlement)})]


def prime_paid_response(cache, model, structured, messages, *, operation, closure_path,
                        allowed_parent_run_ids, output_dir, ordinal):
    """Validate complete paid provenance and prime this exact structured request.

    Returns provenance only. The caller still invokes its existing structured
    LangChain runnable and emits a confirmed cache-hit event after it returns.
    Historical usage is retained in the AIMessage and is NOT new provider usage.
    """
    analyst_operations = {"authority_analyst", "cross_validation_analyst", "arbitrate"}
    if operation not in {"strict_selection", "strict_annotation", *analyst_operations}:
        _fail("paid_cache_operation_forbidden")
    if type(ordinal) is not int or ordinal < 1:
        _fail("paid_cache_ordinal_invalid")
    if not isinstance(allowed_parent_run_ids, (list, tuple, set)) or any(not isinstance(item, str) or not item for item in allowed_parent_run_ids):
        _fail("paid_cache_explicit_parent_required")
    closure_path = Path(closure_path).resolve()
    closure = _json(closure_path)
    if closure.get("closed") is not True or closure.get("references_read") is not False:
        _fail("paid_cache_closed_reference_free_parent_required")
    _verify_descriptors(closure)
    artifacts = closure.get("artifacts", {})
    run = _json(_descriptor(artifacts.get("run")))
    native = _json(_descriptor(artifacts.get("native_episode_manifest")))
    bundle_tail = native.get("format") == "bundle-front-and-strict-tail-files-v1"
    paid_front = []
    if bundle_tail:
        if (run.get("source_bundle_enabled") is not True or run.get("source_bundle_front_framework") != "langchain"
                or not isinstance(native.get("tail"), dict) or not isinstance(native.get("front"), dict)
                or not isinstance(native["tail"].get("requests"), list) or native["front"].get("native_requests") != []):
            _fail("paid_cache_bundle_sealed_tail_required")
        if "sealed-front-recovery.json" in native["front"]:
            receipt = _json(_descriptor(native["front"]["sealed-front-recovery.json"]))
            if (receipt.get("format") != "lc-sealed-front-recovery-v1" or receipt.get("recovery_run_id") != run.get("id")
                    or receipt.get("origin_topic_id") != run.get("topic_id")
                    or receipt.get("origin_question_id") not in run.get("question_ids", [])
                    or receipt.get("new_model_calls") != 0 or receipt.get("new_tokens") != 0
                    or receipt.get("original_response_modified") is not False):
                _fail("paid_cache_bundle_sealed_owner_invalid")
            _verify_descriptors(receipt)
        else:
            if len(run.get("question_ids", [])) != 1:
                _fail("paid_cache_bundle_native_front_question")
            paid_front = _paid_bundle_front(native, run, closure)
        episode = _json(_descriptor(native["tail"].get("episode.json")))
        native_requests = native["tail"]["requests"]
        directory = Path(native.get("episode_directory", "")).resolve()
        if any(_descriptor(row.get("finished")).parent != directory for row in native_requests):
            _fail("paid_cache_bundle_tail_directory_binding")
    else:
        episode = _json(_descriptor(native.get("episode")))
        native_requests = native.get("requests", [])
    parent = run.get("id")
    if (parent not in set(allowed_parent_run_ids) or not bundle_tail and native.get("run_id") != parent
            or run.get("status") not in {"succeeded", "failed", "cancelled"} or closure.get("status") != run["status"]
            # Durable DB captures precede API-derived feature flags. Missing is
            # proved by the bound native strict episode and exact strict wire;
            # an explicitly non-strict run must still be rejected.
            or ("strict_evidence_enabled" in run and run["strict_evidence_enabled"] is not True
                and not (run.get("source_bundle_enabled") is True and run.get("source_bundle_front_framework") == "langchain"))
            or len(run.get("question_ids", [])) != 1
            or not bundle_tail and (native.get("closed") is not True or native.get("fixture") is not False or native.get("real_provider") is not True)
            or episode.get("strict") is not True or episode.get("method") != "function_calling"
            or episode.get("model") != model.model_name or episode.get("thinking") != "disabled"):
        _fail("paid_cache_parent_binding_invalid")
    body, prompt, cache_key = _wire_and_key(cache, model, structured, messages)
    qid = run["question_ids"][0]
    records = run.get("usage", {}).get(qid)
    if not isinstance(records, list) or any(not isinstance(row, dict) or row.get("usage_source") != "provider"
                                           or row.get("model") != model.model_name for row in records):
        _fail("paid_cache_run_usage_unknown")
    run_usage = {key: sum(_usage(row)[key] for row in records) for key in ("input_tokens", "output_tokens", "total_tokens")}
    if not bundle_tail and _usage(native.get("known_usage")) != run_usage:
        _fail("paid_cache_run_usage_binding_invalid")
    matched = []
    all_settlements = [usage for _, usage in paid_front]
    settled_requests = {(str(_descriptor(entry["settlement"]).resolve()), entry["settlement"]["sha256"])
        for entry, _ in paid_front}
    for request in native_requests:
        if not isinstance(request, dict):
            _fail("paid_cache_request_descriptor_invalid")
        finished = _json(_descriptor(request.get("finished")))
        if finished.get("request_started") is not True:
            continue
        settlement = _json(_descriptor(request.get("settlement")))
        settlement_identity = str(_descriptor(request["settlement"]).resolve()), request["settlement"]["sha256"]
        if settlement_identity in settled_requests:
            _fail("paid_cache_duplicate_paid_request")
        settled_requests.add(settlement_identity)
        usage = _usage(settlement)
        if (settlement.get("actual_request_started") is not True or settlement.get("usage_known") is not True
                or settlement.get("is_usage") is not True or settlement.get("provider_http_status") != 200
                or settlement.get("usage_total_basis") != "provider_reported_consistent_full_total"
                or finished.get("unknown_usage_requests") != 0):
            _fail("paid_cache_paid_settlement_unproved")
        all_settlements.append({"operation": settlement.get("operation"), **usage})
        if finished.get("operation") != operation:
            continue
        wire = _json(_descriptor(request.get("wire")))
        if _canonical(wire) != _canonical(body):
            continue
        preflight = _json(_descriptor(request.get("preflight")))
        response = _json(_descriptor(request.get("response")))
        if (settlement.get("operation") != operation or preflight.get("operation") != operation
                or preflight.get("admitted") is not True or preflight.get("wire") != request["wire"]
                or preflight.get("reserved_output_tokens") != wire["max_tokens"]
                or _usage(response.get("usage")) != usage or response.get("finish_reason") not in {"tool_calls", "stop"}):
            _fail("paid_cache_complete_success_unproved")
        normalized, visible = response.get("tool_calls"), response.get("tool_calls_wire")
        if not isinstance(normalized, list) or not isinstance(visible, list) or len(visible) != 1:
            _fail("paid_cache_single_tool_required")
        successful = settlement.get("outcome") == finished.get("outcome") == "success"
        syntax_recovery_expected = (operation in analyst_operations and normalized == []
            and settlement.get("outcome") == finished.get("outcome") == "invalid_output"
            and response.get("finish_reason") == "tool_calls")
        if not successful and not syntax_recovery_expected:
            _fail("paid_cache_complete_success_unproved")
        if successful and len(normalized) != 1:
            _fail("paid_cache_single_tool_required")
        expected_name = wire["tools"][0]["function"]["name"]
        try:
            if (visible[0].get("type") != "function" or visible[0]["function"]["name"] != expected_name
                    or not isinstance(visible[0].get("id"), str) or not visible[0]["id"]):
                _fail("paid_cache_recorded_tool_values_changed")
            if syntax_recovery_expected:
                # Prove only the narrow recovery is possible, while retaining
                # the original malformed wire and empty normalized tool list.
                # Actual repair, schema validation and a separate repair audit
                # remain the structured caller's responsibility after cache hit.
                from app.agents.strict_json_syntax import decode_visible_arguments
                args, recovery_audit = decode_visible_arguments(visible[0]["function"]["arguments"])
                if not isinstance(args, dict) or recovery_audit is None:
                    _fail("paid_cache_quote_syntax_recovery_unproved")
            else:
                args = json.loads(visible[0]["function"]["arguments"], object_pairs_hook=_unique)
                if (normalized[0].get("name") != expected_name or not isinstance(args, dict)
                        or args != normalized[0].get("args") or visible[0].get("id") != normalized[0].get("id")):
                    _fail("paid_cache_recorded_tool_values_changed")
        except PaidResponseCacheError:
            raise
        except (KeyError, TypeError, ValueError):
            _fail("paid_cache_recorded_tool_invalid")
        matches = [re.match(r"request-(\d+)-", Path(desc["file"]).name)
                   for desc in (request["wire"], request["response"], request["finished"], request["settlement"], request["preflight"])]
        if any(match is None for match in matches):
            _fail("paid_cache_request_ordinal_binding_invalid")
        numbers = {int(match.group(1)) for match in matches}
        if len(numbers) != 1:
            _fail("paid_cache_request_ordinal_binding_invalid")
        matched.append((request, response, usage, numbers.pop(), syntax_recovery_expected))
    expected_records = sorted(({"operation": row.get("operation"), **_usage(row)} for row in records), key=_canonical)
    if sorted(all_settlements, key=_canonical) != expected_records:
        _fail("paid_cache_run_does_not_include_paid_response")
    if len(matched) != 1:
        _fail("paid_cache_exact_request_not_unique")
    request, response, usage, parent_ordinal, syntax_recovery_expected = matched[0]
    message = AIMessage(content=response.get("content", ""), additional_kwargs={"tool_calls": deepcopy(response["tool_calls_wire"])},
        tool_calls=deepcopy(response["tool_calls"]), response_metadata={"finish_reason": response["finish_reason"], "model_name": model.model_name},
        usage_metadata=deepcopy(response["usage"]))
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    wire_file = _save(output_dir / f"cache-{ordinal:02d}-wire.json", body)
    response_file = _save(output_dir / f"cache-{ordinal:02d}-response.json", response)
    cache.update(prompt, cache_key, [ChatGeneration(message=message, generation_info={"finish_reason": response["finish_reason"]})])
    if not isinstance(cache.lookup(prompt, cache_key), list):
        _fail("paid_cache_prime_failed")
    hit = {"format": "langchain-paid-tool-response-cache-v1", "operation": operation,
           "parent_run_id": parent, "parent_request_ordinal": parent_ordinal,
           "parent_wire_sha256": request["wire"]["sha256"], "parent_response_sha256": request["response"]["sha256"],
           "exact_request_JSON_values_equal": True, "no_provider_request_started": True,
           "scope": ("exact_prior_visible_args_requires_local_quote_syntax_recovery" if syntax_recovery_expected
                     else "recorded_visible_structured_tool_call_only"),
           "syntax_recovery_expected": syntax_recovery_expected, "cache_primed": True,
           "structured_invoke_confirmation_owned_by_caller": True, "new_provider_tokens": 0,
           "parent_usage": usage, "parent_new_run_usage": run_usage,
           "source_closure": {"file": str(closure_path), "sha256": _sha(closure_path)},
           "cache_key_sha256": sha256((prompt + "\0" + cache_key).encode()).hexdigest(),
           "wire_file": wire_file, "response_file": response_file}
    hit_file = _save(output_dir / f"cache-{ordinal:02d}-hit.json", hit)
    return {"parent_run_id": parent, "parent_request_ordinal": parent_ordinal, "source_response": deepcopy(response),
            "parent_usage": usage, "parent_new_run_usage": run_usage, "cache_hit_file": hit_file,
            "wire_file": wire_file, "response_file": response_file, "syntax_recovery_expected": syntax_recovery_expected}
