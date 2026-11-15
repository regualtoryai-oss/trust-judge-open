"""Single-shot LangChain strict tools, audited at the actual HTTP boundary.

This is a provider adapter, not an agent loop. LangGraph owns orchestration;
the provider never repairs, retries, or falls back to another protocol/model.
"""
import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from uuid import uuid4

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.agents.provider import (LangChainOpenAIProvider, ModelInvocationFailed,
    ModelOutputInvalid, SYSTEM_BOUNDARY)
from app.domain.model_cost_guard import ModelCostGuard, CostGuardPolicy, CostGuardStopped
from app.domain.model_transport import classify_model_transport

ROOT = Path(__file__).resolve().parents[2]
VERSION = "langchain-provider-strict-full-parent-v1"
LIMITS = {"strict_selection": 4096, "strict_annotation": 8192,
          "authority_analyst": 4096, "cross_validation_analyst": 4096, "arbitrate": 8192}
TOKENIZER = ROOT / ".runtime/v32-deepseek-tokenizer-r2/tokenizer.json"
TOKENIZER_SHA = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (canonical(value) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
    return {"file": str(path.resolve()), "sha256": sha256(raw).hexdigest()}


def strict_tool(schema):
    """Remove unsupported wire constraints, keep full Pydantic validation locally."""
    tool = convert_to_openai_tool(schema, strict=True)
    unsupported = {"minItems", "maxItems", "uniqueItems", "minLength", "maxLength",
                   "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                   "pattern", "default", "allOf", "if", "then", "else", "examples"}
    def walk(value):
        if isinstance(value, list):
            return [walk(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: walk(child) for key, child in value.items() if key not in unsupported}
        if "const" in result:
            # Equivalent singleton enum is supported for string tool fields.
            result["enum"] = [result.pop("const")]
        if result.get("type") == "object":
            result["additionalProperties"] = False
            result["required"] = list(result.get("properties", {}))
            # Non-empty-list predicates lose minItems on this provider. LC's
            # strict conversion also closes each partial predicate object,
            # making it contradict the complete required object. Remove only
            # these now-empty predicates; source Pydantic/business validation
            # still enforces evidence/prerequisite grounding locally.
            alternatives = result.get("anyOf")
            if (isinstance(alternatives, list) and alternatives and all(
                    isinstance(branch, dict)
                    and not (set(branch) - {"properties", "required", "additionalProperties"})
                    and set(branch.get("required", [])) <= set(result["required"])
                    and all(child == {} for child in branch.get("properties", {}).values())
                    for branch in alternatives)):
                result.pop("anyOf")
        return result
    tool["function"]["parameters"] = walk(tool["function"]["parameters"])
    return tool


def _remap(value, mapping):
    if isinstance(value, dict):
        return {mapping.get(key, key): _remap(child, mapping) for key, child in value.items()}
    if isinstance(value, list):
        return [_remap(child, mapping) for child in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def prepare_strict_request(operation, instructions, payload, schema, *, balanced_scoped_enabled=False,
                           selection_hypotheses=(), compact_analyses_enabled=False, issue_frame_enabled=False):
    """The shared model view for actual execution and offline budget planning.

    Data messages, projected source view and the exact dynamic schema are
    returned together. Local reversible audits are never put in the messages.
    This function does no IO, model invocation or semantic summarization.
    """
    if (type(balanced_scoped_enabled) is not bool or type(compact_analyses_enabled) is not bool
            or type(issue_frame_enabled) is not bool
            or not isinstance(payload, dict) or not isinstance(instructions, str)):
        raise ValueError("strict_request_preparation_shape")
    projected = deepcopy(payload)
    analysis_projection = None
    if compact_analyses_enabled and operation == "arbitrate":
        from app.domain.analysis_wire_projection import project_analyses
        original = deepcopy(projected.get("analyses"))
        compact = project_analyses(original)
        projected["analyses"] = compact["projected"]
        analysis_projection = {"format": "strict-analysis-input-projection-v1",
            "trusted_original": original, **compact}
    evidence = projected.get("evidence", [])
    aliases = {row["id"]: f"E{index + 1}" for index, row in enumerate(evidence)}
    fields = {"id", "quote", "fact_des", "raw_doc_id", "raw_para_id", "grade_score", "source_group_id"}
    if balanced_scoped_enabled:
        fields.update({"selection_relevance", "selection_stance", "selection_hypothesis_id", "selection_semantics_verified"})
        projected["selection_hypotheses"] = deepcopy(selection_hypotheses)
        projected["selection_context_basis"] = "上游主张和立场是未验证检索线索，不能替代本轮原文核验；principal_claim可以与H1不同方向。"
    projected["evidence"] = [{key: value for key, value in row.items() if key in fields} for row in evidence]
    groups = {key: f"G{index + 1}" for index, key in enumerate(dict.fromkeys(
        row.get("source_group_id") for row in evidence if row.get("source_group_id")))}
    for row in projected["evidence"]:
        if row.get("source_group_id") in groups:
            row["source_group_id"] = groups[row["source_group_id"]]
    projected["grouping_basis"] = "已知来源关联分组不证明各组独立；同组不能重复投票。"
    projected = _remap(projected, aliases)
    scoped = balanced_scoped_enabled and operation == "arbitrate"
    if scoped:
        from app.agents.scoped_arbitration import prepare_scoped_arbitration
        schema, scope_instructions = prepare_scoped_arbitration(schema, projected["evidence"],
            **({"question": projected["question"], "issue_frame": True} if issue_frame_enabled else {}))
        instructions += "\n" + scope_instructions
    if analysis_projection is not None:
        analysis_projection.update(evidence_aliases=deepcopy(aliases), wire_analyses=deepcopy(projected["analyses"]))
    return {"schema": schema, "tool": strict_tool(schema), "messages": [
        {"role": "system", "content": SYSTEM_BOUNDARY + instructions},
        {"role": "user", "content": canonical(projected)}], "projected": projected,
        "evidence_aliases": aliases, "source_group_aliases": groups, "scoped": scoped,
        "analysis_projection": analysis_projection, "issue_frame_enabled": issue_frame_enabled and scoped}


def _strict_chat_model(model_name, capacity, *, api_key, http_async_client=None, cache=False):
    kwargs = {"model": model_name, "api_key": api_key, "base_url": "https://api.deepseek.com/beta",
        "max_retries": 0, "temperature": 0, "use_responses_api": False, "cache": cache,
        "extra_body": {"thinking": {"type": "disabled"}, "max_tokens": capacity}}
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client
    return ChatOpenAI(**kwargs)


def _native_request_payload(model, schema, messages):
    """Use LC conversion and the installed SDK request builder, never send it."""
    from openai._models import FinalRequestOptions
    from openai._utils import maybe_transform
    from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
    tool = strict_tool(schema)
    binding = model.bind_tools([tool], tool_choice=tool["function"]["name"], strict=True, parallel_tool_calls=False)
    parameters = model._get_request_payload(messages, **binding.kwargs)
    extra = parameters.pop("extra_body", None)
    request = model.root_client._build_request(FinalRequestOptions.construct(method="post", url="/chat/completions",
        json_data=maybe_transform(parameters, CompletionCreateParamsNonStreaming), extra_json=extra))
    serialized = bytes(request.content)
    body = json.loads(serialized)
    # SDK-created headers (including authentication) are intentionally neither
    # returned nor logged. Only the JSON body is a planner/model input.
    return {"body": body, "canonical_wire_json": canonical(body) + "\n",
        "canonical_wire_sha256": sha256((canonical(body) + "\n").encode()).hexdigest(),
        "sdk_preview_body_json": serialized.decode("utf8")}


def build_strict_request_payload(prepared, *, model_name="deepseek-v4-pro", max_output_tokens):
    """Offline native serializer for a shared preparation; zero provider calls.

    The returned canonical JSON is exactly the format counted by count_wire.
    sdk_preview_body_json is a local preview, never a delivery/billing proof.
    Actual transport independently compares its parsed HTTP JSON to this view.
    """
    if (not isinstance(prepared, dict) or type(max_output_tokens) is not int or max_output_tokens < 1
            or not isinstance(model_name, str) or not model_name.strip()
            or prepared.get("tool") != strict_tool(prepared["schema"])):
        raise ValueError("strict_request_serializer_shape")
    model = _strict_chat_model(model_name, max_output_tokens, api_key="local-preview-not-an-api-key")
    try:
        return _native_request_payload(model, prepared["schema"], prepared["messages"])
    finally:
        model.root_client.close()


def count_wire(body, directory, ordinal):
    """Offline serialized JSON estimate, explicitly distinct from provider billing."""
    if sha256(TOKENIZER.read_bytes()).hexdigest() != TOKENIZER_SHA:
        raise ValueError("strict_tokenizer_changed")
    source = freeze(directory / f"request-{ordinal:02d}-wire.json", body)
    command = [str(ROOT / ".runtime/v20-embedding-env/Scripts/python.exe"), "-c",
        "import sys;from tokenizers import Tokenizer;from pathlib import Path;"
        "t=Tokenizer.from_file(sys.argv[1]);print(len(t.encode(Path(sys.argv[2]).read_text('utf8')).ids))",
        str(TOKENIZER), source["file"]]
    run = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if run.returncode != 0 or not run.stdout.strip().isdigit():
        raise ValueError("strict_tokenizer_failed")
    return int(run.stdout.strip()), source


class AuditedTransport(httpx.AsyncBaseTransport):
    def __init__(self, owner, operation, capacity, inner=None):
        self.owner, self.operation, self.capacity = owner, operation, capacity
        self.inner = inner or httpx.AsyncHTTPTransport(retries=0)
        self.ticket = self.body = self.response = None
        owner.attempt_index += 1
        self.ordinal = owner.attempt_index
        self.started = False
        self.preflight_error = None
        self.cache_only = False
        self.expected_body = None

    async def handle_async_request(self, request):
        if self.cache_only:
            self.preflight_error = ModelOutputInvalid("已付响应未命中缓存，已阻止重复付费请求。",
                stage=self.operation, code="paid_response_cache_miss")
            raise self.preflight_error
        if (request.method != "POST" or request.url.scheme != "https" or
                request.url.host != "api.deepseek.com" or request.url.path != "/beta/chat/completions"):
            raise ValueError("strict_official_endpoint_required")
        raw_body = await request.aread()
        self.body = json.loads(raw_body)
        if self.expected_body is not None and self.body != self.expected_body:
            self.preflight_error = ModelOutputInvalid("原生请求与共享请求计划不一致，未调用模型。", stage=self.operation,
                code="strict_request_plan_mismatch")
            raise self.preflight_error
        tools = self.body.get("tools", [])
        if (len(tools) != 1 or tools[0].get("function", {}).get("strict") is not True
                or self.body.get("tool_choice", {}).get("function", {}).get("name") != tools[0]["function"]["name"]
                or self.body.get("thinking") != {"type": "disabled"}
                or self.body.get("max_tokens") != self.capacity or self.body.get("stream") is True):
            raise ValueError("strict_actual_wire_contract_failed")
        estimate, binding = await asyncio.to_thread(count_wire, self.body, self.owner.directory, self.ordinal)
        actual_path = self.owner.directory / f"request-{self.ordinal:02d}-actual-wire.json"
        with actual_path.open("xb") as stream:
            stream.write(raw_body)
        actual_binding = {"file": str(actual_path.resolve()), "sha256": sha256(raw_body).hexdigest()}
        known = self.owner.prior_known_tokens + self.owner.guard.report()["known_total_tokens"]
        # A conservative experimental stop, not a permanent user budget or a
        # claim that an offline tokenizer can guarantee exact billing.
        admitted = known + estimate + self.capacity + 512 <= self.owner.threshold
        freeze(self.owner.directory / f"request-{self.ordinal:02d}-preflight.json", {
            "operation": self.operation, "wire": binding, "actual_wire": actual_binding, "known_tokens_before": known,
            "serialized_json_token_estimate": estimate, "reserved_output_tokens": self.capacity,
            "safety_tokens": 512, "experimental_stop_threshold": self.owner.threshold,
            "admitted": admitted, "is_actual_usage": False, "no_credentials_or_headers_logged": True})
        if not admitted:
            self.preflight_error = ModelOutputInvalid("本题已接近实验用量保护线，未发起下一次调用。", stage=self.operation,
                code="experiment_usage_stop", details={"known_tokens": known, "input_estimate": estimate})
            raise self.preflight_error
        messages = deepcopy(self.body["messages"])
        messages.append({"role": "system", "content": canonical({"tools": tools, "tool_choice": self.body["tool_choice"]})})
        try:
            self.ticket = self.owner.guard.prepare(self.operation, messages, self.capacity)
        except CostGuardStopped:
            self.preflight_error = ModelOutputInvalid("本题调用保护已停止后续请求。", stage=self.operation,
                code="experiment_usage_stop", details={"known_tokens": known})
            raise self.preflight_error from None
        try:
            freeze(self.owner.directory / f"request-{self.ordinal:02d}-admitted.json", self.ticket)
        except Exception:
            self.owner.guard.abort_before_invocation(self.ticket)
            raise
        self.owner.guard.mark_invoked(self.ticket)
        self.started = True
        started_hook = getattr(self.owner, "actual_request_started", None)
        if callable(started_hook):
            started_hook(self.operation, self.ordinal)
        response = await self.inner.handle_async_request(request)
        await response.aread()
        self.response = response
        return response

    async def aclose(self):
        await self.inner.aclose()


class StrictWorkbenchProvider(LangChainOpenAIProvider):
    strict_evidence_enabled = True
    repair_analysis = None

    def __init__(self, settings, *, threshold=None):
        threshold = getattr(settings, "strict_evidence_token_stop", 40000) if threshold is None else threshold
        super().__init__(settings)
        self.prompt_version = settings.model_prompt_version
        self.settings = settings
        self.strict_candidate_limit = settings.strict_evidence_candidate_limit
        self.strict_balanced_enabled = settings.strict_evidence_balanced_enabled
        self.strict_balance_policy = {"target":10,"per_doc_soft_cap":2} if self.strict_balanced_enabled else None
        self.strict_recall_strategy = "balanced" if self.strict_balanced_enabled else None
        self.selection_hypotheses = []
        self.compact_analyses_enabled = getattr(settings, "source_bundle_compact_analyses_enabled", False)
        self.issue_frame_enabled = getattr(settings, "source_bundle_issue_frame_enabled", False)
        if type(self.compact_analyses_enabled) is not bool:
            raise ValueError("strict_compact_analyses_flag_boolean_required")
        self.arbitration_scope = None
        self.threshold = threshold
        self.max_requests = getattr(settings, "strict_evidence_max_requests", 5)
        self.attempt_index = 0
        selection_capacity = (8192 if getattr(settings, "source_bundle_paragraph_quality_enabled", False) else
            2048 if getattr(settings, "source_bundle_graded_selection", False) else
            1024 if settings.source_bundle_selection_only else LIMITS["strict_selection"])
        self.output_limits = {**LIMITS, "strict_selection": selection_capacity}
        self.prior_known_tokens = 0
        self.directory = settings.data_dir / "strict-episodes" / str(uuid4())
        self.directory.mkdir(parents=True, exist_ok=False)
        self.guard = ModelCostGuard(CostGuardPolicy(soft_total_tokens=threshold, max_requests=self.max_requests,
            max_message_bytes=2_000_000, max_output_tokens=8192, max_length_recoveries=0,
            max_structure_repairs=0, initial_output_limits=self.output_limits))
        self._strict_semaphore = asyncio.Semaphore(1)
        self.workbench_event = None
        self.response_reuse_by_operation = {}
        self.paid_response_operations = (set(settings.strict_evidence_response_reuse_operations)
            if settings.strict_evidence_response_reuse_closures else set())
        if self.paid_response_operations - {"strict_selection", "strict_annotation", "authority_analyst", "cross_validation_analyst", "arbitrate"}:
            raise ValueError("strict_paid_cache_operation_forbidden")
        source_paths = tuple(ROOT / name for name in (
            "app/agents/strict_transport.py", "app/agents/strict_evidence.py", "app/agents/schemas.py",
            "app/workflows/analysis.py", "app/workflows/strict_workbench.py",
            "app/agents/paid_response_cache.py", "app/domain/numeric_literals.py",
            "app/agents/strict_json_syntax.py", "app/agents/compact_selection.py",
            "app/agents/scoped_arbitration.py", "app/domain/evidence_balance.py",
            "app/domain/question_recall.py", "app/domain/paragraph_ranked.py", "app/domain/retrieval.py",
            "app/domain/source_metadata.py", "app/domain/material_eligibility.py", "app/domain/spans.py",
            "app/domain/provisional_relations.py"))
        if self.compact_analyses_enabled:
            source_paths += (ROOT / "app/domain/analysis_wire_projection.py",)
        source_archive = {}
        for path in source_paths:
            relative = str(path.relative_to(ROOT))
            destination = self.directory / "source-archive" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(path.read_bytes())
            source_archive[relative] = {"file": str(destination.resolve()), "sha256": sha256(destination.read_bytes()).hexdigest()}
        freeze(self.directory / "episode.json", {"version": VERSION, "method": "function_calling",
            "strict": True, "model": settings.model_name, "temperature": 0, "thinking": "disabled",
            "max_requests": self.max_requests, "automatic_retries": 0, "experimental_stop_threshold": threshold,
            "user_20000_is_soft_guidance": True,
            "balanced_scoped_enabled":self.strict_balanced_enabled,
            **({"compact_analyses_enabled": True} if self.compact_analyses_enabled else {}),
            **({"issue_frame_enabled": True} if self.issue_frame_enabled else {}),
            "balance_policy":self.strict_balance_policy,"recall_strategy":self.strict_recall_strategy,
            "sources": {key: descriptor["sha256"] for key, descriptor in source_archive.items()},
            "source_archive": source_archive})

    def capacity_policy(self, operation):
        capacity = self.output_limits.get(operation)
        if capacity is None:
            raise ValueError("strict_operation_not_authorized")
        return {"initial_tokens": capacity, "max_tokens": capacity, "retries": 0}

    def seed_prior_usage(self, records):
        """One immutable resume seed; caller supplies *new* usage of each ancestor."""
        if self.attempt_index or self.prior_known_tokens or not isinstance(records, list):
            raise ValueError("strict_prior_usage_seed_invalid")
        for row in records:
            if (not isinstance(row, dict) or row.get("usage_source") != "provider"
                    or any(type(row.get(k)) is not int or row[k] < 0 for k in ("input_tokens", "output_tokens", "total_tokens"))
                    or row["input_tokens"] + row["output_tokens"] != row["total_tokens"]):
                raise ModelOutputInvalid("历史任务用量未知，未调用模型续跑。", stage="resume",
                    code="experiment_usage_unknown")
        total = sum(row["total_tokens"] for row in records)
        if len(records) >= self.max_requests or total >= self.threshold:
            raise ModelOutputInvalid("本题历史调用已到实验保护线，未调用模型续跑。", stage="resume",
                code="experiment_usage_stop", details={"known_tokens": total})
        self.prior_known_tokens = total
        self.guard = ModelCostGuard(CostGuardPolicy(soft_total_tokens=self.threshold - total,
            max_requests=self.max_requests - len(records), max_message_bytes=2_000_000, max_output_tokens=8192,
            max_length_recoveries=0, max_structure_repairs=0, initial_output_limits=self.output_limits))
        freeze(self.directory / "resume-prior-usage.json", {"records": deepcopy(records),
            "prior_known_tokens": total, "prior_calls": len(records), "not_new_usage": True})

    async def invoke_strict(self, operation, schema, messages):
        capacity = self.output_limits[operation]
        async with self._strict_semaphore:
            wire = AuditedTransport(self, operation, capacity)
            started = perf_counter()
            output = None
            error = None
            usage = None
            outcome = "transport_failure"
            finish = None
            cache_binding = None
            want_cache = operation in self.paid_response_operations
            wire.cache_only = want_cache
            try:
                async with httpx.AsyncClient(transport=wire, timeout=240) as client:
                    from langchain_core.caches import InMemoryCache
                    cache = InMemoryCache() if want_cache else False
                    model = _strict_chat_model(self.model_name, capacity,
                        api_key=self.settings.model_api_key.get_secret_value(), http_async_client=client, cache=cache)
                    tool = strict_tool(schema)
                    if want_cache:
                        from app.agents.paid_response_cache import prime_paid_response
                        closures = self.settings.strict_evidence_response_reuse_closures
                        try:
                            # Select only a recorded logical operation. The helper
                            # subsequently verifies every descriptor, full native
                            # request and explicit parent; no outbound fallback.
                            matching = []
                            for closure in closures:
                                metadata = json.loads(Path(closure).read_text("utf-8-sig"))
                                manifest = json.loads(Path(metadata["artifacts"]["native_episode_manifest"]["file"]).read_text("utf-8-sig"))
                                entries = (manifest.get("tail", {}).get("requests", [])
                                    if manifest.get("format") == "bundle-front-and-strict-tail-files-v1" else manifest.get("requests", []))
                                for entry in entries:
                                    recorded = json.loads(Path(entry["finished"]["file"]).read_text("utf-8-sig"))
                                    if recorded.get("operation") == operation and recorded.get("request_started") is True:
                                        matching.append((closure, entry))
                            if len(matching) != 1:
                                raise ValueError("paid_cache_parent_operation_not_unique")
                            prior_closure, prior_entry = matching[0]
                            recorded_wire = json.loads(Path(prior_entry["wire"]["file"]).read_text("utf-8-sig"))
                            archived_tool = recorded_wire["tools"][0]
                            if archived_tool["function"]["name"] != tool["function"]["name"]:
                                raise ValueError("paid_cache_schema_identity_changed")
                            # Historical cache deserializes its original wire;
                            # the complete source schema still validates below.
                            # This archived tool is never sent to the provider.
                            runnable = model.with_structured_output(archived_tool,
                                method="function_calling", strict=True, include_raw=True)
                            cache_binding = prime_paid_response(cache, model, runnable, messages,
                                operation=operation, closure_path=prior_closure,
                                allowed_parent_run_ids=self.settings.strict_evidence_cost_parent_run_ids,
                                output_dir=self.directory, ordinal=wire.ordinal)
                        except (ValueError, OSError, KeyError, TypeError):
                            raise ModelOutputInvalid("已付响应与本次原文、结构或模型配置不一致，未重复调用模型。",
                                stage=operation, code="paid_response_cache_binding_invalid") from None
                        self.response_reuse_by_operation[operation] = cache_binding
                    else:
                        runnable = model.with_structured_output(tool,
                            method="function_calling", strict=True, include_raw=True)
                        wire.expected_body = _native_request_payload(model, schema, messages)["body"]
                    output = await runnable.ainvoke(messages)
                raw = output["raw"]
                usage = raw.usage_metadata
                finish = raw.response_metadata.get("finish_reason")
                visible_wire = (cache_binding["source_response"]["tool_calls_wire"] if cache_binding is not None else
                    (wire.response.json().get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                     if wire.response is not None else []))
                visible = {"tool_calls": raw.tool_calls, "tool_calls_wire": visible_wire,
                    "usage": usage, "finish_reason": finish}
                if cache_binding is None:
                    freeze(self.directory / f"request-{wire.ordinal:02d}-response.json", visible)
                if finish == "length":
                    outcome = "length"
                    raise ModelOutputInvalid("模型输出达到长度上限，未发布。", stage=operation,
                        code="model_output_truncated", details={"finish_reason": "length"})
                if finish not in {"tool_calls", "stop"}:
                    outcome = "filtered"
                    raise ModelOutputInvalid("模型未完整返回结构化结果。", stage=operation, code="model_output_incomplete")
                outcome = "invalid_output"
                expected_name = strict_tool(schema)["function"]["name"]
                if (not isinstance(visible_wire, list) or len(visible_wire) != 1
                        or visible_wire[0].get("type") != "function"
                        or visible_wire[0].get("function", {}).get("name") != expected_name
                        or not isinstance(visible_wire[0].get("function", {}).get("arguments"), str)):
                    raise ModelOutputInvalid("模型必须完整返回唯一的指定结构化工具结果。", stage=operation,
                        code="model_output_invalid", details={"check": "strict_single_named_tool"})
                try:
                    from app.agents.strict_json_syntax import decode_visible_arguments
                    decoded, syntax_recovery = decode_visible_arguments(visible_wire[0]["function"]["arguments"])
                except (ValueError, TypeError) as syntax_error:
                    duplicate = getattr(syntax_error, "code", None) == "duplicate_json_key"
                    raise ModelOutputInvalid("结构化结果含重复字段，未发布。" if duplicate else "结构化工具参数存在无法安全处理的JSON语法错误，未发布。", stage=operation,
                        code="model_output_invalid", details={"check": "strict_unique_tool_keys" if duplicate else "strict_structured_json_syntax"}) from None
                if syntax_recovery is None and (output.get("parsing_error") is not None or output.get("parsed") is None
                        or len(raw.tool_calls) != 1 or raw.invalid_tool_calls
                        or raw.tool_calls[0].get("args") != decoded):
                    raise ModelOutputInvalid("严格结构输出未通过解析，未发布。", stage=operation, code="model_output_invalid")
                try:
                    parsed = schema.model_validate(decoded).model_dump(by_alias=True)
                except ValidationError as failure:
                    raise ModelOutputInvalid("严格结构输出未通过业务字段校验。", stage=operation,
                        code="model_output_invalid", details={"schema_error_types": sorted({row["type"] for row in failure.errors(include_input=False)})}) from None
                if syntax_recovery is not None:
                    recovery_scope = ("single_unmatched_closing_delimiter_deletion_only"
                        if syntax_recovery.get("format") == "strict-json-single-unmatched-closer-v1"
                        else "quote_escape_insertions_only")
                    prefix = "cache" if cache_binding is not None else "request"
                    original = self.directory / f"{prefix}-{wire.ordinal:02d}-response.json"
                    freeze(self.directory / f"{prefix}-{wire.ordinal:02d}-syntax-recovery.json", {
                        "operation":operation, **syntax_recovery,
                        "original_response_sha256":sha256(original.read_bytes()).hexdigest(),
                        "original_response_preserved":True, "schema_validation_passed":True,
                        "new_model_calls":0, "scope":recovery_scope})
                    if callable(self.workbench_event):
                        self.workbench_event("model_syntax_recovered", {"operation":operation,
                            "new_provider_calls":0,"new_tokens":0,"scope":recovery_scope})
                outcome = "success"
                if cache_binding is not None and callable(self.workbench_event):
                    self.workbench_event("model_response_reused", {"operation":operation,
                        "parent_run_id":cache_binding["parent_run_id"], "new_provider_calls":0,
                        "new_tokens":0, "cache_hit_sha256":cache_binding["cache_hit_file"]["sha256"],
                        "scope":"exact_prior_paid_structured_tool_response"})
                return parsed
            except (ModelOutputInvalid, CostGuardStopped) as failure:
                error = failure
                raise
            except Exception as failure:
                if wire.preflight_error is not None:
                    error = wire.preflight_error
                    raise wire.preflight_error from None
                error = failure
                public = classify_model_transport(failure)
                raise ModelInvocationFailed(public["message"], stage=operation,
                    code=public["code"], details=public["details"]) from None
            finally:
                if wire.started:
                    if usage is None and wire.response is not None:
                        try:
                            original = wire.response.json().get("usage") or {}
                            if all(type(original.get(key)) is int for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
                                usage = {"input_tokens": original["prompt_tokens"], "output_tokens": original["completion_tokens"], "total_tokens": original["total_tokens"]}
                        except (ValueError, TypeError):
                            pass
                    self._record_usage(operation, usage)
                try:
                    if wire.started:
                        record = self.guard.settle(wire.ticket, usage, outcome=outcome, finish_reason=finish,
                            provider_http_status=wire.response.status_code if wire.response else None)
                        freeze(self.directory / f"request-{wire.ordinal:02d}-settlement.json", {**record,
                            "elapsed_seconds": round(perf_counter() - started, 3),
                            "business_validation_is_separate": True})
                    prefix = "cache" if want_cache else "request"
                    freeze(self.directory / f"{prefix}-{wire.ordinal:02d}-finished.json", {
                        "operation": operation, "request_started": wire.started, "outcome": outcome,
                        "cache_hit":cache_binding is not None,
                        "safe_error_code": getattr(error, "code", None),
                        "known_total_tokens": self.guard.report()["known_total_tokens"],
                        "known_episode_tokens_with_ancestors": self.prior_known_tokens + self.guard.report()["known_total_tokens"],
                        "unknown_usage_requests": self.guard.report()["unknown_usage_requests"]})
                except Exception:
                    self.guard.stop_after_audit_failure()
                    raise ModelOutputInvalid("执行审计写入失败，已停止后续模型调用。", stage=operation,
                        code="model_audit_write_failed") from None

    async def _request(self, operation, instructions, payload, schema, *, reasoning_repair=False):
        prepared = prepare_strict_request(operation, instructions, payload, schema,
            balanced_scoped_enabled=self.strict_balanced_enabled, selection_hypotheses=self.selection_hypotheses,
            compact_analyses_enabled=self.compact_analyses_enabled, issue_frame_enabled=self.issue_frame_enabled)
        aliases, projected = prepared["evidence_aliases"], prepared["projected"]
        if prepared["analysis_projection"] is not None:
            projection_prefix = "cache" if operation in self.paid_response_operations else "request"
            freeze(self.directory / f"{projection_prefix}-{self.attempt_index + 1:02d}-analysis-input-projection.json", {
                "operation": operation, **prepared["analysis_projection"],
                "projected_payload_sha256": sha256(canonical(projected).encode()).hexdigest(),
                "new_model_calls": 0})
        result = await self.invoke_strict(operation, prepared["schema"], prepared["messages"])
        if prepared["scoped"]:
            from app.agents.scoped_arbitration import canonical_scoped_arbitration
            try:
                result, scope = canonical_scoped_arbitration(result, projected["evidence"],
                    **({"question": projected["question"], "issue_frame": True} if self.issue_frame_enabled else {}))
            except ValueError:
                raise ModelOutputInvalid("仲裁的主张、证据角色或依赖未通过校验，未发布。",stage=operation,
                    code="model_output_invalid",details={"check":"principal_claim_relations"}) from None
            inverse = {alias:original for original,alias in aliases.items()}
            self.arbitration_scope = _remap(scope, inverse)
            if self.arbitration_scope.get("format") in {"issue-frame-evidence-relations-v2", "conclusion-aligned-evidence-relations-v3"}:
                self.arbitration_scope["evidence_order_sha256"] = sha256(
                    canonical(self.arbitration_scope["evidence_order"]).encode()).hexdigest()
            response_prefix = "cache" if operation in self.paid_response_operations else "request"
            freeze(self.directory / f"{response_prefix}-{self.attempt_index:02d}-relation-projection.json", {
                "operation":operation,"source_response_sha256":sha256((self.directory / f"{response_prefix}-{self.attempt_index:02d}-response.json").read_bytes()).hexdigest(),
                "scope":self.arbitration_scope,"canonical_output_sha256":sha256(canonical(_remap(result,inverse)).encode()).hexdigest(),
                "role_assignment_origin":"model","new_model_calls":0})
        return _remap(result, {alias: original for original, alias in aliases.items()})

    async def strict_selection(self, raw_case):
        from app.agents.strict_evidence import prepare_selection_request, canonical_selection_wire, StrictEvidenceError
        if self.strict_balanced_enabled:
            from app.agents.compact_selection import prepare_compact_selection_request, canonical_compact_selection
            prepare_selection_request, canonical_selection_wire = prepare_compact_selection_request, canonical_compact_selection
        schema, messages = prepare_selection_request(raw_case)
        value = await self.invoke_strict("strict_selection", schema, messages)
        try:
            return canonical_selection_wire(raw_case, value).model_dump()
        except StrictEvidenceError:
            raise ModelOutputInvalid("选证结果未完整覆盖原段或含未知来源。", stage="strict_selection",
                code="validation_failed", details={"check":"strict_selection_contract"}) from None

    async def strict_annotation(self, raw_case, selection):
        from app.agents.strict_evidence import (prepare_annotation_request, canonical_annotation_wire,
            annotation_time_projection, StrictEvidenceError)
        schema, messages = prepare_annotation_request(raw_case, selection)
        value = await self.invoke_strict("strict_annotation", schema, messages)
        try:
            result = canonical_annotation_wire(raw_case, selection, value).model_dump()
            _projected, changes = annotation_time_projection(raw_case, value)
            prefix = "cache" if "strict_annotation" in self.response_reuse_by_operation else "request"
            freeze(self.directory / f"{prefix}-{self.attempt_index:02d}-field-projection.json", {
                "operation":"strict_annotation", "changes":changes,
                "original_model_response_preserved":True,
                "allowed_projection":"nonliteral_argument_time_to_empty_only"})
            return result
        except StrictEvidenceError as error:
            messages = {"annotation_number_not_in_source":"注释含原段中不存在的数字，未发布结果。",
                "nonempty_fact_des_required":"注释缺少实质事实描述，未发布结果。",
                "nonempty_argument_content_required":"注释缺少实质论元内容，未发布结果。"}
            raise ModelOutputInvalid(messages.get(error.code,"入选证据的注释未通过原文与字段核验。"),
                stage="strict_annotation", code="validation_failed", details={"check":error.code}) from None
