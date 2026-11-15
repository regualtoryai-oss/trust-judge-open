"""One LangChain strict source-selection call and the existing three-call tail.

This adapter adds no model loop, retries, synthetic DSH tools or paid replay.
The pure source-catalog compiler and the independent workflow replay own source
geometry. The two inherited transports retain separate real HTTP audit logs.
"""
import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.provider import ModelOutputInvalid
from app.agents.strict_transport import ROOT, StrictWorkbenchProvider, freeze, canonical
from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL


PROMPT_VERSION = "trusted-analysis-langchain-source-bundles-v1"
PROOF_FORMAT = "langchain-source-bundle-workbench-proof-v1"
TAIL_OPERATIONS = {"authority_analyst", "cross_validation_analyst", "arbitrate"}
ANALYST_OPERATIONS = {"authority_analyst", "cross_validation_analyst"}
COMPILE_FAILURE_MESSAGES = {
    "lc_bundle_unknown_or_duplicate_window": "选证包含未知或重复的原文窗口编号，未发布。",
    "lc_bundle_duplicate_original_start_anchor": "多条选证共用同一原始起始段，未发布。",
    "lc_bundle_time_not_literal": "选证时间不是所选原文中的逐字文本，未发布。",
    "lc_bundle_entity_not_literal_or_duplicate": "选证实体未在所选原文中逐字出现或存在重复，未发布。",
    "lc_bundle_annotation_number_not_in_source": "选证注释包含所选原文不支持的数值或单位，未发布。",
    "lc_bundle_empty_annotation": "选证缺少事实概括或内容说明，未发布。",
    "lc_bundle_output_schema": "选证编号或注释字段不符合规定结构，未发布。",
    "lc_bundle_output_object_required": "选证结果不是规定的结构化对象，未发布。",
    "lc_bundle_prepared_contract_changed": "选证目录与绑定的原始资料不一致，未发布。",
    "lc_bundle_graded_selection_inconsistent_background_score": "分级选证把背景或无关资料标为高分，评分与用途不一致；未启动后续研判。",
    "paragraph_quality_output_schema": "完整段落选证缺少必填决定或字段结构不合规，未启动后续研判。",
    "paragraph_quality_duplicate_annotation_source": "同一原段出现重复字段注释，未启动后续研判。",
    "paragraph_quality_retained_annotation_exact_coverage": "已保留段落与字段注释未一一对应，未启动后续研判。",
    "paragraph_quality_prepared_contract_changed": "完整段落选证目录与冻结原始资料不一致，未启动后续研判。",
    "evidence_fields_output_schema": "已选原段的字段结构不合规，未启动后续研判。",
    "evidence_fields_literal_not_in_parent": "字段文本无法在所属完整原段逐字定位，未启动后续研判。",
    "evidence_fields_summary_number_not_in_parent": "事实概括含所属原段未支持的数值，未启动后续研判。",
    "evidence_fields_empty_fact_description": "已选原段缺少事实概括，未启动后续研判。",
    "evidence_fields_duplicate_field_span": "同一字段包含重复的原文片段，未启动后续研判。",
    "evidence_fields_duplicate_entity_text": "同一原段的实体重复，未启动后续研判。",
}


def _compile_failure(error):
    # Exception text may contain source/provider content. Only exact, known
    # compiler codes become public diagnostics; never reflect arbitrary text.
    rule = error.args[0] if error.args else None
    check = rule if type(rule) is str and rule in COMPILE_FAILURE_MESSAGES else "lc_bundle_source_compile_failed"
    message = COMPILE_FAILURE_MESSAGES.get(check, "所选来源编号或注释未通过业务校验，未发布。")
    return ModelOutputInvalid(message, stage="source_bundle_selection", code="lc_bundle_source_compile_failed",
        details={"check": check})


def _bound(path):
    path = Path(path)
    return {"file": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}


def _known(records):
    if (not isinstance(records, list) or any(not isinstance(row, dict)
            or row.get("usage_source") != "provider"
            or any(type(row.get(key)) is not int or row[key] < 0
                   for key in ("input_tokens", "output_tokens", "total_tokens"))
            or row["input_tokens"] + row["output_tokens"] != row["total_tokens"]
            for row in records)):
        raise ModelOutputInvalid("模型用量未完整结算，已阻止后续研判。", stage="source_bundle_selection",
            code="experiment_usage_unknown")
    return sum(row["total_tokens"] for row in records)


def _messages(prepared):
    classes = {"system": SystemMessage, "user": HumanMessage}
    rows = prepared.get("messages")
    if (not isinstance(rows, list) or not rows or any(not isinstance(row, dict)
            or set(row) != {"role", "content"} or row["role"] not in classes
            or not isinstance(row["content"], str) for row in rows)):
        raise ValueError("lc_bundle_exact_prepared_messages_required")
    return [classes[row["role"]](content=row["content"]) for row in rows]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("lc_bundle_duplicate_raw_json_key")
        result[key] = value
    return result


def _verify_front_delivery(prepared, output, requests):
    """Read only visible HTTP JSON and tool arguments; never hidden reasoning."""
    if len(requests) != 1 or "syntax-recovery" in requests[0]:
        raise ValueError("lc_bundle_one_unrepaired_actual_request_required")
    request = requests[0]
    required = {"actual_wire", "wire", "preflight", "settlement", "response", "finished"}
    if not required <= set(request):
        raise ValueError("lc_bundle_actual_http_audit_required")
    def load(name):
        descriptor = request[name]
        data = Path(descriptor["file"]).read_bytes()
        if sha256(data).hexdigest() != descriptor["sha256"]:
            raise ValueError("lc_bundle_actual_http_bytes_changed")
        return json.loads(data, object_pairs_hook=_unique_object)
    actual, counted = load("actual_wire"), load("wire")
    if (actual != counted or actual.get("messages") != prepared["messages"]
            or actual.get("tools") != [prepared["wire_tool"]]
            or actual.get("tool_choice", {}).get("function", {}).get("name") != prepared["function_name"]):
        raise ValueError("lc_bundle_exact_full_prepared_input_required")
    visible = load("response")
    calls = visible.get("tool_calls_wire")
    if (not isinstance(calls, list) or len(calls) != 1 or calls[0].get("type") != "function"
            or calls[0].get("function", {}).get("name") != prepared["function_name"]
            or not isinstance(calls[0]["function"].get("arguments"), str)
            or json.loads(calls[0]["function"]["arguments"], object_pairs_hook=_unique_object) != output):
        raise ValueError("lc_bundle_exact_unique_raw_model_fields_required")
    return True


class _SingleFrontProvider(StrictWorkbenchProvider):
    """Keep exactly one actual strict invocation and observe its existing bill."""

    def __init__(self, settings, *, threshold):
        super().__init__(settings, threshold=threshold)
        self.usage_records = []
        self.single_invocation_entered = False

    def actual_request_started(self, operation, ordinal):
        if callable(self.workbench_event):
            self.workbench_event("strict_request_started", {"operation": operation,
                "request_sequence": ordinal, "actual_request_started": True})

    def _record_usage(self, operation, usage):
        # The normal task-local recorder sees each real call exactly once.
        super()._record_usage(operation, usage)
        self.usage_records.append({"operation": operation, "model": self.model_name,
            "usage_source": "provider" if usage is not None else "unavailable", "cost": None,
            **({key: usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")}
               if usage is not None else {key: None for key in ("input_tokens", "output_tokens", "total_tokens")})})

    async def invoke_strict(self, operation, schema, messages):
        if operation != "strict_selection" or self.single_invocation_entered:
            raise ValueError("lc_bundle_front_exactly_one_strict_selection_attempt")
        self.single_invocation_entered = True
        return await super().invoke_strict(operation, schema, messages)


class LCBundleWorkbenchProvider(StrictWorkbenchProvider):
    """Opt-in single complete-catalog selection; ordinary isolated A/B/arb."""

    source_bundle_enabled = True
    source_bundle_protocol = SOURCE_BUNDLE_PROTOCOL
    strict_evidence_enabled = False

    def __init__(self, settings, *, threshold=None):
        config = deepcopy(settings.source_bundle_config)
        if (not settings.model_source_bundle_enabled or settings.model_provider != "deepseek"
                or settings.model_name != "deepseek-v4-pro" or config.get("front_framework") != "langchain"
                or settings.strict_evidence_response_reuse_closures and not (
                    settings.source_bundle_recover_front_run_id
                    and set(settings.strict_evidence_response_reuse_operations) in (TAIL_OPERATIONS, ANALYST_OPERATIONS))):
            raise ValueError("lc_bundle_explicit_single_call_configuration_required")
        tail_settings = settings.model_copy(update={"strict_evidence_balanced_enabled": True})
        super().__init__(tail_settings, threshold=threshold)
        self.prompt_version = PROMPT_VERSION
        self.source_bundle_config = config
        if settings.source_bundle_context_passages_enabled:
            from app.domain.snapshot import SOURCE_CONTEXT_PROTOCOL
            self.source_bundle_protocol = SOURCE_CONTEXT_PROTOCOL
        self.bundle_directory = self.directory / "source-bundle"
        self.bundle_directory.mkdir(exist_ok=False)
        self._previous_records, self._front_records = [], []
        self._previous_set = self._tail_seeded = self._selection_started = self._selection_failed = False
        self.front_provider = None
        self.front_directory = None
        self._front_started_event = False
        self._tail_forecast_lock = asyncio.Lock()
        self._tail_forecast_recorded = False
        self._optional_field_quality = None
        self._sealed_front_recovery_context = None
        self.source_archives = {}
        sources = list((ROOT / "app").rglob("*.py")) + [ROOT / "scripts/lc_bundle_case.py"]
        if settings.source_bundle_local_rank_enabled:
            sources.append(ROOT / "scripts/local_parent_rank.py")
        for source in sources:
            relative = source.relative_to(ROOT).as_posix()
            destination = self.bundle_directory / "source-archive" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(source.read_bytes())
            self.source_archives[relative] = _bound(destination)
        freeze(self.directory / "source-bundle-episode.json", {
            "format": "langchain-source-bundle-workbench-episode-v1", "prompt_version": PROMPT_VERSION,
            "framework": "langchain", "source_bundle_enabled": True, "strict_evidence_enabled": False,
            "base_episode_scope": "inherited_strict_tail_only", "base_episode": _bound(self.directory / "episode.json"),
            "pipeline": ["langchain_single_source_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"],
            "source_bundle_config": config, "tail_scoped_arbitration": True,
            "whole_token_stop": self.threshold, "max_total_requests": self.max_requests,
            "source_archives": self.source_archives, "front_max_actual_requests": 1,
            "completed_langgraph_source_stage_resume_supported": True,
            "partial_front_stage_resume_supported": False, "fixture": False})

    def _budget_enabled(self):
        if (self.settings.source_bundle_recover_front_run_id
                and self.paid_response_operations in (TAIL_OPERATIONS, ANALYST_OPERATIONS)):
            # Explicit sealed-source recovery uses exact paid A/B caches.
            # Cached operations cannot fall back to HTTP. Any remaining arb
            # still passes the inherited native whole-episode token/call guard;
            # do not reserve paid calls for the two zero-HTTP cache reads.
            return False
        return ((self.settings.source_bundle_graded_selection or self.settings.source_bundle_paragraph_quality_enabled)
                and self.settings.source_bundle_compact_analyses_enabled)

    def _context_features(self):
        return {key: value for feature in ("complete_grading", "hybrid_retrieval", "paragraph_quality", "paragraph_quality_compact", "parent_token_pack")
                if self.source_bundle_config.get(feature) is True
                for key, value in ((feature, True),
                    (feature + "_policy", deepcopy(self.source_bundle_config[feature + "_policy"])))}

    def _check_budget_plan(self, label, plan):
        binding = freeze(self.directory / ("budget-" + label + ".json"), plan)
        self._emit("episode_budget_planned", {"stage": "source_bundle_selection" if label == "before-first-paid" else "analysis",
            "name": "whole_chain_budget_forecast", "kind": "budget", "actual_request_started": False,
            "summary": {key: plan[key] for key in ("phase", "known_tokens", "planned_whole_tokens", "threshold", "admitted")},
            "receipt_sha256": binding["sha256"], "is_actual_usage": False})
        if not plan["admitted"]:
            raise ModelOutputInvalid("整条研判链的预估用量超过已冻结保护线，未启动下一步模型调用。",
                stage="source_bundle_selection" if label == "before-first-paid" else "analysis",
                code="experiment_usage_stop", details={"known_tokens": plan["known_tokens"],
                    "planned_whole_tokens": plan["planned_whole_tokens"],
                    "required_remaining_tokens": plan["required_remaining_planning_tokens"],
                    "frozen_token_stop": plan["threshold"], "budget_phase": label})
        return binding

    def _incremental_front_budget(self):
        return (self.source_bundle_config.get("whole_chain_planning", {}).get("front_token_budget_semantics")
                == "new_front_tokens_excluding_paid_ancestors")

    def _front_limit(self, parent):
        allowance = self.source_bundle_config["front_token_stop"]
        return min(parent + allowance if self._incremental_front_budget() else allowance,
                   self.threshold - self.source_bundle_config["tail_reserved_tokens"])

    def _front_budget_metadata(self, parent):
        return ({"front_token_budget_semantics": "new_front_tokens_excluding_paid_ancestors",
            "configured_new_front_token_budget": self.source_bundle_config["front_token_stop"],
            "cumulative_front_guard_threshold": self._front_limit(parent)}
            if self._incremental_front_budget() else {})

    def bind_sealed_front_recovery(self, context):
        if (self._sealed_front_recovery_context is not None or self._selection_started or self._tail_seeded
                or not isinstance(context, dict) or not self.settings.source_bundle_recover_front_run_id):
            raise ValueError("lc_bundle_sealed_recovery_binding_once_required")
        self._sealed_front_recovery_context = deepcopy(context)

    async def _recover_sealed_front(self, question, documents, paragraphs, excluded_spans):
        from app.workflows.bundle_workbench import compile_result
        context = self._sealed_front_recovery_context
        if (self._selection_started or self._tail_seeded or self.attempt_index or not isinstance(context, dict)
                or context.get("origin_run_id") != self.settings.source_bundle_recover_front_run_id
                or context.get("origin_question_id") != question.get("id")):
            raise ModelOutputInvalid("已付选证恢复缺少原任务、原题或费用绑定，未调用模型。",
                stage="source_bundle_selection", code="sealed_front_recovery_not_authorized")
        self._selection_started = self._selection_failed = True
        descriptor = {"file": str(self.settings.source_bundle_recover_front_response),
            "sha256": self.settings.source_bundle_recover_front_response_sha256}
        original_bytes = Path(descriptor["file"]).read_bytes()
        if sha256(original_bytes).hexdigest() != descriptor["sha256"]:
            raise ModelOutputInvalid("原已付选证封印已变化，未调用模型。", stage="source_bundle_selection",
                code="sealed_front_recovery_changed")
        response = json.loads(original_bytes, object_pairs_hook=_unique_object)
        # Both the original native codec and current canonical source checks
        # are replayed. Nothing is reconstructed from a partial model answer.
        projection = await asyncio.to_thread(compile_result, question, paragraphs, documents, response,
            excluded_spans=sorted(map(list, excluded_spans)))
        prior = _known(self._previous_records)
        if (response.get("usage_report", {}).get("known_usage_complete") is not True
                or type(response["usage_report"].get("whole_episode_tokens")) is not int
                or prior < response["usage_report"]["whole_episode_tokens"]):
            raise ModelOutputInvalid("原已付选证费用未完整继承，未调用模型。", stage="source_bundle_selection",
                code="sealed_front_recovery_cost_not_inherited")
        receipt = freeze(self.bundle_directory / "sealed-front-recovery.json", {
            "format": "lc-sealed-front-recovery-v1", **deepcopy(context), "response": descriptor,
            "current_question_sha256": sha256(canonical(question).encode("utf8")).hexdigest(),
            "current_sources_sha256": sha256(canonical({"documents": documents, "paragraphs": paragraphs}).encode("utf8")).hexdigest(),
            "original_front_proof": deepcopy(response["proof"]["closed_receipt"]),
            "current_recovery_source_archives": deepcopy(self.source_archives),
            "source_proof_replay_verified": True, "current_canonical_projection_verified": True,
            "original_response_modified": False, "fields_modified": False,
            "new_model_calls": 0, "new_tokens": 0, "prior_known_tokens": prior,
            "evidence_count": len(projection["evidence"]), "not_a_new_native_front": True})
        self._emit("sealed_front_recovery_completed", {"stage": "source_bundle_selection",
            "origin_run_id": context["origin_run_id"], "recovery_receipt": receipt,
            "actual_request_started": False, "new_model_calls": 0, "new_tokens": 0})
        self._emit("tool_completed", {"stage": "source_bundle_selection", "kind": "local", "engine": "Program",
            "name": "sealed_front_recovery", "call_id": self.directory.name + ":sealed-front-recovery",
            "actual_request_started": False, "summary": ("原已付选证封印重放通过，原始字段未改；本节点0次模型调用，费用继续累计。"
                + (f" {response['artifact']['optional_field_quality_summary']['quarantined_value_count']} 项可选字段已隔离。"
                   if type(response.get("artifact", {}).get("optional_field_quality_summary", {}).get("quarantined_value_count")) is int else ""))})
        self._selection_failed = False
        self._ensure_tail_seeded()
        return response

    def _tail_budget_arguments(self):
        # Completed source checkpoints resume without entering selection again.
        # Seed their immutable paid ancestors before forecasting any new node.
        self._ensure_tail_seeded()
        report = self.guard.report()
        if report["unknown_usage_requests"]:
            raise ModelOutputInvalid("模型用量尚未完整结算，已停止后续调用。", stage="analysis", code="experiment_usage_unknown")
        arguments = {"known_tokens": self.prior_known_tokens + report["known_total_tokens"],
                "known_calls": len(self._previous_records) + len(self._front_records) + report["actual_requests_started"],
                "threshold": self.threshold, "max_calls": self.max_requests,
                "model_name": self.model_name, "output_limits": self.output_limits,
                "compact_analyses": True, "issue_frame_enabled": self.settings.source_bundle_issue_frame_enabled}
        if self.settings.source_bundle_paragraph_quality_enabled:
            from app.agents.episode_budget import PARAGRAPH_QUALITY_POLICY
            arguments["planning_policy"] = PARAGRAPH_QUALITY_POLICY
            if self._incremental_front_budget():
                from app.agents.episode_budget import PARAGRAPH_QUALITY_COMPACT_V4_POLICY
                arguments["planning_policy"] = PARAGRAPH_QUALITY_COMPACT_V4_POLICY
                if self.settings.source_bundle_parent_token_pack_enabled:
                    from app.agents.episode_budget import PARAGRAPH_QUALITY_BUDGET_V5_POLICY
                    arguments["planning_policy"] = PARAGRAPH_QUALITY_BUDGET_V5_POLICY
        return arguments

    async def analyze(self, role, question, evidence, snapshot_id):
        if self._budget_enabled():
            # The inherited transport serializes real requests. Reserve both
            # independent inputs in one immutable plan before either is sent.
            async with self._tail_forecast_lock:
                if not self._tail_forecast_recorded:
                    from app.agents.episode_budget import plan_tail
                    plan = await asyncio.to_thread(plan_tail, question, evidence,
                        snapshot_id=snapshot_id, **self._tail_budget_arguments())
                    self._check_budget_plan("selected-sources", plan)
                    self._tail_forecast_recorded = True
        return await super().analyze(role, question, evidence, snapshot_id)

    async def arbitrate(self, question, evidence, analyses):
        if (self._budget_enabled() or self.settings.source_bundle_recover_front_run_id
                and self.paid_response_operations == ANALYST_OPERATIONS):
            from app.agents.episode_budget import plan_tail
            plan = await asyncio.to_thread(plan_tail, question, evidence, analyses=analyses,
                                           **self._tail_budget_arguments())
            self._check_budget_plan("actual-analyses", plan)
        return await super().arbitrate(question, evidence, analyses)

    def set_previous_usage(self, records):
        if self._previous_set or self._selection_started or self.attempt_index or self._tail_seeded:
            raise ValueError("lc_bundle_previous_usage_once_before_calls_required")
        _known(records)
        self._previous_records, self._previous_set = deepcopy(records), True

    def _ensure_tail_seeded(self):
        if self._tail_seeded:
            return
        if self._selection_failed or (not self._selection_started and not self._previous_records):
            raise ModelOutputInvalid("完整原文选证尚未完成，未启动后续研判。", stage="source_bundle_selection",
                code="source_bundle_tail_not_admitted")
        # A resumed completed source checkpoint is verified by the workflow.
        # Its previous records include the paid front; this provider does not
        # invoke it again or insert a second front usage record.
        records = deepcopy(self._previous_records + self._front_records)
        _known(records)
        super().seed_prior_usage(records)
        self._tail_seeded = True

    async def invoke_strict(self, operation, schema, messages):
        if operation not in TAIL_OPERATIONS:
            raise ValueError("lc_bundle_outer_only_three_normal_tail_operations")
        self._ensure_tail_seeded()
        return await super().invoke_strict(operation, schema, messages)

    def capacity_policy(self, operation):
        if operation == "source_bundle_selection":
            capacity = self.output_limits["strict_selection"]
            return {"initial_tokens": capacity, "max_tokens": capacity, "retries": 0}
        return super().capacity_policy(operation)

    def _emit(self, event, data):
        if callable(self.workbench_event):
            self.workbench_event(event, deepcopy(data))

    def _front_event(self, event, data):
        # The base HTTP boundary owns this notification; entering a Python
        # provider method or writing a preflight is not a sent request.
        if event in {"strict_request_started", "model_request_started"}:
            if data.get("operation") != "strict_selection" or data.get("actual_request_started") is not True:
                return
            self._front_started_event = True
            self._emit("tool_started", {"stage": "source_bundle_selection", "node": "source_bundle_selection",
                "engine": "LangChain", "kind": "model", "name": "langchain_source_bundle_select",
                "call_id": self.directory.name + ":lc-front:1", "actual_request_started": True,
                "request_sequence": 1, "summary": ("单次读取候选完整原段，自主选择直接论据与关键限定并提取事实、六论元和实体；不设评分阈值或证据条数上限。"
                    if self.settings.source_bundle_paragraph_quality_enabled else "单次读取完整候选原文并给来源编号评0–3分；程序保留最多10项及必要反方输入，临时用途不是最终冲突标签。"
                    if self.settings.source_bundle_graded_selection else "单次读取完整候选原文，只选择合法来源编号；程序取原文首句作为字段预览，研判仍据全文。"
                    if self.settings.source_bundle_selection_only else "单次读取完整候选目录并选择合法来源编号和注释。")})
        else:
            self._emit(event, data)

    def _front_requests(self):
        if self.front_directory is None:
            return []
        rows = []
        for path in sorted(self.front_directory.glob("request-*-preflight.json")):
            prefix = path.name.removesuffix("preflight.json")
            row = {"operation": "strict_selection", "preflight": _bound(path)}
            for suffix, field in (("wire", "wire"), ("actual-wire", "actual_wire"), ("response", "response"),
                    ("settlement", "settlement"), ("finished", "finished"), ("syntax-recovery", "syntax-recovery")):
                item = self.front_directory / (prefix + suffix + ".json")
                if item.is_file(): row[field] = _bound(item)
            rows.append(row)
        return rows

    def _close_front(self, *, status, raw_descriptor, source_binding, artifact_descriptor=None, error_code=None):
        records = deepcopy(self.front_provider.usage_records) if self.front_provider is not None else []
        self._front_records = records
        try:
            new_tokens, unknown = _known(records), False
        except ModelOutputInvalid:
            new_tokens, unknown = None, True
        parent = _known(self._previous_records)
        report = {"framework": "langchain", "execution_mode": "workbench", "real_api_requests": len(records),
            "simulated_usage_not_bill": False, "new_request_count": len(records), "new_tokens": new_tokens,
            "parent_tokens": parent, "whole_episode_tokens": None if unknown else parent + new_tokens,
            "known_usage_complete": not unknown, "front_token_stop": self._front_limit(parent),
            **self._front_budget_metadata(parent),
            "whole_token_stop": self.threshold, "reserved_tail_tokens": self.source_bundle_config["tail_reserved_tokens"],
            "partial_front_stage_resumed": False}
        requests = self._front_requests()
        receipt = {"format": "langchain-source-bundle-closed-receipt-v1", "status": status,
            "episode_closed": True, "known_usage_complete": not unknown, "source_binding": source_binding,
            "raw_case_sha256": raw_descriptor["sha256"],
            "artifact_sha256": artifact_descriptor["sha256"] if artifact_descriptor is not None else None,
            "front_requests": requests, "usage_records": records, "failure_code": error_code, **report}
        report_descriptor = freeze(self.bundle_directory / "usage-report.json", report)
        receipt_descriptor = freeze(self.bundle_directory / "closed-receipt.json", receipt)
        if records:
            self._emit("tool_completed" if status == "succeeded" else "tool_failed", {
                "stage": "source_bundle_selection", "node": "source_bundle_selection", "engine": "LangChain",
                "kind": "model", "name": "langchain_source_bundle_select", "call_id": self.directory.name + ":lc-front:1",
                "actual_request_started": True, "known_usage": not unknown,
                "summary": {"actual_usage": records[0] if len(records) == 1 else None,
                    "known_episode_tokens": report["whole_episode_tokens"], "failure_code": error_code,
                    **(self._optional_field_quality or {})}})
        return report, receipt_descriptor, report_descriptor

    async def select_source_bundles(self, question, documents, paragraphs, excluded_spans=()):
        if self.settings.source_bundle_recover_front_run_id:
            return await self._recover_sealed_front(question, documents, paragraphs, excluded_spans)
        from app.domain.lc_bundle_case import prepare_lc_case, selection_schema, compile_business
        if self._selection_started or self._tail_seeded or self.attempt_index:
            raise ValueError("lc_bundle_selection_once_per_provider_required")
        self._selection_started = self._selection_failed = True
        original = {"question_id": question.get("raw_question_id", question.get("question_id")), "question": question["question"]}
        raw = {"format": "frozen-source-bundle-case-v1", "question": original,
            "documents": deepcopy(documents), "paragraphs": deepcopy(paragraphs),
            "limits": {"max_search_calls": 3, "max_read_calls": 2, "max_paragraphs": 5,
                "max_read_anchors": 12, "max_read_codepoints": 12000, "max_read_payload_bytes": 150000}}
        excluded = sorted(map(list, excluded_spans))
        raw_descriptor = freeze(self.bundle_directory / "raw-case.json", raw)
        documents_descriptor = freeze(self.bundle_directory / "documents.json", documents)
        paragraphs_descriptor = freeze(self.bundle_directory / "paragraphs.json", paragraphs)
        previous = freeze(self.bundle_directory / "previous-usage.json", {"records": self._previous_records})
        rank_plan, rank_sha, rank_proof = None, None, None
        if self.settings.source_bundle_local_rank_enabled:
            from app.domain.local_parent_rank import build_local_rank, verify_local_rank
            call_id = self.directory.name + ":local-parent-rank"
            self._emit("tool_started", {"stage": "source_bundle_selection", "node": "source_bundle_selection",
                "engine": "Local BGE", "kind": "retrieval", "name": "local_complete_parent_rank",
                "call_id": call_id, "actual_request_started": False,
                "summary": "对全部原文窗口做本地正文排序，再提供完整候选原段；不调用付费模型。"})
            try:
                ranked = await asyncio.to_thread(build_local_rank, raw,
                    raw_case_sha256=raw_descriptor["sha256"],
                    model_manifest=self.settings.source_bundle_local_rank_manifest,
                    expected_manifest_sha256=self.settings.source_bundle_local_rank_manifest_sha256,
                    cache_dir=self.settings.data_dir / "local-parent-rank-cache",
                    output_dir=self.bundle_directory / "local-parent-rank")
                rank_plan, rank_sha, rank_proof = ranked["rank_plan"], ranked["rank_plan_sha256"], ranked["proof"]
                verify_local_rank(raw, rank_plan, rank_proof, raw_case_sha256=raw_descriptor["sha256"],
                    expected_model_manifest_sha256=self.settings.source_bundle_local_rank_manifest_sha256)
                freeze(self.bundle_directory / "rank-plan.json", rank_plan)
            except (KeyError, TypeError, ValueError, OSError, RuntimeError):
                self._emit("tool_failed", {"stage": "source_bundle_selection", "node": "source_bundle_selection",
                    "engine": "Local BGE", "kind": "retrieval", "name": "local_complete_parent_rank",
                    "call_id": call_id, "actual_request_started": False,
                    "summary": {"status": "failed", "paid_model_calls": 0}})
                raise ModelOutputInvalid("本地全文排序未通过来源或模型完整性检查，未调用付费模型。",
                    stage="source_bundle_selection", code="lc_bundle_local_rank_failed") from None
            self._emit("tool_completed", {"stage": "source_bundle_selection", "node": "source_bundle_selection",
                "engine": "Local BGE", "kind": "retrieval", "name": "local_complete_parent_rank",
                "call_id": call_id, "actual_request_started": False,
                "summary": {"status": "succeeded", "paid_model_calls": 0,
                    "corpus_paragraphs": len(paragraphs), "rank_plan_sha256": rank_sha}})
        rank_options = {"rank_plan": rank_plan, "rank_plan_sha256": rank_sha} if rank_plan is not None else {}
        if self.settings.source_bundle_selection_only:
            rank_options["selection_only"] = True
        if self.settings.source_bundle_graded_selection:
            rank_options["graded_selection"] = True
        context_inputs = None
        context_options = {}
        try:
            if self.settings.source_bundle_context_passages_enabled:
                from app.domain.context_source_inputs import prepare_inputs
                from app.domain.lc_context_case import selection_schema
                context_options = {"raw_case_sha256": raw_descriptor["sha256"], "rank_plan": rank_plan,
                    "rank_plan_sha256": rank_sha, "local_rank_proof": rank_proof,
                    "expected_model_manifest_sha256": self.settings.source_bundle_local_rank_manifest_sha256,
                    "excluded_spans": excluded}
                for feature in ("complete_grading", "hybrid_retrieval"):
                    if self.source_bundle_config.get(feature) is True:
                        context_options[feature] = True
                context_inputs = await asyncio.to_thread(prepare_inputs, raw, **context_options)
                prepared = context_inputs["prepared"]
                if self.settings.source_bundle_complete_grading_enabled:
                    from app.domain.complete_context_grading import complete_schema as selection_schema
                freeze(self.bundle_directory / "context-plan.json", context_inputs["context_plan"])
                freeze(self.bundle_directory / "context-window-scores.json", context_inputs["window_scores"])
            elif self.settings.source_bundle_paragraph_quality_enabled:
                from app.domain.paragraph_quality_case import prepare_paragraph_quality, paragraph_quality_schema as selection_schema
                if self.settings.source_bundle_paragraph_quality_compact_enabled:
                    from app.domain.paragraph_quality_compact_case import prepare_paragraph_quality, paragraph_quality_schema as selection_schema
                if self.settings.source_bundle_parent_token_pack_enabled:
                    from app.domain.paragraph_quality_budget_case import prepare_paragraph_quality, paragraph_quality_schema as selection_schema
                    self._emit("tool_started", {"stage": "source_bundle_selection", "kind": "local", "engine": "BM25 / token pack",
                        "name": "parent_token_pack", "call_id": self.directory.name + ":parent-token-pack",
                        "actual_request_started": False, "summary": "按原题多视角检索，按排名公平提供完整原段；不以短段优先，未提供仍未知。"})
                prepared = prepare_paragraph_quality(raw, excluded_spans=excluded, **rank_options)
                if self.settings.source_bundle_parent_token_pack_enabled:
                    receipt = prepared["retrieval"]["parent_token_pack_receipt"]
                    pool, provided, proxy = (receipt[key] for key in
                        ("recall_pool_count", "provided_parent_count", "provided_payload_proxy_tokens"))
                    if any(type(value) is not int or value < 0 for value in (pool, provided, proxy)):
                        raise ValueError("lc_bundle_parent_token_pack_receipt_invalid")
                    self._emit("tool_completed", {"stage": "source_bundle_selection", "kind": "local", "engine": "BM25 / token pack",
                        "name": "parent_token_pack", "call_id": self.directory.name + ":parent-token-pack",
                        "actual_request_started": False, "summary": (
                            f"检索{pool}段，预算完整提供{provided}段；未提供仍未知；内容成本估计{proxy} tokens（非账单）。")})
            else:
                prepared = prepare_lc_case(raw, excluded_spans=excluded, **rank_options)
            schema = selection_schema(prepared)
            messages = _messages(prepared)
        except (KeyError, TypeError, ValueError):
            raise ModelOutputInvalid("原始资料目录未通过完整性校验，未调用模型。", stage="source_bundle_selection",
                code="lc_bundle_source_catalog_invalid") from None
        prepared_descriptor = freeze(self.bundle_directory / "prepared-case.json", prepared)
        initial_budget = None
        if self._budget_enabled():
            from app.agents.episode_budget import plan_source_episode
            plan = await asyncio.to_thread(plan_source_episode, question, prepared,
                known_tokens=_known(self._previous_records), known_calls=len(self._previous_records),
                threshold=self.threshold, max_calls=self.max_requests, model_name=self.model_name,
                output_limits=self.output_limits, compact_analyses=True,
                issue_frame_enabled=self.settings.source_bundle_issue_frame_enabled)
            initial_budget = self._check_budget_plan("before-first-paid", plan)
        parent = _known(self._previous_records)
        front_limit = self._front_limit(parent)
        if front_limit <= 0 or self.max_requests - len(self._previous_records) < 4:
            raise ModelOutputInvalid("本题不足以保留单次选证与三次研判调用，未调用模型。", stage="source_bundle_selection",
                code="experiment_usage_stop", details={"known_tokens": parent})
        front_settings = self.settings.model_copy(update={"data_dir": self.bundle_directory / "front",
            "strict_evidence_max_requests": self.max_requests - 3,
            "strict_evidence_response_reuse_closures": [], "strict_evidence_response_reuse_operations": []})
        self.front_provider = _SingleFrontProvider(front_settings, threshold=front_limit)
        self.front_directory = self.front_provider.directory
        self.front_provider.workbench_event = self._front_event
        self.front_provider.seed_prior_usage(deepcopy(self._previous_records))
        launch = freeze(self.bundle_directory / "launch-manifest.json", {
            "format": "langchain-source-bundle-launch-v1", "framework": "langchain",
            "raw_case_sha256": raw_descriptor["sha256"], "source_binding": prepared["source_binding"],
            "prepared_case": prepared_descriptor, "front_episode": _bound(self.front_directory / "episode.json"),
            "model": self.model_name, "front_token_stop": front_limit, "whole_token_stop": self.threshold,
            **self._front_budget_metadata(parent),
            "max_requests": self.max_requests, "front_max_requests": 1,
            "front_guard_max_requests": self.front_provider.max_requests,
            "parent_request_count": len(self._previous_records), "parent_tokens": parent,
            "required_tail_requests": 3, "reserved_output_tokens": self.front_provider.output_limits["strict_selection"], "safety_tokens": 512,
            **({"selection_only": True} if self.settings.source_bundle_selection_only else {}),
            **({"graded_selection": True, "graded_selection_policy": self.source_bundle_config["graded_selection_policy"],
                "selection_output_tokens": 2048} if self.settings.source_bundle_graded_selection else {}),
            **({"compact_analyses": True} if self.settings.source_bundle_compact_analyses_enabled else {}),
            **({"issue_frame": True} if self.settings.source_bundle_issue_frame_enabled else {}),
            **self._context_features(),
            **({"selection_output_tokens": self.output_limits["strict_selection"]}
               if self.settings.source_bundle_paragraph_quality_enabled else {}),
            **({"context_passages": True, "source_bundle_protocol": self.source_bundle_protocol,
                "context_plan": _bound(self.bundle_directory / "context-plan.json"),
                "context_window_scores": _bound(self.bundle_directory / "context-window-scores.json"),
                "context_window_scores_sha256": context_inputs["window_scores_sha256"]}
               if context_inputs is not None else {}),
            **({"whole_chain_planning": self.source_bundle_config["whole_chain_planning"],
                "initial_budget_plan": initial_budget} if initial_budget is not None else {}),
            **({"local_parent_rank": {"plan_sha256": rank_sha, "proof": rank_proof,
                "config": self.source_bundle_config["local_parent_rank"]}} if rank_plan is not None else {}),
            "tail_reserved_tokens": self.source_bundle_config["tail_reserved_tokens"]})
        status, error_code, artifact_descriptor = "failed", None, None
        try:
            output = await self.front_provider.invoke_strict("strict_selection", schema, messages)
            model_output = freeze(self.bundle_directory / "model-output.json", output)
            requests = self._front_requests()
            try:
                _verify_front_delivery(prepared, output, requests)
            except (KeyError, TypeError, ValueError, OSError):
                raise ModelOutputInvalid("单次模型请求缺少完整原生审计，未采纳选证。", stage="source_bundle_selection",
                    code="lc_bundle_actual_http_audit_required") from None
            try:
                if context_inputs is not None:
                    from app.domain.context_source_inputs import compile_inputs
                    compiled = await asyncio.to_thread(compile_inputs, raw, prepared, output, **context_options)
                elif self.settings.source_bundle_paragraph_quality_enabled:
                    from app.domain.paragraph_quality_case import compile_paragraph_quality
                    if self.settings.source_bundle_paragraph_quality_compact_enabled:
                        from app.domain.paragraph_quality_compact_case import compile_paragraph_quality
                    if self.settings.source_bundle_parent_token_pack_enabled:
                        from app.domain.paragraph_quality_budget_case import compile_paragraph_quality
                    compiled = compile_paragraph_quality(raw, prepared, output, excluded_spans=excluded, **rank_options)
                else:
                    compiled = compile_business(raw, prepared, output, excluded_spans=excluded, **rank_options)
            except (KeyError, TypeError, ValueError) as failure:
                raise _compile_failure(failure) from None
            artifact_descriptor = freeze(self.bundle_directory / "artifact.json", compiled["artifact"])
            optional_quality = compiled["artifact"].get("optional_field_quality_summary")
            if isinstance(optional_quality, dict):
                self._optional_field_quality = {key: optional_quality[key] for key in
                    ("quarantined_value_count", "deduplicated_value_count", "affected_annotation_count")
                    if type(optional_quality.get(key)) is int and optional_quality[key] >= 0}
            scope = freeze(self.bundle_directory / "scope.json", compiled["scope"])
            paid_front = _known(self.front_provider.usage_records)
            if (parent + paid_front > front_limit
                    or self.front_provider.guard.report()["unknown_usage_requests"] != 0):
                raise ModelOutputInvalid("本题选证实际用量已触及保护线，未启动后续研判。", stage="source_bundle_selection",
                    code="experiment_usage_stop", details={"known_tokens": parent + paid_front})
            status = "succeeded"
        except BaseException as failure:
            error_code = getattr(failure, "code", None) or ("cancelled" if type(failure).__name__ == "CancelledError" else "lc_bundle_selection_failed")
            raise
        finally:
            report, receipt, _report = self._close_front(status=status, raw_descriptor=raw_descriptor,
                source_binding=prepared["source_binding"], artifact_descriptor=artifact_descriptor, error_code=error_code)
        response = {"artifact": compiled["artifact"], "usage_report": report, "proof": {
            "format": PROOF_FORMAT, "raw_case": raw_descriptor, "documents": documents_descriptor,
            "paragraphs": paragraphs_descriptor, "prepared_case": prepared_descriptor, "model_output": model_output,
            "artifact": artifact_descriptor, "scope": scope, "front_episode": _bound(self.front_directory / "episode.json"),
            "front_requests": self._front_requests(), "previous_usage": previous,
            "launch_manifest": launch, "closed_receipt": receipt, "source_archives": self.source_archives,
            **({"selection_only": True} if self.settings.source_bundle_selection_only else {}),
            **({"graded_selection": True, "graded_selection_policy": self.source_bundle_config["graded_selection_policy"],
                "selection_output_tokens": 2048} if self.settings.source_bundle_graded_selection else {}),
            **({"compact_analyses": True} if self.settings.source_bundle_compact_analyses_enabled else {}),
            **({"issue_frame": True} if self.settings.source_bundle_issue_frame_enabled else {}),
            **self._context_features(),
            **({"selection_output_tokens": self.output_limits["strict_selection"]}
               if self.settings.source_bundle_paragraph_quality_enabled else {}),
            **({"context_passages": True, "source_bundle_protocol": self.source_bundle_protocol,
                "context_plan": _bound(self.bundle_directory / "context-plan.json"),
                "context_window_scores": _bound(self.bundle_directory / "context-window-scores.json"),
                "context_window_scores_sha256": context_inputs["window_scores_sha256"]}
               if context_inputs is not None else {}),
            **({"whole_chain_planning": self.source_bundle_config["whole_chain_planning"],
                "initial_budget_plan": initial_budget} if initial_budget is not None else {}),
            **({"local_parent_rank": {"plan": _bound(self.bundle_directory / "rank-plan.json"),
                "plan_sha256": rank_sha, "proof": rank_proof,
                "config": self.source_bundle_config["local_parent_rank"]}} if rank_plan is not None else {}),
            "excluded_spans": excluded, "usage_report": report}}
        freeze(self.bundle_directory / "provider-response.json", response)
        self._selection_failed = False
        self._ensure_tail_seeded()
        return response


LangChainBundleWorkbenchProvider = LCBundleWorkbenchProvider
