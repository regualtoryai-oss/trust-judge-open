"""Opt-in official DSH source selection with the existing strict three-call tail.

The SDK owns its loop. Each actual native request is independently accounted;
source selection is never a synthetic ModelCostGuard ticket or a single bill.
"""
import asyncio
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from uuid import uuid4

from app.agents.provider import ModelOutputInvalid
from app.agents.strict_transport import ROOT, StrictWorkbenchProvider, freeze
from app.domain.frozen_selection_inputs import canonical
from app.domain.source_bundle_tools import SourceBundleTools
from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL


PROMPT_VERSION = "trusted-analysis-dsh-source-bundles-v1"
TAIL_OPERATIONS = {"authority_analyst", "cross_validation_analyst", "arbitrate"}
PUBLIC_TOOL_FAILURES = {
    "source_bundle_tools_invalid_source_range": "读取范围无效：每份来源范围最多包含5个原始段落。",
    "bundle_bridge_duplicate_selected_start_anchor": "选证提交包含重复起点：同一原始起点只能提交一条证据。",
}


def _known(records):
    if (not isinstance(records, list) or any(not isinstance(row, dict) or row.get("usage_source") != "provider"
            or any(type(row.get(key)) is not int or row[key] < 0 for key in ("input_tokens", "output_tokens", "total_tokens"))
            or row["input_tokens"] + row["output_tokens"] != row["total_tokens"] for row in records)):
        raise ModelOutputInvalid("历史或本次原生调用用量未知，已阻止后续模型请求。", stage="source_bundle_selection", code="experiment_usage_unknown")
    return sum(row["total_tokens"] for row in records)


def _bound(path):
    path = Path(path)
    return {"file": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}


class BundleWorkbenchProvider(StrictWorkbenchProvider):
    source_bundle_enabled = True
    source_bundle_protocol = SOURCE_BUNDLE_PROTOCOL
    strict_evidence_enabled = False

    def __init__(self, settings, *, threshold=None, fixture_plan=None):
        if not settings.model_source_bundle_enabled or settings.model_provider != "deepseek" or settings.model_name != "deepseek-v4-pro":
            raise ValueError("source_bundle_explicit_official_model_configuration_required")
        # The inherited immutable episode describes the actual scoped tail.
        # It is not edited later to pretend DSH was its old strict front half.
        tail_settings = settings.model_copy(update={"strict_evidence_balanced_enabled": True})
        super().__init__(tail_settings, threshold=threshold)
        self.prompt_version = PROMPT_VERSION
        self.source_bundle_config = deepcopy(settings.source_bundle_config)
        self._previous_records = []
        self._previous_set = self._tail_seeded = self._selection_started = self._selection_failed = False
        self._native_records = []
        self._native_tools = {}
        self._fixture_plan = Path(fixture_plan).resolve() if fixture_plan is not None else None
        self.bundle_directory = self.directory / "source-bundle"
        self.bundle_directory.mkdir(exist_ok=False)
        sources = list((ROOT / "app").rglob("*.py")) + [ROOT / "scripts/source_bundle_tool.py"]
        sources += [path for path in (ROOT / "integrations/dsh").glob("*.mjs") if not path.name.endswith(".test.mjs")]
        self.source_archives = {}
        for source in sources:
            relative = source.relative_to(ROOT).as_posix()
            destination = self.bundle_directory / "source-archive" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(source.read_bytes())
            self.source_archives[relative] = _bound(destination)
        freeze(self.directory / "source-bundle-episode.json", {
            "format": "dsh-source-bundle-workbench-episode-v1", "prompt_version": self.prompt_version,
            "source_bundle_enabled": True, "strict_evidence_enabled": False,
            "pipeline": ["official_dsh_source_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"],
            "base_episode_scope": "inherited_strict_tail_only", "base_episode": _bound(self.directory / "episode.json"),
            "tail_scoped_arbitration": True, "source_bundle_config": self.source_bundle_config,
            "whole_token_stop": self.threshold, "max_total_requests": self.max_requests,
            "source_archives": self.source_archives, "fixture": self._fixture_plan is not None,
            "native_partial_stage_resume_supported": False, "completed_langgraph_source_stage_resume_supported": True})

    def set_previous_usage(self, records):
        if self._previous_set or self._selection_started or self.attempt_index or self._tail_seeded:
            raise ValueError("source_bundle_previous_usage_must_be_set_once_before_calls")
        _known(records)
        self._previous_records = deepcopy(records)
        self._previous_set = True

    def _ensure_tail_seeded(self):
        if self._tail_seeded:
            return
        if self._selection_failed or (not self._selection_started and not self._previous_records):
            raise ModelOutputInvalid("证据选取尚未完成或其用量未知，未启动后续研判。", stage="source_bundle_selection", code="source_bundle_tail_not_admitted")
        records = deepcopy(self._previous_records + self._native_records)
        _known(records)
        # No native ModelCostGuard ticket is invented. Its settled request
        # count and bill are transferred exactly once before the first tail.
        super().seed_prior_usage(records)
        self._tail_seeded = True

    async def invoke_strict(self, operation, schema, messages):
        if operation not in TAIL_OPERATIONS:
            raise ValueError("source_bundle_only_normal_independent_tail_authorized")
        self._ensure_tail_seeded()
        return await super().invoke_strict(operation, schema, messages)

    def capacity_policy(self, operation):
        if operation == "source_bundle_selection":
            maximum = self.source_bundle_config["native_max_output_tokens"]
            return {"initial_tokens": maximum, "max_tokens": maximum, "retries": 0}
        return super().capacity_policy(operation)

    def _event(self, event, data):
        if callable(self.workbench_event):
            data = deepcopy(data)
            if event == "tool_completed" and (not isinstance(data.get("failure_code"), str) or data["failure_code"] not in PUBLIC_TOOL_FAILURES):
                data.pop("failure_code", None)
            self.workbench_event("source_bundle_" + event, deepcopy(data))
            base = {"stage": "source_bundle_selection", "node": "source_bundle_selection", "engine": "official-dsh"}
            if event == "prefetch_completed":
                if data.get("model_calls") != 0 or type(data.get("candidate_count")) is not int:
                    raise ModelOutputInvalid("本地检索完成事件无效。", stage="source_bundle_selection", code="source_bundle_event_invalid")
                self.workbench_event("tool_completed", {**base, "kind": "local", "name": "search_source_bundles",
                    "call_id": self.directory.name + ":program-original-question-navigation",
                    "summary": f"程序本地检索完成，展示{data['candidate_count']}个候选，未调用模型。",
                    "model_calls": 0, "completed_at": data.get("completed_at"), "navigation_archive": data.get("navigation_archive")})
            elif event == "tool_started":
                if data.get("tool") not in {"search_source_bundles", "read_source_bundles", "submit_bundle_evidence"} or not isinstance(data.get("call_id"), str):
                    raise ModelOutputInvalid("原生工具事件缺少有效身份，已停止。", stage="source_bundle_selection", code="source_bundle_event_invalid")
                self._native_tools[data["call_id"]] = data["tool"]
                self.workbench_event("tool_started", {**base, "kind": "local", "name": data["tool"],
                    "call_id": self.directory.name + ":dsh-tool:" + data["call_id"], "summary": "执行原始资料业务工具"})
            elif event == "tool_completed" and data.get("call_id") in self._native_tools:
                tool_failure = data.get("failure_code")
                diagnostic = ({"code": tool_failure, "message": PUBLIC_TOOL_FAILURES[tool_failure]}
                              if isinstance(tool_failure, str) and tool_failure in PUBLIC_TOOL_FAILURES else {})
                self.workbench_event("tool_deferred" if data.get("deferred") else "tool_failed" if data.get("failed") else "tool_completed", {**base,
                    "kind": "local", "name": self._native_tools[data["call_id"]],
                    "call_id": self.directory.name + ":dsh-tool:" + data["call_id"],
                    "summary": "未执行：已达到本回合或任务的读取额度，本调用未读取原文、未消耗读取额度。" if data.get("deferred")
                        else diagnostic.get("message", "原始资料工具执行失败") if data.get("failed") else "原始资料工具完成",
                    **diagnostic})
            elif event in {"native_request_started", "native_usage_settled"}:
                sequence = data.get("request_sequence")
                if type(sequence) is int and sequence > 0:
                    complete = data.get("known_usage") is True and data.get("response_completed") is True and data.get("halted") is False
                    kind = "tool_started" if event == "native_request_started" else "tool_completed" if complete else "tool_failed"
                    self.workbench_event(kind, {**base, "kind": "model", "name": "source_bundle_select",
                        "call_id": f"{self.directory.name}:dsh-model:{sequence}", "request_sequence": sequence,
                        "summary": {"request_sequence": sequence, "actual_usage": data.get("provider_usage"),
                            "estimated_input_tokens": data.get("estimated_input_tokens"),
                            "known_episode_tokens": data.get("known_episode_tokens")}})

    def _collect_native_usage(self):
        ledger = self.bundle_directory / "native/budget.jsonl"
        events = [json.loads(line) for line in ledger.read_text("utf-8").splitlines() if line.strip()] if ledger.exists() else []
        started = [row for row in events if row.get("status") == "request_started"]
        settled = {row["request_sequence"]: row for row in events if row.get("status") in {"usage_settled", "usage_unknown"}}
        records, unknown = [], False
        for start in started:
            settlement = settled.get(start["request_sequence"], {})
            usage = settlement.get("provider_usage")
            complete = (settlement.get("status") == "usage_settled"
                        and settlement.get("unknown_usage_requests") == 0
                        and isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens", "total_tokens"))
                        and usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"])
            self._record_usage("source_bundle_select", usage if complete else None)
            record = {"operation": "source_bundle_select", "model": self.model_name,
                      "usage_source": "provider" if complete else "unavailable", "cost": None,
                      **({key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")} if complete else
                         {key: None for key in ("input_tokens", "output_tokens", "total_tokens")})}
            records.append(record)
            unknown |= not complete
        self._native_records = records
        return unknown

    async def _launch(self, manifest, descriptor):
        node = shutil.which("node")
        if node is None:
            raise ModelOutputInvalid("本机官方 DSH 启动环境不可用，未调用模型。", stage="source_bundle_selection", code="source_bundle_runtime_unavailable")
        env = {key: value for key, value in os.environ.items() if key.upper() in {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT"}}
        if self._fixture_plan is None:
            env["TRUST_BUNDLE_WORKBENCH_KEY"] = self.settings.model_api_key.get_secret_value()
        process = await asyncio.create_subprocess_exec(node, str(ROOT / "integrations/dsh/bundle_workbench_launcher.mjs"),
            descriptor["file"], descriptor["sha256"], cwd=str(ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        async def stdout():
            while line := await process.stdout.readline():
                try:
                    row = json.loads(line)
                    if row.get("type") == "bundle_workbench_event":
                        self._event(row["event"], row.get("data", {}))
                except (ValueError, TypeError, KeyError):
                    raise ModelOutputInvalid("原生执行事件未通过审计，已停止。", stage="source_bundle_selection", code="source_bundle_event_invalid") from None
        async def stderr():
            # Drain without reflecting arbitrary SDK stderr, source fragments or credentials.
            while await process.stderr.read(65536):
                pass
        tasks = [asyncio.create_task(stdout()), asyncio.create_task(stderr())]
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=manifest["timeout_seconds"] + 75)
            await asyncio.gather(*tasks)
        except BaseException:
            native = self.bundle_directory / "native"
            native.mkdir(exist_ok=True)
            cancellation = native / "cancel-requested.json"
            if not cancellation.exists():
                freeze(cancellation, {"cancel_requested": True, "no_result_published": True})
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), timeout=20)
            except BaseException:
                if process.returncode is None:
                    if os.name == "nt":
                        killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(process.pid), "/T", "/F",
                            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                        await killer.wait()
                    else:
                        process.terminate()
                    await process.wait()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def select_source_bundles(self, question, documents, paragraphs, excluded_spans=()):
        if self._selection_started or self._tail_seeded or self.attempt_index:
            raise ValueError("source_bundle_selection_once_per_provider_required")
        self._selection_started = True
        self._selection_failed = True
        original_question = {"question_id": question.get("raw_question_id", question.get("question_id")), "question": question["question"]}
        limits = {"max_search_calls": 3, "max_read_calls": 2, "max_paragraphs": 5,
                  "max_read_anchors": 12, "max_read_codepoints": 12000, "max_read_payload_bytes": 150000}
        SourceBundleTools(documents, paragraphs, original_question, **limits)
        raw = {"format": "frozen-source-bundle-case-v1", "question": original_question,
               "documents": deepcopy(documents), "paragraphs": deepcopy(paragraphs), "limits": limits}
        config = self.source_bundle_config
        raw_descriptor = freeze(self.bundle_directory / "raw-case.json", raw)
        documents_descriptor = freeze(self.bundle_directory / "documents.json", documents)
        paragraphs_descriptor = freeze(self.bundle_directory / "paragraphs.json", paragraphs)
        previous = freeze(self.bundle_directory / "previous-usage.json", {"records": self._previous_records})
        excluded = sorted(map(list, excluded_spans))
        manifest = {"format": "dsh-source-bundle-workbench-launch-v1", "authorization": {
            "intent": "workbench-source-bundle-selection", "episode_id": str(uuid4()),
            "question_id": original_question["question_id"], "question_sha256": sha256(canonical(original_question)).hexdigest(),
            "case_sha256": raw_descriptor["sha256"]}, "execution_mode": "fixture" if self._fixture_plan is not None else "workbench",
            "raw_case": raw_descriptor, "documents": documents_descriptor, "paragraphs": paragraphs_descriptor,
            "previous_usage": previous, "source_archives": self.source_archives, "excluded_spans": excluded,
            "output_directory": str((self.bundle_directory / "native").resolve()), "model": self.model_name,
            "whole_token_stop": self.threshold, "front_token_stop": config["front_token_stop"],
            "tail_reserved_tokens": config["tail_reserved_tokens"], "max_total_requests": self.max_requests,
            "max_native_requests": config["native_max_requests"], "required_tail_requests": 3,
            "output_capacity": {"navigation": config["navigation_max_output_tokens"], "submission": config["submit_max_output_tokens"]},
            "safety_tokens": 512, "timeout_seconds": math.ceil(config["timeout_seconds"])}
        if config.get("compact_handles") is True:
            manifest["compact_handles"] = True
        if self._fixture_plan is not None:
            manifest["fixture_plan"] = _bound(self._fixture_plan)
        launch_descriptor = freeze(self.bundle_directory / "launch-manifest.json", manifest)
        self._event("scope_prepared", {"raw_question_id": original_question["question_id"],
            "corpus_document_count": len(documents), "corpus_paragraph_count": len(paragraphs),
            "prior_known_tokens": _known(self._previous_records), "prior_requests": len(self._previous_records),
            "native_partial_stage_resumed": False, "native_partial_resume_supported": False,
            "source_binding_sha256": raw_descriptor["sha256"]})
        try:
            await self._launch(manifest, launch_descriptor)
        finally:
            usage_unknown = self._collect_native_usage()
        native = self.bundle_directory / "native"
        receipt_path = native / "closed-receipt.json"
        receipt = json.loads(receipt_path.read_text("utf-8")) if receipt_path.exists() else {}
        if usage_unknown or receipt.get("known_usage_complete") is not True:
            raise ModelOutputInvalid("原生模型调用用量未完整结算，已阻止后续研判。", stage="source_bundle_selection", code="experiment_usage_unknown")
        if receipt.get("status") != "succeeded":
            code = receipt.get("failure_code")
            if not isinstance(code, str) or not code.replace("_", "").isalnum():
                code = "source_bundle_selection_failed"
            details = {}
            if code == "whole_episode_budget_exceeded":
                rows = [json.loads(line) for line in (native / "budget.jsonl").read_text("utf-8").splitlines() if line.strip()]
                blocked = next((row for row in reversed(rows) if row.get("status") == "blocked"), {})
                for source, target in (("known_episode_tokens_before", "known_tokens"),
                                       ("estimated_input_tokens", "input_estimate"),
                                       ("reserved_output_tokens", "last_max_output_tokens")):
                    if type(blocked.get(source)) is int and blocked[source] >= 0:
                        details[target] = blocked[source]
                details.update(front_token_stop=min(config["front_token_stop"], self.threshold - config["tail_reserved_tokens"]),
                               tail_reserved_tokens=config["tail_reserved_tokens"])
            last_tool = receipt.get("last_business_failure_code")
            cause = PUBLIC_TOOL_FAILURES.get(last_tool, "") if isinstance(last_tool, str) else ""
            message = cause + "后续选证未完成，原始调用记录已保留，未发布结果。" if cause else "原生选证未完整提交，原始调用记录已保留，未发布结果。"
            raise ModelOutputInvalid(message, stage="source_bundle_selection", code=code, details=details)
        report = json.loads((native / "usage-report.json").read_text("utf-8"))
        response = {"artifact": json.loads((native / "artifact.json").read_text("utf-8")), "usage_report": report,
            "proof": {"format": "dsh-source-bundle-workbench-proof-v1", "raw_case": raw_descriptor,
                "documents": documents_descriptor, "paragraphs": paragraphs_descriptor,
                "submit_request": _bound(native / "submit-request.json"), "tool_submission": _bound(native / "tool-submission.json"),
                "artifact": _bound(native / "artifact.json"), "closed_receipt": _bound(receipt_path),
                "native_requests": receipt["native_requests"], "source_archives": self.source_archives,
                "presence": _bound(native / "native-presence.jsonl"), "usage_ledger": _bound(native / "budget.jsonl"),
                "program_navigation": _bound(native / "program-navigation.json"),
                "previous_usage": previous, "excluded_spans": excluded, "usage_report": report,
                "launch_manifest": launch_descriptor, "business_transcript": _bound(native / "business-transcript.jsonl")}}
        if config.get("compact_handles") is True:
            response["proof"].update(program_projection=_bound(native / "handles.json"),
                                     native_projections=_bound(native / "native-projections.jsonl"))
        # Source-only compiler remains independently owned by the workflow.
        freeze(self.bundle_directory / "provider-response.json", response)
        self._selection_failed = False
        self._ensure_tail_seeded()
        return response

