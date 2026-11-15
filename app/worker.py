"""Single local worker, durable SQLite queue, immutable publication and export."""
import asyncio
from contextlib import suppress
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from app.agents.provider import LangChainOpenAIProvider, ModelInvocationFailed, ModelNotConfigured, ModelOutputInvalid
from app.domain.snapshot import build_snapshot
from app.domain.model_transport import public_transport_failure
from app.domain.model_cost_guard import CostGuardStopped
from app.errors import ServiceError
from app.repository import Repository
from app.utils import dumps, loads, now
from app.workflows.analysis import AnalysisCancelled, run_analysis

log = logging.getLogger(__name__)


class WorkerLease:
    """OS releases this lock after a crash. Only its holder may recover runs."""
    def __init__(self, path: Path):
        self.path, self.file = path, None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        try:
            self.file.seek(0)
            if not self.file.read(1):
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            self.file.close()
            self.file = None
            raise RuntimeError("此运行目录已有 worker；请勿启动第二个 worker。") from None

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


class Worker:
    def __init__(self, repo: Repository, provider=None):
        self.repo = repo
        self.provider = provider
        self.stop_event = asyncio.Event()
        self.lease = WorkerLease(repo.settings.data_dir / "worker.lock")

    async def run_forever(self):
        self.lease.acquire()
        try:
            self.repo.recover()
            while not self.stop_event.is_set():
                run = self.repo.claim()
                if run:
                    await self.execute(run)
                elif self.export_one():
                    continue
                else:
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self.stop_event.wait(), timeout=0.15)
        finally:
            self.lease.close()

    async def execute(self, run):
        rid = run["id"]
        task = asyncio.create_task(self.analyze(run))
        try:
            while not task.done():
                if self.stop_event.is_set():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    self.repo.finish_failed(rid, "worker_interrupted", "worker 已停止；保留检查点，可显式重试。")
                    return
                if self.repo.cancel_requested(rid):
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                    self.repo.finish_failed(rid, "cancelled", "已取消，未发布本次结果。", cancelled=True)
                    return
                await asyncio.wait({task}, timeout=0.1)
            await task
        except asyncio.CancelledError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            self.repo.finish_failed(rid, "worker_interrupted", "worker 已中断；保留检查点，可显式重试。")
            raise
        except AnalysisCancelled:
            self.repo.finish_failed(rid, "cancelled", "已取消，未发布本次结果。", cancelled=True)
        except ModelNotConfigured:
            self.repo.finish_failed(rid, "model_not_configured", "worker 未配置真实模型，请检查其 .env 后显式重试。")
        except ModelInvocationFailed as error:
            failure = public_transport_failure(error.code, error.details)
            self.repo.finish_failed(rid, failure["code"], failure["message"],
                stage=error.stage, details=failure["details"])
        except ModelOutputInvalid as error:
            self.repo.finish_failed(rid, error.code, str(error), stage=error.stage, details=error.details)
        except CostGuardStopped as error:
            self.repo.finish_failed(rid, error.code, "已达到本题模型调用或费用保护条件；保留已知用量与检查点，未发布。",
                stage=getattr(error, "stage", None), details={key: value for key, value in error.details.items()
                    if key in {"known_total_tokens", "actual_requests_started", "unknown_usage_requests"}})
        except ValueError:
            self.repo.finish_failed(rid, "validation_failed", "业务产物未通过完整性校验；未发布结果。", stage="snapshot")
        except ServiceError as error:
            self.repo.finish_failed(rid, error.code, error.message)
        except Exception as error:
            # No provider content/configuration in errors or events.
            log.error("run %s failed: %s", rid, type(error).__name__)
            self.repo.finish_failed(rid, "internal_error", f"执行失败（{type(error).__name__}）；未发布本次结果。")

    async def analyze(self, run):
        # Recovery requests never authorize fresh model execution. If an
        # explicit recovery process exits after queue insertion, the ordinary
        # worker must fail closed rather than bill a normal retry under reuse
        # metadata that declares zero new model calls.
        if "checkpoint_recovery" in run.get("options", {}):
            raise ServiceError(409, "checkpoint_recovery_requires_cli", "完成检查点恢复须由显式恢复命令处理；普通 worker 不会为该任务调用模型。")
        if loads(run["model_config_json"]) != self.repo.config():
            raise ServiceError(409, "worker_config_mismatch", "API 与 worker 的模型或流程配置不同，请统一配置后重试。")
        planned = loads(run["model_config_json"])
        if (planned.get("strict_evidence_enabled") is True or planned.get("source_bundle_enabled") is True) and len(loads(run["question_ids_json"])) != 1:
            raise ServiceError(422, "strict_question_limit", "严格选证实验一次只运行一道题，费用按该题完整累计。")
        if self.provider is not None:
            provider = self.provider
        elif self.repo.settings.model_source_bundle_enabled:
            if self.repo.settings.source_bundle_front_framework == "langchain":
                from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
                provider = LCBundleWorkbenchProvider(self.repo.settings)
            else:
                from app.agents.bundle_transport import BundleWorkbenchProvider
                provider = BundleWorkbenchProvider(self.repo.settings)
        elif self.repo.settings.model_strict_evidence_enabled:
            from app.agents.strict_transport import StrictWorkbenchProvider
            provider = StrictWorkbenchProvider(self.repo.settings)
        else:
            provider = LangChainOpenAIProvider(self.repo.settings)
        cost_parents = planned.get("strict_evidence_cost_parent_run_ids", [])
        resume_parent = run.get("options", {}).get("resume_from_run_id")
        if (planned.get("strict_evidence_enabled") is True or planned.get("source_bundle_enabled") is True) and (cost_parents or resume_parent):
            prior_records, seen = [], set()
            pending = list(cost_parents) + ([resume_parent] if resume_parent else [])
            qid = loads(run["question_ids_json"])[0]
            current_manifest = self.repo.revision(run["topic_id"], run["target_revision_id"])["input_manifest_sha256"]
            while pending:
                prior_id = pending.pop()
                if not isinstance(prior_id, str) or not prior_id or prior_id == run["id"]:
                    raise ServiceError(409, "strict_cost_parent_invalid", "费用父任务标识无效，未调用模型。")
                if prior_id in seen:
                    continue
                seen.add(prior_id)
                prior = self.repo.one("SELECT topic_id,question_ids_json,target_revision_id,model_config_json,usage_json FROM runs WHERE id=?", (prior_id,))
                if prior["topic_id"] != run["topic_id"] or qid not in loads(prior["question_ids_json"]):
                    raise ServiceError(409, "strict_cost_parent_binding", "费用父任务不属于当前专题及原题，未调用模型。")
                previous_manifest = self.repo.revision(prior["topic_id"], prior["target_revision_id"])["input_manifest_sha256"]
                if previous_manifest != current_manifest or current_manifest != self.repo.topic(run["topic_id"])["source_manifest_sha256"]:
                    raise ServiceError(409, "strict_cost_parent_source_changed", "费用父任务的原始资料或原题清单不同，未调用模型。")
                prior_records.extend(loads(prior["usage_json"], {}).get(qid, []))
                prior_options = loads(self.repo.one("SELECT options_json FROM run_options WHERE run_id=?", (prior_id,))["options_json"], {})
                if prior_options.get("resume_from_run_id"):
                    pending.append(prior_options["resume_from_run_id"])
                prior_config = loads(prior["model_config_json"], {})
                pending.extend(prior_config.get("strict_evidence_cost_parent_run_ids", []))
            seed = getattr(provider, "set_previous_usage" if planned.get("source_bundle_enabled") is True else "seed_prior_usage", None)
            if not callable(seed):
                raise ServiceError(409, "strict_resume_usage_unavailable", "严格模式适配器不能承接历史用量，未调用模型；原任务用量保留。")
            if any(not isinstance(row, dict) or row.get("usage_source") != "provider"
                    or any(type(row.get(key)) is not int or row[key] < 0 for key in ("input_tokens", "output_tokens", "total_tokens"))
                    or row["input_tokens"] + row["output_tokens"] != row["total_tokens"] for row in prior_records):
                raise ServiceError(409, "strict_cost_parent_usage_unknown", "费用父任务有未知或不一致用量，未调用模型；原失败记录保留。")
            seed(prior_records)
            self.repo.event(run["id"], "cost_lineage_seeded", {"question_id": qid, "prior_run_ids": sorted(seen),
                "prior_known_tokens": sum(row["total_tokens"] for row in prior_records),
                "prior_calls": len(prior_records), "source_manifest_sha256": current_manifest,
                "cost_only": True, "checkpoint_results_reused": False})
            if ((planned.get("strict_evidence_enabled") is True
                    or planned.get("source_bundle_config", {}).get("sealed_front_recovery")
                    and getattr(provider, "paid_response_operations", set()) in (
                        {"authority_analyst", "cross_validation_analyst"},
                        {"authority_analyst", "cross_validation_analyst", "arbitrate"}))
                    and {"authority_analyst", "cross_validation_analyst"}.intersection(getattr(provider, "paid_response_operations", set()))):
                frozen = []
                for parent_id in seen:
                    for event in self.repo.events(parent_id):
                        data = event["data"]
                        if (event["type"] == "node_completed" and data.get("node") == "freeze"
                                and data.get("question_id") == qid):
                            frozen.append({"source_run_id":parent_id, "evidence_snapshot_id":data["evidence_snapshot_id"],
                                           "snapshot_sha256":data["snapshot_sha256"]})
                if len(frozen) != 1:
                    raise ServiceError(409, "strict_frozen_parent_ambiguous", "已付研判未绑定唯一冻结快照，未调用模型。")
                provider.reusable_frozen_snapshot = frozen[0]
        recovery = planned.get("source_bundle_config", {}).get("sealed_front_recovery")
        if recovery is not None:
            origin = recovery.get("origin_run_id") if isinstance(recovery, dict) else None
            if not (cost_parents or resume_parent) or origin not in seen:
                raise ServiceError(409, "sealed_front_recovery_cost_parent_required", "已付选证恢复必须完整继承原任务费用，未调用模型。")
            original = self.repo.one("SELECT * FROM runs WHERE id=?", (origin,))
            descriptor = recovery.get("response", {})
            path = Path(descriptor.get("file", ""))
            outer = path.parent.parent.name
            expected = self.repo.settings.data_dir / "strict-episodes" / outer / "source-bundle" / "provider-response.json"
            matching = [event for event in self.repo.events(origin) if event["type"] == "tool_completed"
                and event["data"].get("name") == "langchain_source_bundle_select"
                and event["data"].get("call_id") == outer + ":lc-front:1"]
            if (original["status"] != "failed" or original["topic_id"] != run["topic_id"]
                    or qid not in loads(original["question_ids_json"]) or path.resolve() != expected.resolve()
                    or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != descriptor.get("sha256")
                    or len(matching) != 1):
                raise ServiceError(409, "sealed_front_recovery_origin_invalid", "原任务、原题或已付选证封印不一致，未调用模型。")
            bind = getattr(provider, "bind_sealed_front_recovery", None)
            if not callable(bind):
                raise ServiceError(409, "sealed_front_recovery_provider_required", "当前适配器不支持已付选证恢复，未调用模型。")
            bind({"origin_run_id": origin, "origin_topic_id": run["topic_id"], "origin_question_id": qid,
                "source_manifest_sha256": current_manifest, "recovery_run_id": run["id"],
                "origin_native_outer_episode": outer})
        if ((planned.get("strict_evidence_enabled") is True) != (getattr(provider, "strict_evidence_enabled", False) is True)
                or planned.get("strict_evidence_enabled") is True and
                (getattr(provider, "prompt_version", None) != planned.get("prompt_version")
                 or getattr(provider, "strict_candidate_limit", 60) != planned.get("strict_evidence_config", {}).get("candidate_limit"))):
            raise ServiceError(409, "worker_protocol_mismatch", "严格选证任务与适配器能力或候选配置不一致，未调用模型。")
        if planned.get("strict_evidence_enabled") is True:
            strict_config = planned.get("strict_evidence_config", {})
            if ((getattr(provider, "strict_balanced_enabled", False) is True) != (strict_config.get("balanced_scoped_enabled") is True)
                    or getattr(provider, "strict_balance_policy", None) != strict_config.get("balance_policy")
                    or getattr(provider, "strict_recall_strategy", None) != strict_config.get("recall_strategy")):
                raise ServiceError(409, "worker_protocol_mismatch", "任务的证据平衡与主张仲裁配置不一致，未调用模型。")
        if (loads(run["model_config_json"]).get("premise_bank_enabled") is True) != (getattr(provider, "premise_bank_enabled", False) is True):
            raise ServiceError(409, "worker_protocol_mismatch", "任务的独立用途审核协议与实际模型适配器能力不一致，未调用模型；请统一配置后新建任务。")
        bundle_enabled = planned.get("source_bundle_enabled") is True
        if (bundle_enabled != (getattr(provider, "source_bundle_enabled", False) is True)
                or bundle_enabled and (getattr(provider, "prompt_version", None) != planned.get("prompt_version")
                    or getattr(provider, "source_bundle_config", None) != planned.get("source_bundle_config"))):
            raise ServiceError(409, "worker_protocol_mismatch", "连续原文任务与选证框架或预算配置不一致，未调用模型。")
        if ((planned.get("paragraph_ranked_enabled") is True) != (getattr(provider, "paragraph_ranked_enabled", False) is True)
                or (planned.get("paragraph_ranked_enabled") is True and
                    (getattr(provider, "paragraph_ranked_config", None) != planned.get("paragraph_ranked_config")
                     or getattr(provider, "prompt_version", None) != planned.get("prompt_version")))):
            raise ServiceError(409, "worker_protocol_mismatch", "任务的段落排序策略与实际模型适配器配置不一致，未调用模型。")
        topic = self.repo.topic(run["topic_id"])
        if self.repo.revision(topic["id"], run["target_revision_id"])["input_manifest_sha256"] != topic["source_manifest_sha256"]:
            raise ServiceError(409, "run_input_mismatch", "原始资料清单已变化，任务未执行；请重新研判。")
        questions = self.repo.questions(topic["id"])
        paragraphs = self.repo.paragraphs(topic["id"])
        selected = set(loads(run["question_ids_json"]))
        results, base_snapshot = {}, {}
        if run["base_revision_id"]:
            base = self.repo.revision(topic["id"], run["base_revision_id"])
            if base["state"] == "ready":
                base_snapshot = self.repo.snapshot(topic["id"], base["id"])
            elif topic["published_revision_id"]:
                base_snapshot = self.repo.snapshot(topic["id"], topic["published_revision_id"])
            results.update(base_snapshot.get("question_results", {}))
        overlays = run["options"].get("overlays", base_snapshot.get("overlays", {"excluded": [], "manual_evidence": [], "interpretation_overrides": {}}))
        fresh = {}
        for question in questions:
            if question["id"] not in selected:
                continue
            if getattr(provider, "strict_evidence_enabled", False) or bundle_enabled:
                provider.workbench_event = lambda kind, data, qid=question["id"]: self.repo.event(
                    run["id"], kind, {"question_id":qid, **data})
            candidate_artifact = self.repo.refine_input(run["id"], question["id"])
            if run["options"].get("execution_mode") == "quality-refine" and candidate_artifact is None:
                raise ServiceError(409, "refine_missing_candidate_ledger", "复核输入工件缺失，未调用模型。")
            if candidate_artifact is not None:
                from app.domain.refine_input import validate_candidate_artifact, RefineInputInvalid
                try:
                    candidate_artifact = validate_candidate_artifact(candidate_artifact, topic_id=topic["id"],
                        question=question, paragraphs=paragraphs, source_manifest_sha256=topic["source_manifest_sha256"], overlays=overlays)
                except RefineInputInvalid as error:
                    raise ServiceError(409, error.check, "候选工件已与当前题目、来源或人工标注不一致，未调用模型。") from None
            result = await run_analysis(question, paragraphs, provider,
                **({"source_documents": self.repo.source_documents(topic["id"])} if bundle_enabled else {}),
                strict_scope_enabled=True, candidate_artifact=candidate_artifact,
                test_only_legacy_precision=self.provider is not None and getattr(self.provider, "test_only_legacy_precision", False) is True,
                checkpoint_path=self.repo.settings.checkpoint_path,
                thread_id=run["id"] + ":" + question["id"],
                cancel_check=lambda: self.repo.cancel_requested(run["id"]),
                emit=lambda kind, data: self.repo.event(run["id"], kind, data),
                excluded_spans={tuple(x) for x in overlays["excluded"]},
                manual_evidence=overlays["manual_evidence"],
                interpretation_overrides=overlays["interpretation_overrides"],
                retrieval_limit=self.repo.settings.retrieval_limit,
                source_batch_codepoints=self.repo.settings.source_batch_codepoints,
                source_batch_spans=self.repo.settings.source_batch_spans,
                usage_callback=lambda records, qid=question["id"]: self.repo.record_usage(run["id"], qid, records),
                resume_from_thread_id=(run["options"].get("resume_checkpoint_sources", {}).get(question["id"], run["options"].get("resume_from_run_id", "")) + ":" + question["id"]) if question["id"] in run["options"].get("resume_checkpoint_question_ids", []) else None,
                include_baseline=run["options"].get("include_baseline", False))
            result["origin_run_id"] = run["id"]
            fresh[question["id"]] = results[question["id"]] = result
        revision = self.repo.revision(topic["id"], run["target_revision_id"])
        snapshot_started = time.perf_counter()
        self.repo.event(run["id"], "tool_started", {"call_id": run["id"] + ":snapshot", "name": "snapshot_integrity_validation", "summary": "组装证据图并校验引用、依赖关系和导出结构"})
        try:
            snapshot = build_snapshot(topic, [q for q in questions if q["id"] in results], results,
                revision_id=revision["id"], graph_revision=revision["revision_no"], run_id=run["id"], source_paragraphs=paragraphs,
                source_documents=self.repo.source_documents(topic["id"]) if any(r.get("source_bundle_protocol") for r in results.values()) else None)
        except ValueError as error:
            self.repo.event(run["id"], "tool_failed", {"call_id": run["id"] + ":snapshot", "name": "snapshot_integrity_validation", "elapsed_ms": round((time.perf_counter() - snapshot_started) * 1000), "summary": "产物未通过完整性校验，未发布"})
            known_rules = {
                "asserted conclusion introduces a numeric literal absent from cited evidence": "结论包含已引用证据中不存在的数字，未通过证据核对。",
                "final supporting refs requires at least one qualified supporting evidence": "结论缺少可定位的支持证据，未发布结果。",
                "resolved conflict must identify exactly one adopted side in final supporting refs": "冲突记录没有明确唯一采纳侧，未发布结果。",
            }
            message = known_rules.get(str(error), "研判产物的引用、依赖图或导出结构未通过完整性校验。")
            raise ModelOutputInvalid(message, code="snapshot_validation_failed", stage="snapshot", details={"validation_rule": str(error) if str(error) in known_rules else "business_snapshot_integrity"}) from None
        self.repo.event(run["id"], "tool_completed", {"call_id": run["id"] + ":snapshot", "name": "snapshot_integrity_validation", "elapsed_ms": round((time.perf_counter() - snapshot_started) * 1000), "summary": f"完整性校验通过：{len(snapshot['evidence'])} 条证据"})
        snapshot["overlays"] = overlays
        snapshot["scope"] = {"completed_question_ids": list(results), "all_question_count": len(questions), "is_complete_topic": len(results) == len(questions)}
        try:
            self.repo.event(run["id"], "publication_started", {})
            self.repo.publish(run["id"], snapshot, fresh)
        except ValueError:
            raise ModelOutputInvalid("发布前的原文、版本或引用完整性校验失败。", code="publish_validation_failed", stage="publish") from None

    def export_one(self):
        with self.repo.connection() as db:
            row = db.execute("SELECT * FROM export_jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if not row:
                return False
            row = dict(row)
            db.execute("UPDATE export_jobs SET status='running' WHERE id=? AND status='queued'", (row["id"],))
        path = self.repo.settings.export_dir / (row["id"] + ".zip")
        temporary = path.with_suffix(".tmp")
        try:
            # Revalidate when executing too: synchronous API eligibility is not
            # permission to bypass source/reference/coverage checks later.
            files = self.repo.formal_submission_files(row["topic_id"], row["revision_id"])
            encoded = {name: (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8") for name, value in files.items()}
            manifest = {"topic_id": row["topic_id"], "revision_id": row["revision_id"], "adapter_version": row["adapter_version"], "created_at": now(), "validation": "local-contract-source-span-reference-and-graph-checks; not official scoring", "files": {name: {"sha256": hashlib.sha256(value).hexdigest(), "records": len(files[name])} for name,value in encoded.items()}}
            path.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
                for name, value in encoded.items():
                    archive.writestr(name, value)
                archive.writestr("manifest.json", dumps(manifest).encode("utf-8"))
                archive.writestr("field_definitions.json", (Path(__file__).resolve().parents[1] / "contracts" / "competition-export.schema.json").read_bytes())
            temporary.replace(path)
            with self.repo.connection() as db:
                db.execute("UPDATE export_jobs SET status='succeeded',manifest_json=?,artifact_path=? WHERE id=?", (dumps(manifest), str(path), row["id"]))
        except Exception as error:
            temporary.unlink(missing_ok=True)
            with self.repo.connection() as db:
                db.execute("UPDATE export_jobs SET status='failed',error_json=? WHERE id=?", (dumps({"code": "export_validation_failed", "message": "导出校验失败或专题问题尚未全部完成。", "error_type": type(error).__name__}), row["id"]))
        return True
