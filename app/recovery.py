"""Explicit publication recovery of already completed local checkpoints.

This is deliberately narrower than workflow resume: no graph is invoked, no
model provider is constructed, and an incomplete checkpoint is never continued.
The original failed run and its billed usage remain intact for audit.
"""

from copy import deepcopy
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.data.db import transaction
from app.domain.snapshot import build_snapshot, SOURCE_BUNDLE_PROTOCOL, SOURCE_DOCUMENT_PROTOCOLS, validate_source_bundle_rows
from app.domain.spans import validate_span
from app.errors import ServiceError
from app.repository import Repository
from app.utils import digest, loads
from app.worker import WorkerLease


_REQUIRED_CHANNELS = {
    "question", "evidence", "authority_analysis", "cross_analysis", "final", "evidence_snapshot_id",
}
_KIND = "completed_checkpoint_reuse"


def _refuse(code: str, message: str) -> ServiceError:
    return ServiceError(409, code, message)


def _existing(repo: Repository, topic_id: str, key: str, request: dict):
    rows = repo.all("SELECT id,request_sha256 FROM runs WHERE topic_id=? AND idempotency_key=?", (topic_id, key))
    if not rows:
        return None
    if rows[0]["request_sha256"] != digest(request):
        raise _refuse("idempotency_conflict", "该幂等键已用于不同请求，不能用于此次检查点恢复。")
    return _response(repo, rows[0]["id"])


def _response(repo: Repository, run_id: str) -> dict:
    options = loads(repo.one("SELECT options_json FROM run_options WHERE run_id=?", (run_id,))["options_json"], {})
    return {**repo.run(run_id), "recovery": options.get("checkpoint_recovery")}


def _ensure_idle(repo: Repository, *, except_run_id: str | None = None, db=None):
    sql = "SELECT id FROM runs WHERE status IN ('queued','running','awaiting_review')"
    args = ()
    if except_run_id is not None:
        sql += " AND id<>?"
        args = (except_run_id,)
    if db is None:
        active = repo.all(sql + " LIMIT 1", args)
    else:
        active = db.execute(sql + " LIMIT 1", args).fetchall()
    if active:
        raise _refuse("recovery_queue_busy", "仍有排队、运行或待复核任务；请在队列空闲时显式恢复。")


def _verify_sources(repo: Repository, topic_id: str, results: dict):
    """Validate every recovered or retained quote against immutable raw text."""
    try:
        paragraphs = repo.paragraphs(topic_id)
        documents = repo.source_documents(topic_id) if any(
            isinstance(result, dict) and result.get("source_bundle_protocol") is not None for result in results.values()) else None
        for result in results.values():
            if not isinstance(result, dict) or not isinstance(result.get("evidence"), list):
                raise ValueError("invalid checkpoint result")
            validate_source_bundle_rows(result["evidence"], protocol=result.get("source_bundle_protocol"),
                source_documents=documents, source_paragraphs=paragraphs)
            for evidence in result["evidence"]:
                paragraph = repo.paragraph(evidence["paragraph_id"])
                if (
                    paragraph["topic_id"] != topic_id
                    or evidence["raw_doc_id"] != paragraph["raw_doc_id"]
                    or evidence["raw_para_id"] != paragraph["raw_para_id"]
                ):
                    raise ValueError("checkpoint source identity mismatch")
                if evidence.get("span_scope") != "document":
                    validate_span(paragraph["text"], evidence["quote"], evidence["start_cp"], evidence["end_cp"])
    except (KeyError, TypeError, ValueError, ServiceError):
        # Do not expose checkpoint content through diagnostic exception bodies.
        raise _refuse("recovery_source_invalid", "检查点引用未通过原文、段落归属或码点校验，未创建恢复任务。") from None


async def recover_completed_run(repo: Repository, source_run_id: str, idempotency_key: str) -> dict:
    """Validate and publish completed source checkpoints without new model calls.

    Only an explicitly selected snapshot/publish failure at the current topic
    head can be recovered. A live worker or any existing active queue entry
    prevents a new recovery. An exact idempotent replay returns its existing
    recovery first.
    """
    if not isinstance(idempotency_key, str) or not idempotency_key.strip() or len(idempotency_key) > 200:
        raise ServiceError(422, "idempotency_key_required", "检查点恢复需要 1–200 字符的幂等键。")
    source = repo.one("SELECT * FROM runs WHERE id=?", (source_run_id,))
    source_options = loads(repo.one("SELECT options_json FROM run_options WHERE run_id=?", (source_run_id,))["options_json"], {})
    selected = loads(source["question_ids_json"])
    request = {
        "question_ids": selected, "base_revision_id": source["target_revision_id"],
        "mode": "retry", "include_baseline": bool(source_options.get("include_baseline", False)),
        "recovery_kind": _KIND, "recovery_source_run_id": source_run_id,
    }
    # Return a previously created recovery even though it changed the topic head.
    existing = _existing(repo, source["topic_id"], idempotency_key, request)
    if existing is not None:
        return existing

    lease = WorkerLease(repo.settings.data_dir / "worker.lock")
    try:
        lease.acquire()
    except RuntimeError:
        raise _refuse("recovery_worker_active", "worker 仍持有运行锁；请先停止 worker，再显式恢复完成的检查点。") from None

    new_run_id = None
    try:
        existing = _existing(repo, source["topic_id"], idempotency_key, request)
        if existing is not None:
            return existing
        source = repo.one("SELECT * FROM runs WHERE id=?", (source_run_id,))
        topic = repo.topic(source["topic_id"])
        source_error = loads(source["error_json"], {})
        if source["status"] != "failed" or source_error.get("stage") not in {"snapshot", "publish"}:
            raise _refuse("recovery_source_not_completed", "仅可恢复在产物组装或发布阶段失败且已有完整检查点的任务。")
        if topic["head_revision_id"] != source["target_revision_id"]:
            raise _refuse("recovery_stale_source", "来源任务已不是专题最新版本，不能恢复覆盖后续工作。")
        revision = repo.revision(topic["id"], source["target_revision_id"])
        if revision["state"] != "failed" or revision["input_manifest_sha256"] != topic["source_manifest_sha256"]:
            raise _refuse("recovery_input_mismatch", "来源版本状态或原始输入清单不一致，不能复用检查点。")
        if loads(source["model_config_json"]) != repo.config():
            raise _refuse("recovery_config_mismatch", "模型、提示词或流程配置已改变，不能将旧检查点作为本次执行。")
        _ensure_idle(repo)
        questions = repo.questions(topic["id"])
        question_map = {question["id"]: question for question in questions}
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)) or not set(selected) <= set(question_map):
            raise _refuse("recovery_question_mismatch", "来源任务的问题集合不属于当前专题。")
        if not repo.settings.checkpoint_path.is_file():
            raise _refuse("recovery_checkpoint_missing", "找不到来源检查点，未创建恢复任务。")

        fresh, checkpoint_ids = {}, {}
        async with AsyncSqliteSaver.from_conn_string(str(repo.settings.checkpoint_path)) as saver:
            for question_id in selected:
                stored = await saver.aget_tuple({"configurable": {"thread_id": source_run_id + ":" + question_id}})
                channels = stored.checkpoint.get("channel_values", {}) if stored else {}
                if not isinstance(channels, dict) or not _REQUIRED_CHANNELS <= channels.keys():
                    raise _refuse("recovery_checkpoint_incomplete", "来源问题尚无完整双路研判和仲裁检查点，未创建恢复任务。")
                checkpoint_id = stored.checkpoint.get("id")
                if not isinstance(checkpoint_id, str) or not checkpoint_id:
                    raise _refuse("recovery_checkpoint_incomplete", "检查点缺少可追溯编号，未创建恢复任务。")
                question = channels["question"]
                original = question_map[question_id]
                if not isinstance(question, dict) or any(question.get(key) != original[key] for key in ("id", "raw_question_id", "question")):
                    raise _refuse("recovery_question_mismatch", "检查点题干或问题标识与原始输入不一致。")
                provenance = {"kind": _KIND, "origin_run_id": source_run_id, "checkpoint_id": checkpoint_id}
                result = {
                    "evidence": deepcopy(channels["evidence"]),
                    "analyses": [deepcopy(channels["authority_analysis"]), deepcopy(channels["cross_analysis"])],
                    "final": deepcopy(channels["final"]), "evidence_snapshot_id": channels["evidence_snapshot_id"],
                    "origin_run_id": source_run_id, "execution_provenance": deepcopy(provenance), "usage": [],
                }
                # Preserve the actual verification protocol, not just the
                # flattened frozen rows. A completed v18 checkpoint still needs
                # bank/purpose/source replay before it can be published.
                for name in ("candidate_evidence", "operator_exclusion_audit", "strict_scope_review",
                             "material_eligibility", "source_coverage", "extraction_reviews",
                             "evidence_selection", "quote_faithful_bank", "purpose_review", "paragraph_ranked_proof", "strict_evidence_proof",
                             "source_bundle_protocol"):
                    if name in channels:
                        result[name] = deepcopy(channels[name])
                if loads(source["model_config_json"]).get("premise_bank_enabled") is True:
                    from app.domain.premise_bank import VERSION
                    result["verification_protocol"] = VERSION
                    if not {"candidate_evidence", "strict_scope_review", "material_eligibility",
                            "quote_faithful_bank", "purpose_review"} <= result.keys():
                        raise _refuse("recovery_checkpoint_incomplete", "完成检查点缺少原句或独立用途许可记录，未创建恢复任务。")
                if loads(source["model_config_json"]).get("paragraph_ranked_enabled") is True:
                    from app.domain.paragraph_ranked import VERSION
                    result["verification_protocol"] = VERSION
                    if not isinstance(result.get("paragraph_ranked_proof"), dict):
                        raise _refuse("recovery_checkpoint_incomplete", "完成检查点缺少完整段落排序与字段验真证明，未创建恢复任务。")
                if loads(source["model_config_json"]).get("strict_evidence_enabled") is True:
                    from app.workflows.strict_workbench import VERSION
                    result["verification_protocol"] = VERSION
                    if not isinstance(result.get("strict_evidence_proof"), dict):
                        raise _refuse("recovery_checkpoint_incomplete", "完成检查点缺少严格选证原文与字段证明，未创建恢复任务。")
                if loads(source["model_config_json"]).get("source_bundle_protocol") is not None:
                    frozen_protocol = loads(source["model_config_json"])["source_bundle_protocol"]
                    if frozen_protocol not in SOURCE_DOCUMENT_PROTOCOLS or result.get("source_bundle_protocol") != frozen_protocol:
                        raise _refuse("recovery_checkpoint_incomplete", "完成检查点缺少明确的文档范围来源协议，未创建恢复任务。")
                if request["include_baseline"]:
                    if "baseline_analysis" not in channels:
                        raise _refuse("recovery_checkpoint_incomplete", "来源任务要求的独立基线尚未完成，未创建恢复任务。")
                    result["baseline"] = deepcopy(channels["baseline_analysis"])
                elif "baseline_analysis" in channels:
                    result["baseline"] = deepcopy(channels["baseline_analysis"])
                for output in [*result["analyses"], result["final"], *([result["baseline"]] if "baseline" in result else [])]:
                    if not isinstance(output, dict):
                        raise _refuse("recovery_checkpoint_incomplete", "完成检查点的研判产物结构不完整。")
                    output["execution_provenance"] = deepcopy(provenance)
                fresh[question_id], checkpoint_ids[question_id] = result, checkpoint_id

        try:
            from app.domain.material_eligibility import validate_workflow_material_eligibility
            paragraphs = repo.paragraphs(topic["id"])
            source_documents = repo.source_documents(topic["id"]) if any(
                result.get("source_bundle_protocol") is not None for result in fresh.values()) else None
            for question_id, result in fresh.items():
                repo._validate_result_protocol(result, source_run_id, topic["id"], question_id)
                validate_workflow_material_eligibility(result, question_map[question_id], paragraphs, source_documents=source_documents)
        except (KeyError, TypeError, ValueError, ServiceError):
            raise _refuse("recovery_validation_failed", "检查点事实库、用途许可或材料资格重放失败，未创建恢复任务。") from None

        # Retain unaffected, previously published questions just as a normal retry.
        base_snapshot = {}
        if source["base_revision_id"]:
            base = repo.revision(topic["id"], source["base_revision_id"])
            if base["state"] == "ready":
                base_snapshot = repo.snapshot(topic["id"], base["id"])
            elif topic["published_revision_id"]:
                base_snapshot = repo.snapshot(topic["id"], topic["published_revision_id"])
        results = {**deepcopy(base_snapshot.get("question_results", {})), **fresh}
        source_documents = repo.source_documents(topic["id"]) if any(
            result.get("source_bundle_protocol") is not None for result in results.values()) else None
        covered = [question for question in questions if question["id"] in results]
        overlays = deepcopy(source_options.get("overlays", base_snapshot.get("overlays", {
            "excluded": [], "manual_evidence": [], "interpretation_overrides": {},
        })))
        _verify_sources(repo, topic["id"], results)
        try:
            # Complete source/reference/export validation happens before any run
            # or new revision is created. No synthetic model output is added.
            build_snapshot(topic, covered, results, revision_id=str(uuid4()),
                           graph_revision=revision["revision_no"] + 1, run_id=source_run_id,
                           source_paragraphs=paragraphs, source_documents=source_documents)
        except (KeyError, TypeError, ValueError):
            raise _refuse("recovery_validation_failed", "完成检查点仍未通过业务产物和导出校验，未创建恢复任务。") from None
        _ensure_idle(repo)
        metadata = {
            "kind": _KIND, "source_run_id": source_run_id, "checkpoint_ids": checkpoint_ids,
            "new_model_calls": 0, "billing": "usage remains on source run; no new requests",
        }
        created = repo.create_run(topic["id"], request, idempotency_key, model_available=True,
                                  options={"include_baseline": request["include_baseline"],
                                           "overlays": overlays, "checkpoint_recovery": metadata})
        new_run_id = created["run_id"]
        # Never call the generic queue claim: an API request may have enqueued a
        # different topic while checkpoints were being read. Claim this ID only.
        with repo.connection() as db, transaction(db):
            _ensure_idle(repo, except_run_id=new_run_id, db=db)
            if db.execute("UPDATE runs SET status='running' WHERE id=? AND status='queued'", (new_run_id,)).rowcount != 1:
                raise _refuse("recovery_claim_failed", "恢复任务不再处于可领取状态，未发布结果。")
            repo._event(db, new_run_id, "run_started", {"execution_kind": _KIND})
        repo.event(new_run_id, "checkpoint_reused", metadata)
        target = repo.revision(topic["id"], created["target_revision_id"])
        snapshot = build_snapshot(topic, covered, results, revision_id=target["id"],
                                  graph_revision=target["revision_no"], run_id=new_run_id,
                                  source_paragraphs=paragraphs, source_documents=source_documents)
        snapshot["overlays"] = overlays
        snapshot["scope"] = {
            "completed_question_ids": list(results), "all_question_count": len(questions),
            "is_complete_topic": len(results) == len(questions),
        }
        snapshot["execution_provenance"] = deepcopy(metadata)
        repo.publish(new_run_id, snapshot, fresh)
        return _response(repo, new_run_id)
    except ServiceError as error:
        if new_run_id is not None:
            repo.finish_failed(new_run_id, error.code, error.message, stage="recovery")
        raise
    except Exception:
        if new_run_id is not None:
            repo.finish_failed(new_run_id, "recovery_publish_failed", "检查点恢复发布失败；来源任务和原始费用保留。", stage="recovery")
        raise _refuse("recovery_publish_failed", "检查点恢复发布未完成；未生成新的模型结果。") from None
    finally:
        lease.close()
