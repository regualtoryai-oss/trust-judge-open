"""Transactional business state. LangGraph checkpoints never act as publication state."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import re
import sqlite3
from uuid import uuid4

from app.config import Settings
from app.data.db import connect, initialize, transaction
from app.domain.graph import downstream_closure, validate_graph
from app.domain.snapshot import competition_files
from app.domain.spans import validate_span
from app.domain.source_bundles import build_source_bundles, paragraph_manifest_sha256
from app.domain.model_transport import MODEL_TRANSPORT_CODES, public_transport_failure
from app.errors import ServiceError
from app.utils import digest, dumps, loads, now

TERMINAL = {"succeeded", "failed", "cancelled", "superseded"}
WORKFLOW_STAGES = {"candidate_import", "scope_review", "purpose_review", "retrieve", "extract", "verify", "freeze", "authority_analyst", "cross_validation_analyst", "arbitrate", "baseline"}
WORKFLOW_STAGES |= {"paragraph_queries", "paragraph_recall", "paragraph_rank", "paragraph_annotate", "paragraph_fields"}
WORKFLOW_STAGES |= {"strict_selection", "strict_annotation", "source_bundle_selection"}
MODEL_REQUEST_STAGES = WORKFLOW_STAGES | {"select_evidence", "verify_evidence_scope", "narrow_evidence_annotations", "review_fact_purpose", "source_bundle_select"}


class Repository:
    def __init__(self, settings: Settings):
        self.settings = settings
        initialize(settings.database_path)
        with self.connection() as db, transaction(db):
            # Additive service migration, separate from the original-data migration.
            db.execute("CREATE TABLE IF NOT EXISTS service_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS revision_artifacts(revision_id TEXT PRIMARY KEY REFERENCES graph_revisions(id), payload_json TEXT NOT NULL CHECK(json_valid(payload_json)))")
            db.execute("CREATE TABLE IF NOT EXISTS run_options(run_id TEXT PRIMARY KEY REFERENCES runs(id), options_json TEXT NOT NULL CHECK(json_valid(options_json)))")
            for action in ("INSERT", "UPDATE", "DELETE"):
                key = "NEW" if action == "INSERT" else "OLD"
                db.execute(f"CREATE TRIGGER IF NOT EXISTS artifacts_ready_{action.lower()} BEFORE {action} ON revision_artifacts WHEN (SELECT state FROM graph_revisions WHERE id={key}.revision_id)='ready' BEGIN SELECT RAISE(ABORT,'immutable_ready_revision'); END")
            db.execute("CREATE TABLE IF NOT EXISTS run_refine_inputs(run_id TEXT NOT NULL REFERENCES runs(id), question_id TEXT NOT NULL REFERENCES questions(id), artifact_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL CHECK(json_valid(payload_json)), PRIMARY KEY(run_id,question_id))")
            for action in ("UPDATE", "DELETE"):
                db.execute(f"CREATE TRIGGER IF NOT EXISTS refine_input_{action.lower()} BEFORE {action} ON run_refine_inputs BEGIN SELECT RAISE(ABORT,'immutable_refine_input'); END")
            db.execute("INSERT OR IGNORE INTO service_migrations VALUES (1,?)", (now(),))

    @contextmanager
    def connection(self):
        db = connect(self.settings.database_path)
        try:
            yield db
        finally:
            db.close()

    def one(self, sql, args=(), *, db=None):
        if db is None:
            with self.connection() as conn:
                return self.one(sql, args, db=conn)
        row = db.execute(sql, args).fetchone()
        if row is None:
            raise ServiceError(404, "not_found", "找不到指定记录。")
        return dict(row)

    def all(self, sql, args=()):
        with self.connection() as db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    def topic(self, topic_id):
        return self.one("SELECT * FROM topics WHERE id=?", (topic_id,))

    def topics(self):
        return self.all("""SELECT t.*, (SELECT COUNT(*) FROM documents d WHERE d.topic_id=t.id) document_count,
          (SELECT COUNT(*) FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE d.topic_id=t.id) paragraph_count,
          (SELECT COUNT(*) FROM questions q WHERE q.topic_id=t.id) question_count FROM topics t ORDER BY t.created_at,t.id""")

    def questions(self, topic_id):
        self.topic(topic_id)
        return self.all("SELECT id,raw_question_id,question FROM questions WHERE topic_id=? ORDER BY rowid", (topic_id,))

    def paragraphs(self, topic_id):
        return self.all("""SELECT p.id,p.document_id,p.id paragraph_id,p.raw_para_id,p.text,p.start_cp document_start_cp,p.end_cp document_end_cp,
          d.raw_doc_id,d.title,d.content_sha256,d.content_url,d.raw_publish_time,d.parsed_publish_time,d.time_precision,d.source_credibility_json
          FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE d.topic_id=? ORDER BY d.rowid,p.rowid""", (topic_id,))

    def paragraph(self, paragraph_id):
        return self.one("SELECT p.*,d.raw_doc_id,d.title,d.topic_id FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE p.id=?", (paragraph_id,))

    def source_documents(self, topic_id, *, document_id=None):
        """Project original documents with a complete, ordered source manifest.

        Original IDs may be opaque and intervals may overlap. Ordering follows
        document codepoints; tied starts retain the database's import order.
        This projection contains no reference annotations or generated text.
        """
        restriction = " AND d.id=?" if document_id is not None else ""
        args = (topic_id, document_id) if document_id is not None else (topic_id,)
        with self.connection() as db:
            documents = [dict(row) for row in db.execute(
                "SELECT d.id,d.raw_doc_id,d.title,d.content,d.content_sha256 FROM documents d WHERE d.topic_id=?" + restriction + " ORDER BY d.rowid", args)]
            parents = [dict(row) for row in db.execute(
                "SELECT p.id,p.document_id,p.raw_para_id,p.start_cp,p.end_cp,p.text_sha256 FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE d.topic_id=?" + restriction + " ORDER BY d.rowid,p.start_cp,p.rowid", args)]
        manifests = {row["id"]: [] for row in documents}
        for parent in parents:
            manifests[parent["document_id"]].append({key: parent[key] for key in ("id", "raw_para_id", "start_cp", "end_cp", "text_sha256")})
        for document in documents:
            manifest = manifests[document["id"]]
            document["paragraph_manifest"] = manifest
            document["paragraph_manifest_sha256"] = paragraph_manifest_sha256(manifest)
        return documents

    def source_context(self, paragraph_id, *, before=1, after=3):
        """Return an exact document slice for inspection, without a model/run."""
        if type(before) is not int or type(after) is not int or min(before, after) < 0 or max(before, after) > 4 or before + 1 + after > 5:
            raise ServiceError(422, "source_context_range_invalid", "相邻原文一次最多读取连续 5 个原始段落。")
        anchor = self.paragraph(paragraph_id)
        documents = self.source_documents(anchor["topic_id"], document_id=anchor["document_id"])
        parents = self.all("""SELECT p.id,p.document_id,p.raw_para_id,p.text,p.start_cp document_start_cp,p.end_cp document_end_cp,p.text_sha256,d.raw_doc_id,d.content_sha256
          FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE p.document_id=? ORDER BY p.start_cp,p.rowid""", (anchor["document_id"],))
        try:
            bundle = build_source_bundles(documents, parents, [paragraph_id], before=before, after=after, max_paragraphs=5)[0]
        except (ValueError, IndexError):
            raise ServiceError(409, "source_context_integrity_failed", "原文完整性校验未通过，未返回相邻原文。") from None
        included = set(bundle["source_paragraph_ids"])
        return {**bundle, "anchor_paragraph_id": paragraph_id, "anchor_raw_para_id": anchor["raw_para_id"],
                "anchor_start_cp": anchor["start_cp"], "anchor_end_cp": anchor["end_cp"],
                "paragraph_count": len(included), "members": [
                    {"paragraph_id": p["id"], "raw_para_id": p["raw_para_id"], "start_cp": p["document_start_cp"], "end_cp": p["document_end_cp"]}
                    for p in parents if p["id"] in included]}

    def revision(self, topic_id, revision_id):
        return self.one("SELECT * FROM graph_revisions WHERE id=? AND topic_id=?", (revision_id, topic_id))

    def snapshot(self, topic_id, revision_id):
        revision = self.revision(topic_id, revision_id)
        if revision["state"] != "ready":
            raise ServiceError(409, "revision_not_ready", "该版本尚未成功发布，不能读取或导出研判成果。")
        return loads(self.one("SELECT payload_json FROM revision_artifacts WHERE revision_id=?", (revision_id,))["payload_json"])

    def evidence(self, evidence_id, revision_id):
        row = self.one("SELECT e.* FROM evidence e JOIN node_states n ON n.node_id=e.id WHERE e.id=? AND n.revision_id=?", (evidence_id, revision_id))
        snapshot = self.snapshot(row["topic_id"], revision_id)
        return next(e for e in snapshot["evidence"] if e["id"] == evidence_id)

    def config(self):
        def effective_budget(stage_budget):
            return min(self.settings.model_capacity_max_output_tokens, max(self.settings.model_max_output_tokens, stage_budget))

        config = {"model_name": self.settings.model_name, "model_provider": self.settings.model_provider,
                "max_output_tokens": self.settings.model_max_output_tokens,
                "output_budgets": {"extract": effective_budget(self.settings.model_extraction_max_output_tokens),
                                   "analysis": effective_budget(self.settings.model_analysis_max_output_tokens),
                                   "arbitrate": effective_budget(self.settings.model_arbitration_max_output_tokens),
                                   "capacity_max": self.settings.model_capacity_max_output_tokens,
                                   "retries": self.settings.model_capacity_retries},
                "generation_timeout_seconds": max(self.settings.model_timeout_seconds, self.settings.model_generation_timeout_seconds),
                "thinking": self.settings.model_thinking, "reasoning_effort": self.settings.model_reasoning_effort,
                "extraction_thinking": "disabled" if self.settings.model_provider == "deepseek" else None,
                "retrieval_limit": self.settings.retrieval_limit,
                "source_strategy": ("langchain_source_book" if self.settings.source_bundle_front_framework == "langchain" else "dsh_source_bundles") if self.settings.model_source_bundle_enabled else (("strict_balanced_scoped" if self.settings.strict_evidence_balanced_enabled else "strict_bm25_full_parents") if self.settings.model_strict_evidence_enabled else ("paragraph_ranked_v19" if self.settings.model_paragraph_ranked_enabled else "full_source_scan")),
                **({"source_bundle_enabled": True, "source_bundle_protocol": ("document-source-context-v1"
                    if self.settings.source_bundle_config.get("context_passages") is True else "document-source-bundle-v1"),
                    "source_bundle_config": self.settings.source_bundle_config} if self.settings.model_source_bundle_enabled else {}),
                "strict_evidence_enabled": self.settings.model_strict_evidence_enabled,
                "strict_evidence_config": {"candidate_limit": self.settings.strict_evidence_candidate_limit,
                    **({"balanced_scoped_enabled": True, "balance_policy": {"target": 10, "per_doc_soft_cap": 2}, "recall_strategy": "balanced"} if self.settings.strict_evidence_balanced_enabled else {})},
                "strict_evidence_max_requests": self.settings.strict_evidence_max_requests,
                "strict_evidence_token_stop": self.settings.strict_evidence_token_stop,
                "strict_evidence_cost_parent_run_ids": list(self.settings.strict_evidence_cost_parent_run_ids),
                "strict_evidence_response_reuse_closures": [str(path.resolve()) for path in self.settings.strict_evidence_response_reuse_closures],
                "strict_evidence_response_reuse_operations": self.settings.strict_evidence_response_reuse_operations,
                "paragraph_ranked_enabled": self.settings.model_paragraph_ranked_enabled,
                "paragraph_ranked_config": self.settings.paragraph_ranked_config,
                "cost_guard_sha256": hashlib.sha256((Path(__file__).parent / "domain/model_cost_guard.py").read_bytes()).hexdigest(),
                "source_batch_codepoints": self.settings.source_batch_codepoints,
                "source_batch_spans": self.settings.source_batch_spans,
                "timeout_seconds": self.settings.model_timeout_seconds, "max_retries": self.settings.model_max_retries,
                "strict_scope_enabled": not (self.settings.model_paragraph_ranked_enabled or self.settings.model_strict_evidence_enabled or self.settings.model_source_bundle_enabled), "premise_bank_enabled": self.settings.model_premise_bank_enabled, "prompt_version": self.settings.model_prompt_version, "prompt_sha256": hashlib.sha256(b"\0".join((Path(__file__).parent / name).read_bytes() for name in (("config.py", "main.py", "repository.py", "worker.py", "recovery.py", "agents/provider.py", "agents/schemas.py", "domain/model_transport.py", "domain/source_catalog.py", "domain/spans.py", "domain/coverage.py", "domain/evidence_selection.py", "domain/strict_verification.py", "domain/compiled_scope.py", "domain/paragraph_assertions.py", "domain/recall_recovery.py", "domain/premise_bank.py", "domain/paragraph_ranked.py", "domain/paragraph_mirror_recall.py", "domain/paragraph_mirrors.py", "domain/source_metadata.py", "domain/material_eligibility.py", "domain/refine_input.py", "domain/scope_batches.py", "domain/manual_identity.py", "domain/source_lineage.py", "domain/program_inventory.py", "domain/snapshot.py", "workflows/analysis.py", "workflows/paragraph_ranked_nodes.py", "workflows/checkpoint_resume.py") + (("agents/strict_evidence.py", "agents/strict_transport.py", "agents/paid_response_cache.py", "agents/strict_json_syntax.py", "domain/numeric_literals.py", "workflows/strict_workbench.py") if self.settings.model_strict_evidence_enabled else ()) + (("agents/compact_selection.py", "agents/scoped_arbitration.py", "domain/evidence_balance.py", "domain/question_recall.py") if self.settings.strict_evidence_balanced_enabled else ())))).hexdigest(), "code_version": "0.1.0"}


        if self.settings.model_source_bundle_enabled:
            # Bind actual bridge/codec/tail code as well as the former prompt
            # surface. Test fixtures and source data never enter this binding.
            root = Path(__file__).resolve().parents[1]
            paths = sorted((root / "app").rglob("*.py"))
            paths += sorted(path for path in (root / "integrations/dsh").glob("*.mjs")
                if not path.name.endswith(".test.mjs") and "fixture" not in path.name)
            paths += [root / "scripts/source_bundle_tool.py"]
            if self.settings.source_bundle_front_framework == "langchain":
                paths += [root / "scripts/lc_bundle_case.py"]
            manifest = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths}
            config["source_bundle_code_manifest"] = manifest
            config["prompt_sha256"] = digest({"base_prompt_sha256": config["prompt_sha256"],
                "source_bundle_code_manifest": manifest})
        return config

    def run(self, run_id):
        row = self.one("SELECT * FROM runs WHERE id=?", (run_id,))
        error = loads(row["error_json"])
        options = loads(self.one("SELECT options_json FROM run_options WHERE run_id=?", (run_id,))["options_json"], {})
        paused = bool(error and error.get("code") == "paused")
        reason = self._resume_blocked_reason(row)
        inputs = self.all("SELECT question_id,artifact_sha256,payload_json FROM run_refine_inputs WHERE run_id=? ORDER BY question_id", (run_id,))
        derived = []
        for item in inputs:
            artifact = loads(item["payload_json"])
            derived.append({"question_id": item["question_id"], "artifact_sha256": item["artifact_sha256"],
                            "source_revision_id": artifact["source_revision_id"], "source_run_id": artifact["source_run_id"],
                            "candidate_count": len(artifact["candidate_evidence"]),
                            "historical_source_coverage": artifact["source_coverage"],
                            "historical_source_usage": artifact["source_usage"]})
        config = loads(row["model_config_json"])
        return {"mode": options.get("execution_mode", row["mode"]),
                "strict_scope_enabled": bool(config.get("strict_scope_enabled")), "premise_bank_enabled": bool(config.get("premise_bank_enabled")),
                "paragraph_ranked_enabled": config.get("paragraph_ranked_enabled") is True, "derived_inputs": derived,
                "strict_evidence_enabled": config.get("strict_evidence_enabled") is True,
                "source_bundle_enabled": config.get("source_bundle_enabled") is True,
                "source_bundle_front_framework": config.get("source_bundle_config", {}).get("front_framework", "dsh") if config.get("source_bundle_enabled") is True else None,
                "source_bundle_selection_only": config.get("source_bundle_config", {}).get("selection_only", False),
                "source_bundle_graded_selection": config.get("source_bundle_config", {}).get("graded_selection", False),
                "source_bundle_compact_analyses": config.get("source_bundle_config", {}).get("compact_analyses", False),
                "source_bundle_issue_frame": config.get("source_bundle_config", {}).get("issue_frame", False),
                "source_bundle_context_passages": config.get("source_bundle_config", {}).get("context_passages", False),
                "source_bundle_complete_grading": config.get("source_bundle_config", {}).get("complete_grading", False),
                "source_bundle_hybrid_retrieval": config.get("source_bundle_config", {}).get("hybrid_retrieval", False),
                "source_bundle_paragraph_quality": config.get("source_bundle_config", {}).get("paragraph_quality", False),
                "source_bundle_paragraph_quality_compact": config.get("source_bundle_config", {}).get("paragraph_quality_compact", False),
                "source_bundle_parent_token_pack": config.get("source_bundle_config", {}).get("parent_token_pack", False),
                "source_bundle_recovered_front_run_id": config.get("source_bundle_config", {}).get("sealed_front_recovery", {}).get("origin_run_id"),
                "id": row["id"], "run_id": row["id"], "topic_id": row["topic_id"], "status": row["status"],
                "view_status": "paused" if paused and row["status"] == "cancelled" else "pausing" if paused and row["status"] not in TERMINAL else row["status"],
                "resumable": reason is None, "resume_blocked_reason": reason,
                "resumed_from_run_id": options.get("resume_from_run_id"),
                "resume_mode": options.get("resume_mode"),
                "base_revision_id": row["base_revision_id"], "target_revision_id": row["target_revision_id"],
                "cancel_requested": bool(row["cancel_requested"]), "question_ids": loads(row["question_ids_json"]),
                "error": error, "usage": loads(row["usage_json"]),
                "created_at": row["created_at"], "completed_at": row["completed_at"]}

    def _resume_blocked_reason(self, row, *, db=None):
        if row["status"] not in {"cancelled", "failed"}:
            return "任务尚未暂停或失败。"
        topic = self.one("SELECT * FROM topics WHERE id=?", (row["topic_id"],), db=db)
        if topic["head_revision_id"] != row["target_revision_id"]:
            return "专题已有后续版本，请基于最新任务继续。"
        revision = self.one("SELECT input_manifest_sha256 FROM graph_revisions WHERE id=?", (row["target_revision_id"],), db=db)
        if revision["input_manifest_sha256"] != topic["source_manifest_sha256"]:
            return "原始资料清单已变化，请重新研判。"
        budget_error = loads(row["error_json"], {})
        detail = budget_error.get("details")
        frozen_config = loads(row["model_config_json"])
        if budget_error.get("code") == "experiment_usage_stop" and isinstance(detail, dict):
            known, estimate, capacity, stop = (detail.get("known_tokens"), detail.get("input_estimate"),
                detail.get("last_max_output_tokens"), frozen_config.get("strict_evidence_token_stop"))
            if (all(type(n) is int and n >= 0 for n in (known, estimate, capacity, stop))
                    and known + estimate + capacity + 512 > stop):
                return (f"本题累计已用 {known:,} tokens，当前保护线剩余 {max(0, stop-known):,}；"
                    f"下一阶段预留至少需要 {estimate+capacity+512:,}，当前配置不能续跑。检查点与费用记录保留。")
        if loads(row["model_config_json"]) != self.config():
            return "模型或流程已更新，旧检查点不可续跑；请重新研判。"
        if self.settings.checkpoint_path.is_file():
            try:
                with sqlite3.connect(f"file:{self.settings.checkpoint_path.as_posix()}?mode=ro", uri=True) as connection:
                    connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            except (sqlite3.DatabaseError, OSError):
                return "检查点存储无法读取，不能续跑；任务历史和已发布成果保留。"
        if not self.checkpoint_sources(row["id"], loads(row["question_ids_json"])):
            began_work = self.one("SELECT COUNT(*) AS n FROM run_events WHERE run_id=? AND type IN ('node_started','node_completed','tool_started','tool_completed')", (row["id"],), db=db)["n"]
            error = loads(row["error_json"], {})
            if began_work or row["status"] == "failed" and error.get("code") not in {"worker_interrupted"}:
                return "任务没有可恢复的检查点，请重新研判。"
        return None

    def runs(self, topic_id, *, limit=50):
        self.topic(topic_id)
        return [self.run(row["id"]) for row in self.all("SELECT id FROM runs WHERE topic_id=? ORDER BY created_at DESC,rowid DESC LIMIT ?", (topic_id, limit))]

    def checkpoint_questions(self, run_id, question_ids):
        """Read only checkpoint identities; workflow validates their full state."""
        if not self.settings.checkpoint_path.is_file():
            return []
        try:
            with sqlite3.connect(f"file:{self.settings.checkpoint_path.as_posix()}?mode=ro", uri=True) as connection:
                return [qid for qid in question_ids if connection.execute("SELECT 1 FROM checkpoints WHERE thread_id=? LIMIT 1", (run_id + ":" + qid,)).fetchone()]
        except sqlite3.DatabaseError:
            return []

    def checkpoint_sources(self, run_id, question_ids):
        """Keep nearest per-question checkpoint across repeated pause/resume."""
        source = self.one("SELECT * FROM runs WHERE id=?", (run_id,))
        wanted, found, seen = set(question_ids), {}, set()
        while run_id and run_id not in seen and wanted:
            seen.add(run_id)
            rows = self.all("SELECT * FROM runs WHERE id=?", (run_id,))
            if not rows:
                break
            row = rows[0]
            if row["topic_id"] != source["topic_id"] or row["model_config_json"] != source["model_config_json"]:
                break
            for qid in self.checkpoint_questions(run_id, sorted(wanted.intersection(loads(row["question_ids_json"])))):
                found[qid] = run_id
                wanted.remove(qid)
            options = loads(self.one("SELECT options_json FROM run_options WHERE run_id=?", (run_id,))["options_json"], {})
            run_id = options.get("resume_from_run_id")
        return found

    def resume(self, run_id, key, *, model_available):
        source = self.one("SELECT * FROM runs WHERE id=?", (run_id,))
        options = loads(self.one("SELECT options_json FROM run_options WHERE run_id=?", (run_id,))["options_json"], {})
        safe = {name: options[name] for name in ("overlays", "include_baseline", "execution_mode") if name in options}
        checkpoint_sources = self.checkpoint_sources(run_id, loads(source["question_ids_json"]))
        safe.update({"resume_from_run_id": run_id, "resume_checkpoint_sources": checkpoint_sources, "resume_checkpoint_question_ids": list(checkpoint_sources)})
        safe["resume_mode"] = "checkpoint" if safe["resume_checkpoint_question_ids"] else "start_unstarted"
        request = {"question_ids": loads(source["question_ids_json"]), "base_revision_id": source["target_revision_id"],
                   "include_baseline": bool(options.get("include_baseline", False)), "mode": "retry", "resume_from_run_id": run_id}
        return self.create_run(source["topic_id"], request, key, model_available=model_available, options=safe)

    def activity(self, run_id, after=0):
        run, events = self.run(run_id), self.events(run_id)
        def diagnostic(data):
            # Public observability must never copy provider bodies, free-form
            # identifiers or exception payloads into tool records.
            result = {}
            code = data.get("code")
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
                result["code"] = code
            message = data.get("message")
            if (isinstance(message, str) and 0 < len(message) <= 400
                    and not re.search(r"[\r\n]|^[{\[]|```|\bTraceback\b|\bBearer\s+|\bsk-[a-z0-9]{12,}", message, re.I)):
                result["message"] = message
            details = data.get("details")
            safe = {}
            if isinstance(details, dict):
                if isinstance(code, str) and code in MODEL_TRANSPORT_CODES:
                    safe.update(public_transport_failure(code, details)["details"])
                check = details.get("check")
                if isinstance(check, str) and check in {"paragraph_contract", "paragraph_fields_not_verified",
                        "paragraph_source_proof_replay", "paragraph_ranked_freeze_replay", "rank_selection_or_critical_qualifier",
                        "paragraph_ranked_duplicate_json_key", "ranked_checkpoint_projection"}:
                    safe["check"] = check
                if isinstance(check, str) and check in {"schema", "json", "structured_output", "candidate_schema", "duplicate_judgment_id", "ungrounded_judgment", "unknown_prerequisite", "assertion_without_support", "prerequisite_cycle", "duplicate_conflict_id", "invalid_evidence_reference", "unknown_paragraph", "unseen_paragraph", "quote_occurrence", "quote_span", "unknown_interpretation_target", "duplicate_prerequisite", "overlapping_evidence_relations", "overlapping_conflict_sides", "ambiguous_conflict_resolution", "uncited_numeric_literal", "source_selection", "unknown_source_selection", "source_catalog", "source_catalog_id", "source_catalog_limit", "source_interval_selection", "source_window_catalog", "source_window_selection", "evidence_selection", "precision_decisions", "precision_duplicate_json_key", "precision_candidate_catalog", "scope_question_invalid", "scope_catalog_invalid", "scope_decision_contract", "scope_target_contract", "scope_compiled_contract", "scope_compiled_repair_binding", "scope_verification_duplicate_json_key", "scope_annotation_narrowing", "manual_identity", "scope_ledger_invalid", "annotation_refocus_catalog_invalid", "annotation_refocus_manual_protected", "annotation_correction_contract", "annotation_correction_empty", "annotation_correction_no_supported_nonempty", "annotation_refocus_ledger_invalid", "annotation_refocus_duplicate_json_key", "refine_artifact_hash", "refine_question_binding", "refine_source_manifest", "refine_overlay_binding", "refine_incomplete_coverage", "refine_candidate_source", "premise_bank_binding", "premise_bank_replay", "premise_purpose_contract", "premise_purpose_batch_identity", "premise_purpose_coverage", "premise_purpose_full_coverage", "premise_purpose_equivalence", "premise_purpose_ledger_replay", "premise_purpose_duplicate_json_key", "premise_formal_authorization_replay", "premise_formal_projection_replay", "premise_manual_purpose_unverified"}:
                    safe["check"] = check
                def public_path(value):
                    return (isinstance(value, str) and len(value) <= 160 and re.fullmatch(r"(?:judgments|conflicts|candidates|evidence|assessments)(?:(?:\[\d{1,6}\]|\.\d{1,6}))?(?:\.(?:id|statement|evidence_ids|prerequisite_judgment_ids|supporting_evidence_ids|conflicting_evidence_ids|paragraph_id|quote|start_cp|end_cp|source_span_ids|source_start_id|source_end_id|source_window_id|evidence_id|relation|reason|equivalent_to)(?:\[\d{1,6}\]|\.\d{1,6})?)?|decisions(?:\[\d{1,6}\]|\.\d{1,6})?(?:\.(?:relation|reason|equivalent_to|fact_entailment|target_relation|target_question_text|qualifier_dimension|requires_unstated_bridge|target_fact_present_but_annotation_incomplete|target_binding_id|quote_requires_unstated_bridge|target_requires_unstated_bridge|field_checks(?:\.(?:fact_des|entities|argument(?:\.(?:时间|主体|地点|内容|影响|数值口径))?))?))?|(?:decision|conclusion|supporting_evidence_ids|conflicting_evidence_ids|source_span_ids|source_start_id|source_end_id|source_window_id|source_catalog_id|candidate_catalog_id|target_catalog_id|selected_evidence_ids)(?:\[\d{1,6}\]|\.\d{1,6})?", value))
                field = details.get("field_path")
                if public_path(field):
                    safe["field_path"] = field
                for key in ("judgment_index", "conflict_index", "candidate_index", "evidence_count", "prerequisite_count", "judgment_count", "candidate_count", "invalid_candidate_count", "duplicate_key_count", "total_count", "attempt", "repair_attempt"):
                    if type(details.get(key)) is int and 0 <= details[key] <= 1_000_000:
                        safe[key] = details[key]
                if details.get("repair_exhausted") is True:
                    safe["repair_exhausted"] = True
                if isinstance(details.get("selection_issues"), list):
                    issue_checks = {"unknown_source_start", "unknown_source_end", "cross_paragraph_interval", "reversed_source_interval", "invalid_source_interval", "source_catalog_mismatch", "unknown_source_window", "invalid_source_window_catalog", "invalid_source_window", "precision_duplicate_self", "precision_duplicate_target_not_retained", "precision_unexpected_equivalent_to", "precision_decision_coverage"}
                    issues = []
                    for item in details["selection_issues"][:3]:
                        if not isinstance(item, dict) or not isinstance(item.get("check"), str) or item["check"] not in issue_checks:
                            continue
                        issue = {"check": item["check"]}
                        index = item.get("candidate_index")
                        if type(index) is int and 0 <= index <= 1_000_000:
                            issue["candidate_index"] = index
                        if public_path(item.get("field_path")):
                            issue["field_path"] = item["field_path"]
                        issues.append(issue)
                    if issues:
                        safe["selection_issues"] = issues
                if isinstance(details.get("schema_errors"), list):
                    fields = {"evidence", "paragraph_id", "quote", "fact_des", "argument", "entities", "start_cp", "end_cp", "source_span_ids", "source_start_id", "source_end_id", "source_window_id", "source_catalog_id", "candidate_catalog_id", "target_catalog_id", "bank_id", "purpose_catalog_id", "purpose_binding_id", "introduces_unstated_source_fact", "decisions", "selected_evidence_ids", "assessments", "evidence_id", "relation", "reason", "equivalent_to", "fact_entailment", "target_relation", "target_question_text", "qualifier_dimension", "requires_unstated_bridge", "quote_requires_unstated_bridge", "target_requires_unstated_bridge", "target_binding_id", "field_checks", "target_fact_present_but_annotation_incomplete", "annotations", "annotation", "corrections", "时间", "主体", "地点", "内容", "影响", "数值口径", "decision", "conclusion", "judgments", "supporting_evidence_ids", "conflicting_evidence_ids", "id", "statement", "evidence_ids", "prerequisite_judgment_ids", "conflicts", "status", "resolution_type", "basis", "accepted_reason", "rejected_reason", "minority_opinion", "reopen_conditions"}
                    types = {"missing", "extra_forbidden", "string_type", "string_too_short", "string_too_long", "list_type", "dict_type", "model_type", "model_attributes_type", "int_type", "int_parsing", "float_type", "bool_type", "literal_error", "enum", "greater_than", "greater_than_equal", "less_than", "less_than_equal", "too_short", "too_long", "finite_number", "value_error", "json_invalid", "json_type", "none_required", "union_tag_invalid", "union_tag_not_found"}
                    fields.update({"time", "subject", "location", "content", "impact", "numeric_scope"})
                    errors = []
                    for item in details["schema_errors"][:3]:
                        if not isinstance(item, dict) or not isinstance(item.get("loc"), list):
                            continue
                        location = [part if type(part) is int and 0 <= part <= 1_000_000 or isinstance(part, str) and part in fields else "<field>" for part in item["loc"][:8]]
                        error_type = item.get("type")
                        errors.append({"loc": location, "type": error_type if isinstance(error_type, str) and error_type in types else "schema"})
                    if errors:
                        safe["schema_errors"] = errors
                if safe:
                    result["details"] = safe
            return result
        def summary(value):
            if isinstance(value, str):
                return value
            labels = {"paragraph_count": "段原文", "candidate_count": "条候选", "evidence_count": "条证据",
                      "quarantined_value_count": "项可选字段原文不符已隔离", "deduplicated_value_count": "项可选字段重复已去重", "affected_annotation_count": "条证据含字段质量提示",
                      "selected_count": "条精筛保留", "rejected_count": "条精筛排除", "manual_preserved_count": "条人工保留", "fact_view_count": "个事实版本", "reviewed_count": "个已审版本", "verified_fact_count": "个通过事实", "narrowing_count": "个待缩窄", "unresolved_count": "个待处理"}
            if isinstance(value, dict):
                counts = [f"{value[key]} {label}" for key, label in labels.items() if type(value.get(key)) is int and value[key] >= 0]
                if (type(value.get("batch_index")) is int and type(value.get("batch_count")) is int
                        and 1 <= value["batch_index"] <= value["batch_count"]):
                    counts.insert(0, f"第 {value['batch_index']} / {value['batch_count']} 批")
                return " · ".join(counts)
            return ""
        completed, active, calls, failed = set(), {}, {}, {}
        for event in events:
            data, kind = event["data"], event["type"]
            qid, stage = data.get("question_id"), data.get("node") or data.get("stage")
            if kind == "node_started" and stage in WORKFLOW_STAGES:
                active[(qid, stage)] = event["created_at"]
            elif kind == "node_completed" and stage in WORKFLOW_STAGES:
                completed.add((qid, stage))
                active.pop((qid, stage), None)
                failed.pop((qid, stage), None)
            elif kind == "checkpoint_reused":
                for reused in data.get("reused_nodes", []):
                    if reused in WORKFLOW_STAGES:
                        completed.add((qid, reused))
                        failed.pop((qid, reused), None)
            elif kind == "publication_started":
                active[(None, "publish")] = event["created_at"]
            elif kind == "run_succeeded":
                completed.add((None, "publish"))
                active.clear()
                failed.clear()
            if kind.startswith("tool_"):
                call_id = data.get("call_id") or str(event["seq"])
                call = calls.setdefault(call_id, {"call_id": call_id, "name": data.get("name") or data.get("tool") or data.get("tool_name") or stage or "工具调用", "question_id": qid,
                                                   "kind": data.get("kind") if data.get("kind") in {"local", "model"} else "local",
                                                   "engine": data.get("engine") or None, "stage": stage,
                                                   "status": "running", "started_at": event["created_at"], "completed_at": None, "elapsed_ms": None, "summary": summary(data.get("summary"))})
                if kind != "tool_started":
                    local_finished = (kind == "tool_finished" and data.get("name") == "local_complete_parent_rank"
                        and isinstance(data.get("summary"), dict) and data["summary"].get("status") == "succeeded")
                    call.update(status="completed" if kind == "tool_completed" or local_finished else "deferred" if kind == "tool_deferred" else "cancelled" if kind == "tool_cancelled" or data.get("code") == "cancelled" else "failed", completed_at=event["created_at"], elapsed_ms=data.get("elapsed_ms"), summary=summary(data.get("summary")) or call["summary"])
                    call.update(diagnostic(data))
                    if call["status"] == "failed" and stage in WORKFLOW_STAGES | {"snapshot", "publish"} and (not str(call["name"]).endswith("_repair") or run["status"] in TERMINAL):
                        failed[(qid, stage)] = diagnostic(data)
        if run["status"] == "failed" and isinstance(run.get("error"), dict):
            error = run["error"]
            stage = error.get("stage")
            if stage in MODEL_REQUEST_STAGES | {"snapshot", "build_snapshot", "publish"}:
                details = error.get("details")
                qid = details.get("question_id") if isinstance(details, dict) else None
                if qid not in run["question_ids"]:
                    qid = run["question_ids"][0] if len(run["question_ids"]) == 1 and stage in WORKFLOW_STAGES else None
                if (qid, stage) not in completed:
                    failed[(qid, stage)] = diagnostic(error)
        failed = {key: value for key, value in failed.items() if key not in completed}
        terminal = run["status"] in TERMINAL
        if terminal:
            for call in calls.values():
                if call["status"] == "running":
                    call.update(status="cancelled" if run["status"] == "cancelled" else "failed", completed_at=run["completed_at"])
        started = next((e["created_at"] for e in events if e["type"] == "run_started"), run["created_at"])
        elapsed = max(0, (datetime.fromisoformat(run["completed_at"] or now()) - datetime.fromisoformat(started)).total_seconds())
        options = loads(self.one("SELECT options_json FROM run_options WHERE run_id=?", (run_id,))["options_json"], {})
        per_question = 5 if run.get("source_bundle_enabled") else (6 if run.get("strict_evidence_enabled") else (9 if run.get("paragraph_ranked_enabled") else (6 if run.get("mode") == "quality-refine" else (8 if run.get("strict_scope_enabled") else 7))))
        total = len(run["question_ids"]) * (per_question + int(run.get("premise_bank_enabled", False)) + int(bool(options.get("include_baseline")))) + 1
        active_rows = [{"question_id": qid, "stage": stage} for qid, stage in active] if not terminal else []
        paused_rows = [{"question_id": qid, "stage": stage} for qid, stage in active if (qid, stage) not in completed] if run.get("view_status") in {"paused", "pausing"} else []
        return {"run": run, "events": [{key: value for key, value in event.items() if key != "data_json"} for event in events if event["seq"] > after],
                "progress": {"completed_stages": [{"question_id": qid, "stage": stage} for qid, stage in sorted(completed, key=lambda value: (value[0] or "", value[1]))],
                             "active_stages": active_rows, "failed_stages": [{"question_id": qid, "stage": stage, **error} for (qid, stage), error in failed.items()],
                             "paused_stages": paused_rows, "current_stage": active_rows[-1]["stage"] if active_rows else None,
                             "completed_count": len(completed), "total_count": total, "elapsed_seconds": round(elapsed, 3)},
                "tool_calls": list(calls.values()), "usage": run["usage"]}

    def _event(self, db, run_id, event_type, data):
        seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM run_events WHERE run_id=?", (run_id,)).fetchone()[0]
        db.execute("INSERT INTO run_events VALUES (?,?,?,?,?)", (run_id, seq, event_type, dumps(data), now()))

    def event(self, run_id, event_type, data):
        with self.connection() as db, transaction(db):
            if self.one("SELECT status FROM runs WHERE id=?", (run_id,), db=db)["status"] in TERMINAL:
                return
            self._event(db, run_id, event_type, data)

    def events(self, run_id, after=0):
        self.run(run_id)
        return [{**r, "data": loads(r["data_json"])} for r in self.all("SELECT * FROM run_events WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after))]

    def create_run(self, topic_id, request, key, *, model_available, options=None, preview=None):
        request_hash = digest(request)
        with self.connection() as db, transaction(db):
            old = db.execute("SELECT id,request_sha256 FROM runs WHERE topic_id=? AND idempotency_key=?", (topic_id, key)).fetchone()
            if old:
                if old["request_sha256"] != request_hash:
                    raise ServiceError(409, "idempotency_conflict", "同一个幂等键已用于不同请求。")
                return self.run(old["id"])
            topic = self.one("SELECT * FROM topics WHERE id=?", (topic_id,), db=db)
            if request.get("resume_from_run_id"):
                source = self.one("SELECT * FROM runs WHERE id=? AND topic_id=?", (request["resume_from_run_id"], topic_id), db=db)
                reason = self._resume_blocked_reason(source, db=db)
                if reason:
                    raise ServiceError(409, "resume_not_available", reason)
            base = request["base_revision_id"]
            if base != topic["head_revision_id"]:
                raise ServiceError(409, "stale_base", "专题版本已变化，请刷新后重新预览。")
            if options is None and base:
                previous = db.execute("SELECT o.options_json FROM run_options o JOIN runs r ON r.id=o.run_id WHERE r.target_revision_id=?", (base,)).fetchone()
                if previous:
                    # Execution controls belong only to their original task.
                    # Inheriting checkpoint_recovery made every later UI retry
                    # hit the explicit CLI recovery guard before any work ran.
                    inherited = loads(previous["options_json"], {})
                    options = {name: inherited[name] for name in ("overlays", "include_baseline") if name in inherited}
                    options["include_baseline"] = request.get("include_baseline", False)
            qids = request["question_ids"]
            if not qids or len(qids) != len(set(qids)) or len(qids) > self.settings.max_questions_per_run:
                raise ServiceError(422, "invalid_questions", "问题列表不能为空、重复或超出运行上限。")
            if (self.settings.model_strict_evidence_enabled or self.settings.model_source_bundle_enabled) and (len(qids) != 1 or request.get("include_baseline")):
                raise ServiceError(422, "strict_question_limit", "严格选证实验一次只运行一道题，并使用真实双分析与仲裁；暂不附加基线调用。")
            known = {r[0] for r in db.execute("SELECT id FROM questions WHERE topic_id=?", (topic_id,))}
            if not set(qids) <= known:
                raise ServiceError(422, "foreign_question", "问题不属于当前专题。")
            if not model_available:
                raise ServiceError(503, "model_not_configured", "请先在 .env 配置真实模型名称和密钥，然后重启 API 与 worker。")
            if db.execute("SELECT 1 FROM runs WHERE topic_id=? AND status IN ('queued','running','awaiting_review')", (topic_id,)).fetchone():
                raise ServiceError(409, "topic_busy", "该专题已有运行任务，请等待或取消。")
            if preview:
                p = self.one("SELECT * FROM change_previews WHERE id=? AND topic_id=?", (preview, topic_id), db=db)
                if p["consumed_by_run_id"] or p["expires_at"] < now():
                    raise ServiceError(409, "expired_preview", "预览已过期或已提交，请重新预览。")
                if p["operations_sha256"] != request["preview_content_hash"] or p["base_revision_id"] != base or p["config_sha256"] != digest(self.config()):
                    raise ServiceError(409, "preview_mismatch", "预览内容、配置或版本不一致，请重新预览。")
            from app.domain.refine_input import create_candidate_artifact, RefineInputInvalid
            options = dict(options or {"include_baseline": request.get("include_baseline", False)})
            refine_inputs = {}
            if request.get("resume_from_run_id"):
                # Same-version normal checkpoint continuation: copy only the
                # immutable internal input, never synthesize cross-version state.
                refine_inputs = {row["question_id"]: loads(row["payload_json"]) for row in db.execute(
                    "SELECT question_id,payload_json FROM run_refine_inputs WHERE run_id=?", (request["resume_from_run_id"],))}
            elif request["mode"] == "quality-refine":
                if self.settings.model_paragraph_ranked_enabled:
                    raise ServiceError(409, "paragraph_refine_requires_fresh_ranking", "v19使用段落候选排序；旧完整扫描候选不能冒充本策略输入，请新建研判。")
                source_revision_id = request.get("source_revision_id")
                if not isinstance(source_revision_id, str) or not source_revision_id:
                    raise ServiceError(422, "refine_source_required", "请选择已发布的完整候选来源版本。")
                source_revision = self.one("SELECT * FROM graph_revisions WHERE id=? AND topic_id=?", (source_revision_id, topic_id), db=db)
                if source_revision["state"] != "ready":
                    raise ServiceError(409, "refine_source_not_ready", "质量复核必须指定已发布的固定版本。")
                if source_revision["input_manifest_sha256"] != topic["source_manifest_sha256"]:
                    raise ServiceError(409, "refine_source_manifest", "原文集合已变化，不能复用该候选工件。")
                artifact_json = self.one("SELECT payload_json FROM revision_artifacts WHERE revision_id=?", (source_revision_id,), db=db)["payload_json"]
                source_snapshot = loads(artifact_json)
                source_overlays = source_snapshot.get("overlays", {"excluded": [], "manual_evidence": [], "interpretation_overrides": {}})
                current_overlays = options.get("overlays", source_overlays)
                if digest(current_overlays) != digest(source_overlays):
                    raise ServiceError(409, "refine_overlay_binding", "人工标注或排除项已变化，请重新生成候选输入。")
                options.update({"execution_mode": "quality-refine", "overlays": current_overlays})
                source_paragraphs = self.paragraphs(topic_id)
                for qid in qids:
                    result = source_snapshot.get("question_results", {}).get(qid)
                    if not isinstance(result, dict):
                        raise ServiceError(409, "refine_missing_candidate_ledger", "该固定版本未保存本题完整候选池。")
                    origin = self.one("SELECT * FROM runs WHERE id=? AND topic_id=?", (result.get("origin_run_id"), topic_id), db=db)
                    if origin["status"] != "succeeded" or qid not in loads(origin["question_ids_json"]):
                        raise ServiceError(409, "refine_source_origin", "候选工件来源任务未完整成功。")
                    try:
                        refine_inputs[qid] = create_candidate_artifact(topic_id=topic_id,
                            source_revision_id=source_revision_id,
                            source_artifact_sha256=hashlib.sha256(artifact_json.encode("utf-8")).hexdigest(),
                            source_manifest_sha256=topic["source_manifest_sha256"],
                            question=self.one("SELECT id,raw_question_id,question FROM questions WHERE id=? AND topic_id=?", (qid, topic_id), db=db),
                            paragraphs=source_paragraphs, result=result, source_config=loads(origin["model_config_json"]),
                            overlays=current_overlays, created_at=now())
                    except RefineInputInvalid as error:
                        raise ServiceError(409, error.check, "候选工件未通过原文、覆盖或身份校验，尚未调用模型。") from None
            elif request.get("source_revision_id") is not None:
                raise ServiceError(422, "unexpected_refine_source", "只有 quality-refine 新任务接受候选来源版本。")
            if request["mode"] == "quality-refine" and not refine_inputs:
                raise ServiceError(422, "refine_missing_candidate_ledger", "质量复核须指定完整候选工件。")
            run_id, revision_id = str(uuid4()), str(uuid4())
            number = db.execute("SELECT COALESCE(MAX(revision_no),0)+1 FROM graph_revisions WHERE topic_id=?", (topic_id,)).fetchone()[0]
            db.execute("INSERT INTO graph_revisions(id,topic_id,revision_no,parent_revision_id,state,input_manifest_sha256,config_sha256,created_at) VALUES (?,?,?,?,'building',?,?,?)", (revision_id, topic_id, number, base, topic["source_manifest_sha256"], digest(self.config()), now()))
            db.execute("""INSERT INTO runs(id,topic_id,base_revision_id,target_revision_id,mode,status,question_ids_json,model_config_json,prompt_bundle_sha256,code_version,actor_id,idempotency_key,request_sha256,created_at)
              VALUES (?,?,?,?,?,'queued',?,?,?,?,?,?,?,?)""", (run_id, topic_id, base, revision_id, ("recompute" if request["mode"] == "quality-refine" else request["mode"]), dumps(qids), dumps(self.config()), self.config()["prompt_sha256"], "0.1.0", self.settings.local_actor, key, request_hash, now()))
            db.execute("INSERT INTO run_options VALUES (?,?)", (run_id, dumps(options)))
            for qid, artifact in refine_inputs.items():
                db.execute("INSERT INTO run_refine_inputs VALUES (?,?,?,?)", (run_id, qid, artifact["artifact_sha256"], dumps(artifact)))
                self._event(db, run_id, "candidate_artifact_attached", {"question_id": qid,
                    "source_revision_id": artifact["source_revision_id"], "source_run_id": artifact["source_run_id"],
                    "artifact_sha256": artifact["artifact_sha256"], "candidate_count": len(artifact["candidate_evidence"]),
                    "kind": "new_quality_refine", "new_source_scan_performed": False})
            db.execute("UPDATE topics SET head_revision_id=? WHERE id=?", (revision_id, topic_id))
            if preview:
                db.execute("UPDATE change_previews SET consumed_by_run_id=? WHERE id=?", (run_id, preview))
                db.execute("INSERT INTO changesets VALUES (?,?,?,?,?,?)", (str(uuid4()), preview, run_id, self.settings.local_actor, p["reason"], now()))
            self._event(db, run_id, "run_queued", {"revision_id": revision_id, "question_ids": qids})
            if request.get("resume_from_run_id"):
                self._event(db, run_id, "resume_queued", {"source_run_id": request["resume_from_run_id"], "resume_mode": options.get("resume_mode"), "checkpoint_question_ids": options.get("resume_checkpoint_question_ids", []), "billing": "completed checkpoint calls are not repeated; unfinished requests may already have been billed"})
        return self.run(run_id)

    def refine_input(self, run_id, question_id):
        rows = self.all("SELECT artifact_sha256,payload_json FROM run_refine_inputs WHERE run_id=? AND question_id=?", (run_id, question_id))
        if not rows:
            return None
        artifact = loads(rows[0]["payload_json"])
        if artifact.get("artifact_sha256") != rows[0]["artifact_sha256"]:
            raise ServiceError(409, "refine_artifact_hash", "候选工件身份不一致，尚未调用模型。")
        return artifact

    def claim(self):
        with self.connection() as db, transaction(db):
            row = db.execute("SELECT * FROM runs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE runs SET status='running' WHERE id=? AND status='queued'", (row["id"],))
            self._event(db, row["id"], "run_started", {})
            return {**dict(row), "status": "running", "options": loads(self.one("SELECT options_json FROM run_options WHERE run_id=?", (row["id"],), db=db)["options_json"])}

    def cancel_requested(self, run_id):
        row = self.one("SELECT cancel_requested,status FROM runs WHERE id=?", (run_id,))
        return bool(row["cancel_requested"]) or row["status"] != "running"

    def cancel(self, run_id):
        with self.connection() as db, transaction(db):
            row = self.one("SELECT * FROM runs WHERE id=?", (run_id,), db=db)
            if row["status"] in TERMINAL:
                return self.run(run_id)
            db.execute("UPDATE runs SET cancel_requested=1 WHERE id=?", (run_id,))
            self._event(db, run_id, "cancel_requested", {})
            if row["status"] == "queued":
                db.execute("UPDATE runs SET status='cancelled',completed_at=? WHERE id=?", (now(), run_id))
                db.execute("UPDATE graph_revisions SET state='cancelled',completed_at=? WHERE id=?", (now(), row["target_revision_id"]))
                self._event(db, run_id, "run_cancelled", {})
        return self.run(run_id)

    def pause(self, run_id):
        with self.connection() as db, transaction(db):
            row = self.one("SELECT * FROM runs WHERE id=?", (run_id,), db=db)
            if row["status"] in TERMINAL:
                return self.run(run_id)
            previous = loads(row["error_json"], {})
            if row["cancel_requested"] and previous.get("code") == "paused":
                return self.run(run_id)
            marker = {"code": "paused", "message": "暂停请求已保存；已完成节点保留，当前模型请求可能已经计费。"}
            db.execute("UPDATE runs SET cancel_requested=1,error_json=? WHERE id=?", (dumps(marker), run_id))
            self._event(db, run_id, "pause_requested", {"message": marker["message"]})
            if row["status"] == "queued":
                db.execute("UPDATE runs SET status='cancelled',completed_at=? WHERE id=?", (now(), run_id))
                db.execute("UPDATE graph_revisions SET state='cancelled',completed_at=? WHERE id=?", (now(), row["target_revision_id"]))
                self._event(db, run_id, "run_paused", {"message": "排队任务已暂停，尚未调用模型。"})
        return self.run(run_id)

    def record_usage(self, run_id, question_id, records):
        """Persist known counters on success, failure or cancellation; no bodies."""
        with self.connection() as db, transaction(db):
            run = self.one("SELECT topic_id,status,usage_json FROM runs WHERE id=?", (run_id,), db=db)
            if run["status"] in TERMINAL:
                return
            self.one("SELECT id FROM questions WHERE id=? AND topic_id=?", (question_id, run["topic_id"]), db=db)
            cleaned = []
            for record in records:
                counts = {key: value if type(value := record.get(key)) is int and value >= 0 else None for key in ("input_tokens", "output_tokens", "total_tokens")}
                cleaned.append({"operation": record.get("operation") if isinstance(record.get("operation"), str) and record["operation"] in {"extract", "select_evidence", "verify_evidence_scope", "narrow_evidence_annotations", "review_fact_purpose", "paragraph_queries", "paragraph_rank", "paragraph_annotate", "paragraph_fields", "strict_selection", "strict_annotation", "source_bundle_select", "authority_analyst", "cross_validation_analyst", "arbitrate", "baseline"} else "unknown", "model": self.settings.model_name or "explicit-test-provider", "usage_source": "provider" if record.get("usage_source") == "provider" else "unavailable", **counts, "cost": None})
            usage = loads(run["usage_json"], {})
            usage[question_id] = cleaned
            db.execute("UPDATE runs SET usage_json=? WHERE id=?", (dumps(usage), run_id))

    def finish_failed(self, run_id, code, message, *, cancelled=False, stage=None, details=None):
        if isinstance(code, str) and code in MODEL_TRANSPORT_CODES:
            failure = public_transport_failure(code, details)
            code, message, details = failure["code"], failure["message"], failure["details"]
            stage = stage if isinstance(stage, str) and stage in MODEL_REQUEST_STAGES else None
        with self.connection() as db, transaction(db):
            r = self.one("SELECT * FROM runs WHERE id=?", (run_id,), db=db)
            if r["status"] in TERMINAL:
                return
            status = "cancelled" if cancelled or r["cancel_requested"] else "failed"
            requested = loads(r["error_json"], {})
            if r["cancel_requested"] and requested.get("code") == "paused":
                code, message = "paused", "已暂停；完成的检查点保留，可继续执行。当前模型请求可能已经计费。"
            error = {"code": code, "message": message}
            if stage is not None:
                error["stage"] = stage
            if details:
                error["details"] = details
            db.execute("UPDATE runs SET status=?,completed_at=?,error_json=? WHERE id=?", (status, now(), dumps(error), run_id))
            db.execute("UPDATE graph_revisions SET state=?,completed_at=? WHERE id=?", (status, now(), r["target_revision_id"]))
            self._event(db, run_id, "run_paused" if code == "paused" else f"run_{status}", error)

    def recover(self):
        for row in self.all("SELECT id FROM runs WHERE status='running'"):
            self.finish_failed(row["id"], "worker_interrupted", "上次 worker 中断；检查点已保留，请显式重试。")
        with self.connection() as db:
            db.execute("UPDATE export_jobs SET status='failed',error_json=? WHERE status='running'", (dumps({"code": "worker_interrupted", "message": "导出中断，请使用新幂等键重新创建。"}),))

    def _validate_result_protocol(self, result, origin_run_id, topic_id, question_id, *, db=None):
        """A payload cannot downgrade the verification protocol of its run."""
        origin = self.one("SELECT topic_id,question_ids_json,model_config_json FROM runs WHERE id=?", (origin_run_id,), db=db)
        if origin["topic_id"] != topic_id or question_id not in loads(origin["question_ids_json"]):
            raise ValueError("question result origin does not belong to this topic/question")
        config = loads(origin["model_config_json"])
        from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL, SOURCE_CONTEXT_PROTOCOL, SOURCE_DOCUMENT_PROTOCOLS, validate_source_bundle_rows
        configured_bundle = config.get("source_bundle_protocol")
        declared_bundle = result.get("source_bundle_protocol")
        if configured_bundle is not None:
            context = config.get("source_bundle_config", {}).get("context_passages", False)
            expected_protocol = SOURCE_CONTEXT_PROTOCOL if context else SOURCE_BUNDLE_PROTOCOL
            if (type(context) is not bool or configured_bundle not in SOURCE_DOCUMENT_PROTOCOLS
                    or configured_bundle != expected_protocol or declared_bundle != configured_bundle
                    or any(config.get(key) is True for key in ("strict_evidence_enabled", "paragraph_ranked_enabled", "premise_bank_enabled"))
                    or any(key in result for key in ("strict_evidence_proof", "paragraph_ranked_proof", "quote_faithful_bank", "purpose_review"))):
                raise ValueError("source bundle protocol must match its independent trusted run configuration")
            validate_source_bundle_rows(result.get("evidence", []), protocol=declared_bundle,
                source_documents=self.source_documents(topic_id), source_paragraphs=self.paragraphs(topic_id))
            if config.get("source_bundle_enabled") is True:
                from app.workflows.bundle_workbench import PROMPT_VERSION, validate_result
                framework = config.get("source_bundle_config", {}).get("front_framework", "dsh")
                expected_prompt = "trusted-analysis-langchain-source-bundles-v1" if framework == "langchain" else PROMPT_VERSION
                expected_strategy = "langchain_source_book" if framework == "langchain" else "dsh_source_bundles"
                expected_front_proof = "langchain-source-bundle-workbench-proof-v1" if framework == "langchain" else "dsh-source-bundle-workbench-proof-v1"
                actual_front_proof = result.get("source_bundle_proof", {}).get("response", {}).get("proof", {}).get("format")
                if (framework not in {"dsh", "langchain"} or config.get("prompt_version") != expected_prompt
                        or config.get("source_strategy") != expected_strategy or actual_front_proof != expected_front_proof):
                    raise ValueError("source bundle run configuration does not match the native workflow")
                local_rank = result.get("source_bundle_proof", {}).get("response", {}).get("proof", {}).get("local_parent_rank")
                if (local_rank.get("config") if isinstance(local_rank, dict) else None) != config.get("source_bundle_config", {}).get("local_parent_rank"):
                    raise ValueError("source bundle local ranking does not match its frozen run configuration")
                selected_only = result.get("source_bundle_proof", {}).get("response", {}).get("proof", {}).get("selection_only", False)
                if type(selected_only) is not bool or selected_only != config.get("source_bundle_config", {}).get("selection_only", False):
                    raise ValueError("source bundle selection mode does not match its frozen run configuration")
                front_proof = result.get("source_bundle_proof", {}).get("response", {}).get("proof", {})
                if (front_proof.get("context_passages", False) is not context
                        or context and (framework != "langchain" or front_proof.get("source_bundle_protocol") != configured_bundle)
                        or "source_bundle_protocol" in front_proof and front_proof["source_bundle_protocol"] != configured_bundle):
                    raise ValueError("source context geometry does not match its frozen run configuration")
                complete = front_proof.get("complete_grading", False)
                expected_complete = config.get("source_bundle_config", {}).get("complete_grading", False)
                complete_policy = config.get("source_bundle_config", {}).get("complete_grading_policy")
                if (type(complete) is not bool or type(expected_complete) is not bool or complete != expected_complete
                        or complete and (not context or framework != "langchain" or not isinstance(complete_policy, dict)
                            or not complete_policy or front_proof.get("complete_grading_policy") != complete_policy)
                        or not complete and "complete_grading_policy" in front_proof):
                    raise ValueError("source bundle complete grading does not match its frozen run configuration")
                hybrid = front_proof.get("hybrid_retrieval", False)
                expected_hybrid = config.get("source_bundle_config", {}).get("hybrid_retrieval", False)
                hybrid_policy = config.get("source_bundle_config", {}).get("hybrid_retrieval_policy")
                if (type(hybrid) is not bool or type(expected_hybrid) is not bool or hybrid != expected_hybrid
                        or hybrid and (not context or framework != "langchain" or not isinstance(hybrid_policy, dict)
                            or not hybrid_policy or front_proof.get("hybrid_retrieval_policy") != hybrid_policy)
                        or not hybrid and "hybrid_retrieval_policy" in front_proof):
                    raise ValueError("source bundle hybrid retrieval does not match its frozen run configuration")
                paragraph = front_proof.get("paragraph_quality", False)
                expected_paragraph = config.get("source_bundle_config", {}).get("paragraph_quality", False)
                paragraph_policy = config.get("source_bundle_config", {}).get("paragraph_quality_policy")
                if (type(paragraph) is not bool or type(expected_paragraph) is not bool or paragraph != expected_paragraph
                        or paragraph and (context or selected_only or complete or hybrid or framework != "langchain"
                            or not isinstance(paragraph_policy, dict) or not paragraph_policy
                            or front_proof.get("paragraph_quality_policy") != paragraph_policy
                            or front_proof.get("selection_output_tokens") != 8192
                            or config.get("source_bundle_config", {}).get("selection_output_tokens") != 8192)
                        or not paragraph and "paragraph_quality_policy" in front_proof):
                    raise ValueError("source bundle paragraph quality does not match its frozen run configuration")
                paragraph_compact = front_proof.get("paragraph_quality_compact", False)
                frozen_bundle = config.get("source_bundle_config", {})
                expected_paragraph_compact = frozen_bundle.get("paragraph_quality_compact", False)
                if (type(paragraph_compact) is not bool or type(expected_paragraph_compact) is not bool
                        or paragraph_compact != expected_paragraph_compact
                        or paragraph_compact and (not paragraph or framework != "langchain"
                            or not isinstance(frozen_bundle.get("paragraph_quality_compact_policy"), dict)
                            or front_proof.get("paragraph_quality_compact_policy") != frozen_bundle["paragraph_quality_compact_policy"])
                        or not paragraph_compact and "paragraph_quality_compact_policy" in front_proof):
                    raise ValueError("source bundle compact paragraph quality does not match its frozen run configuration")
                parent_pack = front_proof.get("parent_token_pack", False)
                expected_parent_pack = frozen_bundle.get("parent_token_pack", False)
                if (type(parent_pack) is not bool or type(expected_parent_pack) is not bool or parent_pack != expected_parent_pack
                        or parent_pack and (not paragraph_compact or local_rank or framework != "langchain"
                            or not isinstance(frozen_bundle.get("parent_token_pack_policy"), dict)
                            or front_proof.get("parent_token_pack_policy") != frozen_bundle["parent_token_pack_policy"])
                        or not parent_pack and "parent_token_pack_policy" in front_proof):
                    raise ValueError("source bundle parent token pack does not match its frozen run configuration")
                graded = front_proof.get("graded_selection", False)
                frozen_bundle = config.get("source_bundle_config", {})
                expected_graded = frozen_bundle.get("graded_selection", False)
                if type(graded) is not bool or type(expected_graded) is not bool or graded != expected_graded:
                    raise ValueError("source bundle graded selection does not match its frozen run configuration")
                if graded and (selected_only is not True or framework != "langchain"
                        or front_proof.get("graded_selection_policy") != frozen_bundle.get("graded_selection_policy")
                        or front_proof.get("selection_output_tokens") != 2048 or frozen_bundle.get("selection_output_tokens") != 2048):
                    raise ValueError("source bundle graded policy does not match its frozen run configuration")
                compact_analyses = front_proof.get("compact_analyses", False)
                expected_compact = frozen_bundle.get("compact_analyses", False)
                if (type(compact_analyses) is not bool or type(expected_compact) is not bool
                        or compact_analyses != expected_compact or compact_analyses and framework != "langchain"):
                    raise ValueError("source bundle analysis projection does not match its frozen run configuration")
                if front_proof.get("whole_chain_planning") != frozen_bundle.get("whole_chain_planning"):
                    raise ValueError("source bundle whole-chain planning does not match its frozen run configuration")
                issue = front_proof.get("issue_frame", False)
                expected_issue = frozen_bundle.get("issue_frame", False)
                if (type(issue) is not bool or type(expected_issue) is not bool or issue != expected_issue
                        or issue and framework != "langchain"):
                    raise ValueError("source bundle issue frame does not match its frozen run configuration")
                if (result.get("evidence_relation_scope", {}).get("format") in {"issue-frame-evidence-relations-v2", "conclusion-aligned-evidence-relations-v3"}) != issue:
                    raise ValueError("source bundle issue frame requires directional final roles")
                question = self.one("SELECT id,raw_question_id,question FROM questions WHERE id=? AND topic_id=?", (question_id, topic_id), db=db)
                if issue:
                    from app.agents.scoped_arbitration import validate_scoped_result
                    validate_scoped_result(result["evidence_relation_scope"], result["final"], result["evidence"], question=question)
                validate_result(result, question, self.paragraphs(topic_id), self.source_documents(topic_id))
            elif "source_bundle_proof" in result:
                raise ValueError("source-only publication cannot introduce native workflow proof")
            return
        if declared_bundle is not None or any(row.get("span_scope") == "document" for row in result.get("evidence", [])):
            raise ValueError("legacy run cannot introduce source bundle protocol")
        from app.workflows.strict_workbench import VERSION as strict_version, BALANCED_VERSION, PROMPT_VERSION as strict_prompt, validate_result
        has_strict = result.get("verification_protocol") in {strict_version, BALANCED_VERSION} or "strict_evidence_proof" in result
        if config.get("strict_evidence_enabled") is True:
            strict_config = config.get("strict_evidence_config", {})
            balanced = strict_config.get("balanced_scoped_enabled") is True
            expected_strategy = "strict_balanced_scoped" if balanced else "strict_bm25_full_parents"
            if (not has_strict or config.get("premise_bank_enabled") or config.get("paragraph_ranked_enabled")
                    or config.get("prompt_version") != strict_prompt or config.get("source_strategy") != expected_strategy):
                raise ValueError("strict protocol required by trusted run configuration")
            if balanced:
                proof = result.get("strict_evidence_proof", {})
                if (result.get("verification_protocol") != BALANCED_VERSION
                        or proof.get("balance_policy") != strict_config.get("balance_policy")
                        or proof.get("recall_strategy") != strict_config.get("recall_strategy")
                        or strict_config.get("balance_policy") != {"target": 10, "per_doc_soft_cap": 2}
                        or strict_config.get("recall_strategy") != "balanced"):
                    raise ValueError("balanced protocol cannot be downgraded or change its trusted policy")
                from app.agents.scoped_arbitration import validate_scoped_result
                validate_scoped_result(result.get("evidence_relation_scope"), result["final"], result["evidence"])
            elif result.get("verification_protocol") != strict_version or result.get("evidence_relation_scope") is not None:
                raise ValueError("legacy strict origin cannot introduce balanced protocol")
            question = self.one("SELECT id,raw_question_id,question FROM questions WHERE id=? AND topic_id=?", (question_id, topic_id), db=db)
            options = loads(self.one("SELECT options_json FROM run_options WHERE run_id=?", (origin_run_id,), db=db)["options_json"], {})
            validate_result(result, question, self.paragraphs(topic_id),
                candidate_limit=config["strict_evidence_config"]["candidate_limit"],
                excluded_spans={tuple(row) for row in options.get("overlays", {}).get("excluded", [])})
            return
        if has_strict:
            raise ValueError("legacy run cannot introduce strict evidence protocol")
        from app.domain.paragraph_ranked import VERSION as ranked_version, validate_ranked_result
        has_ranked = (result.get("verification_protocol") == ranked_version or "paragraph_ranked_proof" in result
            or any(isinstance(row, dict) and (row.get("paragraph_fields_verified") is True
                or row.get("span_location_method") == "program_whole_original_paragraph") for row in result.get("evidence", []))
            or result.get("source_coverage", {}).get("strategy") == "ranked_paragraph_subset")
        if config.get("paragraph_ranked_enabled") is True:
            from app.domain.paragraph_ranked import PROMPT_VERSION
            if (config.get("premise_bank_enabled") is True or not has_ranked or not isinstance(config.get("paragraph_ranked_config"), dict)
                    or config.get("prompt_version") != PROMPT_VERSION or config.get("source_strategy") != "paragraph_ranked_v19"):
                raise ValueError("paragraph protocol required by trusted run configuration")
            options = loads(self.one("SELECT options_json FROM run_options WHERE run_id=?", (origin_run_id,), db=db)["options_json"], {})
            question = self.one("SELECT id,raw_question_id,question FROM questions WHERE id=? AND topic_id=?", (question_id, topic_id), db=db)
            # The original workflow projection includes topic_id and raw IDs.
            question["topic_id"] = topic_id
            validate_ranked_result(result, question, self.paragraphs(topic_id), expected_config=config.get("paragraph_ranked_config"),
                excluded_spans={tuple(row) for row in options.get("overlays", {}).get("excluded", [])})
            return
        if has_ranked:
            raise ValueError("legacy run cannot introduce paragraph-ranked protocol")
        if config.get("premise_bank_enabled") is True:
            from app.domain.premise_bank import VERSION
            if (result.get("verification_protocol") != VERSION
                    or not all(isinstance(result.get(key), dict) for key in ("quote_faithful_bank", "purpose_review", "material_eligibility"))):
                raise ValueError("premise protocol required by trusted run configuration")

    def _validate_document_source(self, evidence, origin_run_id, source_documents, source_paragraphs, *, db=None):
        """Authorize source geometry with the immutable origin run, not the row."""
        from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL, SOURCE_CONTEXT_PROTOCOL, validate_source_bundle_rows
        origin = self.one("SELECT model_config_json FROM runs WHERE id=?", (origin_run_id,), db=db)
        config = loads(origin["model_config_json"])
        context = config.get("source_bundle_config", {}).get("context_passages", False)
        protocol = config.get("source_bundle_protocol")
        if (type(context) is not bool or protocol != (SOURCE_CONTEXT_PROTOCOL if context else SOURCE_BUNDLE_PROTOCOL)):
            raise ValueError("document source geometry lacks trusted origin protocol")
        validate_source_bundle_rows([evidence], protocol=protocol,
            source_documents=source_documents, source_paragraphs=source_paragraphs)

    @staticmethod
    def _validate_snapshot_evidence_projection(snapshot):
        expected = {}
        core = ("id", "paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote",
                "fact_des", "argument", "entities", "source_group_id", "source_group_basis")
        from app.domain.source_passages import SOURCE_PASSAGE_CORE_FIELDS
        core = (*core, *SOURCE_PASSAGE_CORE_FIELDS)
        for result in snapshot["question_results"].values():
            for row in result["evidence"]:
                peers = expected.setdefault(row["id"], [])
                if peers and any(row.get(key) != peers[0].get(key) for key in core):
                    raise ValueError("shared evidence identity has inconsistent source/annotation")
                peers.append(row)
        actual = {row["id"]: row for row in snapshot["evidence"]}
        # The builder chooses the first authorized row, but canonical JSON
        # sorts question keys. Do not guess that original order after reading
        # the artifact: the full global row must equal one authorized local
        # projection, while every local permission is independently replayed.
        if (len(actual) != len(snapshot["evidence"]) or set(actual) != set(expected)
                or any(row not in expected[identity] for identity, row in actual.items())):
            raise ValueError("snapshot evidence differs from authorized question projection")

    def _validate_inherited_results(self, run, snapshot, *, db=None):
        """Inherited payloads must be identical to an immutable ready ancestor."""
        fresh = set(loads(run["question_ids_json"]))
        inherited = {qid: row for qid, row in snapshot["question_results"].items() if qid not in fresh}
        if not inherited:
            return
        revision_id, seen, prior = run["base_revision_id"], set(), {}
        while revision_id and revision_id not in seen:
            seen.add(revision_id)
            revision = self.one("SELECT topic_id,state,parent_revision_id FROM graph_revisions WHERE id=?", (revision_id,), db=db)
            if revision["topic_id"] != run["topic_id"]:
                raise ValueError("inherited revision belongs to another topic")
            if revision["state"] == "ready":
                artifact = self.one("SELECT payload_json FROM revision_artifacts WHERE revision_id=?", (revision_id,), db=db)
                prior = loads(artifact["payload_json"])["question_results"]
                break
            revision_id = revision["parent_revision_id"]
        if any(prior.get(qid) != row for qid, row in inherited.items()):
            raise ValueError("snapshot changed an inherited immutable question result")

    def _request_result_validator(self, topic_id, *, db=None):
        """Deduplicate successful full replays only inside one caller request.

        The closure is never stored on Repository or shared across revisions.
        Authorization origin and exact result bytes are both part of the key;
        unsuccessful checks are never cached.
        """
        validated = set()
        def validate_once(result, origin_run_id, question_id):
            key = (origin_run_id, question_id, digest(result))
            if key not in validated:
                self._validate_result_protocol(result, origin_run_id, topic_id, question_id, db=db)
                validated.add(key)
        return validate_once

    def publish(self, run_id, snapshot, fresh_results):
        from app.domain.snapshot import competition_files
        validate_graph(snapshot["nodes"], snapshot["edges"])
        source_documents = self.source_documents(snapshot["topic_id"]) if any(
            result.get("source_bundle_protocol") is not None for result in snapshot["question_results"].values()) else None
        competition_files(snapshot, source_paragraphs=self.paragraphs(snapshot["topic_id"]), source_documents=source_documents)
        self._validate_snapshot_evidence_projection(snapshot)
        with self.connection() as db, transaction(db):
            run = self.one("SELECT * FROM runs WHERE id=?", (run_id,), db=db)
            rid, tid = run["target_revision_id"], run["topic_id"]
            head = self.one("SELECT head_revision_id FROM topics WHERE id=?", (tid,), db=db)["head_revision_id"]
            if run["status"] != "running" or run["cancel_requested"] or head != rid:
                raise ServiceError(409, "publish_rejected", "任务已取消或版本已变化，拒绝发布迟到结果。")
            if any(snapshot[k] != value for k, value in (("topic_id", tid), ("revision_id", rid), ("run_id", run_id))):
                raise ValueError("snapshot does not belong to this run/revision/topic")
            known_questions = {r[0] for r in db.execute("SELECT id FROM questions WHERE topic_id=?", (tid,))}
            if set(fresh_results) != set(loads(run["question_ids_json"])) or not set(snapshot["question_results"]) <= known_questions:
                raise ValueError("snapshot question scope does not match run")
            if any(snapshot["question_results"].get(qid) != result for qid, result in fresh_results.items()):
                raise ValueError("snapshot changed a fresh question result")
            self._validate_inherited_results(run, snapshot, db=db)
            if any(n.get("question_id") and n["question_id"] not in known_questions for n in snapshot["nodes"]):
                raise ValueError("foreign question in business graph")
            if self.one("SELECT state FROM graph_revisions WHERE id=?", (rid,), db=db)["state"] != "building":
                raise ValueError("publication requires a building revision")
            from app.domain.program_inventory import validate_published_inventory
            from app.domain.material_eligibility import validate_workflow_material_eligibility
            excluded = {tuple(row) for row in snapshot.get("overlays", {}).get("excluded", [])}
            current_paragraphs = self.paragraphs(tid)
            validate_result_once = self._request_result_validator(tid, db=db)
            for question_id, result in fresh_results.items():
                if result.get("verification_protocol") != "paragraph-ranked-source-v19":
                    validate_published_inventory(result, current_paragraphs, excluded_spans=excluded)
                validate_result_once(result, run_id, question_id)
            for question_id, result in snapshot["question_results"].items():
                validate_result_once(result, result.get("origin_run_id", run_id), question_id)
                source_question = self.one("SELECT id,raw_question_id,question FROM questions WHERE id=? AND topic_id=?", (question_id, tid), db=db)
                validate_workflow_material_eligibility(result, source_question, current_paragraphs, source_documents=source_documents)
            origin_by_evidence = {e["id"]: result.get("origin_run_id", run_id) for result in snapshot["question_results"].values() for e in result["evidence"]}
            for e in snapshot["evidence"]:
                p = self.one("SELECT p.*,d.topic_id,d.raw_doc_id FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE p.id=?", (e["paragraph_id"],), db=db)
                if p["topic_id"] != tid or e["raw_doc_id"] != p["raw_doc_id"] or e["raw_para_id"] != p["raw_para_id"]:
                    raise ValueError("evidence source identity mismatch")
                if e.get("span_scope") == "document":
                    self._validate_document_source(e, origin_by_evidence[e["id"]], source_documents, current_paragraphs, db=db)
                else:
                    validate_span(p["text"], e["quote"], e["start_cp"], e["end_cp"])
                values = (e["id"], tid, e["paragraph_id"], e["start_cp"], e["end_cp"], e["quote"], hashlib.sha256(e["quote"].encode()).hexdigest(), now())
                old = db.execute("SELECT paragraph_id,start_cp,end_cp,quote FROM evidence WHERE id=?", (e["id"],)).fetchone()
                if old and tuple(old) != (e["paragraph_id"], e["start_cp"], e["end_cp"], e["quote"]):
                    raise ValueError("immutable evidence identity collision")
                if old:
                    prior = db.execute("SELECT content_json FROM node_states WHERE node_id=? LIMIT 1", (e["id"],)).fetchone()
                    if prior:
                        from app.domain.source_passages import SOURCE_PASSAGE_CORE_FIELDS
                        original_source = loads(prior["content_json"])
                        bundle_scope = original_source.get("span_scope") == "document" or e.get("span_scope") == "document"
                        if bundle_scope and any(original_source.get(key) != e.get(key) for key in SOURCE_PASSAGE_CORE_FIELDS):
                            raise ValueError("immutable evidence source scope collision")
                db.execute("INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?,?,?,?)", values)
            for n in snapshot["nodes"]:
                identity = (n["id"], tid, n.get("question_id"), n.get("evidence_id"), n["kind"])
                old = db.execute("SELECT id,topic_id,question_id,evidence_id,kind FROM node_identities WHERE id=?", (n["id"],)).fetchone()
                if old and tuple(old) != identity:
                    raise ValueError("immutable node identity collision")
                db.execute("INSERT OR IGNORE INTO node_identities VALUES (?,?,?,?,?)", identity)
                origin_run = snapshot["question_results"][n["question_id"]].get("origin_run_id", run_id) if n.get("question_id") else origin_by_evidence[n["evidence_id"]]
                owner = snapshot["question_results"][n["question_id"]] if n.get("question_id") else next(result for result in snapshot["question_results"].values() if any(item["id"] == n["evidence_id"] for item in result["evidence"]))
                continuation = owner.get("execution_provenance", {})
                reused = ("freeze" if n["kind"] == "evidence" else "arbitrate") in continuation.get("reused_nodes", [])
                if reused and continuation.get("source_thread_id"):
                    origin_run = continuation["source_thread_id"].rsplit(":", 1)[0]
                provenance = {"run_id": origin_run, "origin": "workflow", "reused": origin_run != run_id}
                if continuation:
                    provenance["execution_provenance"] = continuation
                db.execute("INSERT INTO node_states(revision_id,node_id,status,authority,locked,content_json,provenance_json,node_version) VALUES (?,?,?,?,?,?,?,?)", (rid, n["id"], n["status"], n["authority"], int(n["locked"]), dumps(n["content"]), dumps(provenance), n["node_version"]))
            for e in snapshot["edges"]:
                db.execute("INSERT INTO graph_edges VALUES (?,?,?,?,?)", (rid, e["from_node_id"], e["to_node_id"], e["relation"], dumps(e.get("basis", {}))))
            groups = {}
            for e in snapshot["evidence"]:
                groups.setdefault(e["source_group_id"], e.get("source_group_basis", {}))
            for gid, basis in groups.items():
                db.execute("INSERT INTO source_groups VALUES (?,?,?)", (rid, gid, dumps(basis)))
            for e in snapshot["evidence"]:
                db.execute("INSERT INTO source_group_members VALUES (?,?,?)", (rid, e["source_group_id"], e["id"]))
            for qid, result in fresh_results.items():
                sid = result["evidence_snapshot_id"]
                frozen = {"question_id": qid, "evidence": result["evidence"]}
                db.execute("INSERT INTO evidence_snapshots VALUES (?,?,?,?,?)", (sid, run_id, digest(frozen), dumps(frozen), now()))
                outputs = [(x["role"], x) for x in result["analyses"]] + [("arbitrate", result["final"])]
                if result.get("baseline"):
                    outputs.append(("baseline", result["baseline"]))
                for role, output in outputs:
                    input_payload = {"question": self.one("SELECT id,raw_question_id,question FROM questions WHERE id=?", (qid,), db=db), "evidence": result["evidence"], "evidence_snapshot_id": sid}
                    if role == "arbitrate":
                        input_payload["analyses"] = result["analyses"]
                    db.execute("""INSERT INTO agent_executions(id,run_id,question_id,role,round,evidence_snapshot_id,input_sha256,prompt_version,model_name,status,input_json,output_json,usage_json)
                      VALUES (?,?,?,?,0,?,?,?,?,'succeeded',?,?,?)""", (str(uuid4()), run_id, qid, role, sid, digest(input_payload), self.config()["prompt_version"], self.settings.model_name or "explicit-test-provider", dumps(input_payload), dumps(output), dumps([u for u in result.get("usage", []) if u.get("operation") == role])))
            for quality in snapshot["confidence"]:
                ref = quality["object_ref"]
                if quality["object_type"] == "conclusion":
                    node_id = next(c["conclusion_node_id"] for c in snapshot["conclusions"] if c["raw_question_id"] == ref["question_id"])
                else:
                    node_id = next(e["id"] for e in snapshot["evidence"] if (e["raw_doc_id"],e["raw_para_id"],e["quote"]) == (ref["doc_id"],ref["para_id"],ref["quote"]))
                score = quality["score"]
                db.execute("INSERT OR IGNORE INTO confidence_scores VALUES (?,?,?,?,?,?,?,?,?,?)", (rid, node_id, "quality_index", score, "high" if score >= .8 else "medium" if score >= .5 else "low", quality["formula_version"], quality["parameter_version"], dumps(quality["components"]), dumps(quality["inputs"]), dumps(quality["calculation"])))
            for c in snapshot["conflicts"]:
                status = {"adopt_a": "resolved", "adopt_b": "resolved", "coexist": "coexisting", "unresolved": "unresolved"}[c["result"]]
                db.execute("INSERT INTO conflicts VALUES (?,?,?,?,?)", (rid, c["id"], c["question_id"], status, dumps(c)))
            # confidence export schema owns reproducible object references; retain the
            # exact projection in immutable artifacts rather than inventing node IDs.
            db.execute("INSERT INTO revision_artifacts VALUES (?,?)", (rid, dumps(snapshot)))
            if db.execute("UPDATE graph_revisions SET state='ready',completed_at=? WHERE id=? AND state='building'", (now(), rid)).rowcount != 1:
                raise ValueError("revision publication compare-and-swap failed")
            if db.execute("UPDATE topics SET published_revision_id=? WHERE id=? AND head_revision_id=?", (rid, tid, rid)).rowcount != 1:
                raise ValueError("topic publication compare-and-swap failed")
            db.execute("UPDATE runs SET status='succeeded',completed_at=?,usage_json=? WHERE id=?", (now(), dumps({qid: r.get("usage", []) for qid, r in fresh_results.items()}), run_id))
            self._event(db, run_id, "run_succeeded", {"revision_id": rid, "evidence_count": len(snapshot["evidence"]), "conclusion_count": len(snapshot["conclusions"])})

    def preview_change(self, topic_id, base, operations, reason):
        if self.topic(topic_id)["head_revision_id"] != base:
            raise ServiceError(409, "stale_base", "请基于专题最新版本预览。")
        snapshot = self.snapshot(topic_id, base)
        evidence = {e["id"]: e for e in snapshot["evidence"]}
        evidence.update({e["id"]: e for e in snapshot.get("overlays", {}).get("known_evidence", [])})
        changed, all_questions = set(), False
        for op in operations:
            if op["op"] in {"exclude_evidence", "include_evidence", "set_interpretation"}:
                if op["evidence_id"] not in evidence:
                    raise ServiceError(422, "foreign_evidence", "引用不在当前版本中。")
                if evidence[op["evidence_id"]].get("span_scope") == "document":
                    self._document_overlay_spans(snapshot, evidence[op["evidence_id"]])
                changed.add(op["evidence_id"])
            elif op["op"] == "add_evidence":
                all_questions = True
                for span in op["source_spans"]:
                    p = self.paragraph(span["paragraph_id"])
                    if p["topic_id"] != topic_id or not 0 <= span["start"] < span["end"] <= len(p["text"]):
                        raise ServiceError(422, "invalid_span", "人工引用须是当前专题原文的有效码点范围。")
            else:
                raise ServiceError(501, "review_not_implemented", "首版支持引用增补、停用和解释修订；人工锁定判断与冲突复核将在下一阶段实现。")
        affected = downstream_closure(snapshot["edges"], changed)
        qids = {n["question_id"] for n in snapshot["nodes"] if n["id"] in affected and n.get("question_id")}
        # A frozen candidate may be unused in today's graph yet affect the next
        # independent judgment. Invalidate every question that received it.
        qids.update(qid for qid,result in snapshot["question_results"].items() if changed.intersection(e["id"] for e in result["evidence"]))
        if all_questions or not qids:
            qids = set(snapshot["question_results"])
        impact = {"affected_node_ids": sorted(affected), "affected_question_ids": sorted(qids), "must_rerun_agents": ["authority_analyst", "cross_validation_analyst", "arbitrate"], "scope_basis": "全量重审所影响问题，保留未受影响问题"}
        pid, h = str(uuid4()), digest({"base_revision_id": base, "operations": operations, "config": self.config()})
        with self.connection() as db:
            db.execute("INSERT INTO change_previews VALUES (?,?,?,?,?,?,?,?,?,NULL)", (pid, topic_id, base, dumps(operations), h, digest(self.config()), dumps(impact), reason, (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()))
        return {"preview_id": pid, "content_hash": h, **impact}

    def _document_overlay_spans(self, snapshot, evidence):
        """Translate authorized document CP to exact original-parent overlays."""
        origins = {e["id"]: result.get("origin_run_id", snapshot["run_id"])
            for result in snapshot["question_results"].values() for e in result["evidence"]}
        try:
            parents = self.paragraphs(snapshot["topic_id"])
            self._validate_document_source(evidence, origins.get(evidence["id"], snapshot["run_id"]),
                self.source_documents(snapshot["topic_id"]), parents)
            by_id = {row["id"]: row for row in parents}
            spans = []
            for pid in evidence["source_paragraph_ids"]:
                parent = by_id[pid]
                start = max(evidence["start_cp"], parent["document_start_cp"]) - parent["document_start_cp"]
                end = min(evidence["end_cp"], parent["document_end_cp"]) - parent["document_start_cp"]
                if start < end:
                    spans.append((pid, start, end, parent["text"][start:end]))
            if not spans:
                raise ValueError("document overlay requires a nonempty original slice")
            return spans
        except (KeyError, TypeError, ValueError):
            raise ServiceError(409, "source_context_integrity_failed", "引用的原文几何或来源任务配置未通过校验，未修改人工标注。") from None

    def commit_change(self, topic_id, body, key, *, model_available):
        # Reconstruct only from the saved preview; clients cannot swap operations.
        p = self.one("SELECT * FROM change_previews WHERE id=? AND topic_id=?", (body["preview_id"], topic_id))
        if p["consumed_by_run_id"]:
            old = self.one("SELECT idempotency_key FROM runs WHERE id=?", (p["consumed_by_run_id"],))
            if old["idempotency_key"] != key:
                raise ServiceError(409, "preview_consumed", "该预览已提交。")
        base_snapshot = self.snapshot(topic_id, p["base_revision_id"])
        overlays = loads(dumps(base_snapshot.get("overlays", {"excluded": [], "manual_evidence": [], "interpretation_overrides": {}})))
        ev = {e["id"]: e for e in base_snapshot["evidence"]}
        ev.update({e["id"]: e for e in overlays.get("known_evidence", [])})
        overlays["known_evidence"] = list(ev.values())
        excluded = {tuple(x) for x in overlays["excluded"]}
        for op in loads(p["operations_json"]):
            kind = op["op"]
            if kind in {"exclude_evidence", "include_evidence"}:
                e = ev[op["evidence_id"]]
                spans = (self._document_overlay_spans(base_snapshot, e) if e.get("span_scope") == "document" else
                         [(e["paragraph_id"], e["start_cp"], e["end_cp"], e["quote"])])
                for span in spans:
                    excluded.add(span) if kind == "exclude_evidence" else excluded.discard(span)
                if kind == "include_evidence":
                    overlays["manual_evidence"].append(e)
            elif kind == "set_interpretation":
                e = ev[op["evidence_id"]]
                if e.get("span_scope") == "document":
                    self._document_overlay_spans(base_snapshot, e)
                overlays["manual_evidence"].append(e)
                overlays["interpretation_overrides"][e["id"]] = self._interpretation(op["interpretation"])
            elif kind == "add_evidence":
                span = op["source_spans"][0]
                para = self.paragraph(span["paragraph_id"])
                overlays["manual_evidence"].append({"paragraph_id": para["id"], "start_cp": span["start"], "end_cp": span["end"], "quote": para["text"][span["start"]:span["end"]], **self._interpretation(op["interpretation"])})
        overlays["excluded"] = [list(x) for x in sorted(excluded)]
        request = {**body, "question_ids": loads(p["impact_json"])["affected_question_ids"], "mode": "recompute", "include_baseline": False}
        return self.create_run(topic_id, request, key, model_available=model_available, options={"overlays": overlays}, preview=p["id"])

    @staticmethod
    def _interpretation(value):
        args = value.get("argument", {})
        names = {"time": "时间", "subject": "主体", "location": "地点", "content": "内容", "impact": "影响", "numeric_scope": "数值口径"}
        return {"fact_des": value["fact_description"], "stance": value.get("stance", "unclassified"), "argument": {names.get(k,k): v for k,v in args.items()}, "entities": value.get("entities", [])}

    def submission_projection(self, topic_id, revision_id):
        """Recheck the immutable revision against raw sources before any download.

        A partial revision is a valid preview, but never a formal submission.
        This method does not combine results from other revisions or call a model.
        """
        snapshot = self.snapshot(topic_id, revision_id)
        questions = self.questions(topic_id)
        by_question = {question["id"]: question for question in questions}
        try:
            if snapshot["topic_id"] != topic_id or snapshot["revision_id"] != revision_id:
                raise ValueError("snapshot identity mismatch")
            owner = self.one("SELECT * FROM runs WHERE target_revision_id=? AND topic_id=?", (revision_id, topic_id))
            if snapshot["run_id"] != owner["id"]:
                raise ValueError("snapshot owner mismatch")
            self._validate_inherited_results(owner, snapshot)
            completed = [row["question_id"] for row in snapshot["conclusions"]]
            if (len(set(completed)) != len(completed)
                    or set(completed) != set(snapshot["question_results"])
                    or not set(completed) <= set(by_question)):
                raise ValueError("snapshot question coverage mismatch")
            for conclusion in snapshot["conclusions"]:
                question = by_question[conclusion["question_id"]]
                if (conclusion["raw_question_id"] != question["raw_question_id"]
                        or conclusion["question"] != question["question"]):
                    raise ValueError("raw question identity mismatch")
            from app.domain.material_eligibility import validate_workflow_material_eligibility
            current_paragraphs = self.paragraphs(topic_id)
            source_documents = self.source_documents(topic_id) if any(
                result.get("source_bundle_protocol") is not None for result in snapshot["question_results"].values()) else None
            self._validate_snapshot_evidence_projection(snapshot)
            validate_result_once = self._request_result_validator(topic_id)
            for question_id, result in snapshot["question_results"].items():
                if question_id in loads(owner["question_ids_json"]):
                    validate_result_once(result, owner["id"], question_id)
                validate_result_once(result, result.get("origin_run_id", snapshot["run_id"]), question_id)
                validate_workflow_material_eligibility(result, by_question[question_id], current_paragraphs, source_documents=source_documents)
            scope = snapshot.get("scope", {})
            complete = set(completed) == set(by_question)
            if (set(scope.get("completed_question_ids", [])) != set(completed)
                    or scope.get("all_question_count") != len(questions)
                    or scope.get("is_complete_topic") is not complete):
                raise ValueError("snapshot declared scope mismatch")
            validate_graph(snapshot["nodes"], snapshot["edges"])
            for row in snapshot["evidence"]:
                paragraph = self.paragraph(row["paragraph_id"])
                if (paragraph["topic_id"] != topic_id
                        or row["raw_doc_id"] != paragraph["raw_doc_id"]
                        or row["raw_para_id"] != paragraph["raw_para_id"]):
                    raise ValueError("evidence source identity mismatch")
                if row.get("span_scope") == "document":
                    origins = {e["id"]: result.get("origin_run_id", snapshot["run_id"])
                        for result in snapshot["question_results"].values() for e in result["evidence"]}
                    self._validate_document_source(row, origins[row["id"]], source_documents, current_paragraphs)
                else:
                    validate_span(paragraph["text"], row["quote"], row["start_cp"], row["end_cp"])
            files = competition_files(snapshot, source_paragraphs=current_paragraphs, source_documents=source_documents)
        except (KeyError, TypeError, ValueError, ServiceError):
            # No source excerpts or arbitrary internal exception messages are
            # exposed by the API. The ready revision is still preserved.
            raise ServiceError(409, "export_validation_failed", "该版本未通过原文、问题覆盖、引用或证据图完整性校验，无法导出。") from None
        return snapshot, files, questions

    def export_readiness(self, topic_id, revision_id):
        topic = self.topic(topic_id)
        snapshot = self.snapshot(topic_id, revision_id)
        questions = self.questions(topic_id)
        completed = {row.get("question_id") for row in snapshot.get("conclusions", [])}
        missing = [question for question in questions if question["id"] not in completed]
        blockers = []
        if any(row.get("manual_review_required") is True or (row.get("interpretation_origin") == "human" and row.get("scope_verified") is False) for row in snapshot.get("evidence", [])):
            blockers.append({"code": "export_manual_fact_unverified", "message": "存在尚未通过事实复核的人工解释；原意已保留，请修正后重新复核再导出正式提交包。"})
        try:
            self.submission_projection(topic_id, revision_id)
        except ServiceError as error:
            blockers.append({"code": error.code, "message": error.message})
        if missing:
            blockers.append({"code": "export_incomplete_topic", "message": f"该版本已完成 {len(questions) - len(missing)}/{len(questions)} 个问题；正式提交包须覆盖专题全部问题。"})
        existing = None
        if not blockers:
            # A result from another revision must never appear as this revision's
            # download, even when it is the newest successful job of the topic.
            for row in self.all("SELECT id,status,artifact_path FROM export_jobs WHERE topic_id=? AND revision_id=? AND status IN ('queued','running','succeeded') ORDER BY created_at DESC,id DESC", (topic_id, revision_id)):
                if row["status"] in {"queued", "running"}:
                    existing = self.export(row["id"])
                    break
                path = Path(row["artifact_path"] or "")
                if path.is_file() and path.resolve().is_relative_to(self.settings.export_dir.resolve()):
                    existing = self.export(row["id"])
                    break
        return {"topic_id": topic_id, "revision_id": revision_id,
                "published_revision_id": topic["published_revision_id"],
                "total_questions": len(questions), "completed_questions": len(questions) - len(missing),
                "missing_questions": missing, "can_export": not blockers, "blockers": blockers,
                "existing_export": existing,
                "preview_url": f"/api/topics/{topic_id}/revisions/{revision_id}/submission-preview"}

    def submission_preview(self, topic_id, revision_id):
        snapshot, files, questions = self.submission_projection(topic_id, revision_id)
        completed = set(snapshot["question_results"])
        missing = [question["raw_question_id"] for question in questions if question["id"] not in completed]
        manual_warnings = [{"evidence_id": row["id"], "doc_id": row["raw_doc_id"], "para_id": row["raw_para_id"],
                            "scope_verified": False, "manual_review_required": True}
                           for row in snapshot.get("evidence", []) if row.get("manual_review_required") is True]
        return {"kind": "nonformal-submission-preview", "formal_submission": False,
                "manual_review_warnings": manual_warnings,
                "partial": bool(missing), "topic_id": topic_id, "revision_id": revision_id,
                "notice": "仅供本地核对的结构化结果预览，不能作为正式比赛提交包；不含官方评分。",
                "validation": "local-contract-source-span-reference-and-graph-checks; not official scoring",
                "scope": {"completed_questions": len(completed), "total_questions": len(questions),
                          "completed_question_ids": [question["raw_question_id"] for question in questions if question["id"] in completed],
                          "missing_question_ids": missing},
                "files": files}

    def formal_submission_files(self, topic_id, revision_id):
        snapshot, files, questions = self.submission_projection(topic_id, revision_id)
        if any(row.get("manual_review_required") is True or (row.get("interpretation_origin") == "human" and row.get("scope_verified") is False) for row in snapshot.get("evidence", [])):
            raise ServiceError(409, "export_manual_fact_unverified", "存在尚未通过事实复核的人工解释；原意已保留，请修正后重新复核再导出正式提交包。")
        if set(snapshot["question_results"]) != {question["id"] for question in questions}:
            raise ServiceError(409, "export_incomplete_topic", "该版本尚未完成专题全部问题；请先运行缺失问题，或下载明确标注为非正式的结果预览。")
        return files

    def create_export(self, topic_id, revision_id, key):
        request_hash = digest({"revision_id": revision_id})
        with self.connection() as db, transaction(db):
            old = db.execute("SELECT id,request_sha256 FROM export_jobs WHERE topic_id=? AND idempotency_key=?", (topic_id, key)).fetchone()
            if old:
                if old["request_sha256"] != request_hash:
                    raise ServiceError(409, "idempotency_conflict", "同一个幂等键已用于不同导出。")
                return self.export(old["id"])
            # Fail synchronously before queuing instead of making the user wait
            # for a guaranteed worker rejection of a partial topic.
            self.formal_submission_files(topic_id, revision_id)
            eid = str(uuid4())
            db.execute("INSERT INTO export_jobs(id,topic_id,revision_id,status,adapter_version,idempotency_key,request_sha256,created_at) VALUES (?,?,?,'queued','competition-v0.2',?,?,?)", (eid, topic_id, revision_id, key, request_hash, now()))
        return self.export(eid)

    def export(self, export_id):
        r = self.one("SELECT * FROM export_jobs WHERE id=?", (export_id,))
        return {"id": r["id"], "export_id": r["id"], "topic_id": r["topic_id"], "revision_id": r["revision_id"], "status": r["status"], "manifest": loads(r["manifest_json"]), "error": loads(r["error_json"]), "download_url": f"/api/exports/{r['id']}/download" if r["status"] == "succeeded" else None}
