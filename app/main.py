import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import Body, FastAPI, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

from app.config import PROJECT_ROOT, Settings
from app.data.import_service import import_router
from app.errors import ServiceError
from app.repository import Repository, TERMINAL
from app.utils import dumps
from app.worker import Worker


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question_ids: list[str] = Field(min_length=1)
    base_revision_id: str | None
    include_baseline: bool = False
    mode: Literal["initial", "retry", "quality-refine"] = "initial"
    source_revision_id: str | None = None

    @model_validator(mode="after")
    def quality_refine_source(self):
        if self.mode == "quality-refine" and not self.source_revision_id:
            raise ValueError("quality-refine requires a fixed ready source_revision_id")
        if self.mode != "quality-refine" and self.source_revision_id is not None:
            raise ValueError("source_revision_id is accepted only for quality-refine")
        return self



class ExportCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    revision_id: str


def require_key(key):
    if key is None or not key.strip() or len(key) > 200:
        raise ServiceError(422, "idempotency_key_required", "写入请求必须提供 1–200 字符 Idempotency-Key。")
    return key


def validate_contract(name, body):
    contract = json.loads((PROJECT_ROOT / "contracts" / "openapi.json").read_text(encoding="utf-8-sig"))
    validator = Draft202012Validator({"$ref": "#/components/schemas/" + name, "components": contract["components"]}, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(body), key=lambda e: str(e.path))
    if errors:
        raise ServiceError(422, "invalid_request", "请求未通过设计契约校验：" + "/".join(map(str, errors[0].path)))


def create_app(settings=None, provider=None, *, embedded_worker=False):
    settings = settings or Settings()
    repo = Repository(settings)
    available = provider is not None or settings.model_configured

    @asynccontextmanager
    async def lifespan(application):
        task = worker = None
        if embedded_worker:
            worker = Worker(repo, provider)
            task = asyncio.create_task(worker.run_forever())
            await asyncio.sleep(0)
            if task.done():
                await task
        try:
            yield
        finally:
            if worker:
                worker.stop_event.set()
                await task

    app = FastAPI(title="可信研判工作台 · 技术底座", version="0.1.0", lifespan=lifespan)
    app.state.repo = repo
    app.include_router(import_router(repo))

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return JSONResponse(status_code=error.status, content={"detail": {"code": error.code, "message": error.message}})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.1.0", "model_configured": available, "workflow_available": True,
                "source_bundle_enabled": settings.model_source_bundle_enabled,
                "source_bundle_front_framework": settings.source_bundle_front_framework if settings.model_source_bundle_enabled else None,
                "max_questions_per_run": settings.max_questions_per_run,
                "model_name": settings.model_name, "model_provider": settings.model_provider,
                "foundation_stage": True, "mode": "local-single-user", "provider": "explicit-test-injection" if provider is not None else ("langchain-selection-langgraph-analysis" if settings.source_bundle_front_framework == "langchain" else "dsh-selection-langgraph-analysis") if settings.model_source_bundle_enabled else "langchain-openai-compatible"}

    @app.get("/api/topics")
    def topics():
        return repo.topics()

    @app.get("/api/topics/{topic_id}")
    def topic(topic_id: str):
        return repo.topic(topic_id)

    @app.get("/api/topics/{topic_id}/questions")
    def questions(topic_id: str):
        return repo.questions(topic_id)

    @app.get("/api/topics/{topic_id}/revisions")
    def revisions(topic_id: str):
        repo.topic(topic_id)
        return repo.all("SELECT * FROM graph_revisions WHERE topic_id=? ORDER BY revision_no DESC", (topic_id,))

    @app.get("/api/topics/{topic_id}/runs")
    def topic_runs(topic_id: str, limit: int = Query(default=50, ge=1, le=200)):
        return repo.runs(topic_id, limit=limit)

    @app.post("/api/topics/{topic_id}/runs", status_code=202)
    def start_run(topic_id: str, body: RunCreate, idempotency_key: str | None = Header(default=None)):
        return repo.create_run(topic_id, body.model_dump(), require_key(idempotency_key), model_available=available)

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str):
        return repo.run(run_id)

    @app.get("/api/runs/{run_id}/failure-appendix")
    def failure_appendix(run_id: str):
        from app.domain.failure_appendix import build_failure_appendix
        from app.utils import loads
        run = repo.run(run_id)
        if run["status"] != "failed":
            raise ServiceError(409, "failure_appendix_requires_failed_run", "只有失败任务可下载失败附录；该附录不属于比赛提交包。")
        topic = repo.topic(run["topic_id"])
        revision = repo.revision(topic["id"], run["target_revision_id"])
        if revision["input_manifest_sha256"] != topic["source_manifest_sha256"]:
            raise ServiceError(409, "failure_appendix_source_changed", "原始资料清单已变化，不能使用新原文重建旧任务的失败附录。")
        options = loads(repo.one("SELECT options_json FROM run_options WHERE run_id=?", (run_id,))["options_json"], {})
        try:
            appendix = build_failure_appendix(run, repo.questions(topic["id"]), repo.paragraphs(topic["id"]),
                source_manifest_sha256=topic["source_manifest_sha256"],
                excluded_spans=options.get("overlays", {}).get("excluded", []))
        except (KeyError, TypeError, ValueError):
            raise ServiceError(409, "failure_appendix_source_changed", "失败附录的原文、题目或排除项校验失败，未生成替代结果。") from None
        return JSONResponse(appendix, headers={"Content-Disposition": f'attachment; filename="failure-{run_id}.json"', "Cache-Control": "no-store"})

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    def cancel_run(run_id: str):
        return repo.cancel(run_id)

    @app.post("/api/runs/{run_id}/pause", status_code=202)
    def pause_run(run_id: str):
        return repo.pause(run_id)

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    def resume_run(run_id: str, idempotency_key: str | None = Header(default=None)):
        return repo.resume(run_id, require_key(idempotency_key), model_available=available)

    @app.get("/api/runs/{run_id}/activity")
    def run_activity(run_id: str, after: int = Query(default=0, ge=0)):
        return repo.activity(run_id, after)

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request, after: int = Query(default=0, ge=0), last_event_id: str | None = Header(default=None)):
        repo.run(run_id)
        if last_event_id is not None:
            if not last_event_id.isdigit():
                raise ServiceError(422, "invalid_event_cursor", "Last-Event-ID 应为非负整数序号。")
            after = max(after, int(last_event_id))

        async def stream():
            cursor, ticks = after, 0
            while not await request.is_disconnected():
                batch = repo.events(run_id, cursor)
                for event in batch:
                    cursor = event["seq"]
                    yield f"id: {cursor}\nevent: {event['type']}\ndata: {dumps({'run_id':run_id,'seq':cursor,'type':event['type'],'data':event['data'],'created_at':event['created_at']})}\n\n"
                if repo.run(run_id)["status"] in TERMINAL:
                    # A terminal status/event is written in one transaction. Fetch
                    # once more if the transaction committed between these reads.
                    if not repo.events(run_id, cursor):
                        break
                    continue
                ticks += 1
                if ticks % 40 == 0:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/topics/{topic_id}/revisions/{revision_id}/graph")
    def graph(topic_id: str, revision_id: str):
        snapshot = repo.snapshot(topic_id, revision_id)
        return {key: snapshot[key] for key in ("revision_id", "graph_revision", "nodes", "edges", "scope")}

    @app.get("/api/topics/{topic_id}/revisions/{revision_id}/artifacts")
    def artifacts(topic_id: str, revision_id: str):
        snapshot = repo.snapshot(topic_id, revision_id)
        result = {key: snapshot[key] for key in ("revision_id", "conclusions", "conflicts", "confidence", "agent_outputs", "source_groups", "scope")}
        result["quality_reviews"] = [{"question_id": qid, **row["strict_scope_review"]}
            for qid, row in snapshot["question_results"].items() if isinstance(row.get("strict_scope_review"), dict)]
        result["candidate_artifacts"] = [{"question_id": qid,
            "can_refine": isinstance(row.get("candidate_evidence"), list) and row.get("source_coverage", {}).get("complete") is True,
            "candidate_count": len(row.get("candidate_evidence", [])), "origin_run_id": row.get("origin_run_id")}
            for qid, row in snapshot["question_results"].items()]
        return result

    @app.get("/api/evidence/{evidence_id}")
    def evidence(evidence_id: str, revision_id: str):
        return repo.evidence(evidence_id, revision_id)

    @app.get("/api/paragraphs/{paragraph_id}")
    def paragraph(paragraph_id: str):
        return repo.paragraph(paragraph_id)

    @app.get("/api/paragraphs/{paragraph_id}/context")
    def paragraph_context(paragraph_id: str, before: int = Query(default=1, ge=0, le=4), after: int = Query(default=3, ge=0, le=4)):
        return repo.source_context(paragraph_id, before=before, after=after)

    @app.post("/api/topics/{topic_id}/changes/preview")
    def preview(topic_id: str, body: dict = Body(...)):
        validate_contract("ChangePreviewRequest", body)
        return repo.preview_change(topic_id, body["base_revision_id"], body["operations"], body["reason"])

    @app.post("/api/topics/{topic_id}/changes", status_code=202)
    def commit(topic_id: str, body: dict = Body(...), idempotency_key: str | None = Header(default=None)):
        validate_contract("ChangeCommitRequest", body)
        return repo.commit_change(topic_id, body, require_key(idempotency_key), model_available=available)

    @app.post("/api/topics/{topic_id}/exports", status_code=202)
    def create_export(topic_id: str, body: ExportCreate, idempotency_key: str | None = Header(default=None)):
        return repo.create_export(topic_id, body.revision_id, require_key(idempotency_key))

    @app.get("/api/topics/{topic_id}/revisions/{revision_id}/export-readiness")
    def export_readiness(topic_id: str, revision_id: str):
        return repo.export_readiness(topic_id, revision_id)

    @app.get("/api/topics/{topic_id}/revisions/{revision_id}/submission-preview")
    def submission_preview(topic_id: str, revision_id: str):
        return JSONResponse(repo.submission_preview(topic_id, revision_id), headers={
            "Content-Disposition": f'attachment; filename="NONFORMAL-preview-{revision_id}.json"',
            "Cache-Control": "no-store",
        })

    @app.get("/api/exports/{export_id}")
    def export_status(export_id: str):
        return repo.export(export_id)

    @app.get("/api/exports/{export_id}/download")
    def download(export_id: str):
        job = repo.export(export_id)
        if job["status"] != "succeeded":
            raise ServiceError(409, "export_not_ready", "导出尚未完成或已失败。")
        repo.formal_submission_files(job["topic_id"], job["revision_id"])
        stored = repo.one("SELECT artifact_path FROM export_jobs WHERE id=?", (export_id,))["artifact_path"]
        path = Path(stored or "")
        if not path.is_file() or not path.resolve().is_relative_to(settings.export_dir.resolve()):
            raise ServiceError(404, "artifact_missing", "导出文件缺失，请重新创建。")
        return FileResponse(path, media_type="application/zip", filename=f"trust-judge-{export_id}.zip")

    return app


def app_factory():
    return create_app()
