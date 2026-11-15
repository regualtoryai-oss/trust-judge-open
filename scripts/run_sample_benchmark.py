"""Isolated, resumable development-sample runs through the real application.

``plan`` never constructs Settings or a provider and never opens gold answers.
``execute`` requires a reviewed immutable plan and explicit paid-call flag.
Failed attempts are retained; ordinary continuation starts only unstarted work.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN_FORMAT = "trust-judge-sample-benchmark-plan-v1"
STATUS_FORMAT = "trust-judge-sample-benchmark-status-v1"
TERMINAL = {"succeeded", "failed", "cancelled", "superseded"}
PROGRESS_RETRY_DELAYS = (0.02, 0.04, 0.08, 0.16)


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def contained(base, relative):
    path = (Path(base) / relative).resolve()
    if not path.is_relative_to(Path(base).resolve()):
        raise ValueError("path escapes its declared benchmark/input root")
    return path


def source_bundle():
    """No environment, runtime, model outputs, or authentication files."""
    paths = sorted([*ROOT.joinpath("app").rglob("*.py"), *ROOT.joinpath("app").rglob("*.sql"),
                    *ROOT.joinpath("contracts").rglob("*.json"), ROOT / "pyproject.toml", ROOT / "uv.lock",
                    Path(__file__).resolve(), ROOT / "scripts/prepare_sample_inputs.py"])
    files = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths if p.is_file()}
    return {"bundle_sha256": digest(files), "files": files}


def prepare_plan(input_manifest: Path, output_dir: Path):
    """Copy strictly raw-only prepared inputs; do not inspect Settings/gold."""
    from app.data.ingest import validate_inputs

    input_manifest, output_dir = input_manifest.resolve(), output_dir.resolve()
    if output_dir == ROOT or output_dir == ROOT / ".runtime" or output_dir.is_relative_to(input_manifest.parent):
        raise ValueError("choose a new benchmark directory outside the prepared source directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("plan requires a new empty benchmark directory")
    source = read_json(input_manifest)
    if source.get("input_format") != "trust-judge-runtime-input-v1" or not source.get("topics"):
        raise ValueError("expected prepare_sample_inputs.py manifest")
    topics, copies, seen = [], [], set()
    for index, item in enumerate(source["topics"], 1):
        path = contained(input_manifest.parent, item["input_file"])
        if file_sha(path) != item["sha256"]:
            raise ValueError("prepared source input hash mismatch")
        data = read_json(path)
        if set(data) != {"title", "documents", "questions"} or data["title"] != item["title"] or data["title"] in seen:
            raise ValueError("invalid or duplicate raw-only topic")
        seen.add(data["title"])
        # Unlike the browser projection, this boundary rejects answer-bearing
        # questions outright, including fields an importer could discard.
        if not all(isinstance(q, dict) and set(q) == {"question_id", "question"} for q in data["questions"]):
            raise ValueError("benchmark runtime input must not contain gold fields")
        docs, questions = validate_inputs(data["documents"], data["questions"])
        counts = {"documents": len(docs), "paragraphs": sum(len(d["paragraphs"]) for d in docs), "questions": len(questions)}
        if counts != item["counts"]:
            raise ValueError("prepared source counts mismatch")
        key = f"topic_{index:02d}"
        relative = f"inputs/{key}/input.json"
        copies.append((relative, path.read_bytes()))
        topics.append({"topic_key": key, "title": data["title"], "input_file": relative,
                       "input_sha256": item["sha256"], "counts": counts,
                       "data_dir": f"topics/{key}/runtime",
                       "questions": [{"raw_question_id": q["question_id"], "question_sha256": hashlib.sha256(q["question"].encode()).hexdigest()}
                                     for q in questions]})
    bundle = source_bundle()
    plan = {"format": PLAN_FORMAT, "created_at": now(), "purpose": "visible public development examples, not blind/official accuracy",
            "source_input_manifest": {"path": str(input_manifest), "sha256": file_sha(input_manifest)},
            "source_bundle": bundle, "source_bundle_sha256": bundle["bundle_sha256"],
            "total_topics": len(topics), "total_questions": sum(len(t["questions"]) for t in topics), "topics": topics,
            "settings_read": False, "http_calls": 0, "gold_answers_read": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative, content in copies:
        target = contained(output_dir, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    write_json(output_dir / "benchmark-plan.json", plan)
    return plan


def validate_plan(output_dir: Path, reviewed_sha256: str):
    output_dir = output_dir.resolve()
    path = output_dir / "benchmark-plan.json"
    if file_sha(path) != reviewed_sha256:
        raise ValueError("reviewed plan SHA does not match")
    plan = read_json(path)
    if plan.get("format") != PLAN_FORMAT or plan.get("source_bundle") != source_bundle():
        raise ValueError("source bundle changed; use a new benchmark plan, never cross-version reuse")
    if plan.get("source_bundle_sha256") != plan["source_bundle"]["bundle_sha256"]:
        raise ValueError("source bundle identity mismatch")
    seen = set()
    for topic in plan["topics"]:
        key = topic["topic_key"]
        if key in seen or key != f"topic_{len(seen)+1:02d}":
            raise ValueError("invalid topic ordering/identity")
        seen.add(key)
        path = contained(output_dir, topic["input_file"])
        if file_sha(path) != topic["input_sha256"]:
            raise ValueError("frozen raw input changed")
        payload = read_json(path)
        if topic["title"] != payload["title"] or topic["questions"] != [
            {"raw_question_id": q["question_id"], "question_sha256": hashlib.sha256(q["question"].encode()).hexdigest()}
            for q in payload["questions"]
        ]:
            raise ValueError("frozen question identities changed")
        if topic["data_dir"] != f"topics/{key}/runtime":
            raise ValueError("benchmark runtime path is not canonical")
    if plan["total_questions"] != sum(len(t["questions"]) for t in plan["topics"]):
        raise ValueError("question denominator mismatch")
    return plan


def isolated_settings(settings, output_dir, topic):
    """Check paths before Repository can create or open any SQLite file."""
    target = contained(output_dir, topic["data_dir"])
    original = {settings.database_path.resolve(), settings.checkpoint_path.resolve()}
    candidate = {(target / "business.sqlite3").resolve(), (target / "checkpoints.sqlite3").resolve()}
    exports = (target / "exports").resolve()
    if (candidate & original or target == settings.data_dir.resolve() or target.is_relative_to(settings.export_dir.resolve())
            or exports == settings.export_dir.resolve() or exports.is_relative_to(settings.export_dir.resolve())):
        raise ValueError("benchmark directory collides with configured production storage")
    return settings.model_copy(update={"data_dir": target})


class SharedRequestBudget:
    """Hold a global permit around the actual provider _request implementation.

    Logical request entries are counted, not claimed as wire-level attempts:
    the provider SDK can perform additional internal transport retries.
    """
    def __init__(self, concurrency):
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("HTTP concurrency must be positive")
        self.semaphore = asyncio.Semaphore(concurrency)
        self.limit, self.active, self.peak, self.calls = concurrency, 0, 0, 0

    def wrap(self, provider):
        original = provider._request
        budget = self

        async def limited(_self, *args, **kwargs):
            async with budget.semaphore:
                budget.active += 1
                budget.peak = max(budget.peak, budget.active)
                budget.calls += 1
                try:
                    return await original(*args, **kwargs)
                finally:
                    budget.active -= 1
        provider._request = MethodType(limited, provider)
        return provider

    def report(self):
        return {"concurrency_limit": self.limit, "logical_provider_request_calls_this_process": self.calls,
                "peak_active_provider_requests_this_process": self.peak, "wire_http_attempts": None,
                "wire_attempts_note": "SDK retries are not counted as separate logical requests; no wire-level attempt claim"}


def usage_summary(usage):
    records = [r for value in (usage or {}).values() for r in value]
    return {"records": len(records), **{key: {"known_sum": sum(r[key] for r in records if type(r.get(key)) is int),
                                            "unknown_records": sum(type(r.get(key)) is not int for r in records)}
                                     for key in ("input_tokens", "output_tokens", "total_tokens")}}


class ObserverPersistenceIncomplete(RuntimeError):
    """Workers finished, but the final observation ledger was not persisted."""
    def __init__(self, details):
        super().__init__("benchmark observation ledger is incomplete; inspect observer receipts and database terminal states")
        self.details = details


class ProgressLedger:
    """Only status-file PermissionError is a recoverable observer fault.

    Plans, execution freezes, task DB writes and terminal/result artifacts keep
    using strict persistence. No exception text or payload enters receipts.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.session = uuid4().hex
        self.failures = self.retries = 0
        self.dirty = False
        self.receipts = []
        self.receipt_write_failures = 0
        self.stderr_write_failures = 0

    def report(self, *, current=None):
        return {"ledger_current": not self.dirty if current is None else current,
                "permission_error_episodes": self.failures, "retry_attempts": self.retries,
                "receipt_paths": list(self.receipts), "receipt_write_failures": self.receipt_write_failures,
                "stderr_write_failures": self.stderr_write_failures}

    def receipt(self, error, attempts):
        self.failures += 1
        self.dirty = True
        data = {"format": "benchmark-observer-error-v1", "code": "benchmark_progress_permission_error",
                "observed_at": now(), "operation": "atomic_status_write", "file": self.path.name,
                "exception_type": "PermissionError", "attempts": attempts,
                "episode": self.failures, "worker_cancel_requested_by_observer": False}
        for name in ("errno", "winerror"):
            if type(value := getattr(error, name, None)) is int:
                data[name] = value
        relative = Path("observer-errors") / f"{self.session}-{self.failures:06d}.json"
        try:
            target = self.path.parent / relative
            target.parent.mkdir(exist_ok=True)
            # New immutable receipt: no replace of a file a viewer may hold.
            with target.open("xb") as handle:
                handle.write(encoded(data))
            self.receipts.append(relative.as_posix())
        except OSError as receipt_error:
            self.receipt_write_failures += 1
            data["receipt_write_exception_type"] = type(receipt_error).__name__
        # A completely unwritable output directory cannot guarantee a file;
        # stderr plus the eventual non-success outcome remain explicit.
        try:
            print(json.dumps(data, ensure_ascii=False), file=sys.stderr, flush=True)
        except OSError:
            self.stderr_write_failures += 1

    async def write(self, payload):
        """Bounded retries; old complete status is preserved on exhaustion."""
        for attempt in range(len(PROGRESS_RETRY_DELAYS) + 1):
            payload["observer"] = self.report(current=True)
            try:
                write_json(self.path, payload)
            except PermissionError as error:
                if attempt == len(PROGRESS_RETRY_DELAYS):
                    self.receipt(error, attempt + 1)
                    return False
                self.retries += 1
                await asyncio.sleep(PROGRESS_RETRY_DELAYS[attempt])
            else:
                self.dirty = False
                return True


async def observe_worker(task, observe, *, interval=1.0):
    """Observer errors cannot manufacture Worker cancellation.

    Other observer errors stay strict, but are raised only after the already
    started Worker is awaited. Only actual parent cancellation cancels it.
    """
    observer_error = None
    try:
        while not task.done():
            try:
                await observe()
            except Exception as error:
                observer_error = error
                break
            await asyncio.wait({task}, timeout=interval)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
            # A Worker-internal cancellation has its own backend terminal state.
            # The caller must reconcile that state, never rerun the task.
        if observer_error is not None:
            raise observer_error
    except asyncio.CancelledError:
        if asyncio.current_task().cancelling():
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as error:
                try:
                    print(json.dumps({"code": "benchmark_cancel_cleanup_error", "exception_type": type(error).__name__}),
                          file=sys.stderr, flush=True)
                except OSError:
                    pass  # Broken observer stderr must not mask actual parent cancellation.
        raise


async def execute_plan(output_dir: Path, reviewed_sha256: str, *, execute_authorized=False,
                       topic_concurrency=6, http_concurrency=3, settings_factory=None, provider_factory=None):
    if not execute_authorized:
        raise ValueError("paid execution requires --execute-authorized")
    if type(topic_concurrency) is not int or not 1 <= topic_concurrency <= 6:
        raise ValueError("topic concurrency must be 1..6")
    output_dir = output_dir.resolve()
    plan = validate_plan(output_dir, reviewed_sha256)
    # Imports/factories below this point may load the execution environment.
    from app.config import Settings
    from app.repository import Repository
    from app.worker import Worker, WorkerLease
    from app.agents.provider import LangChainOpenAIProvider
    from app.data.import_service import ImportService

    root_lease = WorkerLease(output_dir / "benchmark.lock")
    root_lease.acquire()
    try:
        settings = (settings_factory or Settings)()
        isolated = [isolated_settings(settings, output_dir, t) for t in plan["topics"]]
        # A temporary isolated repo supplies precisely the production config;
        # it never opens settings.database_path.
        config = Repository(isolated[0]).config()
        safe_config = {**config, "endpoint_sha256": hashlib.sha256(settings.model_base_url.encode()).hexdigest()}
        config_sha = digest(safe_config)
        freeze_path = output_dir / "execution-freeze.json"
        freeze_binding = {"format": "trust-judge-sample-benchmark-execution-freeze-v1", "plan_sha256": reviewed_sha256,
                          "source_bundle_sha256": plan["source_bundle_sha256"], "safe_model_config": safe_config,
                          "config_sha256": config_sha, "topic_concurrency": topic_concurrency,
                          "http_concurrency": http_concurrency}
        if freeze_path.exists():
            frozen = read_json(freeze_path)
            if {k: v for k, v in frozen.items() if k != "created_at"} != freeze_binding:
                raise ValueError("execution config, version, or concurrency changed; use a new benchmark directory")
        else:
            write_json(freeze_path, {**freeze_binding, "created_at": now()})
        freeze_sha = file_sha(freeze_path)
        status_path = output_dir / "benchmark-status.json"
        expected = [(t["topic_key"], q["raw_question_id"]) for t in plan["topics"] for q in t["questions"]]
        if status_path.exists():
            status = read_json(status_path)
            if (status.get("format") != STATUS_FORMAT or status.get("plan_sha256") != reviewed_sha256
                    or status.get("config_sha256") != config_sha or status.get("source_bundle_sha256") != plan["source_bundle_sha256"]
                    or status.get("execution_freeze_sha256") != freeze_sha
                    or [(r["topic_key"], r["raw_question_id"]) for r in status["questions"]] != expected
                    or status.get("http_concurrency") != http_concurrency or status.get("topic_concurrency") != topic_concurrency):
                raise ValueError("execution config, version, or case denominator changed; no cross-version continuation")
        else:
            status = {"format": STATUS_FORMAT, "plan_sha256": reviewed_sha256,
                      "source_bundle_sha256": plan["source_bundle_sha256"], "safe_model_config": safe_config,
                      "config_sha256": config_sha, "created_at": now(), "topic_concurrency": topic_concurrency,
                      "execution_freeze_path": str(freeze_path), "execution_freeze_sha256": freeze_sha,
                      "http_concurrency": http_concurrency, "total_questions": len(expected), "topics": [], "questions": []}
            for t, ts in zip(plan["topics"], isolated):
                status["topics"].append({"topic_key": t["topic_key"], "title": t["title"], "database": str(ts.database_path),
                                         "checkpoint_path": str(ts.checkpoint_path), "topic_id": None, "export": None})
                for q in t["questions"]:
                    status["questions"].append({"topic_key": t["topic_key"], "title": t["title"],
                        "database": str(ts.database_path), "checkpoint_path": str(ts.checkpoint_path),
                        "topic_id": None, "raw_question_id": q["raw_question_id"], "internal_question_id": None,
                        "run_id": None, "status": "pending", "revision_id": None, "ready_revision_id": None,
                        "terminal_path": None, "snapshot_path": None, "projection_path": None,
                        "question_result_path": None, "usage": {}, "error": None})
        budget = SharedRequestBudget(http_concurrency)
        write_lock, topic_slots = asyncio.Lock(), asyncio.Semaphore(topic_concurrency)
        observer = ProgressLedger(status_path)

        async def persist(*, final=False):
            async with write_lock:
                status["updated_at"] = now()
                status["status_counts"] = dict(__import__("collections").Counter(r["status"] for r in status["questions"]))
                status["request_budget"] = budget.report()
                status["known_usage"] = usage_summary({str(i): [r for v in (row["usage"] or {}).values() for r in v]
                                                        for i, row in enumerate(status["questions"])})
                payload = deepcopy(status)
                payload["observer"] = observer.report(current=True)
                written = await observer.write(payload)
                status["observer"] = observer.report(current=written)
                if final and not written:
                    raise ObserverPersistenceIncomplete(observer.report(current=False))
                return written

        async def topic_run(topic, ts):
            async with topic_slots:
                repo = Repository(ts)
                worker = Worker(repo)
                worker.lease.acquire()
                try:
                    repo.recover()  # interrupted running attempts become failed, never silently retried
                    payload_path = contained(output_dir, topic["input_file"])
                    service = ImportService(repo)
                    preview = service.preview({"entries": [{"name": "input.json", "content": payload_path.read_text(encoding="utf-8-sig")} ]})
                    if preview["excluded_fields"]:
                        raise ValueError("raw-only benchmark unexpectedly required answer projection")
                    imported = service.commit({"preview_id": preview["preview_id"], "fingerprint": preview["fingerprint"]})
                    tid = imported["topic_id"]
                    summary = next(t for t in status["topics"] if t["topic_key"] == topic["topic_key"])
                    if summary["topic_id"] not in (None, tid):
                        raise ValueError("runtime topic identity changed")
                    summary.update({"topic_id": tid, "source_manifest_sha256": repo.topic(tid)["source_manifest_sha256"]})
                    questions = {q["raw_question_id"]: q for q in repo.questions(tid)}
                    for planned in topic["questions"]:
                        if source_bundle() != plan["source_bundle"]:
                            raise ValueError("source bundle changed during execution; no later question may start")
                        raw = planned["raw_question_id"]
                        row = next(r for r in status["questions"] if (r["topic_key"], r["raw_question_id"]) == (topic["topic_key"], raw))
                        qid = questions[raw]["id"]
                        if row["topic_id"] not in (None, tid) or row["internal_question_id"] not in (None, qid):
                            raise ValueError("runtime question identity changed")
                        row.update({"topic_id": tid, "internal_question_id": qid})
                        key = "benchmark:" + reviewed_sha256 + ":" + raw
                        old = repo.all("SELECT id FROM runs WHERE topic_id=? AND idempotency_key=?", (tid, key))
                        if row["run_id"] and (not old or old[0]["id"] != row["run_id"]):
                            raise ValueError("ledger attempt is absent or replaced")
                        if old:
                            run = repo.run(old[0]["id"])
                        else:
                            if row["status"] != "pending":
                                raise ValueError("terminal ledger must never create a replacement attempt")
                            run = repo.create_run(tid, {"question_ids": [qid], "base_revision_id": repo.topic(tid)["head_revision_id"],
                                "mode": "recompute", "include_baseline": False}, key,
                                model_available=bool(provider_factory) or ts.model_configured)
                        row.update({"run_id": run["id"], "revision_id": run["target_revision_id"], "status": run["status"]})
                        await persist()
                        if run["status"] == "queued":
                            if worker.provider is None:
                                # Test doubles are possible only by explicit Python API injection.
                                worker.provider = (provider_factory(ts, budget) if provider_factory else budget.wrap(LangChainOpenAIProvider(ts)))
                            claimed = repo.claim()
                            if claimed is None or claimed["id"] != run["id"]:
                                raise ValueError("isolated worker claimed an unexpected attempt")
                            task = asyncio.create_task(worker.execute(claimed))
                            async def observe():
                                current = repo.run(run["id"])
                                row.update({"status": current["status"], "usage": current["usage"] or {}, "error": current["error"]})
                                await persist()
                            await observe_worker(task, observe)
                        run = repo.run(run["id"])
                        if run["status"] not in TERMINAL:
                            raise RuntimeError("benchmark Worker exited without an authoritative terminal state")
                        row.update({"status": run["status"], "usage": run["usage"] or {}, "error": run["error"], "ready_revision_id": None})
                        result_dir = output_dir / "topics" / topic["topic_key"] / "questions" / hashlib.sha256(raw.encode()).hexdigest()
                        terminal = result_dir / "terminal.json"
                        write_json(terminal, run)
                        row["terminal_path"] = str(terminal)
                        if run["status"] == "succeeded":
                            revision = run["target_revision_id"]
                            snapshot = repo.snapshot(tid, revision)
                            projection = repo.submission_preview(tid, revision)
                            for name, value in (("snapshot", snapshot), ("projection", projection),
                                                ("question_result", snapshot["question_results"][qid])):
                                target = result_dir / (name + ".json")
                                write_json(target, value)
                                row[name + "_path"] = str(target)
                            row["ready_revision_id"] = revision
                        await persist()
                        print(json.dumps({"topic_key": topic["topic_key"], "raw_question_id": raw,
                                          "status": run["status"], "ready_revision_id": row["ready_revision_id"]}, ensure_ascii=False), flush=True)
                    published = repo.topic(tid)["published_revision_id"]
                    if published:
                        readiness = repo.export_readiness(tid, published)
                        summary["export_readiness"] = readiness
                        if readiness["can_export"]:
                            export = repo.create_export(tid, published, "benchmark-export:" + reviewed_sha256)
                            while export["status"] == "queued":
                                worker.export_one()
                                export = repo.export(export["id"])
                            if export["status"] == "succeeded":
                                export["artifact_path"] = repo.one("SELECT artifact_path FROM export_jobs WHERE id=?", (export["id"],))["artifact_path"]
                            summary["export"] = export
                    await persist()
                finally:
                    worker.lease.close()

        await persist()
        results = await asyncio.gather(*(topic_run(t, ts) for t, ts in zip(plan["topics"], isolated)), return_exceptions=True)
        for topic, result in zip(plan["topics"], results):
            if isinstance(result, BaseException):
                # Never serialize arbitrary exception messages/configuration.
                summary = next(t for t in status["topics"] if t["topic_key"] == topic["topic_key"])
                summary["runner_error"] = {"code": "benchmark_topic_execution_error", "exception_type": type(result).__name__}
        status["process_completed_at"] = now()
        status["all_questions_terminal"] = all(r["status"] in TERMINAL for r in status["questions"])
        await persist(final=True)
        return status
    finally:
        root_lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--input-manifest", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--reviewed-manifest-sha256", required=True)
    execute.add_argument("--execute-authorized", action="store_true")
    execute.add_argument("--topic-concurrency", type=int, default=6)
    execute.add_argument("--http-concurrency", type=int, default=3)
    args = parser.parse_args()
    if args.command == "plan":
        result = prepare_plan(args.input_manifest, args.output_dir)
        print(json.dumps({"plan": str(args.output_dir.resolve() / "benchmark-plan.json"),
                          "sha256": file_sha(args.output_dir / "benchmark-plan.json"),
                          "topics": result["total_topics"], "questions": result["total_questions"], "http_calls": 0}, ensure_ascii=False))
    else:
        try:
            result = asyncio.run(execute_plan(args.output_dir, args.reviewed_manifest_sha256,
                execute_authorized=args.execute_authorized, topic_concurrency=args.topic_concurrency, http_concurrency=args.http_concurrency))
        except ObserverPersistenceIncomplete as error:
            print(json.dumps({"code": "benchmark_observer_incomplete", "observer": error.details}, ensure_ascii=False), file=sys.stderr)
            raise SystemExit(2) from None
        print(json.dumps({"status": str(args.output_dir.resolve() / "benchmark-status.json"),
                          "all_questions_terminal": result["all_questions_terminal"], "counts": result["status_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
