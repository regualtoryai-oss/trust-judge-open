"""Real Repository/Worker, explicit fake providers, isolated temporary DBs."""
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys

import httpx
import pytest

from scripts import run_sample_benchmark as runner
from test_sample_benchmark_runner import ExplicitProvider, fixture_plan, settings


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    original_connect = socket.socket.connect
    def forbidden(*_a, **_k):
        raise AssertionError("observer regression tests are offline")
    def local_event_loop_pipe(sock, address):
        # Windows asyncio's socketpair is implemented with a loopback connect.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(sock, address)
        return forbidden()
    monkeypatch.setattr(socket.socket, "connect", local_event_loop_pipe)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(runner, "PROGRESS_RETRY_DELAYS", (0, 0, 0, 0))


class BlockingProvider(ExplicitProvider):
    def __init__(self, calls):
        super().__init__(calls)
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.cancelled = False

    async def extract(self, question, paragraphs):
        self.account("extract", question)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return []


def db_runs(output):
    path = output / "topics/topic_01/runtime/business.sqlite3"
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT id,status,cancel_requested,error_json FROM runs ORDER BY created_at,id")]


async def run_with_status_fault(tmp_path, monkeypatch, *, failures):
    output, sha, _ = fixture_plan(tmp_path, topics=1)
    calls, seen = Counter(), []
    provider = BlockingProvider(calls)
    exhausted = asyncio.Event()
    original_write = runner.write_json
    def injected(path, value):
        if Path(path).name == "benchmark-status.json" and provider.started.is_set() and (failures is None or len(seen) < failures):
            seen.append(True)
            if len(seen) >= min(failures or 5, 5):
                exhausted.set()
            raise PermissionError(13, "SECRET_IO_MESSAGE_MUST_NOT_ENTER_RECEIPT")
        return original_write(path, value)
    monkeypatch.setattr(runner, "write_json", injected)
    task = asyncio.create_task(runner.execute_plan(output, sha, execute_authorized=True, topic_concurrency=1,
        settings_factory=lambda: settings(tmp_path), provider_factory=lambda *_: provider))
    await asyncio.wait_for(provider.started.wait(), timeout=5)
    old_status = (output / "benchmark-status.json").read_bytes()
    await asyncio.wait_for(exhausted.wait(), timeout=5)
    assert not provider.cancelled
    provider.release.set()
    return output, sha, calls, provider, task, seen, old_status, original_write


async def test_transient_permission_retried_without_cancelling_real_worker(tmp_path, monkeypatch):
    output, _sha, calls, provider, task, seen, _old, _write = await run_with_status_fault(tmp_path, monkeypatch, failures=2)
    result = await task
    assert len(seen) == 2 and result["observer"]["retry_attempts"] == 2
    assert result["observer"]["permission_error_episodes"] == 0
    assert result["status_counts"] == {"succeeded": 3} and not provider.cancelled
    assert all(calls[(q, "extract")] == 1 for q in ("q_001", "q_002", "q_003"))
    assert all(r["status"] == "succeeded" and not r["cancel_requested"] for r in db_runs(output))
    assert runner.read_json(output / "benchmark-status.json")["observer"]["retry_attempts"] == 2


async def test_exhausted_tick_receipt_then_worker_finishes_and_terminal_is_collected(tmp_path, monkeypatch):
    output, _sha, calls, provider, task, seen, _old, _write = await run_with_status_fault(tmp_path, monkeypatch, failures=5)
    result = await task
    assert len(seen) == 5 and result["all_questions_terminal"] and not provider.cancelled
    assert result["observer"]["ledger_current"] and result["observer"]["permission_error_episodes"] == 1
    receipt = runner.read_json(output / result["observer"]["receipt_paths"][0])
    assert receipt["exception_type"] == "PermissionError" and receipt["attempts"] == 5
    assert receipt["worker_cancel_requested_by_observer"] is False
    assert "SECRET_IO_MESSAGE" not in json.dumps(receipt)
    assert len(db_runs(output)) == 3 and all(r["status"] == "succeeded" for r in db_runs(output))
    assert all(calls[(q, "extract")] == 1 for q in ("q_001", "q_002", "q_003"))
    assert all(Path(q["terminal_path"]).is_file() and Path(q["question_result_path"]).is_file() for q in result["questions"])


async def test_persistent_observer_failure_finishes_real_work_but_reports_incomplete_and_no_rerun(tmp_path, monkeypatch):
    output, sha, calls, provider, task, _seen, old_status, original_write = await run_with_status_fault(tmp_path, monkeypatch, failures=None)
    with pytest.raises(runner.ObserverPersistenceIncomplete) as exc:
        await task
    assert not provider.cancelled and exc.value.details["ledger_current"] is False
    assert exc.value.details["permission_error_episodes"] >= 2
    assert exc.value.details["receipt_paths"]
    assert (output / "benchmark-status.json").read_bytes() == old_status
    assert len(db_runs(output)) == 3 and all(r["status"] == "succeeded" for r in db_runs(output))
    assert len(list((output / "topics/topic_01/questions").glob("*/terminal.json"))) == 3
    assert len(list((output / "topics/topic_01/questions").glob("*/question_result.json"))) == 3
    before = dict(calls)
    monkeypatch.setattr(runner, "write_json", original_write)
    result = await runner.execute_plan(output, sha, execute_authorized=True, topic_concurrency=1,
        settings_factory=lambda: settings(tmp_path), provider_factory=lambda *_: pytest.fail("terminal runs cannot construct a new model provider"))
    assert result["all_questions_terminal"] and dict(calls) == before
    assert len(db_runs(output)) == 3 and runner.read_json(output / "benchmark-status.json")["all_questions_terminal"]


async def test_true_parent_cancel_still_cancels_and_awaits_real_worker(tmp_path):
    output, sha, _ = fixture_plan(tmp_path, topics=1)
    calls, provider = Counter(), None
    provider = BlockingProvider(calls)
    task = asyncio.create_task(runner.execute_plan(output, sha, execute_authorized=True, topic_concurrency=1,
        settings_factory=lambda: settings(tmp_path), provider_factory=lambda *_: provider))
    await asyncio.wait_for(provider.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.cancelled
    rows = db_runs(output)
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert json.loads(rows[0]["error_json"])["code"] == "worker_interrupted"
    assert calls[("q_002", "extract")] == calls[("q_003", "extract")] == 0
    from app.worker import WorkerLease
    lease = WorkerLease(output / "topics/topic_01/runtime/worker.lock")
    lease.acquire();lease.close()


async def test_child_cancel_without_parent_cancel_keeps_failed_attempt_and_later_questions_run(tmp_path, monkeypatch):
    output, sha, _ = fixture_plan(tmp_path, topics=1)
    calls, children = Counter(), []
    provider = BlockingProvider(calls)
    original_observe = runner.observe_worker
    async def capture_child(child, observe, **kwargs):
        children.append(child)
        return await original_observe(child, observe, **kwargs)
    monkeypatch.setattr(runner, "observe_worker", capture_child)
    task = asyncio.create_task(runner.execute_plan(output, sha, execute_authorized=True, topic_concurrency=1,
        settings_factory=lambda: settings(tmp_path), provider_factory=lambda *_: provider))
    await asyncio.wait_for(provider.started.wait(), timeout=5)
    children[0].cancel()
    await asyncio.wait_for(asyncio.wait({children[0]}), timeout=5)
    provider.release.set()
    result = await task
    assert result["all_questions_terminal"] and result["status_counts"] == {"failed": 1, "succeeded": 2}
    first = result["questions"][0]
    assert first["error"]["code"] == "worker_interrupted" and Path(first["terminal_path"]).is_file()
    assert first["ready_revision_id"] is None
    assert len(db_runs(output)) == 3 and not any(r["cancel_requested"] for r in db_runs(output))
    assert all(calls[(q, "extract")] == 1 for q in ("q_001", "q_002", "q_003"))


async def test_six_topic_observer_failure_keeps_all_real_workers_and_full_denominator(tmp_path, monkeypatch):
    output, sha, _ = fixture_plan(tmp_path, topics=6)
    calls, providers, failures = Counter(), [], []
    exhausted = asyncio.Event()
    original_write = runner.write_json
    def provider_factory(*_):
        provider = BlockingProvider(calls)
        providers.append(provider)
        return provider
    def injected(path, value):
        all_started = len(providers) == 6 and all(p.started.is_set() for p in providers)
        if Path(path).name == "benchmark-status.json" and all_started and len(failures) < 7:
            failures.append(True)
            if len(failures) >= 5:
                exhausted.set()
            raise PermissionError("safe concurrent observer fault")
        return original_write(path, value)
    monkeypatch.setattr(runner, "write_json", injected)
    task = asyncio.create_task(runner.execute_plan(output, sha, execute_authorized=True,
        topic_concurrency=6, settings_factory=lambda: settings(tmp_path), provider_factory=provider_factory))
    await asyncio.wait_for(exhausted.wait(), timeout=10)
    assert len(providers) == 6 and not any(p.cancelled for p in providers)
    for provider in providers:
        provider.release.set()
    result = await task
    assert len(result["questions"]) == 18 and result["status_counts"] == {"succeeded": 18}
    assert result["all_questions_terminal"] and result["observer"]["permission_error_episodes"] == 1
    assert not any(p.cancelled for p in providers)
    assert all(calls[(q, "extract")] == 6 for q in ("q_001", "q_002", "q_003"))
    for topic in result["topics"]:
        with sqlite3.connect(Path(topic["database"]).as_uri() + "?mode=ro", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 3
            assert db.execute("SELECT COUNT(*) FROM runs WHERE status!='succeeded' OR cancel_requested!=0").fetchone()[0] == 0
    disk = runner.read_json(output / "benchmark-status.json")
    assert disk["all_questions_terminal"] and disk["status_counts"] == {"succeeded": 18}


async def test_non_status_terminal_artifact_permission_error_remains_strict(tmp_path, monkeypatch):
    output, sha, _ = fixture_plan(tmp_path, topics=1)
    calls = Counter()
    original_write = runner.write_json
    def fail_terminal(path, value):
        if Path(path).name == "terminal.json":
            raise PermissionError("not an observer status write")
        return original_write(path, value)
    monkeypatch.setattr(runner, "write_json", fail_terminal)
    result = await runner.execute_plan(output, sha, execute_authorized=True, topic_concurrency=1,
        settings_factory=lambda: settings(tmp_path), provider_factory=lambda *_: ExplicitProvider(calls))
    assert result["topics"][0]["runner_error"]["exception_type"] == "PermissionError"
    assert not result["all_questions_terminal"] and result["observer"]["permission_error_episodes"] == 0
    assert len(db_runs(output)) == 1 and calls[("q_002", "extract")] == 0


async def test_business_task_write_failure_is_not_observer_fallback(tmp_path, monkeypatch):
    output, sha, _ = fixture_plan(tmp_path, topics=1)
    from app.repository import Repository
    def reject(*_a, **_k):
        raise sqlite3.OperationalError("synthetic business write denial")
    monkeypatch.setattr(Repository, "create_run", reject)
    result = await runner.execute_plan(output, sha, execute_authorized=True, topic_concurrency=1,
        settings_factory=lambda: settings(tmp_path), provider_factory=lambda *_: pytest.fail("business failure cannot call a model"))
    assert result["topics"][0]["runner_error"]["exception_type"] == "OperationalError"
    assert not result["all_questions_terminal"] and not db_runs(output)
    assert result["observer"]["permission_error_episodes"] == 0


async def test_other_observer_exception_does_not_cancel_child_but_is_still_raised():
    entered, release = asyncio.Event(), asyncio.Event()
    cancelled = False
    async def worker():
        nonlocal cancelled
        entered.set()
        try: await release.wait()
        except asyncio.CancelledError: cancelled = True;raise
    child = asyncio.create_task(worker())
    await entered.wait()
    async def bad_observer():
        release.set()
        raise sqlite3.OperationalError("strict read failure")
    with pytest.raises(sqlite3.OperationalError):
        await runner.observe_worker(child, bad_observer, interval=0)
    assert child.done() and not cancelled


@pytest.mark.skipif(os.name != "nt", reason="Windows ordinary reader handle replace semantics")
async def test_windows_ordinary_read_handle_recovers_during_bounded_replace_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROGRESS_RETRY_DELAYS", (0.02, 0.04, 0.08, 0.16))
    path = tmp_path / "benchmark-status.json"
    runner.write_json(path, {"version": 1})
    observer = runner.ProgressLedger(path)
    with path.open("r", encoding="utf-8") as reader:
        assert json.loads(reader.read())["version"] == 1
        task = asyncio.create_task(observer.write({"version": 2}))
        await asyncio.sleep(0.03)
    assert await task
    assert runner.read_json(path)["version"] == 2 and observer.retries >= 1
    assert not observer.receipts and not observer.dirty


async def test_failed_receipt_file_has_safe_stderr_and_incomplete_state(tmp_path, monkeypatch, capsys):
    path = tmp_path / "benchmark-status.json"
    original_open = Path.open
    def fail_receipt(self, *args, **kwargs):
        if self.parent.name == "observer-errors":
            raise PermissionError("SECRET_RECEIPT_BODY")
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_receipt)
    monkeypatch.setattr(runner, "write_json", lambda *_: (_ for _ in ()).throw(PermissionError("SECRET_STATUS_BODY")))
    observer = runner.ProgressLedger(path)
    assert not await observer.write({"private": "must never enter receipt"})
    assert observer.dirty and observer.receipt_write_failures == 1
    stderr = capsys.readouterr().err
    assert "benchmark_progress_permission_error" in stderr
    assert "SECRET" not in stderr and "private" not in stderr


def test_cli_incomplete_observation_exits_nonzero_without_success_json(monkeypatch, capsys, tmp_path):
    def incomplete(coroutine):
        coroutine.close()
        raise runner.ObserverPersistenceIncomplete({"ledger_current": False})
    monkeypatch.setattr(asyncio, "run", incomplete)
    monkeypatch.setattr(sys, "argv", ["benchmark", "execute", "--output-dir", str(tmp_path), "--reviewed-manifest-sha256", "0"*64, "--execute-authorized"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "benchmark_observer_incomplete" in captured.err and not captured.out
