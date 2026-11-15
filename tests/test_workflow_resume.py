import asyncio
import copy
import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents import provider as provider_module
from app.agents.provider import ModelOutputInvalid
from app.workflows.analysis import AnalysisCancelled, run_analysis


QUESTION = {"id": "resume-question", "question": "通报中失联多少人？"}
PARAGRAPHS = [{
    "id": "resume-paragraph", "document_id": "resume-document", "raw_doc_id": "doc_resume",
    "raw_para_id": "para_1", "text": "通报称失联3人。", "start_cp": 0, "end_cp": 8,
}]


class RecordingProvider:
    """A deterministic provider that is explicitly injected into tests only."""

    def __init__(self, *, fail_cross=False, pause_after_extract=None, record_usage=False):
        self.calls = []
        self.fail_cross = fail_cross
        self.pause_after_extract = pause_after_extract
        self.record_usage = record_usage
        self.authority_finished = asyncio.Event()

    def account(self, operation):
        if self.record_usage:
            provider_module._usage_records.get().append({"operation": operation, "total_tokens": 10})

    async def extract(self, question, paragraphs):
        self.calls.append(("extract", copy.deepcopy(question), copy.deepcopy(paragraphs)))
        self.account("extract")
        if self.pause_after_extract:
            self.pause_after_extract()
        return [{"paragraph_id": paragraphs[0]["id"], "quote": "失联3人", "fact_des": "失联3人", "argument": {}, "entities": []}]

    def answer(self, evidence):
        return {
            "decision": "asserted", "conclusion": "失联3人。",
            "supporting_evidence_ids": [evidence[0]["id"]], "conflicting_evidence_ids": [],
            "judgments": [{"id": "J", "statement": "通报明确人数", "evidence_ids": [evidence[0]["id"]], "prerequisite_judgment_ids": []}],
        }

    async def analyze(self, role, question, evidence, snapshot_id):
        self.calls.append((role, copy.deepcopy(question), copy.deepcopy(evidence), snapshot_id))
        self.account(role)
        if role == "cross_validation_analyst" and self.fail_cross:
            await self.authority_finished.wait()
            # Let LangGraph persist the successful sibling's pending writes.
            await asyncio.sleep(0.02)
            raise ModelOutputInvalid("测试的显式失败", stage=role)
        self.authority_finished.set()
        return self.answer(evidence)

    async def arbitrate(self, question, evidence, analyses):
        self.calls.append(("arbitrate", copy.deepcopy(question), copy.deepcopy(evidence), copy.deepcopy(analyses)))
        self.account("arbitrate")
        return {**self.answer(evidence), "conflicts": []}


async def execute(tmp_path, provider, thread="original", **kwargs):
    return await run_analysis(
        kwargs.pop("question", QUESTION), kwargs.pop("paragraphs", PARAGRAPHS), provider,
        test_only_legacy_precision=True, checkpoint_path=tmp_path / "resume.sqlite3", thread_id=thread,
        cancel_check=kwargs.pop("cancel_check", lambda: False), emit=kwargs.pop("emit", lambda *_: None), **kwargs,
    )


def checkpoint_bytes(path, thread):
    with sqlite3.connect(path) as db:
        return (
            db.execute("SELECT * FROM checkpoints WHERE thread_id=? ORDER BY checkpoint_id", (thread,)).fetchall(),
            db.execute("SELECT * FROM writes WHERE thread_id=? ORDER BY checkpoint_id,task_id,idx", (thread,)).fetchall(),
        )


async def test_cooperative_pause_after_paid_extract_resumes_without_extracting_again(tmp_path):
    paused = False

    def pause():
        nonlocal paused
        paused = True

    first = RecordingProvider(pause_after_extract=pause)
    with pytest.raises(AnalysisCancelled):
        await execute(tmp_path, first, cancel_check=lambda: paused)
    original_bytes = checkpoint_bytes(tmp_path / "resume.sqlite3", "original")
    resumed = RecordingProvider()
    result = await execute(tmp_path, resumed, "continued", resume_from_thread_id="original")
    assert [call[0] for call in resumed.calls] == ["authority_analyst", "cross_validation_analyst", "arbitrate"]
    assert "extract" in result["execution_provenance"]["reused_nodes"]
    assert checkpoint_bytes(tmp_path / "resume.sqlite3", "original") == original_bytes


async def test_parallel_branch_failure_reuses_successful_sibling_without_peer_input(tmp_path):
    first = RecordingProvider(fail_cross=True)
    with pytest.raises(ModelOutputInvalid):
        await execute(tmp_path, first)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "resume.sqlite3")) as saver:
        source = await saver.aget_tuple({"configurable": {"thread_id": "original"}})
    assert any(channel == "authority_analysis" for _, channel, _ in source.pending_writes)
    original_snapshot_id = source.checkpoint["channel_values"]["evidence_snapshot_id"]
    events = []
    resumed = RecordingProvider(record_usage=True)
    result = await execute(tmp_path, resumed, "continued", resume_from_thread_id="original", emit=lambda *args: events.append(args))
    assert [call[0] for call in resumed.calls] == ["cross_validation_analyst", "arbitrate"]
    analyst_call = resumed.calls[0]
    assert analyst_call[1] == QUESTION
    assert "authority_analysis" not in analyst_call[1] and "analyses" not in analyst_call[1]
    assert result["evidence_snapshot_id"] != original_snapshot_id
    assert all(analysis["evidence_snapshot_id"] == result["evidence_snapshot_id"] for analysis in result["analyses"])
    assert "authority_analyst" in result["execution_provenance"]["reused_nodes"]
    assert [record["operation"] for record in result["usage"]] == ["cross_validation_analyst", "arbitrate"]
    assert not any(kind == "tool_started" and data["name"] == "authority_analyst" for kind, data in events)


async def test_authority_failure_reuses_completed_cross_branch_with_new_snapshot_and_truthful_usage(tmp_path):
    """The observed A-fails/B-completes case must be symmetric to B failure."""
    cross_finished = asyncio.Event()

    class FailingAuthorityProvider(RecordingProvider):
        async def analyze(self, role, question, evidence, snapshot_id):
            if role == "authority_analyst":
                self.calls.append((role, copy.deepcopy(question), copy.deepcopy(evidence), snapshot_id))
                self.account(role)
                await cross_finished.wait()
                await asyncio.sleep(0.02)
                raise ModelOutputInvalid("判断必须具有证据或已声明的前提。", stage=role, details={"check": "ungrounded_judgment"})
            result = await super().analyze(role, question, evidence, snapshot_id)
            cross_finished.set()
            return result

    reports = []
    with pytest.raises(ModelOutputInvalid):
        await execute(tmp_path, FailingAuthorityProvider(record_usage=True), usage_callback=reports.append)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "resume.sqlite3")) as saver:
        source = await saver.aget_tuple({"configurable": {"thread_id": "original"}})
    assert any(channel == "cross_analysis" for _, channel, _ in source.pending_writes)
    source_snapshot = source.checkpoint["channel_values"]["evidence_snapshot_id"]
    original_bytes = checkpoint_bytes(tmp_path / "resume.sqlite3", "original")
    events, resumed = [], RecordingProvider(record_usage=True)
    result = await execute(tmp_path, resumed, "continued", resume_from_thread_id="original", emit=lambda *args: events.append(args))
    assert [call[0] for call in resumed.calls] == ["authority_analyst", "arbitrate"]
    assert resumed.calls[0][1] == QUESTION
    assert "analyses" not in resumed.calls[0][1] and "cross_analysis" not in resumed.calls[0][1]
    assert result["evidence_snapshot_id"] != source_snapshot
    assert all(a["evidence_snapshot_id"] == result["evidence_snapshot_id"] for a in result["analyses"])
    assert "cross_validation_analyst" in result["execution_provenance"]["reused_nodes"]
    assert result["analyses"][1]["execution_provenance"]["reused"] is True
    assert [row["operation"] for row in result["usage"]] == ["authority_analyst", "arbitrate"]
    assert {row["operation"] for row in reports[-1]} == {"extract", "authority_analyst", "cross_validation_analyst"}
    assert not any(kind == "tool_started" and data["name"] == "cross_validation_analyst" for kind, data in events)
    assert checkpoint_bytes(tmp_path / "resume.sqlite3", "original") == original_bytes


async def test_completed_checkpoint_continuation_makes_zero_model_calls_with_fresh_snapshot_id(tmp_path):
    original = await execute(tmp_path, RecordingProvider())
    before = checkpoint_bytes(tmp_path / "resume.sqlite3", "original")
    resumed = RecordingProvider()
    events = []
    result = await execute(tmp_path, resumed, "continued", resume_from_thread_id="original", emit=lambda *args: events.append(args))
    assert resumed.calls == []
    assert result["usage"] == []
    assert result["final"]["conclusion"] == original["final"]["conclusion"]
    assert result["evidence_snapshot_id"] != original["evidence_snapshot_id"]
    assert result["execution_provenance"]["source_evidence_snapshot_id"] == original["evidence_snapshot_id"]
    assert checkpoint_bytes(tmp_path / "resume.sqlite3", "original") == before
    assert [kind for kind, _ in events] == ["checkpoint_reused", "source_coverage_progress"]
    assert events[1][1]["reused"] is True and events[1][1]["complete"] is True
    assert events[1][1]["scanned_paragraphs"] == len(PARAGRAPHS)


async def test_immediate_async_cancellation_keeps_finished_parallel_branch(tmp_path):
    cross_started, authority_completed = asyncio.Event(), asyncio.Event()

    class BlockingProvider(RecordingProvider):
        async def analyze(self, role, question, evidence, snapshot_id):
            if role == "cross_validation_analyst":
                self.calls.append((role,))
                cross_started.set()
                await asyncio.Event().wait()
            return await super().analyze(role, question, evidence, snapshot_id)

    def emit(kind, data):
        if kind == "tool_completed" and data["name"] == "authority_analyst":
            authority_completed.set()

    task = asyncio.create_task(execute(tmp_path, BlockingProvider(), emit=emit))
    await asyncio.wait_for(cross_started.wait(), 2)
    await asyncio.wait_for(authority_completed.wait(), 2)
    # Observe the durable sibling write before cancelling the in-flight peer.
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "resume.sqlite3")) as saver:
        for _ in range(100):
            state = await saver.aget_tuple({"configurable": {"thread_id": "original"}})
            if state and any(channel == "authority_analysis" for _, channel, _ in state.pending_writes or []):
                break
            await asyncio.sleep(0.005)
        else:
            pytest.fail("completed authority branch was not checkpointed")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    resumed = RecordingProvider()
    result = await execute(tmp_path, resumed, "continued", resume_from_thread_id="original")
    assert [call[0] for call in resumed.calls] == ["cross_validation_analyst", "arbitrate"]
    assert result["analyses"][0]["execution_provenance"]["reused"] is True
    assert result["analyses"][1]["execution_provenance"]["reused"] is False
    assert result["final"]["execution_provenance"]["reused"] is False


async def test_pause_at_initial_input_checkpoint_resumes_start_without_claiming_reuse(tmp_path):
    await execute(tmp_path, RecordingProvider())
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "resume.sqlite3")) as saver:
        checkpoints = [state async for state in saver.alist({"configurable": {"thread_id": "original"}})]
        initial = next(state for state in checkpoints if "__start__" in state.checkpoint["channel_values"])
        await saver.aput(
            {"configurable": {"thread_id": "early-pause", "checkpoint_ns": ""}},
            copy.deepcopy(initial.checkpoint), initial.metadata, initial.checkpoint["channel_versions"],
        )
    provider = RecordingProvider()
    result = await execute(tmp_path, provider, "continued", resume_from_thread_id="early-pause")
    assert [call[0] for call in provider.calls] == ["extract", "authority_analyst", "cross_validation_analyst", "arbitrate"]
    assert result["execution_provenance"]["reused_nodes"] == []
    assert result["final"]["execution_provenance"]["reused"] is False


@pytest.mark.parametrize("modified", [
    {"question": {**QUESTION, "question": "另一个问题？"}},
    {"paragraphs": [{**PARAGRAPHS[0], "text": "来源变更"}]},
    {"retrieval_limit": 2},
    {"include_baseline": True},
    {"excluded_spans": {(PARAGRAPHS[0]["id"], 3, 7, "失联3人")}},
    {"manual_evidence": [{"paragraph_id": PARAGRAPHS[0]["id"], "quote": "失联3人", "fact_des": "人工", "argument": {}, "entities": []}]},
    {"interpretation_overrides": {"unknown": {"fact_des": "修改"}}},
])
async def test_resume_rejects_changed_runtime_inputs_before_any_model_call(tmp_path, modified):
    await execute(tmp_path, RecordingProvider())
    provider = RecordingProvider()
    with pytest.raises(ModelOutputInvalid) as failure:
        await execute(tmp_path, provider, "continued", resume_from_thread_id="original", **modified)
    assert failure.value.code == "resume_checkpoint_invalid"
    assert provider.calls == []


async def test_trace_reports_actual_local_and_model_calls_and_usage_before_arbitration(tmp_path):
    events, reports = [], []
    provider = RecordingProvider(record_usage=True)
    result = await execute(tmp_path, provider, emit=lambda *args: events.append(args), usage_callback=reports.append)
    starts = {data["call_id"]: data for kind, data in events if kind == "tool_started"}
    completed = {data["call_id"]: data for kind, data in events if kind == "tool_completed"}
    assert starts.keys() == completed.keys()
    assert len(starts) == 7
    assert {data["name"] for data in starts.values() if data["kind"] == "model"} == {
        "evidence_batch_extraction", "authority_analyst", "cross_validation_analyst", "arbitrate",
    }
    batch_calls = [data for data in starts.values() if data["name"] == "evidence_batch_extraction"]
    assert len(batch_calls) == 1 and batch_calls[0]["batch_index"] == batch_calls[0]["batch_count"] == 1
    assert batch_calls[0]["paragraph_count"] == len(PARAGRAPHS)
    assert all(data["elapsed_ms"] >= 0 and data["completed_at"] for data in completed.values())
    assert reports[0] == [{"operation": "extract", "total_tokens": 10}]
    assert reports[-1] == result["usage"]
    assert len(reports) >= 3
    serialized = json.dumps([data for kind, data in events if kind.startswith("tool_")], ensure_ascii=False)
    assert "失联3人" not in serialized and "通报中" not in serialized


async def test_failed_tool_trace_contains_only_safe_failure_code(tmp_path):
    class FailingProvider(RecordingProvider):
        async def extract(self, question, paragraphs):
            raise RuntimeError("sensitive-provider-body sk-do-not-emit")

    events = []
    with pytest.raises(RuntimeError):
        await execute(tmp_path, FailingProvider(), emit=lambda *args: events.append(args))
    failed = next(data for kind, data in events if kind == "tool_failed")
    assert failed["name"] == "evidence_batch_extraction" and failed["code"] == "tool_execution_failed"
    assert failed["batch_index"] == 1 and failed["batch_count"] == 1
    assert "sensitive" not in json.dumps(events) and "sk-do-not-emit" not in json.dumps(events)
