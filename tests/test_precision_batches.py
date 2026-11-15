"""Bounded deterministic review fixtures; no competition gold or paid models."""

from copy import deepcopy
import json

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.provider import ModelOutputInvalid
from app.workflows.analysis import AnalysisCancelled, _plan_precision_batches
from app.workflows import analysis as workflow_module
from test_full_source_workflow import SelectionProvider, paragraph, run


def test_precision_partition_preserves_all_500_candidates_and_stable_equal_quote_groups():
    rows = [{"id": f"e{i}", "quote": f"事实{i}", "argument": {}} for i in range(500)]
    rows[80]["quote"] = rows[1]["quote"]
    original = deepcopy(rows)
    batches = _plan_precision_batches(rows)
    flattened = [identity for batch in batches for identity in batch]
    assert len(flattened) == len(set(flattened)) == 500
    assert set(flattened) == {row["id"] for row in rows}
    assert all(len(batch) <= 48 for batch in batches)
    assert any("e1" in batch and "e80" in batch for batch in batches)
    assert batches == _plan_precision_batches(rows)
    assert rows == original


def test_precision_partition_bounds_json_characters_and_keeps_oversized_single_candidate():
    rows = [{"id": f"e{i}", "quote": str(i) + "甲" * 1000} for i in range(12)]
    batches = _plan_precision_batches(rows, max_codepoints=3100)
    by_id = {row["id"]: row for row in rows}
    for batch in batches:
        assert len(json.dumps([by_id[identity] for identity in batch], ensure_ascii=False, sort_keys=True, separators=(",", ":"))) <= 3100
    huge = {"id": "huge", "quote": "甲" * 40_001}
    batches = _plan_precision_batches([rows[0], huge, rows[1]])
    assert ["huge"] in batches
    assert {identity for batch in batches for identity in batch} == {"e0", "e1", "huge"}
    with pytest.raises(ValueError, match="duplicate_precision_candidate"):
        _plan_precision_batches([rows[0], rows[0]])


def test_overlapping_source_windows_are_adjacent_without_merging_their_facts():
    rows = [
        {"id": "random-a", "raw_doc_id": "d2", "raw_para_id": "p1", "start_cp": 0, "end_cp": 12, "quote": "异源陈述"},
        {"id": "random-z", "raw_doc_id": "d1", "raw_para_id": "p1", "start_cp": 5, "end_cp": 14, "quote": "窗口二"},
        {"id": "random-m", "raw_doc_id": "d1", "raw_para_id": "p1", "start_cp": 0, "end_cp": 10, "quote": "窗口一"},
    ]
    original = deepcopy(rows)
    assert _plan_precision_batches(rows, max_candidates=2) == [["random-m", "random-z"], ["random-a"]]
    assert rows == original


class BatchedReview(SelectionProvider):
    def __init__(self, *, pause_after=None, pause=None, fail_on=None):
        super().__init__()
        self.pause_after, self.pause_callback, self.fail_on = pause_after, pause, fail_on

    def selection(self, evidence):
        return {"selected_evidence_ids": [row["id"] for row in evidence], "assessments": [
            {"evidence_id": row["id"], "relation": "support", "reason": "直接陈述问题相关事实", "equivalent_to": None}
            for row in evidence]}

    async def select_evidence(self, question, evidence):
        self.selection_calls.append(deepcopy(evidence))
        self.account("select_evidence")
        output = self.selection(evidence)
        if self.fail_on == len(self.selection_calls):
            output["selected_evidence_ids"] = ["PRIVATE_UNKNOWN_REFERENCE"]
        if self.pause_after == len(self.selection_calls):
            self.pause_callback()
        return output


async def test_precision_runs_all_bounded_batches_and_completes_verify_only_once(tmp_path):
    provider, events = BatchedReview(), []
    result = await run(tmp_path, provider, [paragraph(i) for i in range(105)],
                       emit=lambda kind, data: events.append((kind, deepcopy(data))))
    assert [len(batch) for batch in provider.selection_calls] == [48, 48, 9]
    assert len(result["candidate_evidence"]) == len(result["evidence"]) == 105
    review = result["evidence_selection"]
    assert review["candidate_count"] == review["selected_count"] == 105
    assert review["rejected_count"] == 0
    assert review["batch_count"] == review["completed_batches"] == 3
    assert len(review["assessments"]) == len({row["evidence_id"] for row in review["assessments"]}) == 105
    ends = [data for kind, data in events if kind == "node_completed" and data["node"] == "verify"]
    assert len(ends) == 1 and ends[0]["verified_count"] == 105
    progress = [data for kind, data in events if kind == "evidence_precision_progress"]
    assert [data["reviewed_count"] for data in progress] == [48, 96, 105]
    assert [data["complete"] for data in progress] == [False, False, True]
    calls = [data for kind, data in events if kind == "tool_completed" and data["name"] == "evidence_precision_review"]
    assert [data["summary"]["batch_index"] for data in calls] == [1, 2, 3]
    assert [data["summary"]["candidate_count"] for data in calls] == [48, 48, 9]


async def test_successful_precision_batches_checkpoint_and_resume_without_repeating_paid_work(tmp_path):
    sources, stopped = [paragraph(i) for i in range(105)], False

    def pause():
        nonlocal stopped
        stopped = True

    first = BatchedReview(pause_after=2, pause=pause)
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, first, sources, cancel_check=lambda: stopped)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "scan.sqlite3")) as saver:
        stored = await saver.aget_tuple({"configurable": {"thread_id": "scan"}})
    state = stored.checkpoint["channel_values"]
    assert state["precision_batch_index"] == 2
    assert len(state["precision_assessments"]) == 96
    assert "evidence_selection" not in state  # Partial review is not the completed verify stage.
    assert len(state["candidate_evidence"]) == 105
    resumed = BatchedReview()
    result = await run(tmp_path, resumed, sources, thread_id="review-resumed", resume_from_thread_id="scan")
    assert [len(batch) for batch in resumed.selection_calls] == [9]
    previous = {row["id"] for batch in first.selection_calls for row in batch}
    later = {row["id"] for batch in resumed.selection_calls for row in batch}
    assert not previous & later
    assert len(previous | later) == 105
    assert not any(name == "extract" for name, _ in resumed.calls)
    assert "verify" not in result["execution_provenance"]["reused_nodes"]
    assert len(result["evidence"]) == 105


async def test_precision_failed_later_batch_preserves_prior_ledger_and_resumes_only_unfinished_batches(tmp_path):
    sources, events = [paragraph(i) for i in range(105)], []
    first = BatchedReview(fail_on=2)
    with pytest.raises(ModelOutputInvalid) as failure:
        await run(tmp_path, first, sources, emit=lambda kind, data: events.append((kind, deepcopy(data))))
    assert failure.value.stage == "verify"
    assert failure.value.details["repair_exhausted"] is True
    assert len(first.selection_repairs) == 1
    assert len(first.selection_repairs[0][0]) == 48
    assert not any(kind == "node_completed" and data["node"] == "verify" for kind, data in events)
    resumed = BatchedReview()
    result = await run(tmp_path, resumed, sources, thread_id="review-retried", resume_from_thread_id="scan")
    assert [len(batch) for batch in resumed.selection_calls] == [48, 9]
    first_completed = {row["id"] for row in first.selection_calls[0]}
    assert not first_completed & {row["id"] for batch in resumed.selection_calls for row in batch}
    assert len(result["evidence_selection"]["assessments"]) == 105


async def test_precision_multi_batch_manual_sources_and_exclusions_keep_user_intent(tmp_path):
    sources, provider = [paragraph(i) for i in range(105)], BatchedReview()
    manual = {"paragraph_id": "p104", "quote": sources[104]["text"], "start_cp": 0,
              "end_cp": len(sources[104]["text"]), "fact_des": "明确人工保留", "argument": {}, "entities": []}
    excluded = {("p103", 0, len(sources[103]["text"]), sources[103]["text"])}
    result = await run(tmp_path, provider, sources, manual_evidence=[manual], excluded_spans=excluded)
    assert [len(batch) for batch in provider.selection_calls] == [48, 48, 7]
    assert all(row["interpretation_origin"] == "model" for batch in provider.selection_calls for row in batch)
    assert len(result["candidate_evidence"]) == len(result["evidence"]) == 104
    assert "p103" not in {row["paragraph_id"] for row in result["evidence"]}
    assert next(row for row in result["evidence"] if row["paragraph_id"] == "p104")["interpretation_origin"] == "human"
    assert result["evidence_selection"]["candidate_count"] == 103
    assert result["evidence_selection"]["manual_preserved_count"] == 1


async def test_one_source_many_precision_batches_is_not_stopped_by_legacy_recursion_limit(tmp_path, monkeypatch):
    original = workflow_module._plan_precision_batches
    monkeypatch.setattr(workflow_module, "_plan_precision_batches", lambda evidence: original(evidence, max_candidates=1))
    quotes = [f"记录{i}。" for i in range(55)]

    class ManyFacts(BatchedReview):
        async def extract(self, question, sources):
            self.account("extract")
            return [{"paragraph_id": sources[0]["id"], "quote": quote, "fact_des": "材料原子事实",
                     "argument": {}, "entities": []} for quote in quotes]

    provider = ManyFacts()
    result = await run(tmp_path, provider, [paragraph(0, "".join(quotes))])
    assert len(provider.selection_calls) == 55
    assert all(len(batch) == 1 for batch in provider.selection_calls)
    assert len(result["evidence"]) == 55
    assert result["evidence_selection"]["completed_batches"] == 55


async def test_precision_duplicate_target_must_be_retained_in_the_same_batch(tmp_path):
    class CrossBatchDuplicate(BatchedReview):
        def selection(self, evidence):
            if not getattr(self, "prior_representative", None):
                self.prior_representative = evidence[0]["id"]
                return super().selection(evidence)
            return {"selected_evidence_ids": [], "assessments": [
                {"evidence_id": row["id"], "relation": "duplicate", "reason": "声称与上一批相同",
                 "equivalent_to": self.prior_representative} for row in evidence]}

    provider = CrossBatchDuplicate()
    with pytest.raises(ModelOutputInvalid) as failure:
        await run(tmp_path, provider, [paragraph(i, "同一段文字。") for i in range(49)])
    assert failure.value.stage == "verify"
    assert failure.value.details["check"] == "evidence_selection"
    assert [len(batch) for batch in provider.selection_calls] == [48, 1]
    assert len(provider.selection_repairs) == 1
