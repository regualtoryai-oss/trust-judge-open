"""Synthetic complete-source scans and explicit relevance review contracts.

All providers here are deterministic and explicitly injected. No competition
answers, live credentials or network services are consulted by these tests.
"""

from copy import deepcopy

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents import provider as provider_module
from app.agents.provider import ModelOutputInvalid
from app.workflows.analysis import AnalysisCancelled, run_analysis


QUESTION = {"id": "full-source-question", "question": "目标设施是否运行？"}


def paragraph(index, text=None):
    text = text or f"其他记录{index}。"
    return {"id": f"p{index}", "document_id": f"d{index}", "raw_doc_id": f"doc{index}",
            "raw_para_id": "p1", "text": text, "start_cp": 0, "end_cp": len(text)}


def unresolved(arbitration=False):
    value = {"decision": "undetermined", "conclusion": "材料尚不能确认。", "judgments": [],
             "supporting_evidence_ids": [], "conflicting_evidence_ids": []}
    return {**value, "conflicts": []} if arbitration else value


class ScanProvider:
    def __init__(self, *, pause_after=None, callback=None, bad_batch=None, repair_bad=False):
        self.calls, self.repairs = [], []
        self.pause_after, self.callback = pause_after, callback
        self.bad_batch, self.repair_bad = bad_batch, repair_bad

    def account(self, operation):
        provider_module._usage_records.get().append({"operation": operation, "total_tokens": 5})

    def rows(self, sources):
        return [{"paragraph_id": row["id"], "quote": row["text"], "fact_des": "材料中的表述",
                 "argument": {}, "entities": []} for row in sources]

    async def extract(self, question, sources):
        self.calls.append(("extract", deepcopy(sources)))
        self.account("extract")
        value = self.rows(sources)
        if self.bad_batch == len([call for call in self.calls if call[0] == "extract"]):
            value[0]["quote"] = "PRIVATE_INVENTED_QUOTE"
        if self.pause_after == len([call for call in self.calls if call[0] == "extract"]):
            self.callback()
        return value

    async def repair_extract(self, question, sources, invalid_output, diagnostics):
        self.repairs.append((deepcopy(sources), deepcopy(invalid_output), deepcopy(diagnostics)))
        self.account("repair:extract")
        return deepcopy(invalid_output["evidence"]) if self.repair_bad else self.rows(sources)

    async def analyze(self, role, question, evidence, snapshot_id):
        self.calls.append((role, deepcopy(evidence)))
        self.account(role)
        return unresolved()

    async def arbitrate(self, question, evidence, analyses):
        self.calls.append(("arbitrate", deepcopy(evidence)))
        self.account("arbitrate")
        return unresolved(True)


async def run(tmp_path, provider, sources, **kwargs):
    return await run_analysis(QUESTION, sources, provider, test_only_legacy_precision=True, checkpoint_path=tmp_path / "scan.sqlite3",
        thread_id=kwargs.pop("thread_id", "scan"), cancel_check=kwargs.pop("cancel_check", lambda: False),
        emit=kwargs.pop("emit", lambda *_: None), **kwargs)


async def test_complete_scan_covers_zero_matches_and_exceeds_old_top16_and_default_recursion(tmp_path):
    sources = [paragraph(i) for i in range(55)]
    # The last paragraph has no lexical overlap with the question. The planner
    # must still send it to the model; its selection does not prove relevance.
    original, events, provider = deepcopy(sources), [], ScanProvider()
    result = await run(tmp_path, provider, sources, source_batch_codepoints=9,
                       emit=lambda kind, data: events.append((kind, deepcopy(data))))
    seen = [row for name, rows in provider.calls if name == "extract" for row in rows]
    assert {row["id"] for row in seen} == {row["id"] for row in sources}
    assert len(seen) == len(sources) == len(result["evidence"]) == 55
    assert sources == original
    coverage = result["source_coverage"]
    assert coverage["complete"] is True
    assert coverage["completed_batches"] == coverage["batch_count"] == 55
    assert coverage["scanned_paragraphs"] == coverage["total_paragraphs"] == 55
    assert coverage["scanned_codepoints"] == coverage["total_codepoints"] == sum(len(row["text"]) for row in sources)
    assert len([record for record in result["usage"] if record["operation"] == "extract"]) == 55
    planned = next(data for kind, data in events if kind == "source_coverage_planned")
    assert planned["complete"] is False and planned["completed_batches"] == planned["scanned_paragraphs"] == 0
    progress = [data for kind, data in events if kind == "source_coverage_progress"]
    assert [row["completed_batches"] for row in progress] == list(range(1, 56))
    ends = [data for kind, data in events if kind == "node_completed" and data["node"] == "extract"]
    assert len(ends) == 1 and ends[0]["candidate_count"] == 55
    starts = [data for kind, data in events if kind == "tool_started" and data["name"] == "evidence_batch_extraction"]
    completions = [data for kind, data in events if kind == "tool_completed" and data["name"] == "evidence_batch_extraction"]
    assert len(starts) == len(completions) == 55
    assert [row["batch_index"] for row in starts] == list(range(1, 56))
    assert {row["call_id"] for row in starts} == {row["call_id"] for row in completions}


async def test_pause_retains_each_paid_batch_and_resume_reports_cumulative_coverage(tmp_path):
    sources, stopped, events = [paragraph(i) for i in range(5)], False, []

    def pause():
        nonlocal stopped
        stopped = True

    first = ScanProvider(pause_after=2, callback=pause)
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, first, sources, source_batch_codepoints=9, cancel_check=lambda: stopped)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "scan.sqlite3")) as saver:
        prior = await saver.aget_tuple({"configurable": {"thread_id": "scan"}})
    assert prior.checkpoint["channel_values"]["source_coverage"]["completed_batches"] == 2
    resumed = ScanProvider()
    result = await run(tmp_path, resumed, sources, source_batch_codepoints=9, thread_id="continued",
        resume_from_thread_id="scan", emit=lambda kind, data: events.append((kind, deepcopy(data))))
    first_ids = {row["id"] for name, rows in first.calls if name == "extract" for row in rows}
    resumed_ids = {row["id"] for name, rows in resumed.calls if name == "extract" for row in rows}
    assert not first_ids & resumed_ids
    assert first_ids | resumed_ids == {row["id"] for row in sources}
    assert result["execution_provenance"]["reused_source_batches"] == 2
    assert "extract" not in result["execution_provenance"]["reused_nodes"]
    reused = next(data for kind, data in events if kind == "source_coverage_progress" and data.get("reused"))
    assert reused["completed_batches"] == reused["scanned_paragraphs"] == 2
    assert reused["complete"] is False
    assert result["source_coverage"]["completed_batches"] == 5
    assert len([row for row in result["usage"] if row["operation"] == "extract"]) == 3
    assert not any(kind == "node_completed" and data["node"] == "extract" and data.get("reused") for kind, data in events)


async def test_batch_failure_does_not_mark_scanned_or_discard_previous_paid_batches(tmp_path):
    sources, events = [paragraph(i) for i in range(3)], []
    first = ScanProvider(bad_batch=2, repair_bad=True)
    with pytest.raises(ModelOutputInvalid) as failure:
        await run(tmp_path, first, sources, source_batch_codepoints=9,
                  emit=lambda kind, data: events.append((kind, deepcopy(data))))
    assert failure.value.stage == "verify"
    assert len(first.repairs) == 1
    assert len(first.repairs[0][0]) == 1
    progress = [data for kind, data in events if kind == "source_coverage_progress"]
    assert [data["completed_batches"] for data in progress] == [1]
    assert not any(kind == "node_completed" and data["node"] == "extract" for kind, data in events)
    resumed = ScanProvider()
    result = await run(tmp_path, resumed, sources, source_batch_codepoints=9,
                       thread_id="retry", resume_from_thread_id="scan")
    assert len([row for row in resumed.calls if row[0] == "extract"]) == 2
    assert len(result["evidence"]) == 3 and result["source_coverage"]["complete"] is True


async def test_quote_repair_receives_only_the_current_bounded_source_batch(tmp_path):
    sources = [paragraph(i, f"记录{i}：" + "甲" * 29_990 + "。") for i in range(3)]
    provider = ScanProvider(bad_batch=2)
    result = await run(tmp_path, provider, sources, source_batch_codepoints=32_000, source_batch_spans=256)
    assert len(provider.repairs) == 1 and len(provider.repairs[0][0]) == 1
    assert sum(len(row["text"]) for row in provider.repairs[0][0]) < 32_000
    assert len(result["evidence"]) == 3
    assert result["source_coverage"]["total_codepoints"] > 64_000


async def test_hard_oversized_paragraph_fails_before_model_with_accurate_input_error(tmp_path):
    provider = ScanProvider()
    with pytest.raises(ModelOutputInvalid) as failure:
        await run(tmp_path, provider, [paragraph(0, "甲" * 64_001)])
    assert failure.value.stage == "retrieve"
    assert failure.value.code == "model_input_too_large"
    assert failure.value.details == {"check": "source_catalog_limit"}
    assert provider.calls == []


class SelectionProvider(ScanProvider):
    def __init__(self, *, pause=None, invalid=False, repair=False):
        super().__init__()
        self.selection_calls, self.selection_repairs = [], []
        self.pause, self.invalid, self.repair = pause, invalid, repair

    def selection(self, evidence):
        selected = evidence[:1]
        return {"selected_evidence_ids": [row["id"] for row in selected], "assessments": [
            {"evidence_id": row["id"], "relation": "support" if row in selected else "background",
             "reason": "回答本题" if row in selected else "仅背景关联", "equivalent_to": None}
            for row in evidence]}

    async def select_evidence(self, question, evidence):
        self.selection_calls.append(deepcopy(evidence))
        self.account("select_evidence")
        output = self.selection(evidence)
        if self.invalid:
            output["selected_evidence_ids"] = ["PRIVATE_UNKNOWN_REFERENCE"]
        if self.pause:
            self.pause()
        return output

    async def repair_evidence_selection(self, question, evidence, invalid_output, diagnostics):
        self.selection_repairs.append((deepcopy(evidence), deepcopy(diagnostics)))
        self.account("repair:select_evidence")
        return self.selection(evidence) if self.repair else deepcopy(invalid_output)


async def test_precision_review_keeps_auditable_pool_preserves_manual_and_respects_exclusion(tmp_path):
    sources, provider, events = [paragraph(i) for i in range(4)], SelectionProvider(), []
    manual = {"paragraph_id": "p2", "quote": sources[2]["text"], "start_cp": 0,
              "end_cp": len(sources[2]["text"]), "fact_des": "人工意见", "argument": {}, "entities": []}
    excluded = {("p3", 0, len(sources[3]["text"]), sources[3]["text"])}
    result = await run(tmp_path, provider, sources, manual_evidence=[manual], excluded_spans=excluded,
                       emit=lambda kind, data: events.append((kind, deepcopy(data))))
    assert len(provider.selection_calls) == 1
    assert {row["paragraph_id"] for row in provider.selection_calls[0]} == {"p0", "p1"}
    assert {row["paragraph_id"] for row in result["candidate_evidence"]} == {"p0", "p1", "p2"}
    selected = set(result["evidence_selection"]["selected_evidence_ids"])
    assert {row["id"] for row in result["evidence"] if row["interpretation_origin"] == "model"} == selected
    assert next(row for row in result["evidence"] if row["paragraph_id"] == "p2")["interpretation_origin"] == "human"
    assert result["evidence_selection"]["selected_count"] == result["evidence_selection"]["rejected_count"] == 1
    assert result["evidence_selection"]["manual_preserved_count"] == 1
    assert len(result["evidence_selection"]["assessments"]) == 2
    assert len([1 for kind, data in events if kind == "tool_started" and data["name"] == "evidence_precision_review"]) == 1
    assert all(call[1] == result["evidence"] for call in provider.calls if call[0] in {"authority_analyst", "cross_validation_analyst"})


@pytest.mark.parametrize("repair", [True, False])
async def test_precision_review_requires_complete_safe_dispositions_and_at_most_one_repair(tmp_path, repair):
    provider = SelectionProvider(invalid=True, repair=repair)
    if repair:
        result = await run(tmp_path, provider, [paragraph(0), paragraph(1)])
        assert len(result["evidence"]) == 1
    else:
        with pytest.raises(ModelOutputInvalid) as failure:
            await run(tmp_path, provider, [paragraph(0), paragraph(1)])
        assert failure.value.stage == "verify"
        assert failure.value.details["repair_exhausted"] is True
        assert not any(call[0] == "arbitrate" for call in provider.calls)
        assert "PRIVATE_UNKNOWN_REFERENCE" not in str(failure.value)
    assert len(provider.selection_calls) == len(provider.selection_repairs) == 1


async def test_successful_paid_precision_review_checkpoint_is_not_repeated_after_pause(tmp_path):
    stopped, sources = False, [paragraph(0), paragraph(1)]

    def pause():
        nonlocal stopped
        stopped = True

    first = SelectionProvider(pause=pause)
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, first, sources, cancel_check=lambda: stopped)
    resumed = SelectionProvider()
    result = await run(tmp_path, resumed, sources, thread_id="review-continued", resume_from_thread_id="scan")
    assert resumed.selection_calls == []
    assert not any(call[0] == "extract" for call in resumed.calls)
    assert len(result["evidence"]) == 1 and result["evidence_selection"]["applied"] is True
    assert result["execution_provenance"]["reused_nodes"] == ["extract", "retrieve", "verify"]
    assert [row["operation"] for row in result["usage"]] == ["authority_analyst", "cross_validation_analyst", "arbitrate"]


async def test_resume_refuses_changed_scan_budgets_before_any_new_paid_request(tmp_path):
    sources = [paragraph(0)]
    await run(tmp_path, ScanProvider(), sources)
    provider = ScanProvider()
    with pytest.raises(ModelOutputInvalid) as failure:
        await run(tmp_path, provider, sources, source_batch_codepoints=32_000,
                  thread_id="changed-plan", resume_from_thread_id="scan")
    assert failure.value.code == "resume_checkpoint_invalid" and provider.calls == []


async def test_empty_source_pool_has_zero_paid_extraction_and_explicit_complete_empty_coverage(tmp_path):
    provider = ScanProvider()
    result = await run(tmp_path, provider, [])
    assert not any(call[0] == "extract" for call in provider.calls)
    assert result["evidence"] == [] and result["final"]["decision"] == "undetermined"
    assert result["source_coverage"]["complete"] is True
    assert result["source_coverage"]["batch_count"] == result["source_coverage"]["scanned_paragraphs"] == 0


async def test_same_quote_multiple_atomic_interpretations_remain_visible_to_review_and_analysts(tmp_path):
    class MultipleFacts(SelectionProvider):
        async def extract(self, question, sources):
            rows = await super().extract(question, sources)
            first = {**rows[0], "fact_des": "运行状态仍待确认", "argument": {"主体": "泵站"}}
            second = {**rows[0], "fact_des": "损失金额仍待确认", "argument": {"数值口径": "未核实"}}
            return [first, second, deepcopy(second)]

    provider = MultipleFacts()
    result = await run(tmp_path, provider, [paragraph(0, "泵站状态及损失均待核实。")])
    assert len(result["evidence"]) == 1
    item = result["evidence"][0]
    assert item["fact_des"] == "运行状态仍待确认"
    assert len(item["additional_interpretations"]) == 1
    assert item["additional_interpretations"][0]["fact_des"] == "损失金额仍待确认"
    assert provider.selection_calls[0][0]["additional_interpretations"] == item["additional_interpretations"]
    assert all(call[1][0]["additional_interpretations"] == item["additional_interpretations"]
               for call in provider.calls if call[0] in {"authority_analyst", "cross_validation_analyst"})


async def test_precision_repair_uses_reasoning_capacity_instead_of_lower_extraction_capacity(tmp_path):
    from app.agents.provider import get_requested_output_budget

    class CapacitySelection(SelectionProvider):
        def __init__(self):
            super().__init__(invalid=True, repair=True)
            self.budgets = []

        def capacity_policy(self, operation):
            return {"initial_tokens": 32_768 if operation == "verify" else 16_384,
                    "max_tokens": 131_072, "retries": 0}

        async def select_evidence(self, *args):
            self.budgets.append(get_requested_output_budget())
            return await super().select_evidence(*args)

        async def repair_evidence_selection(self, *args):
            self.budgets.append(get_requested_output_budget())
            return await super().repair_evidence_selection(*args)

    provider = CapacitySelection()
    result = await run(tmp_path, provider, [paragraph(0), paragraph(1)])
    assert result["evidence_selection"]["applied"] is True
    assert provider.budgets == [32_768, 32_768]
