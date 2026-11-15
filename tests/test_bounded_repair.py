"""Synthetic, source-only adversarial checks for one explicit model repair.

These providers are injected only into tests. No competition answers, external
requests, or application credentials are used to manufacture passing output.
"""

import copy
import json

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents import provider as provider_module
from app.agents.provider import ModelInvocationFailed, ModelOutputInvalid
from app.workflows.analysis import AnalysisCancelled, run_analysis


QUESTION = {"id": "synthetic-repair-question", "question": "合同履行情况能否确认？"}
TEXT = "🧭甲方通报：合同仍在核对，履行情况尚未确认。"
QUOTE = "履行情况尚未确认"
PARAGRAPHS = [{
    "id": "synthetic-repair-paragraph", "document_id": "synthetic-repair-document",
    "raw_doc_id": "synthetic_doc", "raw_para_id": "synthetic_para",
    "text": TEXT, "start_cp": 0, "end_cp": len(TEXT),
}]
ROLES = ["authority_analyst", "cross_validation_analyst", "baseline", "arbitrate"]


def unresolved(*, arbitration=False):
    result = {
        "decision": "undetermined", "conclusion": "给定资料不能确认合同是否已履行。",
        "judgments": [], "supporting_evidence_ids": [], "conflicting_evidence_ids": [],
    }
    return {**result, "conflicts": []} if arbitration else result


def grounded(evidence, *, arbitration=False):
    if not evidence:
        return unresolved(arbitration=arbitration)
    result = {
        "decision": "asserted", "conclusion": "通报称履行情况尚未确认。",
        "judgments": [{
            "id": "J1", "statement": "通报明确表示尚未确认。",
            "evidence_ids": [evidence[0]["id"]], "prerequisite_judgment_ids": [],
        }],
        "supporting_evidence_ids": [evidence[0]["id"]], "conflicting_evidence_ids": [],
    }
    return {**result, "conflicts": []} if arbitration else result


def corrupt(output, failure):
    result = copy.deepcopy(output)
    if failure == "schema":
        return {"decision": "not-a-decision"}
    if failure == "ungrounded_judgment":
        result["judgments"] = [{
            "id": "J1", "statement": "DO_NOT_RENDER_INVALID_MODEL_STATEMENT",
            "evidence_ids": [], "prerequisite_judgment_ids": [],
        }]
    elif failure == "invalid_evidence_reference":
        result["supporting_evidence_ids"] = ["not-in-this-frozen-snapshot"]
    elif failure == "unknown_prerequisite":
        result["judgments"][0]["prerequisite_judgment_ids"] = ["undeclared"]
    elif failure == "prerequisite_cycle":
        result["judgments"] = [
            {"id": "J1", "statement": "第一判断", "evidence_ids": [], "prerequisite_judgment_ids": ["J2"]},
            {"id": "J2", "statement": "第二判断", "evidence_ids": [], "prerequisite_judgment_ids": ["J1"]},
        ]
    elif failure == "duplicate_judgment_id":
        result["judgments"].append(copy.deepcopy(result["judgments"][0]))
    else:
        raise AssertionError(f"Unknown synthetic failure: {failure}")
    return result


class RepairProvider:
    """Deterministic test double with opt-in, recorded repair operations."""

    def __init__(self, *, bad_stage="authority_analyst", failure="ungrounded_judgment", repair_fails=False):
        self.bad_stage = bad_stage
        self.failure = failure
        self.repair_fails = repair_fails
        self.calls = []
        self.repairs = []

    def account(self, operation):
        provider_module._usage_records.get().append({"operation": operation, "total_tokens": 11})

    def extraction(self, paragraphs):
        return [{
            "paragraph_id": paragraphs[0]["id"], "quote": QUOTE,
            "fact_des": "通报表示尚未确认", "argument": {}, "entities": [],
        }]

    async def extract(self, question, paragraphs):
        self.calls.append(("extract", copy.deepcopy(question), copy.deepcopy(paragraphs)))
        self.account("extract")
        output = self.extraction(paragraphs)
        if self.bad_stage == "extract":
            output[0]["quote"] = "合同已经全部履行"
        return output

    async def analyze(self, role, question, evidence, snapshot_id):
        self.calls.append((role, copy.deepcopy(question), copy.deepcopy(evidence), snapshot_id))
        self.account(role)
        output = grounded(evidence)
        return corrupt(output, self.failure) if role == self.bad_stage else output

    async def arbitrate(self, question, evidence, analyses):
        self.calls.append(("arbitrate", copy.deepcopy(question), copy.deepcopy(evidence), copy.deepcopy(analyses)))
        self.account("arbitrate")
        output = unresolved(arbitration=True) if any(a["decision"] == "undetermined" for a in analyses) else grounded(evidence, arbitration=True)
        return corrupt(output, self.failure) if self.bad_stage == "arbitrate" else output

    async def repair_analysis(self, role, question, evidence, snapshot_id, invalid_output, diagnostics, *, analyses=None):
        self.account("repair:" + role)
        self.repairs.append({
            "role": role, "question": copy.deepcopy(question), "evidence": copy.deepcopy(evidence),
            "snapshot_id": snapshot_id, "invalid_output": copy.deepcopy(invalid_output),
            "diagnostics": copy.deepcopy(diagnostics), "analyses": copy.deepcopy(analyses),
        })
        if self.repair_fails:
            return copy.deepcopy(invalid_output)
        return unresolved(arbitration=role == "arbitrate")

    async def repair_extract(self, question, paragraphs, invalid_output, diagnostics):
        self.account("repair:extract")
        self.repairs.append({
            "role": "extract", "question": copy.deepcopy(question), "paragraphs": copy.deepcopy(paragraphs),
            "invalid_output": copy.deepcopy(invalid_output), "diagnostics": copy.deepcopy(diagnostics),
        })
        return copy.deepcopy(invalid_output) if self.repair_fails else self.extraction(paragraphs)


async def execute(tmp_path, provider, **kwargs):
    return await run_analysis(
        kwargs.pop("question", QUESTION), kwargs.pop("paragraphs", PARAGRAPHS), provider,
        test_only_legacy_precision=True, checkpoint_path=tmp_path / "repair.sqlite3", thread_id="repair-test",
        cancel_check=kwargs.pop("cancel_check", lambda: False), emit=kwargs.pop("emit", lambda *_: None), **kwargs,
    )


async def assert_no_final_checkpoint(tmp_path):
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "repair.sqlite3")) as saver:
        snapshots = [item async for item in saver.alist({"configurable": {"thread_id": "repair-test"}})]
    assert snapshots
    assert all("final" not in item.checkpoint["channel_values"] for item in snapshots)


@pytest.mark.parametrize("role", ROLES)
async def test_ungrounded_judgment_can_be_repaired_to_honest_undetermined(tmp_path, role):
    provider, events = RepairProvider(bad_stage=role), []
    result = await execute(tmp_path, provider, include_baseline=role == "baseline", emit=lambda *event: events.append(event))
    repaired = result["baseline"] if role == "baseline" else (
        result["final"] if role == "arbitrate" else next(a for a in result["analyses"] if a["role"] == role)
    )
    assert repaired["decision"] == "undetermined"
    assert repaired["judgments"] == repaired["supporting_evidence_ids"] == []
    assert len(provider.repairs) == 1
    request = provider.repairs[0]
    assert request["question"] == QUESTION
    assert request["evidence"] == result["evidence"]
    assert request["snapshot_id"] == result["evidence_snapshot_id"]
    assert request["analyses"] is None if role != "arbitrate" else len(request["analyses"]) == 2
    assert len([1 for kind, data in events if kind == "model_repair_started" and data["attempt"] == 1]) == 1
    assert len([1 for kind, _ in events if kind == "model_repair_completed"]) == 1
    assert "DO_NOT_RENDER_INVALID_MODEL_STATEMENT" not in json.dumps(events)
    assert sum(record["operation"] == "repair:" + role for record in result["usage"]) == 1


@pytest.mark.parametrize("failure", [
    "schema", "invalid_evidence_reference", "unknown_prerequisite", "prerequisite_cycle", "duplicate_judgment_id",
])
async def test_schema_reference_and_dependency_failures_get_one_validated_repair(tmp_path, failure):
    provider = RepairProvider(failure=failure)
    result = await execute(tmp_path, provider)
    assert result["analyses"][0]["decision"] == "undetermined"
    assert len(provider.repairs) == 1
    assert result["evidence"][0]["quote"] == QUOTE
    assert "not-in-this-frozen-snapshot" not in json.dumps(result)


@pytest.mark.parametrize("failure", ["ungrounded_judgment", "invalid_evidence_reference", "prerequisite_cycle"])
async def test_second_invalid_output_stops_without_third_call_or_final_checkpoint(tmp_path, failure):
    provider, events, reports = RepairProvider(failure=failure, repair_fails=True), [], []
    with pytest.raises(ModelOutputInvalid) as caught:
        await execute(tmp_path, provider, emit=lambda *event: events.append(event), usage_callback=reports.append)
    assert caught.value.stage == "authority_analyst"
    assert len(provider.repairs) == 1
    assert not any(call[0] == "arbitrate" for call in provider.calls)
    assert any(kind == "model_repair_failed" for kind, _ in events)
    assert not any(kind == "model_repair_completed" for kind, _ in events)
    assert {record["operation"] for record in reports[-1]} >= {"authority_analyst", "repair:authority_analyst"}
    await assert_no_final_checkpoint(tmp_path)


async def test_extraction_repair_preserves_original_source_and_requires_exact_quote(tmp_path):
    original = copy.deepcopy(PARAGRAPHS)
    provider = RepairProvider(bad_stage="extract")
    result = await execute(tmp_path, provider)
    assert len(provider.repairs) == 1
    assert provider.repairs[0]["paragraphs"] == provider.calls[0][2]
    assert PARAGRAPHS == original
    evidence = result["evidence"][0]
    assert TEXT[evidence["start_cp"]:evidence["end_cp"]] == evidence["quote"] == QUOTE
    assert [row["operation"] for row in result["usage"]].count("extract") == 1
    assert [row["operation"] for row in result["usage"]].count("repair:extract") == 1


async def test_second_inexact_extraction_never_reaches_analysts(tmp_path):
    provider = RepairProvider(bad_stage="extract", repair_fails=True)
    with pytest.raises(ModelOutputInvalid):
        await execute(tmp_path, provider)
    assert len(provider.repairs) == 1
    assert [call[0] for call in provider.calls] == ["extract"]
    await assert_no_final_checkpoint(tmp_path)


async def test_repair_receives_no_peer_analysis_and_cannot_mutate_frozen_or_peer_input(tmp_path):
    class MutatingRepair(RepairProvider):
        async def repair_analysis(self, role, question, evidence, snapshot_id, invalid_output, diagnostics, *, analyses=None):
            output = await super().repair_analysis(role, question, evidence, snapshot_id, invalid_output, diagnostics, analyses=analyses)
            question["question"] = "MUTATED_REPAIR_QUESTION"
            evidence[0]["quote"] = "MUTATED_REPAIR_QUOTE"
            invalid_output.clear()
            return output

    provider = MutatingRepair()
    result = await execute(tmp_path, provider)
    peer = next(call for call in provider.calls if call[0] == "cross_validation_analyst")
    assert peer[1] == QUESTION and peer[2] == result["evidence"]
    assert peer[3] == result["evidence_snapshot_id"]
    assert provider.repairs[0]["analyses"] is None
    assert result["evidence"][0]["quote"] == QUOTE
    assert "MUTATED_REPAIR" not in json.dumps(result)


@pytest.mark.parametrize("cancel_during_repair", [False, True])
async def test_cancellation_does_not_launch_or_publish_an_unwanted_repair(tmp_path, cancel_during_repair):
    stopped, reports = False, []

    class CancellableProvider(RepairProvider):
        async def analyze(self, role, question, evidence, snapshot_id):
            nonlocal stopped
            result = await super().analyze(role, question, evidence, snapshot_id)
            if role == "authority_analyst" and not cancel_during_repair:
                stopped = True
            return result

        async def repair_analysis(self, *args, **kwargs):
            nonlocal stopped
            result = await super().repair_analysis(*args, **kwargs)
            stopped = True
            return result

    provider = CancellableProvider()
    with pytest.raises(AnalysisCancelled):
        await execute(tmp_path, provider, cancel_check=lambda: stopped, usage_callback=reports.append)
    assert len(provider.repairs) == int(cancel_during_repair)
    assert not any(call[0] == "arbitrate" for call in provider.calls)
    operations = [record["operation"] for record in reports[-1]]
    assert "authority_analyst" in operations
    assert ("repair:authority_analyst" in operations) == cancel_during_repair
    await assert_no_final_checkpoint(tmp_path)


async def test_repair_transport_failure_keeps_usage_and_safe_events_without_retry(tmp_path):
    class BrokenRepair(RepairProvider):
        async def repair_analysis(self, *args, **kwargs):
            await super().repair_analysis(*args, **kwargs)
            raise ModelInvocationFailed("DO_NOT_RENDER_PROVIDER_RESPONSE")

    provider, events, reports = BrokenRepair(), [], []
    with pytest.raises(ModelInvocationFailed):
        await execute(tmp_path, provider, emit=lambda *event: events.append(event), usage_callback=reports.append)
    assert len(provider.repairs) == 1
    assert "DO_NOT_RENDER_PROVIDER_RESPONSE" not in json.dumps(events)
    assert any(record["operation"] == "repair:authority_analyst" for record in reports[-1])
    await assert_no_final_checkpoint(tmp_path)


async def test_valid_outputs_do_not_pay_for_repairs(tmp_path):
    provider, events = RepairProvider(bad_stage=None), []
    result = await execute(tmp_path, provider, emit=lambda *event: events.append(event))
    assert provider.repairs == []
    assert len(result["usage"]) == 4
    assert not any(kind.startswith("model_repair_") for kind, _ in events)


async def test_invalid_human_selected_quote_is_never_rewritten_by_model_repair(tmp_path):
    provider = RepairProvider(bad_stage=None)
    manual = [{
        "paragraph_id": PARAGRAPHS[0]["id"], "quote": "原文中不存在的人工引用",
        "fact_des": "人工观点", "argument": {}, "entities": [],
    }]
    with pytest.raises(ModelOutputInvalid):
        await execute(tmp_path, provider, manual_evidence=manual)
    assert provider.repairs == []
    assert provider.calls == []
    # Invalid deliberate coordinates are rejected before starting the graph;
    # there is no paid response or checkpoint to preserve in this case.
    assert not (tmp_path / "repair.sqlite3").exists()


SECOND_TEXT = "乙方提到编号999，但尚需等待双方确认。"
TWO_PARAGRAPHS = [*PARAGRAPHS, {
    "id": "synthetic-other-paragraph", "document_id": "synthetic-other-document",
    "raw_doc_id": "synthetic_other_doc", "raw_para_id": "synthetic_other_para",
    "text": SECOND_TEXT, "start_cp": 0, "end_cp": len(SECOND_TEXT),
}]
FINAL_FAILURES = [
    "uncited_numeric_literal", "overlapping_evidence_relations",
    "overlapping_conflict_sides", "ambiguous_conflict_resolution",
]


class InvalidFinalProvider(RepairProvider):
    def __init__(self, failure, *, repair_fails=False):
        super().__init__(bad_stage="arbitrate", failure=failure, repair_fails=repair_fails)

    def extraction(self, paragraphs):
        return [{
            "paragraph_id": paragraph["id"],
            "quote": QUOTE if paragraph["id"] == PARAGRAPHS[0]["id"] else SECOND_TEXT,
            "fact_des": "合成测试的原文陈述", "argument": {}, "entities": [],
        } for paragraph in paragraphs]

    async def arbitrate(self, question, evidence, analyses):
        self.calls.append(("arbitrate", copy.deepcopy(question), copy.deepcopy(evidence), copy.deepcopy(analyses)))
        self.account("arbitrate")
        primary = next(item for item in evidence if item["quote"] == QUOTE)
        secondary = next(item for item in evidence if item["quote"] == SECOND_TEXT)
        left, right = primary["id"], secondary["id"]
        output = grounded([primary], arbitration=True)
        if self.failure == "uncited_numeric_literal":
            # 999 exists in a different available evidence item, but is not
            # cited by this final conclusion. Mere presence is insufficient.
            output["conclusion"] = "合同编号999的履行情况尚未确认。"
        elif self.failure == "overlapping_evidence_relations":
            output["conflicting_evidence_ids"] = [left]
        else:
            output["conflicts"] = [{
                "id": "C1", "statement": "双方履行说法需核对。", "status": "resolved",
                "supporting_evidence_ids": [left],
                "conflicting_evidence_ids": [left if self.failure == "overlapping_conflict_sides" else right],
                "resolution_type": "其他", "basis": "核对直接陈述", "accepted_reason": "保留可核验陈述",
                "rejected_reason": "不可据此推定已履行", "minority_opinion": "仍存在确认缺口",
                "reopen_conditions": "出现双方共同确认的原始记录",
            }]
            if self.failure == "ambiguous_conflict_resolution":
                output["supporting_evidence_ids"] = [left, right]
        return output


@pytest.mark.parametrize("failure", FINAL_FAILURES)
async def test_final_snapshot_invariants_are_checked_inside_the_repair_boundary(tmp_path, failure):
    provider = InvalidFinalProvider(failure)
    result = await execute(tmp_path, provider, paragraphs=TWO_PARAGRAPHS)
    assert len(provider.repairs) == 1
    assert provider.repairs[0]["diagnostics"]["check"] == failure
    assert provider.repairs[0]["analyses"] == result["analyses"]
    assert result["final"]["decision"] == "undetermined"
    assert result["final"]["supporting_evidence_ids"] == result["final"]["conflicts"] == []
    assert len(result["evidence"]) == 2


@pytest.mark.parametrize("failure", FINAL_FAILURES)
async def test_repeated_invalid_final_still_fails_after_repair(tmp_path, failure):
    provider = InvalidFinalProvider(failure, repair_fails=True)
    with pytest.raises(ModelOutputInvalid) as caught:
        await execute(tmp_path, provider, paragraphs=TWO_PARAGRAPHS)
    assert caught.value.stage == "arbitrate"
    assert caught.value.details["check"] == failure
    assert caught.value.details["repair_exhausted"] is True
    assert len(provider.repairs) == 1
    await assert_no_final_checkpoint(tmp_path)


@pytest.mark.parametrize("failure", FINAL_FAILURES)
async def test_provider_without_repair_preserves_strict_final_validation(tmp_path, failure):
    provider, events = InvalidFinalProvider(failure), []
    provider.repair_analysis = None
    with pytest.raises(ModelOutputInvalid) as caught:
        await execute(tmp_path, provider, paragraphs=TWO_PARAGRAPHS, emit=lambda *event: events.append(event))
    assert caught.value.details["check"] == failure
    assert provider.repairs == []
    assert not any(kind.startswith("model_repair_") for kind, _ in events)
    await assert_no_final_checkpoint(tmp_path)


async def test_extraction_schema_repair_cannot_claim_a_second_quote_repair_budget(tmp_path):
    class InvalidWireThenInvalidQuote(RepairProvider):
        async def extract(self, question, paragraphs):
            self.calls.append(("extract", copy.deepcopy(question), copy.deepcopy(paragraphs)))
            self.account("extract")
            raise ModelOutputInvalid("模型输出未通过结构化契约校验。", stage="extract", code="model_output_invalid", details={"check": "schema"})

        async def repair_extract(self, question, paragraphs, invalid_output, diagnostics):
            output = await super().repair_extract(question, paragraphs, invalid_output, diagnostics)
            output[0]["quote"] = "修正输出结构合法但这段引用不存在"
            return output

    provider = InvalidWireThenInvalidQuote(bad_stage="extract")
    with pytest.raises(ModelOutputInvalid) as caught:
        await execute(tmp_path, provider)
    assert len(provider.repairs) == 1
    assert caught.value.details["repair_exhausted"] is True
    assert [call[0] for call in provider.calls] == ["extract"]
    await assert_no_final_checkpoint(tmp_path)


async def test_repaired_extraction_cannot_reinstate_explicitly_excluded_evidence(tmp_path):
    provider = RepairProvider(bad_stage="extract")
    start = TEXT.index(QUOTE)
    result = await execute(tmp_path, provider, excluded_spans={(PARAGRAPHS[0]["id"], start, start + len(QUOTE), QUOTE)})
    assert len(provider.repairs) == 1
    assert result["evidence"] == []
    assert all(analysis["decision"] == "undetermined" for analysis in result["analyses"])
    assert result["final"]["decision"] == "undetermined"
    assert result["final"]["supporting_evidence_ids"] == []


async def test_no_evidence_repair_must_allow_unresolved_without_fabricating_references(tmp_path):
    class NoEvidenceProvider(RepairProvider):
        def extraction(self, paragraphs):
            return []

    provider = NoEvidenceProvider(failure="invalid_evidence_reference")
    result = await execute(tmp_path, provider)
    assert result["evidence"] == []
    assert provider.repairs[0]["evidence"] == []
    assert result["analyses"][0]["judgments"] == []
    assert result["final"]["decision"] == "undetermined"
    assert result["final"]["supporting_evidence_ids"] == []


@pytest.mark.parametrize("code", [
    "model_output_truncated", "model_output_filtered", "model_resource_unavailable", "model_output_aborted",
])
async def test_incomplete_or_filtered_response_is_not_a_licence_for_paid_content_repair(tmp_path, code):
    class IncompleteProvider(RepairProvider):
        async def analyze(self, role, question, evidence, snapshot_id):
            result = await super().analyze(role, question, evidence, snapshot_id)
            if role == "authority_analyst":
                raise ModelOutputInvalid("模型未正常完成，未采纳结果。", stage=role, code=code)
            return result

    provider, events, reports = IncompleteProvider(), [], []
    with pytest.raises(ModelOutputInvalid) as caught:
        await execute(tmp_path, provider, emit=lambda *event: events.append(event), usage_callback=reports.append)
    assert caught.value.code == code
    assert provider.repairs == []
    assert not any(kind.startswith("model_repair_") for kind, _ in events)
    assert any(row["operation"] == "authority_analyst" for row in reports[-1])
    await assert_no_final_checkpoint(tmp_path)
