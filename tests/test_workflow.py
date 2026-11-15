import asyncio
import copy
import json
import sqlite3
from uuid import UUID

import pytest

from langchain_core.messages import AIMessage

from app.agents.provider import LangChainOpenAIProvider, ModelNotConfigured, ModelOutputInvalid, capture_usage
from app.config import Settings
from app.workflows.analysis import AnalysisCancelled, run_analysis


QUESTION = {"id": "question-internal-1", "raw_question_id": "q_001", "question": "通报中失联人员为多少？"}
PARAGRAPHS = [{
    "id": "paragraph-internal-1", "document_id": "document-internal-1", "raw_doc_id": "doc_001",
    "raw_para_id": "para_001", "start_cp": 0, "end_cp": 15, "text": "🦊通报称，截至当日失联3人。",
}]


class FakeProvider:
    test_only_legacy_precision = True
    """Only tests explicitly inject this provider; production has no fake branch."""

    def __init__(self):
        self.calls = []
        self._started = set()
        self._both_started = asyncio.Event()

    async def extract(self, question, paragraphs):
        self.calls.append(("extract", copy.deepcopy(question), copy.deepcopy(paragraphs)))
        paragraph = paragraphs[0]
        return [{
            "paragraph_id": paragraph["id"], "quote": "失联3人", "fact_des": "通报称失联3人",
            "argument": {"时间": None, "主体": "失联人员", "地点": None, "内容": "失联3人", "影响": None, "数值口径": "3人"},
            "entities": ["失联人员"],
        }]

    async def analyze(self, role, question, evidence, snapshot_id):
        self.calls.append((role, copy.deepcopy(question), copy.deepcopy(evidence), snapshot_id))
        if role != "baseline":
            self._started.add(role)
            if len(self._started) == 2:
                self._both_started.set()
            # A sequential implementation would time out here.
            await asyncio.wait_for(self._both_started.wait(), 3)
        output = self._answer(evidence)
        # Mutating a provider's private input must never mutate peer/frozen data.
        if evidence:
            evidence[0]["quote"] = "MUTATED_PRIVATE_INPUT"
        return output

    def _answer(self, evidence):
        refs = [row["id"] for row in evidence]
        return {
            "decision": "asserted" if refs else "undetermined",
            "conclusion": "通报称失联3人。" if refs else "缺少可定位证据，无法定论。",
            "judgments": [{"id": "J1", "statement": "通报有明确人数", "evidence_ids": refs, "prerequisite_judgment_ids": []}] if refs else [],
            "supporting_evidence_ids": refs, "conflicting_evidence_ids": [],
        }

    async def arbitrate(self, question, evidence, analyses):
        self.calls.append(("arbitrate", copy.deepcopy(question), copy.deepcopy(evidence), copy.deepcopy(analyses)))
        return {**self._answer(evidence), "conflicts": []}


async def run(tmp_path, provider, **kwargs):
    return await run_analysis(
        QUESTION, PARAGRAPHS, provider, test_only_legacy_precision=not isinstance(provider, LangChainOpenAIProvider), checkpoint_path=tmp_path / "checkpoints.sqlite3",
        thread_id="run-1:question-1", cancel_check=kwargs.pop("cancel_check", lambda: False),
        emit=kwargs.pop("emit", lambda *_: None), **kwargs,
    )


async def test_parallel_independent_analysts_share_frozen_snapshot_and_durable_checkpoint(tmp_path):
    provider = FakeProvider()
    events = []
    result = await run(tmp_path, provider, emit=lambda kind, data: events.append((kind, data)))
    analysts = [call for call in provider.calls if call[0] in {"authority_analyst", "cross_validation_analyst"}]
    assert len(analysts) == 2
    assert analysts[0][1:] == analysts[1][1:]
    assert analysts[0][3] == result["evidence_snapshot_id"]
    assert all("analyses" not in call[1] and "conclusion" not in call[1] for call in analysts)
    assert all(call[2][0]["quote"] == "失联3人" for call in analysts)
    assert result["evidence"][0]["quote"] == "失联3人"
    assert result["evidence"][0]["start_cp"] == PARAGRAPHS[0]["text"].index("失联3人")
    assert result["evidence"][0]["source_level"] == "unknown"
    assert result["evidence"][0]["grade_score"] is None
    assert result["usage"] == []
    assert len({UUID(record["judgments"][0]["id"]) for record in result["analyses"]}) == 2
    with sqlite3.connect(tmp_path / "checkpoints.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM checkpoints").fetchone()[0] >= 6
        assert connection.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall() == [("run-1:question-1",)]
    assert any(kind == "node_completed" and data["node"] == "arbitrate" for kind, data in events)


def test_production_provider_refuses_empty_configuration():
    settings = Settings(_env_file=None, model_name="", model_api_key="")
    with pytest.raises(ModelNotConfigured, match="TRUST_MODEL_NAME"):
        LangChainOpenAIProvider(settings)


async def test_invalid_quote_never_reaches_analysts(tmp_path):
    class InvalidQuoteProvider(FakeProvider):
        async def extract(self, question, paragraphs):
            candidates = await super().extract(question, paragraphs)
            candidates[0]["quote"] = "失联4人"
            return candidates

    provider = InvalidQuoteProvider()
    with pytest.raises(ModelOutputInvalid, match="引用") as failure:
        await run(tmp_path, provider)
    assert failure.value.stage == "verify"
    assert failure.value.details == {"question_id": QUESTION["id"], "check": "quote_occurrence",
                                     "candidate_index": 0, "field_path": "candidates[0].quote"}
    assert [call[0] for call in provider.calls] == ["extract"]


async def test_cancellation_after_model_request_prevents_analysis_and_arbitration(tmp_path):
    should_cancel = False

    class CancelProvider(FakeProvider):
        async def extract(self, question, paragraphs):
            nonlocal should_cancel
            result = await super().extract(question, paragraphs)
            should_cancel = True
            return result

    provider = CancelProvider()
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, cancel_check=lambda: should_cancel)
    assert [call[0] for call in provider.calls] == ["extract"]


async def test_explicitly_excluded_span_cannot_be_reextracted(tmp_path):
    start = PARAGRAPHS[0]["text"].index("失联3人")
    excluded = {(PARAGRAPHS[0]["id"], start, start + len("失联3人"), "失联3人")}
    result = await run(tmp_path, FakeProvider(), excluded_spans=excluded)
    assert result["evidence"] == []
    assert result["final"]["decision"] == "undetermined"


async def test_baseline_uses_same_snapshot_but_never_enters_arbitration(tmp_path):
    provider = FakeProvider()
    result = await run(tmp_path, provider, include_baseline=True)
    assert result["baseline"]["role"] == "baseline"
    assert result["baseline"]["evidence_snapshot_id"] == result["evidence_snapshot_id"]
    arbitration = next(call for call in provider.calls if call[0] == "arbitrate")
    assert {analysis["role"] for analysis in arbitration[3]} == {"authority_analyst", "cross_validation_analyst"}
    assert len(result["analyses"]) == 2


async def test_reference_to_unseen_evidence_is_rejected(tmp_path):
    class PhantomProvider(FakeProvider):
        async def arbitrate(self, question, evidence, analyses):
            result = await super().arbitrate(question, evidence, analyses)
            result["supporting_evidence_ids"].append("not-in-the-snapshot")
            return result

    with pytest.raises(ModelOutputInvalid, match="不属于") as failure:
        await run(tmp_path, PhantomProvider())
    assert failure.value.stage == "arbitrate"
    assert failure.value.details == {"question_id": QUESTION["id"], "check": "invalid_evidence_reference"}


@pytest.mark.parametrize("invalid_role", ["authority_analyst", "cross_validation_analyst"])
async def test_invalid_parallel_branch_identifies_its_own_role(tmp_path, invalid_role):
    class InvalidAnalystProvider(FakeProvider):
        async def analyze(self, role, question, evidence, snapshot_id):
            result = await super().analyze(role, question, evidence, snapshot_id)
            if role == invalid_role:
                result["supporting_evidence_ids"].append("unseen-evidence")
            return result

    provider = InvalidAnalystProvider()
    with pytest.raises(ModelOutputInvalid, match="不属于") as failure:
        await run(tmp_path, provider)
    assert failure.value.stage == invalid_role
    assert failure.value.details == {"question_id": QUESTION["id"], "check": "invalid_evidence_reference"}
    assert all(call[0] != "arbitrate" for call in provider.calls)


@pytest.mark.parametrize("invalid_role", ["authority_analyst", "cross_validation_analyst"])
async def test_parallel_branch_schema_failure_has_role_without_returned_content(tmp_path, invalid_role):
    class MalformedAnalystProvider(FakeProvider):
        async def analyze(self, role, question, evidence, snapshot_id):
            result = await super().analyze(role, question, evidence, snapshot_id)
            if role == invalid_role:
                result["decision"] = "sensitive-model-content"
            return result

    with pytest.raises(ModelOutputInvalid) as failure:
        await run(tmp_path, MalformedAnalystProvider())
    assert failure.value.stage == invalid_role
    assert failure.value.details == {"question_id": QUESTION["id"], "check": "schema"}
    assert "sensitive-model-content" not in str(failure.value)


async def test_repeated_quote_requires_explicit_occurrence(tmp_path):
    paragraphs = copy.deepcopy(PARAGRAPHS)
    paragraphs[0]["text"] = "失联3人。失联3人。"
    with pytest.raises(ModelOutputInvalid, match="重复"):
        await run_analysis(QUESTION, paragraphs, FakeProvider(), test_only_legacy_precision=True, checkpoint_path=tmp_path / "cp.sqlite3",
                           thread_id="repeat", cancel_check=lambda: False, emit=lambda *_: None)


@pytest.mark.parametrize("wrong_hint", [(0, 4), (4, 20), (9, 3), (None, 3)])
async def test_unique_exact_quote_uses_codepoint_location_and_stable_id_despite_wrong_model_hint(tmp_path, wrong_hint):
    paragraphs = copy.deepcopy(PARAGRAPHS)
    paragraphs[0]["text"] = "前言🦊通报称失联3人，等待复核。"
    literal = "🦊通报称失联3人"
    expected_start = paragraphs[0]["text"].index(literal)
    expected_end = expected_start + len(literal)

    class HintedProvider(FakeProvider):
        def __init__(self, hint):
            super().__init__()
            self.hint = hint

        async def extract(self, question, selected):
            candidates = await super().extract(question, selected)
            candidates[0].update(quote=literal, start_cp=self.hint[0], end_cp=self.hint[1])
            return candidates

    async def run_hinted(hint, thread_id):
        return await run_analysis(
            QUESTION, paragraphs, HintedProvider(hint), test_only_legacy_precision=True, checkpoint_path=tmp_path / "cp.sqlite3",
            thread_id=thread_id, cancel_check=lambda: False, emit=lambda *_: None,
        )

    corrected = await run_hinted(wrong_hint, "incorrect-hint")
    already_correct = await run_hinted((expected_start, expected_end), "correct-hint")
    no_hint = await run_hinted((None, None), "no-hint")
    for result in (corrected, already_correct, no_hint):
        item = result["evidence"][0]
        assert item["quote"] == literal
        assert item["start_cp"] == expected_start and item["end_cp"] == expected_end
        assert paragraphs[0]["text"][item["start_cp"]:item["end_cp"]] == literal
        assert item["span_location_method"] == "exact_unique_quote"
    assert corrected["evidence"][0]["model_span_corrected"] is True
    assert already_correct["evidence"][0]["model_span_corrected"] is False
    assert no_hint["evidence"][0]["model_span_corrected"] is False
    assert len({result["evidence"][0]["id"] for result in (corrected, already_correct, no_hint)}) == 1
    assert paragraphs[0]["text"] == "前言🦊通报称失联3人，等待复核。"


@pytest.mark.parametrize("hint", [(None, None), (1, 5)])
async def test_repeated_quote_does_not_choose_an_occurrence_or_repair_wrong_hint(tmp_path, hint):
    paragraphs = copy.deepcopy(PARAGRAPHS)
    paragraphs[0]["text"] = "失联3人。失联3人。"

    class RepeatedQuoteProvider(FakeProvider):
        async def extract(self, question, selected):
            candidates = await super().extract(question, selected)
            candidates[0].update(start_cp=hint[0], end_cp=hint[1])
            return candidates

    provider = RepeatedQuoteProvider()
    with pytest.raises(ModelOutputInvalid) as failure:
        await run_analysis(
            QUESTION, paragraphs, provider, test_only_legacy_precision=True, checkpoint_path=tmp_path / "cp.sqlite3",
            thread_id="ambiguous", cancel_check=lambda: False, emit=lambda *_: None,
        )
    assert failure.value.stage == "verify"
    assert failure.value.details["check"] in {"quote_occurrence", "quote_span"}
    assert [call[0] for call in provider.calls] == ["extract"]


async def test_repeated_quote_accepts_only_exact_explicit_occurrence(tmp_path):
    paragraphs = copy.deepcopy(PARAGRAPHS)
    paragraphs[0]["text"] = "失联3人。失联3人。"

    class RepeatedQuoteProvider(FakeProvider):
        async def extract(self, question, selected):
            candidates = await super().extract(question, selected)
            candidates[0].update(start_cp=5, end_cp=9)
            return candidates

    result = await run_analysis(
        QUESTION, paragraphs, RepeatedQuoteProvider(), test_only_legacy_precision=True, checkpoint_path=tmp_path / "cp.sqlite3",
        thread_id="explicit-occurrence", cancel_check=lambda: False, emit=lambda *_: None,
    )
    item = result["evidence"][0]
    assert (item["start_cp"], item["end_cp"]) == (5, 9)
    assert item["span_location_method"] == "model_explicit_occurrence"
    assert item["model_span_corrected"] is False


async def test_runtime_projection_does_not_forward_gold_fields(tmp_path):
    provider = FakeProvider()
    contaminated = {**QUESTION, "conclusion": "hidden answer", "supporting_facts": ["gold"]}
    paragraphs = [{**PARAGRAPHS[0], "gold_answer": "hidden answer"}]
    await run_analysis(contaminated, paragraphs, provider, test_only_legacy_precision=True, checkpoint_path=tmp_path / "cp.sqlite3",
                       thread_id="project", cancel_check=lambda: False, emit=lambda *_: None)
    extraction = provider.calls[0]
    assert "conclusion" not in extraction[1]
    assert "supporting_facts" not in extraction[1]
    assert "gold_answer" not in extraction[2][0]


async def test_model_judgment_cycle_is_rejected(tmp_path):
    class CyclicProvider(FakeProvider):
        async def arbitrate(self, question, evidence, analyses):
            output = await super().arbitrate(question, evidence, analyses)
            output["judgments"][0]["prerequisite_judgment_ids"] = ["J2"]
            output["judgments"].append({"id": "J2", "statement": "循环前提", "evidence_ids": [], "prerequisite_judgment_ids": ["J1"]})
            return output

    with pytest.raises(ModelOutputInvalid, match="循环"):
        await run(tmp_path, CyclicProvider())


async def test_evidence_and_judgment_identity_is_stable_between_runs(tmp_path):
    first = await run(tmp_path, FakeProvider())
    second = await run_analysis(QUESTION, PARAGRAPHS, FakeProvider(), test_only_legacy_precision=True, checkpoint_path=tmp_path / "checkpoints.sqlite3",
                                thread_id="run-2:question-1", cancel_check=lambda: False, emit=lambda *_: None)
    assert first["evidence"][0]["id"] == second["evidence"][0]["id"]
    assert first["final"]["judgments"][0]["id"] == second["final"]["judgments"][0]["id"]
    assert first["evidence_snapshot_id"] != second["evidence_snapshot_id"]


@pytest.mark.parametrize("usage", [None, {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}])
async def test_langchain_adapter_json_mode_and_truthful_usage(monkeypatch, usage):
    from app.agents import provider as provider_module

    class StubChat:
        def __init__(self, **kwargs):
            self.binding = None

        def bind(self, **kwargs):
            bound = copy.copy(self)
            bound.binding = {**(self.binding or {}), **kwargs}
            return bound

        async def ainvoke(self, messages):
            assert "JSON Schema" in messages[0].content
            assert self.binding["response_format"] == {"type": "json_object"}
            assert self.binding["max_tokens"] == 32768
            payload = json.loads(messages[1].content)
            decisions = {row["id"]: {"disposition": "no_target_fact", "annotations": [],
                         "reason": "显式合成模型未抽取事实。"} for row in payload["source_paragraphs"]}
            return AIMessage(content=json.dumps({"source_catalog_id": payload["source_catalog_id"],
                "decisions": decisions}), usage_metadata=usage)

    monkeypatch.setattr(provider_module, "ChatOpenAI", StubChat)
    adapter = LangChainOpenAIProvider(Settings(_env_file=None, model_name="explicit-test-model", model_api_key="test-only-value"))
    with capture_usage() as records:
        assert await adapter.extract(QUESTION, PARAGRAPHS) == []
    assert adapter._model.binding == {"response_format": {"type": "json_object"}}
    assert records[0]["input_tokens"] == (11 if usage else None)
    assert records[0]["total_tokens"] == (18 if usage else None)
    assert records[0]["cost"] is None
    assert records[0]["usage_source"] == ("provider" if usage else "unavailable")


async def test_workflow_callback_preserves_provider_failure_stage_and_completed_usage(tmp_path, monkeypatch):
    from app.agents import provider as provider_module

    class StubChat:
        def __init__(self, **kwargs):
            pass

        def bind(self, **kwargs):
            return self

        async def ainvoke(self, messages):
            return AIMessage(
                content="not valid JSON and must not be recorded",
                usage_metadata={"input_tokens": 19, "output_tokens": 5, "total_tokens": 24},
                response_metadata={"finish_reason": "stop"},
            )

    monkeypatch.setattr(provider_module, "ChatOpenAI", StubChat)
    provider = LangChainOpenAIProvider(Settings(_env_file=None, model_name="test-model", model_api_key="test-only-value"))
    reported = []
    with pytest.raises(ModelOutputInvalid) as failure:
        await run(tmp_path, provider, usage_callback=reported.append)
    assert failure.value.stage == "extract"
    assert failure.value.code == "model_output_invalid"
    assert failure.value.details["repair_exhausted"] is True
    # The initial invalid response and the single unsuccessful correction are
    # both billable. Intermediate callbacks must not erase earlier counters.
    assert len(reported) == 2
    expected = {
        "operation": "extract", "model": "test-model", "usage_source": "provider",
        "input_tokens": 19, "output_tokens": 5, "total_tokens": 24, "cost": None,
    }
    assert reported[0] == [expected]
    assert reported[-1] == [expected, expected]
    assert "not valid JSON" not in json.dumps(reported)


async def test_workflow_callback_receives_usage_when_cancellation_follows_model_response(tmp_path, monkeypatch):
    from app.agents import provider as provider_module

    should_cancel = False

    class StubChat:
        def __init__(self, **kwargs):
            pass

        def bind(self, **kwargs):
            return self

        async def ainvoke(self, messages):
            nonlocal should_cancel
            should_cancel = True
            return AIMessage(
                content=json.dumps({"evidence": []}),
                usage_metadata={"input_tokens": 17, "output_tokens": 3, "total_tokens": 20},
            )

    monkeypatch.setattr(provider_module, "ChatOpenAI", StubChat)
    provider = LangChainOpenAIProvider(Settings(_env_file=None, model_name="test-model", model_api_key="test-only-value"))
    reported = []
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, cancel_check=lambda: should_cancel, usage_callback=reported.append)
    assert len(reported) == 1
    assert reported[0][0]["operation"] == "extract"
    assert reported[0][0]["total_tokens"] == 20


async def test_workflow_callback_runs_when_cancelled_before_first_node(tmp_path):
    reported = []
    provider = FakeProvider()
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, cancel_check=lambda: True, usage_callback=reported.append)
    assert reported == [[]]
    assert provider.calls == []


async def test_workflow_success_usage_callback_cannot_mutate_returned_usage(tmp_path):
    reported = []

    def mutate_report(records):
        reported.append(records)
        records.append({"operation": "callback-only"})

    result = await run(tmp_path, FakeProvider(), usage_callback=mutate_report)
    assert reported == [[{"operation": "callback-only"}]]
    assert result["usage"] == []


def manual_candidate(paragraph, quote, **extra):
    start = paragraph["text"].index(quote)
    return {
        "paragraph_id": paragraph["id"], "quote": quote,
        "start_cp": start, "end_cp": start + len(quote),
        "fact_des": "人工提供的待核验解释", "argument": {}, "entities": [], **extra,
    }


async def test_manual_evidence_preserves_annotation_when_complete_scan_includes_zero_match_source(tmp_path):
    provider = FakeProvider()
    outside_top_k = {
        **PARAGRAPHS[0], "id": "paragraph-2", "document_id": "document-2",
        "raw_doc_id": "doc_002", "raw_para_id": "para_002", "text": "三位搜救对象均已获救。",
    }
    manual = manual_candidate(outside_top_k, "均已获救", conclusion="gold must not pass", grade_score=1.0,
                              source_level="official", raw_doc_id="wrong-doc", argument={"gold": "must not pass"})
    result = await run_analysis(
        {**QUESTION, "question": "通报"}, [*PARAGRAPHS, outside_top_k], provider,
        test_only_legacy_precision=True, checkpoint_path=tmp_path / "cp.sqlite3", thread_id="manual", cancel_check=lambda: False,
        emit=lambda *_: None, retrieval_limit=1, manual_evidence=[manual],
    )
    extraction = next(call for call in provider.calls if call[0] == "extract")
    # Retrieval priority no longer cuts off zero-match sources. Explicit human
    # annotations must still be projected safely and survive the complete scan.
    assert [row["id"] for row in extraction[2]] == [PARAGRAPHS[0]["id"], outside_top_k["id"]]
    assert result["source_coverage"]["complete"] is True
    assert result["source_coverage"]["scanned_paragraphs"] == 2
    added = next(row for row in result["evidence"] if row["paragraph_id"] == outside_top_k["id"])
    assert added["raw_doc_id"] == "doc_002"
    assert added["interpretation_origin"] == "human"
    assert added["grade_score"] is None and added["source_level"] == "unknown"
    assert "conclusion" not in added and "gold" not in added["argument"]
    for call in provider.calls:
        if call[0] in {"authority_analyst", "cross_validation_analyst"}:
            assert any(row["id"] == added["id"] for row in call[2])


async def test_human_interpretation_overrides_reach_both_analysts_without_changing_source(tmp_path):
    original = await run(tmp_path, FakeProvider())
    evidence_id = original["evidence"][0]["id"]
    provider = FakeProvider()
    patch = {
        "fact_des": "人工指出该数字仅适用于初报口径", "argument": {"时间": "初报", "gold": "must not pass"},
        "entities": ["初报口径"], "stance": "待核验反证", "quote": "篡改原文", "start_cp": 0,
        "conclusion": "must not pass",
    }
    result = await run(tmp_path, provider, interpretation_overrides={evidence_id: patch})
    final_evidence = result["evidence"][0]
    assert final_evidence["quote"] == original["evidence"][0]["quote"]
    assert final_evidence["start_cp"] == original["evidence"][0]["start_cp"]
    assert final_evidence["fact_des"] == patch["fact_des"]
    assert final_evidence["argument"]["时间"] == "初报"
    assert final_evidence["argument"]["数值口径"] == "3人"
    assert "gold" not in final_evidence["argument"] and "conclusion" not in final_evidence
    for call in provider.calls:
        if call[0] in {"authority_analyst", "cross_validation_analyst"}:
            assert call[2][0]["fact_des"] == patch["fact_des"]
            assert call[2][0]["stance"] == patch["stance"]
            assert call[2][0]["interpretation_origin"] == "human"


async def test_manual_inclusion_and_interpretation_cannot_override_explicit_exclusion(tmp_path):
    original = await run(tmp_path, FakeProvider())
    evidence = original["evidence"][0]
    exclusion = {(evidence["paragraph_id"], evidence["start_cp"], evidence["end_cp"], evidence["quote"])}
    result = await run(
        tmp_path, FakeProvider(), excluded_spans=exclusion,
        manual_evidence=[manual_candidate(PARAGRAPHS[0], "失联3人")],
        interpretation_overrides={evidence["id"]: {"fact_des": "人工仍希望采用"}},
    )
    assert result["evidence"] == []
    assert result["final"]["decision"] == "undetermined"


async def test_bad_manual_span_is_rejected_before_any_paid_corpus_scan(tmp_path):
    provider = FakeProvider()
    manual = manual_candidate(PARAGRAPHS[0], "失联3人", start_cp=0, end_cp=4)
    with pytest.raises(ModelOutputInvalid, match="码点"):
        await run(tmp_path, provider, manual_evidence=[manual])
    assert provider.calls == []


async def test_manual_reference_must_belong_to_current_source_pool(tmp_path):
    manual = manual_candidate(PARAGRAPHS[0], "失联3人", paragraph_id="foreign-topic-paragraph")
    with pytest.raises(ModelOutputInvalid, match="来源资料"):
        await run(tmp_path, FakeProvider(), manual_evidence=[manual])


async def test_unknown_interpretation_target_is_not_silently_ignored(tmp_path):
    with pytest.raises(ModelOutputInvalid, match="必须同时提供"):
        await run(tmp_path, FakeProvider(), interpretation_overrides={"unknown-id": {"fact_des": "人工修改"}})


async def test_new_manual_evidence_stance_reaches_analysts_and_arbitration(tmp_path):
    provider = FakeProvider()
    manual = manual_candidate(PARAGRAPHS[0], "失联3人", stance="context", hidden_answer="must not pass")
    result = await run(tmp_path, provider, manual_evidence=[manual])
    item = result["evidence"][0]
    assert item["stance"] == "context"
    assert item["quote"] == "失联3人" and item["interpretation_origin"] == "human"
    assert "hidden_answer" not in item
    for call in provider.calls:
        if call[0] in {"authority_analyst", "cross_validation_analyst", "arbitrate"}:
            assert call[2][0]["stance"] == "context"
            assert call[2][0]["interpretation_origin"] == "human"


@pytest.mark.parametrize("stance", ["", None, 1])
async def test_invalid_manual_stance_fails_before_model_call(tmp_path, stance):
    provider = FakeProvider()
    manual = manual_candidate(PARAGRAPHS[0], "失联3人", stance=stance)
    with pytest.raises(ValueError, match="立场"):
        await run(tmp_path, provider, manual_evidence=[manual])
    assert provider.calls == []
