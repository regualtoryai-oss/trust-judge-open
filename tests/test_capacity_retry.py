"""Stage-local capacity retries with explicit providers; no network or secrets."""

import asyncio
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agents import provider as provider_module
from app.agents.provider import ModelInvocationFailed, ModelOutputInvalid, get_requested_output_budget
from app.config import Settings
from app.repository import Repository
from app.workflows.analysis import AnalysisCancelled, run_analysis


QUESTION = {"id": "capacity-question", "question": "合成设备是否已复位？"}
TEXT = "🦊合成通报明确：设备已复位。"
PARAGRAPHS = [{"id": "capacity-paragraph", "document_id": "capacity-document", "raw_doc_id": "doc_001",
               "raw_para_id": "p_001", "text": TEXT, "start_cp": 0, "end_cp": len(TEXT)}]


def answer(evidence, *, arbitration=False):
    result = {"decision": "asserted", "conclusion": "合成通报称设备已复位。",
              "judgments": [{"id": "J1", "statement": "通报明确设备已复位。", "evidence_ids": [evidence[0]["id"]], "prerequisite_judgment_ids": []}],
              "supporting_evidence_ids": [evidence[0]["id"]], "conflicting_evidence_ids": []}
    return {**result, "conflicts": []} if arbitration else result


class CapacityProvider:
    """Small token numbers make retry transitions inspectable and inexpensive."""

    def __init__(self, plans=None, *, retries=2, cap=128, on_call=None):
        self.plans = plans or {}
        self.retries = retries
        self.cap = cap
        self.on_call = on_call
        self.counts = Counter()
        self.calls = []

    def capacity_policy(self, stage):
        initial = {"extract": 8, "arbitrate": 32}.get(stage, 16)
        return {"initial_tokens": min(initial, self.cap), "max_tokens": self.cap, "retries": self.retries}

    async def request(self, operation, payload, result):
        index = self.counts[operation]
        self.counts[operation] += 1
        budget = get_requested_output_budget()
        record = {"operation": operation, "budget": budget, "payload": deepcopy(payload)}
        self.calls.append(record)
        provider_module._usage_records.get().append({"operation": operation, "input_tokens": 3, "output_tokens": 7, "total_tokens": 10})
        if self.on_call:
            self.on_call(operation, index)
        # Both analysts overlap here. Their requested budgets must stay local
        # despite one branch resetting/changing its budget during the other.
        await asyncio.sleep(0)
        record["budget_after_await"] = get_requested_output_budget()
        plan = self.plans.get(operation, [])
        failure = plan[index] if index < len(plan) else None
        stage = operation.removesuffix(":repair")
        if failure in {"length", "fake_length", "wrong_code", "filtered", "resource", "schema"}:
            codes = {"length": ("model_output_truncated", "length"), "fake_length": ("model_output_truncated", "unknown"),
                     "wrong_code": ("model_output_incomplete", "length"), "filtered": ("model_output_filtered", "content_filter"),
                     "resource": ("model_resource_unavailable", "insufficient_system_resource"), "schema": ("model_output_invalid", "stop")}
            code, finish = codes[failure]
            raise ModelOutputInvalid("合成受控模型响应错误。", code=code, stage=stage, details={"finish_reason": finish})
        if failure == "transport":
            raise ModelInvocationFailed("合成连接失败。", stage=stage)
        result = deepcopy(result)
        if failure == "ungrounded":
            result["judgments"][0]["evidence_ids"] = []
        if failure == "bad_quote":
            result[0]["quote"] = "原文里不存在的字符串"
        return result

    async def extract(self, question, paragraphs):
        result = [{"paragraph_id": paragraphs[0]["id"], "quote": "设备已复位", "fact_des": "设备已复位", "argument": {}, "entities": []}]
        return await self.request("extract", {"question": question, "paragraphs": paragraphs}, result)

    async def analyze(self, role, question, evidence, snapshot_id):
        return await self.request(role, {"question": question, "evidence": evidence, "snapshot_id": snapshot_id}, answer(evidence))

    async def arbitrate(self, question, evidence, analyses):
        return await self.request("arbitrate", {"question": question, "evidence": evidence, "analyses": analyses}, answer(evidence, arbitration=True))


class ContentRepairProvider(CapacityProvider):
    async def repair_analysis(self, role, question, evidence, snapshot_id, invalid_output, diagnostics, *, analyses=None):
        return await self.request(role + ":repair", {"question": question, "evidence": evidence, "snapshot_id": snapshot_id,
                                                     "invalid_output": invalid_output, "diagnostics": diagnostics, "analyses": analyses},
                                  answer(evidence, arbitration=role == "arbitrate"))

    async def repair_extract(self, question, paragraphs, invalid_output, diagnostics):
        result = [{"paragraph_id": paragraphs[0]["id"], "quote": "设备已复位", "fact_des": "设备已复位", "argument": {}, "entities": []}]
        return await self.request("extract:repair", {"question": question, "paragraphs": paragraphs,
                                                    "invalid_output": invalid_output, "diagnostics": diagnostics}, result)


async def execute(tmp_path, provider, *, events=None, usage=None, cancel_check=lambda: False, **kwargs):
    return await run_analysis(QUESTION, PARAGRAPHS, provider, test_only_legacy_precision=True, checkpoint_path=tmp_path / "capacity.sqlite3",
                              thread_id="capacity-test", cancel_check=cancel_check,
                              emit=lambda kind, detail: events.append((kind, detail)) if events is not None else None,
                              usage_callback=lambda value: usage.append(value) if usage is not None else None, **kwargs)


@pytest.mark.parametrize("stage", ["extract", "authority_analyst", "cross_validation_analyst", "arbitrate", "baseline"])
async def test_length_retry_only_reissues_failed_stage_with_identical_inputs_and_usage(tmp_path, stage):
    provider, events, usage = CapacityProvider({stage: ["length"]}), [], []
    result = await execute(tmp_path, provider, events=events, usage=usage, include_baseline=stage == "baseline")
    attempts = [call for call in provider.calls if call["operation"] == stage]
    initial = provider.capacity_policy(stage)["initial_tokens"]
    assert [call["budget"] for call in attempts] == [initial, initial * 2]
    assert attempts[0]["payload"] == attempts[1]["payload"]
    assert provider.counts["extract"] == (2 if stage == "extract" else 1)
    assert provider.counts["arbitrate"] == (2 if stage == "arbitrate" else 1)
    assert result["final"]["decision"] == "asserted"
    assert len(result["usage"]) == len(provider.calls)
    assert sum(call["total_tokens"] for call in usage[-1]) == len(provider.calls) * 10
    retries = [detail for kind, detail in events if kind == "model_capacity_retry_started"]
    assert len(retries) == 1 and retries[0]["stage"] == stage and retries[0]["node"] == stage
    assert retries[0]["summary"] == {"attempt": 1, "max_output_tokens": initial * 2, "previous_max_output_tokens": initial}
    call_id = retries[0]["call_id"]
    assert any(kind == "tool_started" and detail["name"] == f"{stage}_capacity_retry" and detail["call_id"] == call_id for kind, detail in events)
    assert any(kind == "tool_completed" and detail["call_id"] == call_id for kind, detail in events)
    assert not any(kind == "model_repair_started" for kind, _ in events)
    assert get_requested_output_budget() is None


@pytest.mark.parametrize("failure", ["fake_length", "wrong_code", "filtered", "resource", "schema", "transport"])
async def test_non_length_failures_do_not_trigger_capacity_retries(tmp_path, failure):
    provider, events = CapacityProvider({"extract": [failure]}), []
    with pytest.raises((ModelOutputInvalid, ModelInvocationFailed)):
        await execute(tmp_path, provider, events=events)
    assert provider.counts["extract"] == 1
    assert not any(kind.startswith("model_capacity_retry") for kind, _ in events)
    assert get_requested_output_budget() is None


@pytest.mark.parametrize("retries,cap", [(0, 128), (2, 8)])
async def test_disabled_retry_or_non_growing_budget_stops_after_initial_request(tmp_path, retries, cap):
    provider, events = CapacityProvider({"extract": ["length"]}, retries=retries, cap=cap), []
    with pytest.raises(ModelOutputInvalid) as caught:
        await execute(tmp_path, provider, events=events)
    assert provider.counts["extract"] == 1
    assert caught.value.details["capacity_attempts"] == 0
    assert caught.value.code == "model_output_truncated" and caught.value.stage == "extract"
    assert not any(kind.startswith("model_capacity_retry") for kind, _ in events)


async def test_two_retries_exhaust_without_restarting_prior_nodes_and_preserve_safe_diagnostics(tmp_path):
    provider = CapacityProvider({"arbitrate": ["length", "length", "length"]})
    events, usage = [], []
    with pytest.raises(ModelOutputInvalid) as caught:
        await execute(tmp_path, provider, events=events, usage=usage)
    error = caught.value
    assert error.stage == "arbitrate" and error.code == "model_output_truncated"
    assert error.details == {"finish_reason": "length", "capacity_attempts": 2, "last_max_output_tokens": 128, "max_output_tokens_limit": 128}
    assert provider.counts == {"extract": 1, "authority_analyst": 1, "cross_validation_analyst": 1, "arbitrate": 3}
    assert [call["budget"] for call in provider.calls if call["operation"] == "arbitrate"] == [32, 64, 128]
    assert len([kind for kind, _ in events if kind == "model_capacity_retry_failed"]) == 2
    assert len(usage[-1]) == 6 and sum(item["total_tokens"] for item in usage[-1]) == 60
    assert get_requested_output_budget() is None


async def test_parallel_analyst_budgets_are_context_local_and_successful_peer_not_repeated(tmp_path):
    provider = CapacityProvider({"authority_analyst": ["length", "length"]})
    result = await execute(tmp_path, provider)
    assert provider.counts["authority_analyst"] == 3 and provider.counts["cross_validation_analyst"] == 1
    assert all(call["budget"] == call["budget_after_await"] for call in provider.calls)
    assert [call["budget"] for call in provider.calls if call["operation"] == "authority_analyst"] == [16, 32, 64]
    peers = [call for call in provider.calls if call["operation"] in {"authority_analyst", "cross_validation_analyst"}]
    assert {call["payload"]["snapshot_id"] for call in peers} == {result["evidence_snapshot_id"]}
    assert all(call["payload"]["evidence"] == peers[0]["payload"]["evidence"] for call in peers)
    assert all("analyses" not in call["payload"] for call in peers)


async def test_cancellation_after_billable_truncation_prevents_another_request(tmp_path):
    cancelled = False

    def on_call(operation, _):
        nonlocal cancelled
        if operation == "extract":
            cancelled = True

    provider, events, usage = CapacityProvider({"extract": ["length"]}, on_call=on_call), [], []
    with pytest.raises(AnalysisCancelled):
        await execute(tmp_path, provider, events=events, usage=usage, cancel_check=lambda: cancelled)
    assert provider.counts["extract"] == 1
    assert len(usage[-1]) == 1 and usage[-1][0]["total_tokens"] == 10
    assert not any(kind == "model_capacity_retry_started" for kind, _ in events)


async def test_retry_transport_error_keeps_role_and_recorded_attempt_usage(tmp_path):
    provider, events, usage = CapacityProvider({"arbitrate": ["length", "transport"]}), [], []
    with pytest.raises(ModelInvocationFailed) as caught:
        await execute(tmp_path, provider, events=events, usage=usage)
    assert caught.value.stage == "arbitrate"
    assert provider.counts["arbitrate"] == 2
    assert len([kind for kind, _ in events if kind == "model_capacity_retry_failed"]) == 1
    assert len(usage[-1]) == 5


@pytest.mark.parametrize("stage,initial_failure", [("authority_analyst", "ungrounded"), ("arbitrate", "ungrounded"), ("extract", "bad_quote")])
async def test_content_repair_can_expand_capacity_without_extra_content_repair(tmp_path, stage, initial_failure):
    provider = ContentRepairProvider({stage: [initial_failure], stage + ":repair": ["length"]})
    events = []
    result = await execute(tmp_path, provider, events=events)
    assert result["final"]["decision"] == "asserted"
    assert provider.counts[stage + ":repair"] == 2
    content_repairs = [detail for kind, detail in events if kind == "model_repair_started"]
    assert len(content_repairs) == 1
    capacity = [detail for kind, detail in events if kind == "model_capacity_retry_started"]
    assert len(capacity) == 1 and capacity[0]["stage"] == stage
    attempts = [call for call in provider.calls if call["operation"] == stage + ":repair"]
    assert attempts[0]["payload"] == attempts[1]["payload"]
    initial = provider.capacity_policy(stage)["initial_tokens"]
    assert [call["budget"] for call in attempts] == [initial, initial * 2]


async def test_repair_capacity_failure_keeps_actual_extract_role_not_local_verify(tmp_path):
    provider = ContentRepairProvider({"extract": ["bad_quote"], "extract:repair": ["length"] * 3})
    with pytest.raises(ModelOutputInvalid) as caught:
        await execute(tmp_path, provider)
    assert caught.value.stage == "extract"
    assert caught.value.code == "model_output_truncated"
    assert caught.value.details["capacity_attempts"] == 2
    assert caught.value.details["repair_exhausted"] is True


async def test_providers_without_capacity_policy_retain_legacy_behavior(tmp_path):
    provider = CapacityProvider({"extract": ["length"]})
    provider.capacity_policy = None
    with pytest.raises(ModelOutputInvalid):
        await execute(tmp_path, provider)
    assert provider.counts["extract"] == 1
    assert provider.calls[0]["budget"] is None


@pytest.mark.parametrize("relative_module", ["app/config.py", "app/domain/source_catalog.py", "app/domain/spans.py", "app/domain/compiled_scope.py", "app/domain/paragraph_assertions.py", "app/domain/recall_recovery.py", "app/domain/premise_bank.py", "app/recovery.py"])
def test_effective_capacity_config_is_persistable_and_fingerprints_source_catalog(tmp_path, monkeypatch, relative_module):
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="", model_api_key="", model_max_output_tokens=40000,
                        model_capacity_max_output_tokens=50000, model_generation_timeout_seconds=600, model_timeout_seconds=900,
                        model_premise_bank_enabled=False)
    repo = Repository(settings)
    config = repo.config()
    assert config["output_budgets"] == {"extract": 40000, "analysis": 40000, "arbitrate": 50000, "capacity_max": 50000, "retries": 2}
    assert config["generation_timeout_seconds"] == 900
    assert config["prompt_version"] == "trusted-analysis-json-v17-paragraph-recall-recovery"
    assert config["premise_bank_enabled"] is False
    assert config["source_strategy"] == "full_source_scan"
    assert config["source_batch_codepoints"] == 20000
    assert config["source_batch_spans"] == 128
    assert len(config["prompt_sha256"]) == 64
    catalog = Path(__file__).resolve().parents[1] / relative_module
    assert catalog.is_file()
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: original_read(path) + (b"\n# synthetic catalog code change" if path == catalog else b""))
    assert repo.config()["prompt_sha256"] != config["prompt_sha256"]


@pytest.mark.parametrize("change", [{"model_capacity_retries": -1}, {"model_capacity_retries": 4},
                                  {"model_extraction_max_output_tokens": 0}, {"model_analysis_max_output_tokens": 393217},
                                  {"model_arbitration_max_output_tokens": 393217}, {"model_capacity_max_output_tokens": 393217},
                                  {"model_generation_timeout_seconds": 0},
                                  {"source_batch_codepoints": 0}, {"source_batch_codepoints": 64001},
                                  {"source_batch_spans": 0}, {"source_batch_spans": 513}])
def test_capacity_settings_reject_invalid_values_without_silently_clamping(change):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_name="", model_api_key="", **change)


def test_source_batch_settings_change_execution_identity(tmp_path):
    first = Repository(Settings(_env_file=None, data_dir=tmp_path / "first", model_name="", model_api_key=""))
    second = Repository(Settings(_env_file=None, data_dir=tmp_path / "second", model_name="", model_api_key="", source_batch_spans=256))
    assert first.config() != second.config()
    assert second.config()["source_batch_spans"] == 256


async def execute_capacity_batches(tmp_path, provider, *, count=2, events=None, thread_id="stage-floors",
                                   cancel_check=lambda: False, resume_from_thread_id=None):
    sources = [{**deepcopy(PARAGRAPHS[0]), "id": f"capacity-source-{index}",
                "document_id": f"capacity-document-{index}", "raw_doc_id": f"doc-{index}"}
               for index in range(count)]
    return await run_analysis(QUESTION, sources, provider, test_only_legacy_precision=True, checkpoint_path=tmp_path / "floors.sqlite3",
        thread_id=thread_id, cancel_check=cancel_check, resume_from_thread_id=resume_from_thread_id,
        emit=lambda kind, detail: events.append((kind, deepcopy(detail))) if events is not None else None,
        source_batch_codepoints=len(TEXT), source_batch_spans=128)


async def test_later_source_batches_reuse_learned_output_floor_without_new_retry_or_hiding_usage(tmp_path):
    provider, events = CapacityProvider({"extract": ["length"]}), []
    result = await execute_capacity_batches(tmp_path, provider, events=events)
    attempts = [call for call in provider.calls if call["operation"] == "extract"]
    assert [call["budget"] for call in attempts] == [8, 16, 16]
    assert attempts[0]["payload"] == attempts[1]["payload"]
    assert attempts[1]["payload"] != attempts[2]["payload"]
    assert len([1 for kind, _ in events if kind == "model_capacity_retry_started"]) == 1
    chosen = [value for kind, value in events if kind == "model_capacity_selected" and value["stage"] == "extract"]
    assert [value["max_output_tokens"] for value in chosen] == [8, 16, 16]
    assert [value["budget_origin"] for value in chosen] == ["stage_policy", "length_retry", "learned_stage_floor"]
    assert all(value["is_usage"] is False for value in chosen)
    assert all(call["budget"] == call["budget_after_await"] for call in attempts)
    assert len(result["usage"]) == len(provider.calls)
    assert sum(record["total_tokens"] for record in result["usage"]) == len(provider.calls) * 10
    assert result["source_coverage"]["completed_batches"] == 2
    assert get_requested_output_budget() is None


@pytest.mark.parametrize("plans,extract_budgets,repair_budgets", [
    ({"extract": ["length", "schema"]}, [8, 16, 16], [16]),
    ({"extract": ["bad_quote"], "extract:repair": ["length"]}, [8, 16], [8, 16]),
], ids=["expanded-primary-schema-needs-repair", "repair-expansion-reused-by-next-batch"])
async def test_extraction_repairs_and_later_batches_share_the_same_learned_floor(
    tmp_path, plans, extract_budgets, repair_budgets,
):
    provider, events = ContentRepairProvider(plans), []
    result = await execute_capacity_batches(tmp_path, provider, events=events)
    assert [call["budget"] for call in provider.calls if call["operation"] == "extract"] == extract_budgets
    assert [call["budget"] for call in provider.calls if call["operation"] == "extract:repair"] == repair_budgets
    assert len([1 for kind, _ in events if kind == "model_repair_started"]) == 1
    assert len([1 for kind, _ in events if kind == "model_capacity_retry_started"]) == 1
    assert len(result["evidence"]) == 2
    assert len(result["usage"]) == len(provider.calls)


async def test_output_floor_can_grow_again_in_a_later_batch_but_stays_below_stage_cap(tmp_path):
    provider, events = CapacityProvider({"extract": ["length", None, "length"]}, cap=24), []
    result = await execute_capacity_batches(tmp_path, provider, count=3, events=events)
    assert [call["budget"] for call in provider.calls if call["operation"] == "extract"] == [8, 16, 16, 24, 24]
    assert all(call["budget"] <= 24 for call in provider.calls)
    assert [call["budget"] for call in provider.calls if call["operation"] in {
        "authority_analyst", "cross_validation_analyst"}] == [16, 16]
    retries = [value for kind, value in events if kind == "model_capacity_retry_started"]
    assert [(row["previous_max_output_tokens"], row["max_output_tokens"]) for row in retries] == [(8, 16), (16, 24)]
    assert result["source_coverage"]["complete"] is True


class PrecisionCapacityProvider(ContentRepairProvider):
    def selection(self, evidence):
        return {"selected_evidence_ids": [row["id"] for row in evidence], "assessments": [
            {"evidence_id": row["id"], "relation": "support", "reason": "通报直接说明运行状态。", "equivalent_to": None}
            for row in evidence]}

    async def select_evidence(self, question, evidence):
        return await self.request("select_evidence", {"question": question, "evidence": evidence}, self.selection(evidence))

    async def repair_evidence_selection(self, question, evidence, invalid_output, diagnostics):
        return await self.request("select_evidence:repair", {"question": question, "evidence": evidence,
            "invalid_output": invalid_output, "diagnostics": diagnostics}, self.selection(evidence))


async def test_precision_floor_is_reused_by_its_repair_and_isolated_from_extract_and_analysts(tmp_path):
    provider = PrecisionCapacityProvider({"extract": ["length"], "select_evidence": ["length", "schema"],
                                           "select_evidence:repair": ["length"]})
    result = await execute_capacity_batches(tmp_path, provider)
    budgets = lambda operation: [call["budget"] for call in provider.calls if call["operation"] == operation]
    assert budgets("extract") == [8, 16, 16]
    assert budgets("select_evidence") == [16, 32]
    assert budgets("select_evidence:repair") == [32, 64]
    assert budgets("authority_analyst") == budgets("cross_validation_analyst") == [16]
    assert budgets("arbitrate") == [32]
    assert result["evidence_selection"]["selected_count"] == 2
    assert len(result["usage"]) == len(provider.calls)
    assert all(call["budget"] == call["budget_after_await"] for call in provider.calls)


async def test_learned_output_floor_does_not_escape_one_question_execution(tmp_path):
    provider = CapacityProvider({"extract": ["length"]})
    await execute_capacity_batches(tmp_path, provider, thread_id="first-question-execution")
    offset = len(provider.calls)
    await execute_capacity_batches(tmp_path, provider, thread_id="new-question-execution")
    assert [call["budget"] for call in provider.calls[offset:] if call["operation"] == "extract"] == [8, 8]
    assert get_requested_output_budget() is None


@pytest.mark.parametrize("resumed_cap,expected_budget", [(128, 16), (12, 12)])
async def test_paid_batch_checkpoint_restores_stage_floor_on_resume_with_current_cap(
    tmp_path, resumed_cap, expected_budget,
):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    stopped = False

    def pause_after_expanded_response(operation, index):
        nonlocal stopped
        if operation == "extract" and index == 1:
            stopped = True

    first = CapacityProvider({"extract": ["length"]}, on_call=pause_after_expanded_response)
    with pytest.raises(AnalysisCancelled):
        await execute_capacity_batches(tmp_path, first, cancel_check=lambda: stopped)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "floors.sqlite3")) as saver:
        saved = await saver.aget_tuple({"configurable": {"thread_id": "stage-floors"}})
    assert saved.checkpoint["channel_values"]["source_batch_index"] == 1
    assert saved.checkpoint["channel_values"]["output_capacity_floors"] == {"extract": 16}
    resumed, events = CapacityProvider(cap=resumed_cap), []
    result = await execute_capacity_batches(tmp_path, resumed, thread_id="continued-floors",
        resume_from_thread_id="stage-floors", events=events)
    assert [call["budget"] for call in resumed.calls if call["operation"] == "extract"] == [expected_budget]
    chosen = [value for kind, value in events if kind == "model_capacity_selected" and value["stage"] == "extract"]
    assert len(chosen) == 1 and chosen[0]["budget_origin"] == "learned_stage_floor"
    assert chosen[0]["max_output_tokens"] <= resumed_cap
    assert not any(kind == "model_capacity_retry_started" for kind, _ in events)
    assert result["source_coverage"]["completed_batches"] == 2
    assert len([record for record in result["usage"] if record["operation"] == "extract"]) == 1
