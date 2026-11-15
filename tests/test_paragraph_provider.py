"""Paragraph extraction at the real SDK boundary, intercepted HTTP only.

Only synthetic source data and explicit placeholder credentials are used.
Transport replies are the NEW paragraph wire; no historical window fixtures
are translated into successes. These tests make no semantic-quality claim.
"""
from copy import deepcopy
import json
import socket

import pytest
import pytest_asyncio

from app.agents.provider import ModelOutputInvalid, capture_usage, request_output_budget
from app.domain.paragraph_assertions import (
    ParagraphAssertionOutput, build_paragraph_assertion_request,
)
from app.domain.spans import validate_span
from test_deepseek_provider import intercepted_adapter


QUESTION = {"id": "paragraph-question", "raw_question_id": "q_synthetic",
            "question": "截至周五场馆是否已开放，验收是否完成？"}
PARAGRAPHS = [
    {"id": "paragraph-first", "document_id": "document-one", "raw_doc_id": "doc-one",
     "raw_para_id": "para-one", "raw_publish_time": "2040-01-03",
     "text": "🦊场馆周五试开放。正式验收尚待签字。\r\n负责人称开放不等于验收通过。"},
    {"id": "paragraph-background", "document_id": "document-two", "raw_doc_id": "doc-two",
     "raw_para_id": "para-two", "text": "场馆外墙采用蓝色。"},
]


@pytest_asyncio.fixture(autouse=True)
async def prohibit_socket_network(monkeypatch):
    # Install only after asyncio creates its Windows loopback self-pipe.
    def blocked(*_args, **_kwargs):
        pytest.fail("MockTransport tests must not open a network socket")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def annotation(*, fact="场馆周五试开放，正式验收尚待签字，负责人称开放不等于验收通过。", speaker="负责人"):
    return {"fact_des": fact,
            "argument": {"时间": "周五", "主体": speaker, "地点": None,
                         "内容": fact, "影响": None, "数值口径": None},
            "entities": ["场馆", speaker]}


def reply_value(payload):
    return {"source_catalog_id": payload["source_catalog_id"], "decisions": {
        paragraph["id"]: {
            "disposition": "relevant" if index == 0 else "no_target_fact",
            "annotations": [annotation()] if index == 0 else [],
            "reason": "直接记录开放状态和未验收限定。" if index == 0 else "仅为外墙背景。",
        } for index, paragraph in enumerate(payload["source_paragraphs"])}}


def sdk_reply(*, mutate=None, transform=None):
    def reply(body):
        payload = json.loads(body["messages"][1]["content"])
        value = reply_value(payload)
        if mutate is not None:
            mutate(value, payload)
        encoded = json.dumps(value, ensure_ascii=False)
        return transform(encoded) if transform is not None else encoded
    return reply


def request_payload(request):
    return json.loads(json.loads(request.content)["messages"][1]["content"])


async def test_paragraph_payload_is_fixed_complete_source_with_program_owned_geometry_and_audit():
    sources, q = deepcopy(PARAGRAPHS), deepcopy(QUESTION)
    before = deepcopy((sources, q))
    async with intercepted_adapter(content=sdk_reply(), reasoning_content="PRIVATE_HIDDEN_THOUGHT") as (adapter, requests):
        with capture_usage() as usage:
            result = await adapter.extract(q, sources)
    assert len(requests) == 1 and len(result) == 1
    payload = request_payload(requests[0])
    assert payload["question"] == QUESTION
    assert [(p["id"], p["paragraph_id"], p["text"]) for p in payload["source_paragraphs"]] == [
        (f"P{index + 1}", p["id"], p["text"]) for index, p in enumerate(sources)]
    assert "source_window_catalog" not in payload and "paragraphs" not in payload
    assert not any("source_window_id" in row for row in payload["source_paragraphs"])
    assert result[0]["paragraph_id"] == sources[0]["id"]
    assert result[0]["quote"] == sources[0]["text"]
    assert (result[0]["start_cp"], result[0]["end_cp"]) == (0, len(sources[0]["text"]))
    validate_span(sources[0]["text"], result[0]["quote"], 0, len(sources[0]["text"]))
    audit = result.source_review
    assert audit["complete"] and audit["paragraph_count"] == 2 and audit["candidate_count"] == 1
    assert audit["semantically_verified"] is False
    assert [(d["id"], d["disposition"], d["annotation_count"]) for d in audit["decisions"]] == [
        ("P1", "relevant", 1), ("P2", "no_target_fact", 0)]
    assert audit["source_catalog_id"] == payload["source_catalog_id"]
    assert all(len(row["paragraph_sha256"]) == 64 for row in audit["decisions"])
    assert "PRIVATE_HIDDEN_THOUGHT" not in repr(result) + repr(audit)
    assert (sources, q) == before
    assert len(usage) == 1 and usage[0]["operation"] == "extract" and usage[0]["total_tokens"] == 18


async def test_same_paragraph_keeps_different_speakers_and_opposing_assertions():
    def add_counterclaim(value, _payload):
        value["decisions"]["P1"]["annotations"].append(annotation(
            fact="另一名负责人称场馆周五尚未开放。", speaker="另一名负责人"))
    async with intercepted_adapter(content=sdk_reply(mutate=add_counterclaim)) as (adapter, requests):
        result = await adapter.extract(QUESTION, PARAGRAPHS)
    assert len(requests) == 1 and len(result) == 2
    assert [row["argument"]["主体"] for row in result] == ["负责人", "另一名负责人"]
    assert result[0]["quote"] == result[1]["quote"] == PARAGRAPHS[0]["text"]
    assert result.source_review["decisions"][0]["annotation_count"] == 2


async def test_empty_paragraph_batch_still_makes_one_real_adapter_request_and_records_usage():
    async with intercepted_adapter(content=sdk_reply()) as (adapter, requests):
        with capture_usage() as usage:
            result = await adapter.extract(QUESTION, [])
    assert len(requests) == len(usage) == 1 and result == []
    assert request_payload(requests[0])["source_paragraphs"] == []
    assert result.source_review["complete"] and result.source_review["decisions"] == []
    assert usage[0]["operation"] == "extract" and usage[0]["total_tokens"] == 18


async def test_every_paragraph_needs_a_disposition_and_all_omissions_remain_visible_in_safe_diagnostics():
    paragraphs = [{"id": f"source-{i}", "text": "开放状态未核实。"} for i in range(21)]
    def omit(value, _payload):
        value["decisions"] = {"P1": value["decisions"]["P1"]}
    async with intercepted_adapter(content=sdk_reply(mutate=omit)) as (adapter, requests):
        with capture_usage() as usage:
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.extract(QUESTION, paragraphs)
    assert len(requests) == len(usage) == 1
    error = caught.value
    assert error.stage == "extract" and error.code == "model_output_invalid"
    assert len(error.details["schema_errors"]) == 20
    assert all(row["type"] == "missing" for row in error.details["schema_errors"])
    assert "source-" not in json.dumps(error.details)


@pytest.mark.parametrize("target,field,value", [
    ("decision", "quote", "PRIVATE_FORGED_QUOTE"),
    ("decision", "paragraph_id", "PRIVATE_OTHER_SOURCE"),
    ("annotation", "quote", "PRIVATE_FORGED_QUOTE"),
    ("annotation", "paragraph_id", "PRIVATE_OTHER_SOURCE"),
    ("annotation", "source_window_id", "W_OTHER_PRIVATE"),
    ("annotation", "start_cp", 1),
    ("annotation", "end_cp", 3),
])
async def test_model_cannot_supply_quote_source_ids_or_geometry_at_any_wire_level(target, field, value):
    def forge(output, _payload):
        row = output["decisions"]["P1"]
        if target == "annotation": row = row["annotations"][0]
        row[field] = value
    async with intercepted_adapter(content=sdk_reply(mutate=forge)) as (adapter, requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.extract(QUESTION, PARAGRAPHS)
    assert len(requests) == 1
    assert any(row["type"] == "extra_forbidden" for row in caught.value.details["schema_errors"])
    assert "PRIVATE" not in str(caught.value) + repr(caught.value.details)


@pytest.mark.parametrize("change", ["text", "metadata", "question", "order"])
async def test_previous_request_binding_cannot_be_reused_for_a_changed_request(change):
    seen = []
    def reuse_old_binding(output, payload):
        if not seen: seen.append(payload["source_catalog_id"])
        output["source_catalog_id"] = seen[0]
    paragraphs, q = deepcopy(PARAGRAPHS), deepcopy(QUESTION)
    async with intercepted_adapter(content=sdk_reply(mutate=reuse_old_binding)) as (adapter, requests):
        with capture_usage() as usage:
            first = await adapter.extract(q, paragraphs)
            assert len(first) == 1
            if change == "text": paragraphs[0]["text"] += "更新更正。"
            elif change == "metadata": paragraphs[0]["raw_publish_time"] = "2041-01-03"
            elif change == "question": q["question"] += "实际批准者是谁？"
            else: paragraphs.reverse()
            with pytest.raises(ModelOutputInvalid):
                await adapter.extract(q, paragraphs)
    assert len(requests) == len(usage) == 2
    assert request_payload(requests[0])["source_catalog_id"] != request_payload(requests[1])["source_catalog_id"]


@pytest.mark.parametrize("where", ["binding", "paragraph", "escaped_paragraph", "reason", "annotation", "argument"])
async def test_raw_duplicate_json_keys_are_rejected_before_last_value_overwrites_decisions(where):
    def duplicate(text):
        if where == "binding": return text.replace('"source_catalog_id":', '"source_catalog_id": "old", "source_catalog_id":', 1)
        if where == "paragraph": return text.replace('"P1":', '"P1": {}, "P1":', 1)
        if where == "escaped_paragraph": return text.replace('"P1":', '"P\\u0031": {}, "P1":', 1)
        if where == "reason": return text.replace('"reason":', '"reason": "PRIVATE_DUPLICATE", "reason":', 1)
        if where == "annotation": return text.replace('"fact_des":', '"fact_des": "PRIVATE_DUPLICATE", "fact_des":', 1)
        return text.replace('"主体":', '"主体": "PRIVATE_DUPLICATE", "主体":', 1)
    async with intercepted_adapter(content=sdk_reply(transform=duplicate)) as (adapter, requests):
        with capture_usage() as usage:
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.extract(QUESTION, PARAGRAPHS)
    assert len(requests) == len(usage) == 1
    assert caught.value.details == {"check": "paragraph_assertion_duplicate_json_key", "duplicate_key_count": 1}
    assert caught.value.stage == "extract" and caught.value.code == "model_output_invalid"
    assert "PRIVATE_DUPLICATE" not in str(caught.value) + repr(caught.value.details)


@pytest.mark.parametrize("kind", ["relevant_empty", "background_nonempty", "missing_reason", "extra_alias", "bad_argument"])
async def test_disposition_and_annotation_contracts_are_not_silently_normalized_or_dropped(kind):
    def corrupt(output, _payload):
        if kind == "relevant_empty": output["decisions"]["P1"]["annotations"] = []
        elif kind == "background_nonempty": output["decisions"]["P2"]["annotations"] = [annotation()]
        elif kind == "missing_reason": output["decisions"]["P2"].pop("reason")
        elif kind == "extra_alias": output["decisions"]["P_PRIVATE"] = deepcopy(output["decisions"]["P2"])
        else: output["decisions"]["P1"]["annotations"][0]["argument"]["PRIVATE_ARGUMENT"] = "PRIVATE_VALUE"
    async with intercepted_adapter(content=sdk_reply(mutate=corrupt)) as (adapter, requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.extract(QUESTION, PARAGRAPHS)
    assert len(requests) == 1 and caught.value.code == "model_output_invalid"
    assert "PRIVATE" not in str(caught.value) + repr(caught.value.details)


async def test_identical_annotations_are_explicitly_rejected_after_wire_validation():
    def duplicate(output, _payload):
        output["decisions"]["P1"]["annotations"] *= 2
    async with intercepted_adapter(content=sdk_reply(mutate=duplicate)) as (adapter, requests):
        with capture_usage() as usage:
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.extract(QUESTION, PARAGRAPHS)
    assert len(requests) == len(usage) == 1
    assert caught.value.details == {"check": "paragraph_assertion_duplicate_annotation"}


async def test_unknown_question_answers_and_unknown_metadata_never_enter_the_http_prompt():
    q, paragraphs = deepcopy(QUESTION), deepcopy(PARAGRAPHS)
    q.update(conclusion="PRIVATE_REFERENCE_CONCLUSION", supporting_facts=[{"fact_id": "PRIVATE_GOLD_FACT"}],
             api_key="PRIVATE_NOT_AN_ACTUAL_KEY", unrelated={"nested": "PRIVATE_UNKNOWN"})
    paragraphs[0]["unrecognized_answer"] = "PRIVATE_SOURCE_EXTRA"
    async with intercepted_adapter(content=sdk_reply()) as (adapter, requests):
        result = await adapter.extract(q, paragraphs)
    payload = request_payload(requests[0])
    assert payload["question"] == QUESTION
    assert "PRIVATE" not in requests[0].content.decode()
    assert result.source_review["complete"]


async def test_initial_and_repair_requests_preserve_task_local_budgets_usage_and_distinct_reasoning_modes():
    overrides = {"model_extraction_max_output_tokens": 32768}
    async with intercepted_adapter(content=sdk_reply(), reasoning_effort="low", budget_overrides=overrides) as (adapter, requests):
        with capture_usage() as usage:
            first = await adapter.extract(QUESTION, PARAGRAPHS)
            with request_output_budget(65536):
                repaired = await adapter.repair_extract(QUESTION, PARAGRAPHS,
                    {"PRIVATE_BAD_BODY": "must not become source evidence"}, {"check": "paragraph_assertion_contract"})
            third = await adapter.extract(QUESTION, PARAGRAPHS)
    assert len(requests) == len(usage) == 3
    assert [json.loads(r.content)["max_tokens"] for r in requests] == [32768, 65536, 32768]
    assert [json.loads(r.content)["thinking"] for r in requests] == [
        {"type": "disabled"}, {"type": "enabled"}, {"type": "disabled"}]
    assert [json.loads(r.content)["reasoning_effort"] for r in requests] == ["none", "low", "none"]
    assert all("max_completion_tokens" not in json.loads(r.content) for r in requests)
    assert all(u["operation"] == "extract" and u["total_tokens"] == 18 for u in usage)
    assert first == repaired == third
    assert first.source_review == repaired.source_review == third.source_review
    assert "invalid_output" not in request_payload(requests[1])
    assert "PRIVATE_BAD_BODY" not in requests[1].content.decode()


async def test_one_explicit_structural_repair_reuses_full_sources_and_persists_complete_audit():
    calls = 0
    def first_invalid_then_valid(body):
        nonlocal calls
        calls += 1
        payload = json.loads(body["messages"][1]["content"])
        output = reply_value(payload)
        if calls == 1: output["decisions"].pop("P2")
        return json.dumps(output, ensure_ascii=False)
    async with intercepted_adapter(content=first_invalid_then_valid, reasoning_content="PRIVATE_HIDDEN") as (adapter, requests):
        with capture_usage() as usage:
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.extract(QUESTION, PARAGRAPHS)
            result = await adapter.repair_extract(QUESTION, PARAGRAPHS, caught.value.candidate, caught.value.details)
    assert len(requests) == len(usage) == 2
    first, second = map(request_payload, requests)
    assert first["source_catalog_id"] == second["source_catalog_id"]
    assert first["source_paragraphs"] == second["source_paragraphs"]
    assert len(result) == 1 and len(result.source_review["decisions"]) == 2
    assert result.source_review["complete"] and "PRIVATE_HIDDEN" not in repr(result.source_review)


async def test_truncated_output_preserves_extract_failure_stage_and_known_usage():
    async with intercepted_adapter("length", content=sdk_reply()) as (adapter, requests):
        with capture_usage() as usage:
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.extract(QUESTION, PARAGRAPHS)
    assert len(requests) == len(usage) == 1
    assert caught.value.stage == "extract" and caught.value.code == "model_output_truncated"
    assert usage[0]["total_tokens"] == 18


async def test_catalog_overflow_fails_before_any_mock_or_paid_request(monkeypatch):
    import app.domain.source_catalog as module
    monkeypatch.setattr(module, "MAX_SOURCE_CODEPOINTS", 4)
    async with intercepted_adapter(content=sdk_reply()) as (adapter, requests):
        with capture_usage() as usage:
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.extract(QUESTION, PARAGRAPHS)
    assert requests == [] and usage == []
    assert caught.value.stage == "extract" and caught.value.code == "model_input_too_large"


def test_dynamic_schema_inherits_provider_marker_for_complete_error_and_raw_json_handling():
    _payload, schema, _catalog = build_paragraph_assertion_request(QUESTION, PARAGRAPHS)
    assert issubclass(schema, ParagraphAssertionOutput)
