"""Synthetic safety regression for an UNAPPLIED post-pilot patch."""
from copy import deepcopy
import json
import socket

import pytest
import pytest_asyncio

from app.domain.paragraph_assertions import safe_paragraph_diagnostics
from app.agents.provider import capture_usage
from test_deepseek_provider import intercepted_adapter
from test_paragraph_provider import QUESTION, PARAGRAPHS, sdk_reply, request_payload


@pytest_asyncio.fixture(autouse=True)
async def network_guard(monkeypatch):
    def blocked(*_args, **_kwargs):
        pytest.fail("unapplied safety patch tests must not open network sockets")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def test_diag_filter_keeps_only_fixed_codes_and_paths_without_mutating_input():
    data = {"check": "paragraph_assertion_contract", "duplicate_key_count": 2,
        "source_location_choices": [{"quote": "PRIVATE_WRONG_SOURCE", "paragraph_id": "private-other"}],
        "answer": "PRIVATE_GOLD", "body": "PRIVATE_BODY", "private": {"token": "PRIVATE_TOKEN"},
        "schema_errors": [{"loc": ["decisions", "P2", "PRIVATE_FIELD", "annotations", 0, "argument", "主体"],
                           "type": "missing", "input": "PRIVATE_VALUE", "ctx": {"body": "PRIVATE_CONTEXT"}}]}
    before = deepcopy(data)
    result = safe_paragraph_diagnostics(data, paragraph_count=2)
    assert result == {"check": "paragraph_assertion_contract", "duplicate_key_count": 2,
        "schema_errors": [{"loc": ["decisions", 1, "<field>", "annotations", 0, "argument", "主体"], "type": "missing"}]}
    assert "PRIVATE" not in json.dumps(result)
    assert data == before


@pytest.mark.parametrize("data", [None, [], "PRIVATE_DIAGNOSTIC", {"check": ["PRIVATE"]},
    {"check": "PRIVATE_CHECK", "schema_errors": "PRIVATE_ERRORS"},
    {"check": {}, "duplicate_key_count": True}, {"duplicate_key_count": -1}])
def test_arbitrary_diagnostic_shapes_and_unknown_values_do_not_leak(data):
    result = safe_paragraph_diagnostics(data, paragraph_count=2)
    assert result == {"check": "paragraph_assertion_contract"}


def test_unknown_alias_locations_types_boolean_indexes_and_extra_data_are_redacted():
    result = safe_paragraph_diagnostics({"schema_errors": [
        {"loc": ["decisions", "P999", "PRIVATE", True, -1, {}, "reason"], "type": "PRIVATE_TYPE"},
        {"loc": "PRIVATE_LOC", "type": {"PRIVATE": 1}}, "PRIVATE_ROW"]}, paragraph_count=2)
    assert result == {"check": "paragraph_assertion_contract", "schema_errors": [
        {"loc": ["decisions", "<field>", "<field>", "<field>", "<field>", "<field>", "reason"], "type": "value_error"},
        {"loc": [], "type": "value_error"}]}


@pytest.mark.parametrize("count", [None, True, -1, 100000000, "PRIVATE"])
def test_invalid_catalog_count_does_not_grant_alias_authority(count):
    result = safe_paragraph_diagnostics({"schema_errors": [{"loc": ["decisions", "P1"], "type": "missing"}]}, paragraph_count=count)
    assert result["schema_errors"][0]["loc"] == ["decisions", "<field>"]


def test_all_real_missing_paragraph_diagnostics_remain_complete_not_capped_at_twelve():
    rows = [{"loc": ["decisions", f"P{i + 1}"], "type": "missing"} for i in range(40)]
    safe = safe_paragraph_diagnostics({"schema_errors": rows}, paragraph_count=40)
    assert len(safe["schema_errors"]) == 40
    assert safe["schema_errors"][-1]["loc"] == ["decisions", 39]


async def test_repair_http_ignores_forged_hints_body_gold_and_recompiles_only_original_sources():
    question = {**QUESTION, "conclusion": "PRIVATE_GOLD_CONCLUSION"}
    forged = {"check": "PRIVATE_CHECK", "answer": "PRIVATE_GOLD", "body": "PRIVATE_BODY",
        "source_location_choices": [{"paragraph_id": "PRIVATE_OTHER", "quote": "PRIVATE_FORGED_QUOTE"}],
        "schema_errors": [{"loc": ["decisions", "P1", "PRIVATE_LOC"], "type": "PRIVATE_TYPE", "input": "PRIVATE_INPUT"}]}
    async with intercepted_adapter(content=sdk_reply()) as (adapter, requests):
        with capture_usage() as usage:
            result = await adapter.repair_extract(question, PARAGRAPHS, {"PRIVATE_OLD_OUTPUT": "body"}, forged)
    assert len(requests) == len(usage) == 1
    payload = request_payload(requests[0])
    assert payload["question"] == QUESTION
    assert "PRIVATE" not in requests[0].content.decode()
    assert payload["diagnostics"] == {"check": "paragraph_assertion_contract", "schema_errors": [
        {"loc": ["decisions", 0, "<field>"], "type": "value_error"}]}
    assert [p["text"] for p in payload["source_paragraphs"]] == [p["text"] for p in PARAGRAPHS]
    assert result[0]["quote"] == PARAGRAPHS[0]["text"]
    assert result.source_review["complete"] and result.source_review["candidate_count"] == 1
