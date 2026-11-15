"""v16 single-request contract and actual SDK serialization, synthetic only.

No competition sample/gold, .env, live credentials or external model is read.
All SDK calls use the existing explicit-placeholder MockTransport fixture.
The supplied semantic labels test gates; they do not measure model quality.
"""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from app.agents.provider import ModelOutputInvalid, capture_usage, request_output_budget
from app.domain.compiled_scope import (
    WIRE_VERSION, build_compiled_scope_request, compiled_repair_context,
    decode_compiled_scope_output, literal_target_catalog, resolve_compiled_scope,
)
from app.domain.strict_verification import (
    ScopeVerificationInvalid, build_scope_request, resolve_scope_verification,
    validate_scope_ledger, verified_fact_export_rows,
)
from test_deepseek_provider import intercepted_adapter


Q = {"question_id": "synthetic-compound-venue", "question": "场馆是否通过验收，并已在周五向公众开放？"}
KEYS = ("时间", "主体", "地点", "内容", "影响", "数值口径")


def source(identity="e1", **arguments):
    quote = "场馆已试运行，但正式验收尚待签字。"
    return {"id": identity, "paragraph_id": "p1", "raw_doc_id": "d1", "raw_para_id": "rp1",
        "quote": quote, "start_cp": 4, "end_cp": 4 + len(quote),
        "fact_des": "场馆已试运行，正式验收尚待签字。", "argument": {key: arguments.get(key) for key in KEYS},
        "entities": ["场馆"]}


def row_wire(row, *, token="T0.qualifier.certainty", quote_bridge=False, target_bridge=False):
    return {"fact_entailment": "entailed", "target_binding_id": token,
        "quote_requires_unstated_bridge": quote_bridge, "target_requires_unstated_bridge": target_bridge,
        "field_checks": {"fact_des": "entailed", "entities": "entailed" if row["entities"] else "not_present",
            "argument": {key: "entailed" if value else "not_present" for key, value in row["argument"].items()}},
        "target_fact_present_but_annotation_incomplete": False,
        "reason": "引文限定验收状态；无需独自证明开放状态。"}


def output(rows, question=Q):
    payload, _schema, _views = build_compiled_scope_request(question, rows)
    return {"candidate_catalog_id": payload["candidate_catalog_id"], "target_catalog_id": payload["target_catalog_id"],
        "decisions": {row["id"]: row_wire(row) for row in payload["candidate_catalog"]}}


def sdk_reply(*, mutate=None):
    def reply(body):
        payload = json.loads(body["messages"][1]["content"])
        value = {"candidate_catalog_id": payload["candidate_catalog_id"], "target_catalog_id": payload["target_catalog_id"],
            "decisions": {row["id"]: row_wire(row) for row in payload["candidate_catalog"]}}
        if mutate:
            mutate(value, payload)
        return json.dumps(value, ensure_ascii=False)
    return reply


@pytest.mark.parametrize("text", [Q["question"], "场馆已验收？", "🦊如果甲完成复核，乙是否在周五开放；否则如何？", "  甲；\r\n乙？  "])
def test_target_hints_are_deterministic_exact_codepoint_slices_with_full_question(text):
    targets = literal_target_catalog(text)
    assert targets == literal_target_catalog(text)
    assert targets[0] == {"id": "T0", "start_cp": 0, "end_cp": len(text), "text": text, "kind": "full_question"}
    for target in targets:
        assert text[target["start_cp"]:target["end_cp"]] == target["text"]
        assert target["text"] in text
    assert len({(row["start_cp"], row["end_cp"]) for row in targets}) == len(targets)
    assert all(row["kind"] == "punctuation_clause_hint" for row in targets[1:])


def test_clause_hint_does_not_claim_negation_condition_or_semantic_target_certification():
    q = {"question": "如果设备未通过复核，是否仍可启动；还是必须等待？"}
    payload, _, _ = build_compiled_scope_request(q, [source()])
    assert payload["question"] == q
    assert payload["target_catalog"][0]["text"] == q["question"]
    assert payload["target_catalog"][1]["text"] == "如果设备未通过复核，"
    assert "semantically_verified" not in json.dumps(payload)
    assert "如果" in payload["question"]["question"] and "未" in payload["question"]["question"]


@pytest.mark.parametrize("quote_bridge,target_bridge", [(False, False), (False, True), (True, False), (True, True)])
def test_single_wire_compiles_both_bridge_values_without_flipping_either(quote_bridge, target_bridge):
    rows = [source()]
    wire = output(rows)
    wire["decisions"]["C1"].update(quote_requires_unstated_bridge=quote_bridge, target_requires_unstated_bridge=target_bridge)
    canonical = decode_compiled_scope_output(wire, Q, rows)
    row = canonical["decisions"]["C1"]
    assert row["quote_requires_unstated_bridge"] is quote_bridge
    assert row["target_requires_unstated_bridge"] is target_bridge
    assert row["requires_unstated_bridge"] is (quote_bridge or target_bridge)
    ledger = resolve_compiled_scope(wire, Q, rows)
    assert ledger == resolve_scope_verification(canonical, Q, rows)
    assert bool(ledger["selected_evidence_ids"]) is not (quote_bridge or target_bridge)
    assert validate_scope_ledger(ledger, Q, rows) == ledger


@pytest.mark.parametrize("suffix", ["direct", "counter", *["qualifier." + d for d in ("subject", "time", "scope", "value", "attribution", "certainty", "causality")]])
def test_each_compiled_inside_branch_maps_to_a_legal_original_canonical_decision(suffix):
    rows = [source()]
    wire = output(rows)
    wire["decisions"]["C1"]["target_binding_id"] = "T1." + suffix
    canonical = decode_compiled_scope_output(wire, Q, rows)
    decision = canonical["decisions"]["C1"]
    assert decision["target_question_text"] == "场馆是否通过验收，"
    if suffix.startswith("qualifier."):
        assert decision["target_relation"] == "necessary_qualifier"
        assert decision["qualifier_dimension"] == suffix.split(".")[1]
    else:
        assert decision["target_relation"] == ("direct_answer" if suffix == "direct" else "explicit_counterclaim")
        assert decision["qualifier_dimension"] is None
    assert resolve_compiled_scope(wire, Q, rows) == resolve_scope_verification(canonical, Q, rows)


@pytest.mark.parametrize("token", ["outside_target", "uncertain"])
def test_fixed_nonformal_tokens_remain_audited_not_selected(token):
    rows = [source()]
    wire = output(rows)
    wire["decisions"]["C1"]["target_binding_id"] = token
    result = resolve_compiled_scope(wire, Q, rows)
    assert result["selected_evidence_ids"] == []
    assert result["assessments"][0]["target_relation"] == token
    assert result["assessments"][0]["target_question_text"] is None
    assert result["assessments"][0]["qualifier_dimension"] is None


@pytest.mark.parametrize("field", ["fact_des", "entities", *KEYS])
def test_all_original_field_gates_are_still_enforced(field):
    rows = [source(主体="场馆")]
    wire = output(rows)
    checks = wire["decisions"]["C1"]["field_checks"]
    if field in KEYS: checks["argument"][field] = "unsupported"
    else: checks[field] = "needs_narrowing"
    result = resolve_compiled_scope(wire, Q, rows)
    assert result["selected_evidence_ids"] == []
    assert len(result["narrowing_view_ids"]) == 1
    assert result["assessments"][0]["field_issues"]


@pytest.mark.parametrize("field", ["fact_entailment", "target_binding_id", "quote_requires_unstated_bridge", "target_requires_unstated_bridge", "field_checks", "target_fact_present_but_annotation_incomplete", "reason"])
def test_no_compiled_row_field_can_be_defaulted(field):
    rows = [source()]
    wire = output(rows)
    del wire["decisions"]["C1"][field]
    with pytest.raises(ScopeVerificationInvalid, match="scope_compiled_contract"):
        resolve_compiled_scope(wire, Q, rows)


@pytest.mark.parametrize("mutation", ["unknown_token", "qualifier_without_dimension", "outside_with_dimension", "wrong_catalog", "missing_catalog", "missing_candidate", "extra_candidate", "legacy_target", "aggregate_bridge", "null_bridge", "string_bridge", "null_checks"])
def test_illegal_wire_never_silently_normalizes_to_a_canonical_decision(mutation):
    rows = [source()]
    wire = output(rows)
    row = wire["decisions"]["C1"]
    if mutation == "unknown_token": row["target_binding_id"] = "PRIVATE_UNKNOWN_TOKEN"
    elif mutation == "qualifier_without_dimension": row["target_binding_id"] = "T0.qualifier"
    elif mutation == "outside_with_dimension": row["target_binding_id"] = "outside_target.time"
    elif mutation == "wrong_catalog": wire["target_catalog_id"] = "G_other"
    elif mutation == "missing_catalog": del wire["target_catalog_id"]
    elif mutation == "missing_candidate": del wire["decisions"]["C1"]
    elif mutation == "extra_candidate": wire["decisions"]["C2"] = deepcopy(row)
    elif mutation == "legacy_target": row["target_question_text"] = "验收完成"
    elif mutation == "aggregate_bridge": row["requires_unstated_bridge"] = False
    elif mutation == "null_bridge": row["target_requires_unstated_bridge"] = None
    elif mutation == "string_bridge": row["quote_requires_unstated_bridge"] = "false"
    elif mutation == "null_checks": row["field_checks"] = None
    with pytest.raises(ScopeVerificationInvalid, match="scope_compiled_contract"):
        resolve_compiled_scope(wire, Q, rows)


@pytest.mark.parametrize("mutation", ["question", "question_id", "fact", "source", "quote", "argument"])
def test_catalog_binds_original_question_all_candidates_and_source_geometry(mutation):
    q, rows = deepcopy(Q), [source()]
    wire = output(rows, q)
    if mutation == "question": q["question"] += "为什么？"
    elif mutation == "question_id": q["question_id"] = "other-id"
    elif mutation == "fact": rows[0]["fact_des"] = "不同注释。"
    elif mutation == "source": rows[0]["raw_doc_id"] = "other-doc"
    elif mutation == "quote":
        rows[0]["quote"] += "限定。"
        rows[0]["end_cp"] += 3
    elif mutation == "argument": rows[0]["argument"]["时间"] = "后来"
    with pytest.raises(ScopeVerificationInvalid, match="scope_compiled_contract"):
        resolve_compiled_scope(wire, q, rows)


def test_source_protocol_digest_participates_in_target_catalog_binding(monkeypatch):
    import app.domain.compiled_scope as module
    original, _, _ = build_compiled_scope_request(Q, [source()])
    monkeypatch.setattr(module, "SOURCE_PROTOCOL_SHA256", "0" * 64)
    changed, _, _ = build_compiled_scope_request(Q, [source()])
    assert original["candidate_catalog_id"] == changed["candidate_catalog_id"]
    assert original["target_catalog_id"] != changed["target_catalog_id"]


def test_question_projection_and_catalog_do_not_take_answer_like_metadata():
    payload, _, _ = build_compiled_scope_request({**Q, "conclusion": "SYNTHETIC_FINAL_LEAK", "supporting_facts": ["SYNTHETIC_REF_LEAK"]}, [source()])
    assert payload["question"] == Q
    assert "SYNTHETIC_" not in json.dumps(payload)


def test_old_canonical_ledger_replays_unchanged_but_old_wire_is_not_accepted():
    rows = [source()]
    old_payload, _, _ = build_scope_request(Q, rows)
    decision = {"fact_entailment": "entailed", "target_relation": "necessary_qualifier", "target_question_text": "通过验收",
        "qualifier_dimension": "certainty", "requires_unstated_bridge": False,
        "target_fact_present_but_annotation_incomplete": False, "reason": "历史有归属限定。"}
    old_output = {"candidate_catalog_id": old_payload["candidate_catalog_id"], "decisions": {"C1": decision}}
    ledger = resolve_scope_verification(old_output, Q, rows)
    original = deepcopy(ledger)
    assert validate_scope_ledger(ledger, Q, rows) == original
    assert "quote_requires_unstated_bridge" not in ledger["assessments"][0]
    with pytest.raises(ScopeVerificationInvalid, match="scope_compiled_contract"):
        resolve_compiled_scope(old_output, Q, rows)


def test_canonical_ledger_and_export_contain_no_new_wire_keys_and_sources_stay_fixed():
    rows = [source()]
    original = deepcopy(rows)
    ledger = resolve_compiled_scope(output(rows), Q, rows)
    exports = verified_fact_export_rows(ledger, Q, rows)
    assert len(exports) == 1 and exports[0]["quote"] == rows[0]["quote"]
    assert set(exports[0]) == {"doc_id", "para_id", "quote", "fact_des", "argument", "entities"}
    for key in ("target_catalog_id", "target_binding_id", "scope_wire_version", "source_protocol_sha256"):
        assert key not in json.dumps(ledger) and key not in json.dumps(exports)
    assert rows == original


def test_compilation_does_not_certify_wrong_model_semantics_or_rescue_target_bridge():
    rows = [source()]
    rows[0]["fact_des"] = "场馆正式验收已签字。"  # Deliberately opposite to fixed quote.
    wire = output(rows)  # Deliberately false model entailment label.
    assert resolve_compiled_scope(wire, Q, rows)["selected_evidence_ids"] == ["e1"]
    wire["decisions"]["C1"]["target_requires_unstated_bridge"] = True
    assert resolve_compiled_scope(wire, Q, rows)["selected_evidence_ids"] == []


async def test_actual_provider_is_one_model_request_with_original_usage_operation_and_budget():
    rows = [source()]
    with capture_usage() as usage:
        async with intercepted_adapter(content=sdk_reply()) as (adapter, requests):
            with request_output_budget(24576):
                ledger = await adapter.verify_evidence_scope(Q, rows)
    assert len(requests) == len(usage) == 1
    assert usage[0]["operation"] == "verify_evidence_scope" and usage[0]["total_tokens"] == 18
    request = json.loads(requests[0].content)
    assert request["max_tokens"] == 24576 and request["thinking"] == {"type": "enabled"}
    payload = json.loads(request["messages"][1]["content"])
    assert payload["scope_wire_version"] == WIRE_VERSION
    assert payload["target_catalog"][0]["text"] == Q["question"]
    assert ledger["selected_evidence_ids"] == ["e1"]
    assert adapter.capacity_policy("verify_evidence_scope")["retries"] == 2
    assert "punctuation_clause_hint" in json.dumps(payload)


async def test_actual_raw_duplicate_nested_key_is_rejected_before_pydantic_overwrite():
    def duplicate(body):
        clean = sdk_reply()(body)
        return clean.replace('"quote_requires_unstated_bridge": false',
            '"quote_requires_unstated_bridge": true, "quote_requires_unstated_bridge": false')
    with capture_usage() as usage:
        async with intercepted_adapter(content=duplicate) as (adapter, requests):
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.verify_evidence_scope(Q, [source()])
    assert len(requests) == len(usage) == 1
    assert caught.value.details == {"check": "scope_verification_duplicate_json_key", "duplicate_key_count": 1}
    assert set(caught.value.candidate) == {"scope_wire_version", "candidate_catalog_id", "target_catalog_id", "invalid_response_sha256"}


async def test_scope_truncation_preserves_capacity_error_finish_reason_and_billed_usage():
    with capture_usage() as usage:
        async with intercepted_adapter("length", content=sdk_reply()) as (adapter, requests):
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.verify_evidence_scope(Q, [source()])
    assert len(requests) == len(usage) == 1
    assert caught.value.code == "model_output_truncated"
    assert caught.value.details == {"finish_reason": "length"}
    assert usage[0]["operation"] == "verify_evidence_scope" and usage[0]["total_tokens"] == 18


async def test_actual_provider_unknown_token_has_numeric_safe_field_diagnostics_not_body():
    def bad(wire, payload):
        wire["decisions"]["C1"]["target_binding_id"] = "PRIVATE_UNKNOWN_TOKEN"
        wire["decisions"]["C1"]["reason"] = "PRIVATE_VISIBLE_BODY"
    async with intercepted_adapter(content=sdk_reply(mutate=bad), reasoning_content="PRIVATE_HIDDEN_REASONING") as (adapter, requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.verify_evidence_scope(Q, [source()])
    error = caught.value
    assert len(requests) == 1 and error.stage == "verify"
    assert {"loc": ["decisions", 0, "target_binding_id"], "type": "literal_error"} in error.details["schema_errors"]
    assert "PRIVATE_" not in json.dumps({"details": error.details, "candidate": error.candidate, "message": str(error)})


async def test_actual_provider_reports_each_invalid_row_without_renumbering_or_truncation():
    def bad(wire, payload):
        del wire["decisions"]["C1"]["target_binding_id"]
        del wire["decisions"]["C2"]["target_requires_unstated_bridge"]
    async with intercepted_adapter(content=sdk_reply(mutate=bad)) as (adapter, _requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.verify_evidence_scope(Q, [source("e1"), source("e2")])
    assert {tuple(e["loc"]) for e in caught.value.details["schema_errors"]} == {
        ("decisions", 0, "target_binding_id"), ("decisions", 1, "target_requires_unstated_bridge")}


async def test_budgeted_repair_uses_bound_new_wire_descriptor_and_safe_diagnostics_only():
    calls = 0
    def scripted(body):
        nonlocal calls
        calls += 1
        def bad(wire, payload):
            wire["decisions"]["C1"]["target_binding_id"] = "PRIVATE_INVALID_BINDING"
            wire["decisions"]["C1"]["reason"] = "PRIVATE_PREVIOUS_BODY"
        return sdk_reply(mutate=bad if calls == 1 else None)(body)
    with capture_usage() as usage:
        async with intercepted_adapter(content=scripted, reasoning_content="PRIVATE_HIDDEN_REASONING") as (adapter, requests):
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.verify_evidence_scope(Q, [source()])
            error = caught.value
            diagnostics = {**error.details, "unknown_secret": "PRIVATE_DIAGNOSTIC", "candidate": "PRIVATE_BODY"}
            result = await adapter.repair_evidence_scope(Q, [source()], error.candidate, diagnostics)
    assert result["selected_evidence_ids"] == ["e1"]
    assert len(requests) == len(usage) == 2  # One initial call + one explicit existing repair.
    assert all(row["operation"] == "verify_evidence_scope" for row in usage)
    before, after = [json.loads(json.loads(r.content)["messages"][1]["content"]) for r in requests]
    assert before["candidate_catalog_id"] == after["candidate_catalog_id"]
    assert before["target_catalog_id"] == after["target_catalog_id"]
    assert after["repair_context"] == error.candidate
    assert "invalid_output" not in after and "PRIVATE_" not in json.dumps(after)
    assert after["diagnostics"] == error.details


@pytest.mark.parametrize("change", ["question", "catalog", "old_canonical", "extra_body"])
async def test_repair_wrong_or_legacy_context_is_rejected_before_any_paid_call(change):
    rows = [source()]
    payload, _, _ = build_compiled_scope_request(Q, rows)
    context = compiled_repair_context(payload, "synthetic-invalid-visible-json")
    question = deepcopy(Q)
    if change == "question": question["question"] += "说明原因。"
    elif change == "catalog": context["target_catalog_id"] = "G_other"
    elif change == "old_canonical": context = {"candidate_catalog_id": payload["candidate_catalog_id"], "decisions": {}}
    elif change == "extra_body": context["invalid_output"] = "PRIVATE_BODY"
    async with intercepted_adapter(content=sdk_reply()) as (adapter, requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.repair_evidence_scope(question, rows, context, {"check": "scope_compiled_contract"})
    assert requests == [] and caught.value.details["check"] == "scope_compiled_repair_binding"
