"""Fixed-source annotation refocus protocol; SDK HTTP is fully intercepted.

Synthetic response labels test the boundary, not real model semantic quality.
Caller lifecycle/budget/cancellation lives in quality-refine integration tests.
"""
import copy
import json

import pytest

from app.agents.provider import ModelOutputInvalid, capture_usage
from app.domain.strict_verification import (
    ScopeVerificationInvalid, build_fact_views, build_narrowing_request,
    build_scope_request, resolve_annotation_narrowing, resolve_scope_verification,
    validate_narrowing_ledger, validate_scope_ledger, verified_fact_export_rows,
    safe_annotation_diagnostics,
)
from test_deepseek_provider import intercepted_adapter
from test_strict_verifier_protocol import Q, annotation, decision, evidence, reply_for, http_reply


def ids(rows):
    return [view["id"] for view in build_fact_views(rows)]


def correction(*annotations, status="corrected"):
    return {"status": status, "annotations": list(annotations)}


def narrowing_output(rows, corrections, *, question=Q, view_ids=None):
    payload, _, _ = build_narrowing_request(question, rows, view_ids or ids(rows))
    return {"candidate_catalog_id": payload["candidate_catalog_id"], "corrections": corrections}


def narrowing_http(corrections, *, binding_override=None):
    def reply(payload):
        request = json.loads(payload["messages"][1]["content"])
        return json.dumps({"candidate_catalog_id": binding_override or request["candidate_catalog_id"],
                           "corrections": corrections}, ensure_ascii=False)
    return reply


def test_off_target_correct_annotation_can_trigger_source_refocus_without_being_selected():
    rows = [evidence(fact="甲公司发布了产品公告。")]
    ledger = resolve_scope_verification(reply_for(rows, {"C1": decision(relation="outside_target", incomplete=True, target=None)}), Q, rows)
    assert ledger["selected_fact_view_ids"] == []
    assert ledger["narrowing_view_ids"] == ids(rows)
    assert validate_scope_ledger(ledger, Q, rows) == ledger


def test_multiple_atomic_refocus_versions_keep_source_parent_and_original_pool():
    rows = [evidence(fact="该设备已独立核实通过检测。")]
    original = copy.deepcopy(rows)
    requested = ids(rows)
    raw = narrowing_output(rows, {"C1": correction(annotation("甲公司称设备通过安全检测。"), annotation("该设备的检测报告尚未公开。"))})
    ledger = resolve_annotation_narrowing(raw, Q, rows, requested)
    assert rows == original
    assert len(ledger["corrected_fact_views"]) == len(ledger["annotation_versions"]) == 2
    assert {version["parent_fact_view_id"] for version in ledger["annotation_versions"]} == set(requested)
    assert len(set(version["id"] for version in ledger["annotation_versions"])) == 2
    for version in ledger["annotation_versions"]:
        assert version["semantic_attempt"] == 1 and version["annotation_changed"] is True
    for new in ledger["corrected_fact_views"]:
        for field in ("evidence_id", "paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote"):
            assert new[field] == (rows[0]["id"] if field == "evidence_id" else rows[0][field])
    assert validate_narrowing_ledger(ledger, Q, rows, requested) == ledger


def test_identical_annotations_deduplicated_exactly_without_inventing_more_facts():
    rows = [evidence()]
    same = annotation("甲公司称设备通过安全检测。")
    ledger = resolve_annotation_narrowing(narrowing_output(rows, {"C1": correction(same, same)}), Q, rows, ids(rows))
    assert len(ledger["corrected_fact_views"]) == 1
    assert ledger["corrections"][0]["duplicate_annotation_count"] == 1
    assert validate_narrowing_ledger(ledger, Q, rows, ids(rows)) == ledger


def test_no_supported_annotation_is_an_explicit_nonfatal_empty_result():
    rows = [evidence(quote="该企业宣布更换官网字体。", fact="企业宣布更换字体。")]
    ledger = resolve_annotation_narrowing(narrowing_output(rows, {"C1": correction(status="no_supported_annotation")}), Q, rows, ids(rows))
    assert ledger["corrected_evidence"] == ledger["corrected_fact_views"] == ledger["annotation_versions"] == []
    assert ledger["corrections"][0]["status"] == "no_supported_annotation"
    assert validate_narrowing_ledger(ledger, Q, rows, ids(rows)) == ledger


@pytest.mark.parametrize("wire,check", [
    (correction(), "annotation_correction_empty"),
    (correction(annotation("有归属指称。"), status="no_supported_annotation"), "annotation_correction_no_supported_nonempty"),
])
def test_empty_and_explicit_absence_contracts_have_distinct_fixed_diagnostics(wire, check):
    rows = [evidence()]
    with pytest.raises(ScopeVerificationInvalid) as caught:
        resolve_annotation_narrowing(narrowing_output(rows, {"C1": wire}), Q, rows, ids(rows))
    assert caught.value.details == {"check": check, "invalid_candidate_count": 1, "candidate_indexes": [0]}


def test_unknown_repeated_parent_and_human_override_rejected_before_request():
    rows = [evidence()]
    parent = ids(rows)[0]
    for requested in (["unknown-F"], [parent, parent]):
        with pytest.raises(ScopeVerificationInvalid, match="annotation_refocus_catalog_invalid"):
            build_narrowing_request(Q, rows, requested)
    rows[0]["interpretation_origin"] = "human"
    with pytest.raises(ScopeVerificationInvalid, match="annotation_refocus_manual_protected"):
        build_narrowing_request(Q, rows, [parent])


@pytest.mark.parametrize("mutate", [
    lambda q, rows: q.update(question="另一个目标？"),
    lambda q, rows: rows[0].update(fact_des="另外一个解释。"),
    lambda q, rows: rows[0].update(quote=rows[0]["quote"] + "新", end_cp=rows[0]["end_cp"] + 1),
])
def test_refocus_binding_rejects_stale_source_annotation_or_question(mutate):
    q, rows = copy.deepcopy(Q), [evidence()]
    raw = narrowing_output(rows, {"C1": correction(annotation("甲公司称设备通过安全检测。"))}, question=q)
    mutate(q, rows)
    with pytest.raises(ScopeVerificationInvalid, match="annotation_correction_contract"):
        resolve_annotation_narrowing(raw, q, rows, ids(rows))


def test_unchanged_correction_is_a_version_attempt_not_fake_semantic_success():
    rows = [evidence(fact="该设备已经独立核实通过检测。")]
    old = {field: copy.deepcopy(rows[0][field]) for field in ("fact_des", "argument", "entities")}
    ledger = resolve_annotation_narrowing(narrowing_output(rows, {"C1": correction(old)}), Q, rows, ids(rows))
    assert ledger["annotation_versions"][0]["annotation_changed"] is False
    corrected = ledger["corrected_evidence"]
    verification = resolve_scope_verification(reply_for(corrected, {"C1": decision(entailment="needs_narrowing", incomplete=True)}), Q, corrected)
    assert verification["selected_fact_view_ids"] == [] and verification["narrowing_view_ids"]
    # Caller sees unresolved after its only content attempt; no recursive provider call exists here.


def test_second_verifier_accepts_attributed_claim_but_not_false_independent_confirmation():
    rows = [evidence()]
    narrowed = resolve_annotation_narrowing(narrowing_output(rows, {"C1": correction(
        annotation("甲公司称设备通过安全检测。"), annotation("该设备已独立核实通过安全检测。"))}), Q, rows, ids(rows))
    corrected = narrowed["corrected_evidence"]
    checked = resolve_scope_verification(reply_for(corrected, {"C1": decision(), "C2": decision(entailment="unsupported")}), Q, corrected)
    exports = verified_fact_export_rows(checked, Q, corrected)
    assert len(exports) == 1 and exports[0]["fact_des"] == "甲公司称设备通过安全检测。"
    assert len(narrowed["annotation_versions"]) == 2  # The rejected alternative still has provenance.


def test_formal_export_keeps_all_independently_verified_atomic_views_not_unchecked_additions():
    rows = [evidence(additional=[annotation("该设备的检测报告尚未公开。"), annotation("该设备已获全国所有监管机构许可。")])]
    checked = resolve_scope_verification(reply_for(rows, {
        "C1": decision(), "C2": decision(relation="necessary_qualifier", dimension="certainty"),
        "C3": decision(entailment="unsupported"),
    }), Q, rows)
    exports = verified_fact_export_rows(checked, Q, rows)
    assert len(exports) == 2
    assert exports[0]["quote"] == exports[1]["quote"] == rows[0]["quote"]
    assert len({row["fact_des"] for row in exports}) == 2
    assert all("全国" not in row["fact_des"] for row in exports)


@pytest.mark.parametrize("tamper", [
    lambda ledger: ledger["verified_evidence"][0].update(quote="FORGED_SOURCE"),
    lambda ledger: ledger["verified_fact_views"][0].update(fact_des="FORGED_FACT"),
    lambda ledger: ledger.update(selected_fact_view_ids=[]),
    lambda ledger: ledger["assessments"][0].update(selected=False),
])
def test_provider_scope_ledger_projection_cannot_be_forged(tamper):
    rows = [evidence()]
    checked = resolve_scope_verification(reply_for(rows, {"C1": decision()}), Q, rows)
    tamper(checked)
    with pytest.raises(ScopeVerificationInvalid, match="scope_ledger_invalid"):
        validate_scope_ledger(checked, Q, rows)
    with pytest.raises(ScopeVerificationInvalid):
        verified_fact_export_rows(checked, Q, rows)


@pytest.mark.parametrize("tamper", [
    lambda ledger: ledger["corrected_evidence"][0].update(start_cp=1),
    lambda ledger: ledger["corrected_fact_views"][0].update(quote="FORGED_SOURCE"),
    lambda ledger: ledger["annotation_versions"][0].update(parent_fact_view_id="UNKNOWN_PARENT"),
    lambda ledger: ledger["corrections"][0].update(corrected_fact_view_ids=["UNKNOWN_VIEW"]),
])
def test_provider_narrowing_ledger_source_or_parent_cannot_be_forged(tamper):
    rows = [evidence()]
    checked = resolve_annotation_narrowing(narrowing_output(rows, {"C1": correction(annotation("甲公司称设备通过安全检测。"))}), Q, rows, ids(rows))
    tamper(checked)
    with pytest.raises(ScopeVerificationInvalid, match="annotation_refocus_ledger_invalid"):
        validate_narrowing_ledger(checked, Q, rows, ids(rows))


@pytest.mark.asyncio
async def test_sdk_refocus_then_independent_second_verification_has_no_previous_reason_or_final():
    requests_seen = []
    def scripted(payload):
        actual = json.loads(payload["messages"][1]["content"])
        requests_seen.append(actual)
        if "corrections" in payload["messages"][0]["content"].split("JSON Schema:")[-1]:
            return narrowing_http({"C1": correction(annotation("甲公司称设备通过安全检测。"), annotation("检测报告尚未公开。"))})(payload)
        return http_reply({
            "C1": decision(), "C2": decision(relation="necessary_qualifier", dimension="certainty"),
        })(payload)
    rows = [evidence(fact="错误过强事实。")]
    rows[0]["previous_reason"] = "SYNTHETIC_PRIOR_REASON"
    rows[0]["final"] = "SYNTHETIC_FINAL"
    with capture_usage() as usage:
        async with intercepted_adapter(content=scripted) as (adapter, requests):
            narrowed = await adapter.narrow_evidence_annotations(Q, rows, ids(rows))
            assert "verified_fact_views" not in narrowed  # New annotations are unverified.
            verified = await adapter.verify_evidence_scope(Q, narrowed["corrected_evidence"])
    assert len(requests) == len(usage) == 2
    assert [record["operation"] for record in usage] == ["narrow_evidence_annotations", "verify_evidence_scope"]
    assert len(verified["verified_fact_views"]) == 2
    assert set(requests_seen[0]) == {"question", "candidate_catalog_id", "candidate_catalog"}
    assert all("SYNTHETIC_PRIOR_REASON" not in json.dumps(payload) and "SYNTHETIC_FINAL" not in json.dumps(payload) for payload in requests_seen)
    assert all(json.loads(request.content)["thinking"] == {"type": "enabled"} for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("corrections,loc,kind", [
    ({}, ["corrections", 0], "missing"),
    ({"C1": correction(annotation("有归属事实。")), "C2": correction(status="no_supported_annotation")}, ["corrections", "<field>"], "extra_forbidden"),
    ({"C1": correction({**annotation("有归属事实。"), "quote": "FORGED_SOURCE"})}, ["corrections", 0, "annotations", 0, "<field>"], "extra_forbidden"),
    ({"C1": correction({**annotation("有归属事实。"), "reason": "FORBIDDEN_REASON"})}, ["corrections", 0, "annotations", 0, "<field>"], "extra_forbidden"),
])
async def test_sdk_refocus_missing_extra_and_source_fields_rejected_safely(corrections, loc, kind):
    rows = [evidence()]
    async with intercepted_adapter(content=narrowing_http(corrections)) as (adapter, requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.narrow_evidence_annotations(Q, rows, ids(rows))
    assert len(requests) == 1 and caught.value.stage == "verify"
    assert {"loc": loc, "type": kind} in caught.value.details["schema_errors"]
    assert "FORGED" not in json.dumps(caught.value.details) and "FORBIDDEN" not in json.dumps(caught.value.details)


@pytest.mark.asyncio
async def test_sdk_refocus_repeated_raw_c_key_rejected_not_last_write_wins():
    def duplicate(payload):
        actual = json.loads(payload["messages"][1]["content"])
        row = json.dumps(correction(annotation("有归属事实。")), ensure_ascii=False)
        return '{"candidate_catalog_id":' + json.dumps(actual["candidate_catalog_id"]) + ',"corrections":{"C1":' + row + ',"C1":' + row + '}}'
    rows = [evidence()]
    async with intercepted_adapter(content=duplicate) as (adapter, requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.narrow_evidence_annotations(Q, rows, ids(rows))
    assert len(requests) == 1
    assert caught.value.details == {"check": "annotation_refocus_duplicate_json_key", "duplicate_key_count": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("wire,check", [
    (correction(), "annotation_correction_empty"),
    (correction(annotation("有归属事实。"), status="no_supported_annotation"), "annotation_correction_no_supported_nonempty"),
])
async def test_sdk_refocus_semantic_empty_contracts_are_distinct_safe_failures(wire, check):
    rows = [evidence()]
    async with intercepted_adapter(content=narrowing_http({"C1": wire})) as (adapter, _):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.narrow_evidence_annotations(Q, rows, ids(rows))
    assert caught.value.stage == "verify" and caught.value.details["check"] == check


@pytest.mark.asyncio
async def test_sdk_refocus_has_no_implicit_content_retry_or_automatic_pass():
    rows = [evidence()]
    async with intercepted_adapter(content=narrowing_http({"C1": correction(status="no_supported_annotation")})) as (adapter, requests):
        ledger = await adapter.narrow_evidence_annotations(Q, rows, ids(rows))
        assert len(requests) == 1
        checked = await adapter.verify_evidence_scope(Q, ledger["corrected_evidence"])
    assert len(requests) == 1 and checked["selected_fact_view_ids"] == []


def test_refocus_keeps_all_explicit_atomized_candidates_without_twelve_item_quota():
    quote = "；".join(f"记录项目{index}。" for index in range(22))
    rows = [evidence(quote=quote, fact="过宽的记录汇总。")]
    annotations = [annotation(f"记录项目{index}。") for index in range(22)]
    ledger = resolve_annotation_narrowing(narrowing_output(rows, {"C1": correction(*annotations)}), Q, rows, ids(rows))
    assert len(ledger["corrected_fact_views"]) == 22
    assert len(ledger["corrected_evidence"][0]["additional_interpretations"]) == 21
    assert "verified_fact_views" not in ledger


def test_unicode_and_explicit_offsets_are_copied_not_recomputed_or_relocated():
    rows = [evidence(quote="🦊甲公司称设备已通过检测；尚未公开报告。")]
    rows[0].update(start_cp=17, end_cp=17 + len(rows[0]["quote"]))
    ledger = resolve_annotation_narrowing(narrowing_output(rows, {"C1": correction(annotation("甲公司声称设备通过检测。"))}), Q, rows, ids(rows))
    new = ledger["corrected_evidence"][0]
    assert (new["start_cp"], new["end_cp"], new["quote"]) == (rows[0]["start_cp"], rows[0]["end_cp"], rows[0]["quote"])


def test_prior_interpretation_versions_are_appended_in_projection_not_overwritten_in_pool():
    rows = [evidence()]
    historical = {"id": "historical-version", "note": "prior-input-provenance"}
    rows[0]["annotation_versions"] = [historical]
    original = copy.deepcopy(rows)
    ledger = resolve_annotation_narrowing(narrowing_output(rows, {"C1": correction(annotation("甲公司声称设备通过检测。"))}), Q, rows, ids(rows))
    assert rows == original
    assert ledger["corrected_evidence"][0]["annotation_versions"][0] == historical
    assert len(ledger["corrected_evidence"][0]["annotation_versions"]) == 2
    assert validate_narrowing_ledger(ledger, Q, rows, ids(rows)) == ledger


def test_invented_numbers_and_time_in_new_annotation_do_not_bypass_independent_verifier():
    rows = [evidence()]
    invented = annotation("设备在2099年通过88项检测。")
    invented["argument"].update(时间="2099年", 数值口径="88项检测")
    narrowed = resolve_annotation_narrowing(narrowing_output(rows, {"C1": correction(invented)}), Q, rows, ids(rows))
    corrected = narrowed["corrected_evidence"]
    checked = resolve_scope_verification(reply_for(corrected, {"C1": decision(entailment="unsupported")}), Q, corrected)
    assert checked["verified_fact_views"] == []
    assert verified_fact_export_rows(checked, Q, corrected) == []
    assert narrowed["corrected_fact_views"][0]["fact_des"] == invented["fact_des"]  # Retained as rejected-version audit.


@pytest.mark.asyncio
async def test_sdk_required_refocus_trigger_flag_cannot_be_omitted():
    def missing_flag(payload):
        actual = json.loads(payload["messages"][1]["content"])
        row = decision()
        row.pop("target_fact_present_but_annotation_incomplete")
        return http_reply({"C1": row})(payload)
    async with intercepted_adapter(content=missing_flag) as (adapter, _):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.verify_evidence_scope(Q, [evidence()])
    assert caught.value.details == {"schema_errors": [{"loc": ["decisions", 0, "target_fact_present_but_annotation_incomplete"], "type": "missing"}]}


@pytest.mark.asyncio
async def test_sdk_refocus_invalid_parent_and_manual_override_never_reach_http():
    rows = [evidence()]
    parent = ids(rows)[0]
    async with intercepted_adapter() as (adapter, requests):
        for requested in (["unknown"], [parent, parent]):
            with pytest.raises(ModelOutputInvalid):
                await adapter.narrow_evidence_annotations(Q, rows, requested)
        rows[0]["interpretation_origin"] = "human"
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.narrow_evidence_annotations(Q, rows, [parent])
    assert requests == [] and caught.value.details["check"] == "annotation_refocus_manual_protected"


@pytest.mark.asyncio
async def test_sdk_one_explicit_structural_repair_does_not_receive_previous_body_reason_or_final():
    counter = 0
    def responses(payload):
        nonlocal counter
        counter += 1
        return narrowing_http({"C1": correction() if counter == 1 else correction(annotation("甲公司声称设备通过检测。"))})(payload)
    rows = [evidence()]
    with capture_usage() as usage:
        async with intercepted_adapter(content=responses) as (adapter, requests):
            with pytest.raises(ModelOutputInvalid) as caught:
                await adapter.narrow_evidence_annotations(Q, rows, ids(rows))
            invalid = {"reason": "SYNTHETIC_PREVIOUS_REASON", "final": "SYNTHETIC_PREVIOUS_FINAL"}
            ledger = await adapter.repair_narrow_evidence_annotations(Q, rows, ids(rows), invalid,
                {**caught.value.details, "reason": "SYNTHETIC_REASON_IN_DIAGNOSTIC", "final": "SYNTHETIC_FINAL_IN_DIAGNOSTIC"})
    assert len(requests) == len(usage) == 2
    payload = json.loads(json.loads(requests[1].content)["messages"][1]["content"])
    assert "invalid_output" not in payload
    assert "SYNTHETIC" not in json.dumps(payload)
    assert payload["diagnostics"]["check"] == "annotation_correction_empty"
    assert "verified_fact_views" not in ledger


def test_structural_repair_diagnostics_allow_only_fixed_codes_positions_and_types():
    safe = safe_annotation_diagnostics({"check": {"raw": "SYNTHETIC_BODY"}, "reason": "SYNTHETIC_REASON",
        "schema_errors": [{"loc": ["corrections", 0, "SYNTHETIC_FIELD", {"raw": "SYNTHETIC_BODY"}],
                           "type": {"raw": "SYNTHETIC_TYPE"}, "input": "SYNTHETIC_INPUT"}]})
    assert safe == {"check": "annotation_correction_contract", "schema_errors": [
        {"loc": ["corrections", 0, "<field>", "<field>"], "type": "value_error"}]}
