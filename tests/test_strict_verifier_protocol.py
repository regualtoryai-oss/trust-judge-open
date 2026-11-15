"""Synthetic protocol tests, not evidence of semantic precision or recall.

Every HTTP request is intercepted by the existing MockTransport fixture. No
real model calls, credentials, competition answers or runtime data are read.
"""
import copy
import json

import pytest

from app.agents.provider import ModelOutputInvalid, capture_usage
from app.domain.strict_verification import (
    ScopeVerificationInvalid, build_fact_views, build_scope_request,
    project_question, resolve_scope_verification,
)
from test_deepseek_provider import intercepted_adapter

Q = {"question_id": "synthetic-safety", "question": "材料是否明确说明设备通过安全检测？"}
KEYS = ("时间", "主体", "地点", "内容", "影响", "数值口径")


def evidence(*, quote="甲公司称设备通过安全检测，但检测报告尚未公开。", fact="甲公司称设备通过安全检测。", identity="source-a", additional=None):
    row = {"id": identity, "paragraph_id": "paragraph-a", "raw_doc_id": "doc-a", "raw_para_id": "p-a",
           "start_cp": 0, "end_cp": len(quote), "quote": quote, "fact_des": fact,
           "argument": {key: None for key in KEYS}, "entities": ["甲公司", "设备"],
           "source_group_id": "group-a", "interpretation_origin": "model"}
    if additional is not None:
        row["additional_interpretations"] = additional
    return row


def annotation(fact):
    return {"fact_des": fact, "argument": {key: None for key in KEYS}, "entities": ["设备"]}


def decision(*, entailment="entailed", relation="direct_answer", bridge=False,
             target="设备通过安全检测", dimension=None, incomplete=False):
    return {"fact_entailment": entailment, "target_relation": relation, "target_question_text": target,
            "qualifier_dimension": dimension, "requires_unstated_bridge": bridge,
            "target_fact_present_but_annotation_incomplete": incomplete,
            "reason": "分别核验原句的有归属主张及本题请求，不引入跨属性推断。"}


def reply_for(rows, decisions, question=Q):
    payload, _, _ = build_scope_request(question, rows)
    return {"candidate_catalog_id": payload["candidate_catalog_id"], "decisions": decisions}


def http_reply(decisions, *, binding_override=None):
    """Synthetic v16 model chooses T0 using each test's intended role.

    This is a fixture declaration, not a runtime canonical-response adapter.
    Direct resolver tests below intentionally continue testing legacy ledgers.
    """
    def reply(payload):
        actual = json.loads(payload["messages"][1]["content"])
        detailed = copy.deepcopy(decisions)
        by_id = {row["id"]: row for row in actual["candidate_catalog"]}
        for alias, row in detailed.items():
            if alias not in by_id:
                continue
            source = by_id[alias]
            row.update(quote_requires_unstated_bridge=row["requires_unstated_bridge"], target_requires_unstated_bridge=False,
                field_checks={"fact_des": "entailed", "entities": "entailed" if source["entities"] else "not_present",
                    "argument": {key: "entailed" if source["argument"][key] else "not_present" for key in KEYS}})
            relation = row.pop("target_relation")
            dimension = row.pop("qualifier_dimension")
            row.pop("target_question_text")
            row.pop("requires_unstated_bridge")
            row["target_binding_id"] = ({"direct_answer": "T0.direct", "explicit_counterclaim": "T0.counter",
                "necessary_qualifier": "T0.qualifier." + str(dimension)}.get(relation, relation))
        return json.dumps({"candidate_catalog_id": binding_override or actual["candidate_catalog_id"],
                           "target_catalog_id": actual["target_catalog_id"],
                           "decisions": detailed}, ensure_ascii=False)
    return reply


def test_two_atomic_interpretations_same_quote_are_not_folded_or_dropped():
    rows = [evidence(additional=[annotation("该设备的检测报告尚未公开。")])]
    views = build_fact_views(rows)
    assert len(views) == 2 and views[0]["id"] != views[1]["id"]
    result = resolve_scope_verification(reply_for(rows, {
        "C1": decision(), "C2": decision(relation="necessary_qualifier", dimension="certainty"),
    }), Q, rows)
    assert len(result["verified_fact_views"]) == 2
    assert result["selected_evidence_ids"] == ["source-a"]
    assert len(result["verified_evidence"][0]["additional_interpretations"]) == 1


def test_rejected_first_annotation_does_not_discard_valid_attributed_sibling():
    rows = [evidence(fact="设备已被独立核实通过检测。", additional=[annotation("甲公司称设备通过安全检测。")])]
    original = copy.deepcopy(rows)
    result = resolve_scope_verification(reply_for(rows, {
        "C1": decision(entailment="needs_narrowing"), "C2": decision(),
    }), Q, rows)
    projected = result["verified_evidence"][0]
    assert projected["fact_des"] == "甲公司称设备通过安全检测。"
    assert "additional_interpretations" not in projected
    assert len(result["narrowing_view_ids"]) == 1
    assert rows == original  # The caller's full internal evidence pool is intact.
    for key in ("id", "paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote"):
        assert projected[key] == rows[0][key]


@pytest.mark.parametrize("entailment,relation,bridge,selected", [
    ("entailed", "direct_answer", False, True),
    ("entailed", "explicit_counterclaim", False, True),
    ("entailed", "outside_target", False, False),
    ("entailed", "direct_answer", True, False),
    ("needs_narrowing", "direct_answer", False, False),
    ("unsupported", "direct_answer", False, False),
    ("contradicted", "explicit_counterclaim", False, False),
])
def test_projection_accepts_only_entailed_in_target_without_unstated_bridge(entailment, relation, bridge, selected):
    rows = [evidence()]
    result = resolve_scope_verification(reply_for(rows, {"C1": decision(entailment=entailment, relation=relation, bridge=bridge)}), Q, rows)
    assert bool(result["selected_evidence_ids"]) is selected
    assert len(result["assessments"]) == 1


@pytest.mark.parametrize("dimension", ["subject", "time", "scope", "value", "attribution", "certainty", "causality"])
def test_explicit_necessary_qualifier_dimensions_are_accepted(dimension):
    rows = [evidence()]
    result = resolve_scope_verification(reply_for(rows, {"C1": decision(relation="necessary_qualifier", dimension=dimension)}), Q, rows)
    assert result["selected_evidence_ids"] == ["source-a"]


@pytest.mark.parametrize("change", [
    {"target_question_text": "not-in-question-SENSITIVE"},
    {"target_relation": "necessary_qualifier", "qualifier_dimension": None},
    {"qualifier_dimension": "time"},
])
def test_invalid_question_target_or_qualifier_contract_has_only_safe_diagnostics(change):
    rows = [evidence()]
    with pytest.raises(ScopeVerificationInvalid) as caught:
        resolve_scope_verification(reply_for(rows, {"C1": {**decision(), **change}}), Q, rows)
    assert caught.value.details == {"check": "scope_target_contract", "invalid_candidate_count": 1, "candidate_indexes": [0]}
    assert "SENSITIVE" not in str(caught.value) and "SENSITIVE" not in json.dumps(caught.value.details)


@pytest.mark.parametrize("mutate", [
    lambda q, rows: q.update(question="另一问题？"),
    lambda q, rows: rows[0].update(fact_des="解释版本变更。"),
    lambda q, rows: rows[0]["argument"].update(时间="后来"),
    lambda q, rows: rows[0].update(raw_doc_id="other-doc"),
    lambda q, rows: rows[0].update(quote=rows[0]["quote"] + "新", end_cp=rows[0]["end_cp"] + 1),
])
def test_binding_rejects_stale_question_annotation_or_source(mutate):
    q, rows = copy.deepcopy(Q), [evidence()]
    output = reply_for(rows, {"C1": decision()}, question=q)
    mutate(q, rows)
    with pytest.raises(ScopeVerificationInvalid, match="scope_decision_contract"):
        resolve_scope_verification(output, q, rows)


def test_extra_and_missing_decision_keys_rejected():
    rows = [evidence()]
    for decisions in ({}, {"C1": decision(), "C999": decision()}):
        with pytest.raises(ScopeVerificationInvalid, match="scope_decision_contract"):
            resolve_scope_verification(reply_for(rows, decisions), Q, rows)


def test_question_projection_strips_synthetic_answer_like_metadata():
    projected = project_question({**Q, "conclusion": "SYNTHETIC_ANSWER", "supporting_facts": ["SYNTHETIC_REF"]})
    assert projected == Q
    assert "SYNTHETIC" not in json.dumps(projected)


def test_identity_stable_for_entity_order_but_distinct_for_attribution_and_temporal_versions():
    row = evidence()
    first = build_fact_views([row])[0]["id"]
    row["entities"].reverse()
    assert build_fact_views([row])[0]["id"] == first
    row["fact_des"] = "设备已通过安全检测。"
    assert build_fact_views([row])[0]["id"] != first
    row["argument"]["时间"] = "某一更新时点"
    assert len(build_fact_views([row])) == 1


@pytest.mark.asyncio
async def test_real_sdk_transport_contract_scope_is_independent_and_uses_analysis_thinking():
    rows = [evidence()]
    with capture_usage() as usage:
        async with intercepted_adapter(content=http_reply({"C1": decision()})) as (adapter, requests):
            result = await adapter.verify_evidence_scope({**Q, "conclusion": "SYNTHETIC_LEAK"}, rows)
    assert result["selected_evidence_ids"] == ["source-a"]
    assert len(requests) == 1
    request = json.loads(requests[0].content)
    payload = json.loads(request["messages"][1]["content"])
    assert set(payload) == {"scope_wire_version", "question", "candidate_catalog_id", "candidate_catalog",
                            "source_protocol_sha256", "target_catalog", "target_binding_catalog", "target_catalog_id"}
    assert payload["candidate_catalog"][0]["id"] == "C1"
    assert "SYNTHETIC_LEAK" not in json.dumps(request)
    assert request["thinking"] == {"type": "enabled"}
    assert usage[0]["operation"] == "verify_evidence_scope" and usage[0]["total_tokens"] == 18
    instructions = request["messages"][0]["content"]
    for phrase in ("只核quote是否蕴含", "后来状态", "不同属性B", "requires_unstated_bridge", "存在该有归属指称"):
        assert phrase in instructions
    assert "previous_assessments" not in payload and "analyses" not in payload


@pytest.mark.asyncio
async def test_duplicate_raw_json_decision_keys_rejected_before_last_write_wins():
    def duplicate(payload):
        actual = json.loads(payload["messages"][1]["content"])
        row = json.dumps(decision(), ensure_ascii=False)
        return '{"candidate_catalog_id":' + json.dumps(actual["candidate_catalog_id"]) + ',"decisions":{"C1":' + row + ',"C1":' + row + '}}'
    async with intercepted_adapter(content=duplicate) as (adapter, requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.verify_evidence_scope(Q, [evidence()])
    assert len(requests) == 1 and caught.value.stage == "verify"
    assert caught.value.details == {"check": "scope_verification_duplicate_json_key", "duplicate_key_count": 1}


@pytest.mark.asyncio
async def test_missing_view_reports_safe_numeric_dynamic_location():
    async with intercepted_adapter(content=http_reply({})) as (adapter, _):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.verify_evidence_scope(Q, [evidence()])
    assert caught.value.stage == "verify"
    assert caught.value.details == {"schema_errors": [{"loc": ["decisions", 0], "type": "missing"}]}


@pytest.mark.asyncio
async def test_empty_pool_is_local_and_old_canonical_cannot_enter_new_wire_repair():
    async with intercepted_adapter(content=http_reply({"C1": decision()})) as (adapter, requests):
        empty = await adapter.verify_evidence_scope(Q, [])
        assert empty["selected_evidence_ids"] == [] and requests == []
        original = {"decisions": "SYNTHETIC_INVALID_OUTPUT"}
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.repair_evidence_scope(Q, [evidence()], original, {"check": "scope_decision_contract"})
    assert caught.value.details["check"] == "scope_compiled_repair_binding"
    assert requests == []


@pytest.mark.asyncio
async def test_invalid_source_catalog_fails_before_paid_request():
    async with intercepted_adapter() as (adapter, requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.verify_evidence_scope(Q, [evidence(), evidence()])
    assert requests == []
    assert caught.value.details["check"] == "scope_catalog_invalid"
