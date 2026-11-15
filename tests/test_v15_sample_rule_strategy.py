"""Synthetic contract counterexamples, never model accuracy measurements.

All decisions below are explicitly supplied reviewer judgments. These tests
check independent gates, immutable source geometry and the real SDK schema;
they do not claim the live model always makes these semantic judgments.
"""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from app.agents.provider import ModelOutputInvalid
from app.domain.source_catalog import build_source_windows, resolve_source_window
from app.domain.spans import validate_span
from app.domain.strict_verification import (
    ScopeVerificationInvalid, build_scope_request, resolve_scope_verification,
    validate_scope_ledger,
)
from test_deepseek_provider import intercepted_adapter, extraction_reply


KEYS = ("时间", "主体", "地点", "内容", "影响", "数值口径")
QUESTION = {"question_id": "synthetic-facility", "question": "材料是否证明场馆通过验收并已向公众开放？"}


def source(quote="场馆试运行了，但正式验收尚待签字。", fact="场馆已试运行，正式验收仍待签字。", **arguments):
    return {"id": "e1", "paragraph_id": "p1", "raw_doc_id": "d1", "raw_para_id": "p1",
        "start_cp": 0, "end_cp": len(quote), "quote": quote, "fact_des": fact,
        "argument": {key: arguments.get(key) for key in KEYS}, "entities": ["场馆"],
        "interpretation_origin": "model"}


def detailed(row, *, relation="necessary_qualifier", dimension="certainty", quote_bridge=False, target_bridge=False):
    return {"fact_entailment": "entailed", "target_relation": relation,
        "target_question_text": "通过验收" if relation in {"direct_answer", "explicit_counterclaim", "necessary_qualifier"} else None,
        "qualifier_dimension": dimension, "requires_unstated_bridge": quote_bridge or target_bridge,
        "quote_requires_unstated_bridge": quote_bridge, "target_requires_unstated_bridge": target_bridge,
        "field_checks": {"fact_des": "entailed", "entities": "entailed" if row["entities"] else "not_present",
            "argument": {key: "entailed" if row["argument"][key] else "not_present" for key in KEYS}},
        "target_fact_present_but_annotation_incomplete": False,
        "reason": "原文明确限定验收状态；该限定无需独自证明开放状态。"}


def resolve(row, decision):
    payload, _, _ = build_scope_request(QUESTION, [row])
    return resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"],
        "decisions": {"C1": decision}}, QUESTION, [row])


def test_necessary_condition_cluster_survives_without_proving_entire_compound_question():
    row = source()
    original = deepcopy(row)
    result = resolve(row, detailed(row))
    assert result["selected_evidence_ids"] == ["e1"]
    assert result["verified_evidence"][0] == original
    assert result["assessments"][0]["field_issues"] == []
    assert validate_scope_ledger(result, QUESTION, [row]) == result


@pytest.mark.parametrize("relation", ["outside_target", "uncertain"])
def test_general_policy_or_uncertain_relation_is_audited_but_not_formal_evidence(relation):
    row = source("通常大型场馆验收需要签字。", "一般验收政策要求签字。")
    result = resolve(row, detailed(row, relation=relation, dimension=None))
    assert result["selected_evidence_ids"] == []
    assert result["assessments"][0]["target_relation"] == relation


@pytest.mark.parametrize("quote_bridge,target_bridge", [(True, False), (False, True), (True, True)])
def test_actual_source_or_relation_bridge_is_rejected_separately(quote_bridge, target_bridge):
    row = source("场馆开始试运行。", "场馆已正式验收。")
    result = resolve(row, detailed(row, relation="direct_answer", dimension=None,
        quote_bridge=quote_bridge, target_bridge=target_bridge))
    assert result["selected_evidence_ids"] == []
    assert result["assessments"][0]["quote_requires_unstated_bridge"] is quote_bridge
    assert result["assessments"][0]["target_requires_unstated_bridge"] is target_bridge


@pytest.mark.parametrize("field,quote,claimed", [
    ("内容", "尚未通过验收。", "已通过验收"),
    ("时间", "截至周三仍在试运行。", "周一已完成验收"),
    ("数值口径", "记录两处裂缝，其中一处已修复。", "记录三处裂缝"),
    ("主体", "媒体转述承包商称已验收。", "主管部门证实已验收"),
])
def test_correct_description_cannot_hide_negation_time_count_or_attribution_error(field, quote, claimed):
    row = source(quote, quote, **{field: claimed})
    verdict = detailed(row)
    verdict["field_checks"]["argument"][field] = "needs_narrowing"
    result = resolve(row, verdict)
    assert result["selected_evidence_ids"] == []
    assert result["assessments"][0]["field_issues"] == ["argument." + field]
    assert len(result["narrowing_view_ids"]) == 1
    assert row["argument"][field] == claimed  # Original is retained, not silently fixed.


def test_nonempty_field_cannot_bypass_verification_with_not_present_label():
    row = source(内容="验收尚待签字")
    verdict = detailed(row)
    verdict["field_checks"]["argument"]["内容"] = "not_present"
    assert resolve(row, verdict)["selected_evidence_ids"] == []


@pytest.mark.parametrize("changes", [
    {"quote_requires_unstated_bridge": None}, {"field_checks": None},
    {"requires_unstated_bridge": True},
])
def test_partial_or_inconsistent_new_gate_cannot_downgrade_to_legacy(changes):
    row = source()
    with pytest.raises(ScopeVerificationInvalid, match="scope_target_contract"):
        resolve(row, {**detailed(row), **changes})


def test_legacy_replay_remains_exact_but_live_schema_requires_all_detailed_fields():
    row = source()
    legacy = detailed(row)
    for key in ("quote_requires_unstated_bridge", "target_requires_unstated_bridge", "field_checks"):
        legacy.pop(key)
    old = resolve(row, legacy)
    assert set(old["assessments"][0]) == {"fact_view_id", "evidence_id", "selected", *legacy}
    assert validate_scope_ledger(old, QUESTION, [row]) == old
    payload, schema, _ = build_scope_request(QUESTION, [row], detailed=True)
    with pytest.raises(ValidationError):
        schema.model_validate({"candidate_catalog_id": payload["candidate_catalog_id"], "decisions": {"C1": legacy}})


def test_whole_paragraph_option_keeps_late_qualifications_and_compact_aliases_stable():
    text = "场馆称已验收。天气晴。道路通畅。游客问询。上述验收仅为内部初验，正式验收尚待签字。"
    paragraphs = [{"id": "p1", "text": text}]
    compact = build_source_windows(paragraphs)
    full = build_source_windows(paragraphs, include_paragraphs=True)
    assert full[:len(compact)] == compact
    assert "尚待签字" not in compact[0]["quote"]
    assert full[-1]["quote"] == text
    chosen = {"source_window_id": "W" + str(len(full)), "fact_des": "场馆所称验收仅为内部初验，正式验收尚待签字。",
        "argument": {}, "entities": ["场馆"]}
    result = resolve_source_window(chosen, full, paragraphs, include_paragraphs=True)
    validate_span(text, result["quote"], result["start_cp"], result["end_cp"])
    assert result["quote"] == text
    with pytest.raises(ValueError, match="invalid_source_window_catalog"):
        resolve_source_window(chosen, full, paragraphs)  # No implicit protocol upgrade.
    changed = deepcopy(full)
    changed[-1]["quote"] += "补充事实"
    with pytest.raises(ValueError, match="invalid_source_window_catalog"):
        resolve_source_window(chosen, changed, paragraphs, include_paragraphs=True)


def test_context_options_never_join_paragraphs_or_duplicate_existing_whole_window():
    paragraphs = [{"id": "a", "text": "甲。乙。"}, {"id": "b", "text": "丙。丁。戊。己。"}]
    catalog = build_source_windows(paragraphs, include_paragraphs=True)
    assert sum(row["paragraph_id"] == "a" and row["quote"] == "甲。乙。" for row in catalog) == 1
    for row in catalog:
        original = next(p["text"] for p in paragraphs if p["id"] == row["paragraph_id"])
        validate_span(original, row["quote"], row["start_cp"], row["end_cp"])


@pytest.mark.asyncio
async def test_sdk_selects_program_owned_whole_paragraph_and_prompts_keep_qualified_clusters():
    paragraphs = [{"id": "p1", "text": "称已完成。甲。乙。丙。但此处完成仅指试运行，不是正式验收。"}]
    catalog = build_source_windows(paragraphs, include_paragraphs=True)
    chosen = {"source_window_id": "W" + str(len(catalog)), "fact_des": "所称完成仅指试运行，不是正式验收。", "argument": {}, "entities": []}
    async with intercepted_adapter(content=extraction_reply([chosen])) as (adapter, requests):
        # Retain this v15 window-comparator regression. Production paragraph
        # extraction has its own HTTP contract and whole-source tests.
        result = await adapter.extract_window_candidates(QUESTION, paragraphs)
    assert result[0]["quote"] == paragraphs[0]["text"]
    instructions = json.loads(requests[0].content)["messages"][0]["content"]
    for phrase in ("连贯断言", "完整原段窗口", "不强迫切成极短", "各自出处"):
        assert phrase in instructions


@pytest.mark.asyncio
async def test_live_review_does_not_silently_accept_legacy_model_output():
    row = source()
    old = detailed(row)
    for key in ("quote_requires_unstated_bridge", "target_requires_unstated_bridge", "field_checks"):
        old.pop(key)
    def response(body):
        payload = json.loads(body["messages"][1]["content"])
        return json.dumps({"candidate_catalog_id": payload["candidate_catalog_id"], "decisions": {"C1": old}}, ensure_ascii=False)
    async with intercepted_adapter(content=response) as (adapter, requests):
        with pytest.raises(ModelOutputInvalid) as caught:
            await adapter.verify_evidence_scope(QUESTION, [row])
    assert len(requests) == 1
    missing = {tuple(issue["loc"][-1:]) for issue in caught.value.details["schema_errors"] if issue["type"] == "missing"}
    assert {(key,) for key in ("target_catalog_id", "target_binding_id", "quote_requires_unstated_bridge",
                              "target_requires_unstated_bridge", "field_checks")} <= missing
