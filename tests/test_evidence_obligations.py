"""Synthetic contracts only; no competition source, answers or model calls."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.domain.evidence_obligations import (
    OBLIGATION_INSTRUCTIONS, ObligationPlanOutput, obligation_payload, validate_obligation_plan,
)

QUESTION = {"question": "安排原因还是程序原因导致延迟，哪些前提仍待确认？", "answer": "DO_NOT_FORWARD"}
CATALOG = [
    {"paragraph_ref": "P1", "text": "一方声称安排尚未确定。", "title": "原始标题", "paragraph_complete": True,
     "gold": "GOLD_SENTINEL", "source_credibility": .99, "rank_hint": "RANK_SENTINEL"},
    {"paragraph_ref": "P2", "text": "来源声称存在附条件的程序要求。", "paragraph_complete": False},
    {"paragraph_ref": "P3", "text": "只见主题介绍，后续内容未提供。"},
]


def output():
    return {"obligations": [
        {"obligation_id": "O1", "aspect_id": "A1", "role": "F", "ask": "是否存在影响安排的事实前提？"},
        {"obligation_id": "O2", "aspect_id": "A1", "role": "R", "ask": "是否存在适用的规则及其条件？"}],
        "judgments": [{"judgment_id": "J1", "proposition_to_test": "某来源提出安排未定的说法", "obligation_ids": ["O1"],
            "proposed_state": "supported", "links": [{"obligation_id": "O1", "paragraph_ref": "P1", "span_ref": "S1",
                "relation": "support", "usage_kind": "specific_premise", "usage": "为安排状态的来源说法提供依据"}]},
            {"judgment_id": "J2", "proposition_to_test": "规则是否适用于本题尚待验证", "obligation_ids": ["O2"],
             "depends_on": ["J1"], "links": [{"obligation_id": "O2", "paragraph_ref": "P2", "span_ref": "S1",
                "relation": "limit", "usage_kind": "limitation", "usage": "保留规则适用条件未明这一边界"}]}]}


def test_literal_payload_lossless_once_metadata_allowlist_and_unknown_completeness():
    catalog = deepcopy(CATALOG); text = " \t中英文 sentence. 🙂\r\n" * 120
    catalog[0]["text"] = text
    before = deepcopy(catalog); payload = obligation_payload(QUESTION, catalog)
    assert payload["question"] == {"question": QUESTION["question"]}
    assert all(row["text"] in QUESTION["question"] for row in payload["aspects"])
    assert "".join(block["text"] for block in payload["paragraphs"][0]["blocks"]) == text
    assert sum(len(block["text"]) for block in payload["paragraphs"][0]["blocks"]) == len(text)
    assert not payload["paragraphs"][2]["paragraph_complete"]
    assert all(x not in str(payload) for x in ("GOLD_SENTINEL", "RANK_SENTINEL", "DO_NOT_FORWARD", "source_credibility"))
    assert catalog == before


def test_valid_f_r_premises_remain_proposed_source_bound_and_unpublishable():
    plan = validate_obligation_plan(QUESTION, CATALOG, output())
    assert plan["dependency_order"] == ["J1", "J2"]
    assert plan["obligations"][1]["role"] == "R"
    assert all(row["state"] == "untested" and row["applicability"] == "unknown" for row in plan["obligations"])
    assert plan["source_addresses_resolved"] and not plan["parent_source_mapping_verified"]
    assert all(plan[field] is False for field in ("judgement_states_verified", "semantic_entailment_verified",
        "attribution_or_truth_verified", "rule_applicability_verified", "formal_evidence_generated", "publishable"))
    assert plan["independent_source_votes_created"] == 0
    assert plan["resolved_links"][0]["resolved_source_text"] == CATALOG[0]["text"]
    assert plan["resolved_links"][1]["resolved_source_text"] == CATALOG[1]["text"]
    assert plan["unlinked_paragraph_refs"] == ["P3"]
    assert plan["unlinked_or_omitted_context_status"] == "unknown_not_irrelevant"


def test_unknown_span_never_borrows_address_from_other_paragraph():
    catalog = deepcopy(CATALOG); catalog[0]["text"] = "甲" * 1500
    assert len(obligation_payload(QUESTION, catalog)["paragraphs"][0]["blocks"]) > 1
    value = output(); value["judgments"][1]["links"][0]["span_ref"] = "S2"
    with pytest.raises(ValueError, match="span_not_in_current_paragraph"):
        validate_obligation_plan(QUESTION, catalog, value)


@pytest.mark.parametrize("field,value", [("quote", "改写的引用"), ("offset", 2), ("doc_id", "invented")])
def test_model_cannot_write_text_offsets_or_source_identity(field, value):
    proposed = output(); proposed["judgments"][0]["links"][0][field] = value
    with pytest.raises(ValidationError):
        ObligationPlanOutput.model_validate(proposed)


@pytest.mark.parametrize("damage", ["unknown_obligation", "unknown_link_obligation", "other_judgment_obligation", "unknown_aspect",
                                    "unknown_dependency", "duplicate_obligation", "duplicate_judgment", "unknown_paragraph"])
def test_id_scope_failures_are_explicit(damage):
    value = output(); j = value["judgments"][0]
    if damage == "unknown_obligation": j["obligation_ids"] = ["O99"]
    elif damage == "unknown_link_obligation": j["links"][0]["obligation_id"] = "O99"
    elif damage == "other_judgment_obligation": j["links"][0]["obligation_id"] = "O2"
    elif damage == "unknown_aspect": value["obligations"][0]["aspect_id"] = "A99"
    elif damage == "unknown_dependency": j["depends_on"] = ["J99"]
    elif damage == "duplicate_obligation": value["obligations"].append(deepcopy(value["obligations"][0]))
    elif damage == "duplicate_judgment": value["judgments"].append(deepcopy(j))
    else: j["links"][0]["paragraph_ref"] = "P99"
    with pytest.raises(ValueError):
        validate_obligation_plan(QUESTION, CATALOG, value)


def test_dependency_cycle_rejected_even_with_valid_local_sources():
    value = output(); value["judgments"][0]["depends_on"] = ["J2"]
    value["judgments"][0]["proposed_state"] = "unresolved"
    with pytest.raises(ValueError, match="dependency_cycle"):
        validate_obligation_plan(QUESTION, CATALOG, value)


@pytest.mark.parametrize("damage", ["empty_links", "only_conflict", "missing_conflict", "unresolved_dependency", "conflicted_dependency"])
def test_proposed_closed_judgment_cannot_skip_structural_support(damage):
    value = output(); j = value["judgments"][0]
    if damage == "empty_links": j["links"] = []
    elif damage == "only_conflict": j["links"][0]["relation"] = "conflict"
    elif damage == "missing_conflict": j["proposed_state"] = "conflicted"
    else:
        j["depends_on"] = ["J2"]; value["judgments"][1]["depends_on"] = []
        if damage == "conflicted_dependency":
            other = value["judgments"][1]; other["proposed_state"] = "conflicted"
            for relation in ("support", "conflict"):
                link = deepcopy(other["links"][0]); link["relation"] = relation; other["links"].append(link)
    with pytest.raises(ValueError):
        validate_obligation_plan(QUESTION, CATALOG, value)


def test_minority_conflict_and_limiter_keep_attribution_not_truth():
    value = output(); j = value["judgments"][0]; j["proposed_state"] = "conflicted"
    contrary = deepcopy(j["links"][0]); contrary.update(paragraph_ref="P2", relation="conflict", usage_kind="counterclaim",
        usage="保存相反来源说法，尚不采信其真值")
    j["links"].append(contrary)
    plan = validate_obligation_plan(QUESTION, CATALOG, value)
    assert {link["relation"] for link in plan["resolved_links"]} == {"support", "conflict", "limit"}
    assert not plan["attribution_or_truth_verified"] and not plan["publishable"]


@pytest.mark.parametrize("damage", ["topic_background", "blank_usage", "blank_proposition", "applicability_claim", "known_fact", "missing_time"])
def test_background_unknowns_and_propositions_not_promoted_to_verified_premises(damage):
    value = output()
    if damage == "topic_background": value["judgments"][0]["links"][0]["usage_kind"] = "topic_background"
    elif damage == "blank_usage": value["judgments"][0]["links"][0]["usage"] = " "
    elif damage == "blank_proposition": value["judgments"][0]["proposition_to_test"] = " "
    elif damage == "applicability_claim": value["obligations"][1]["applicability"] = "verified"
    elif damage == "known_fact": value["obligations"][0]["state"] = "known"
    else: value["obligations"][0]["guessed_date"] = "invented"
    with pytest.raises(ValueError):
        validate_obligation_plan(QUESTION, CATALOG, value)


def test_unresolved_is_valid_with_no_links_and_partial_text_is_not_irrelevant():
    value = output(); value["judgments"] = []
    plan = validate_obligation_plan(QUESTION, CATALOG, value)
    assert plan["unlinked_paragraph_refs"] == ["P1", "P2", "P3"]
    assert plan["unlinked_or_omitted_context_status"] == "unknown_not_irrelevant"
    assert not plan["publishable"]
    assert "未见后文未知" in OBLIGATION_INSTRUCTIONS
    assert "最终答案" in OBLIGATION_INSTRUCTIONS


def test_mislabelled_topic_usage_is_not_semantically_certified_by_schema():
    # Text specificity requires semantic examination; do not pretend enum/address
    # validation detects a model hiding background behind a specific-premise label.
    value = output(); value["judgments"][0]["links"][0]["usage"] = "只是同主题"
    plan = validate_obligation_plan(QUESTION, CATALOG, value)
    assert not plan["semantic_entailment_verified"] and not plan["publishable"]


def test_supported_derived_judgment_needs_no_duplicate_direct_source_link():
    value = output(); derived = value["judgments"][1]
    derived["proposed_state"] = "supported"; derived["links"] = []
    plan = validate_obligation_plan(QUESTION, CATALOG, value)
    assert plan["dependency_order"] == ["J1", "J2"]
    assert plan["source_support_structurally_reachable"] == {"J1": True, "J2": True}
    assert len(plan["resolved_links"]) == 1
    assert not plan["judgement_states_verified"] and not plan["semantic_entailment_verified"]
    assert not plan["publishable"]  # Traceability is not validity of the inference.


def test_empty_paragraph_keeps_unknown_identity_without_an_invented_span():
    catalog = deepcopy(CATALOG); catalog[0]["text"] = ""
    payload = obligation_payload(QUESTION, catalog)
    assert payload["paragraphs"][0]["paragraph_ref"] == "P1"
    assert payload["paragraphs"][0]["blocks"] == []
    value = output(); value["judgments"] = []
    plan = validate_obligation_plan(QUESTION, catalog, value)
    assert "P1" in plan["unlinked_paragraph_refs"]
    assert plan["unlinked_or_omitted_context_status"] == "unknown_not_irrelevant"
    with pytest.raises(ValueError, match="span_not_in_current_paragraph"):
        validate_obligation_plan(QUESTION, catalog, output())  # P1/S1 does not exist.


def test_whitespace_is_not_empty_and_nonstring_source_is_rejected():
    catalog = deepcopy(CATALOG); catalog[0]["text"] = " \t\r\n  "
    assert obligation_payload(QUESTION, catalog)["paragraphs"][0]["blocks"] == [
        {"span_ref": "S1", "text": " \t\r\n  "}]
    for invalid in (None, 0, False, [], {}, b"text"):
        catalog[0]["text"] = invalid
        with pytest.raises(ValueError, match="source_text_string_required"):
            obligation_payload(QUESTION, catalog)


def test_answer_paths_include_all_relations_and_preserve_outside_source_nodes():
    value = output(); value["answer_depends_on"] = ["J2"]
    counter = deepcopy(value["judgments"][0]["links"][0]); counter["relation"] = "conflict"
    value["judgments"][0]["links"].append(counter)
    value["judgments"][0]["proposed_state"] = "conflicted"
    orphan = deepcopy(value["judgments"][0]); orphan.update(judgment_id="J3", proposition_to_test="另一个有源但未接入答案的判断")
    orphan["links"] = [deepcopy(orphan["links"][0])]; orphan["links"][0]["paragraph_ref"] = "P3"
    orphan["proposed_state"] = "supported"; value["judgments"].append(orphan)
    plan = validate_obligation_plan(QUESTION, CATALOG, value)
    assert plan["proposed_answer_reachable_judgment_ids"] == ["J1", "J2"]
    assert {row["relation"] for row in plan["proposed_answer_source_refs"]} == {"support", "conflict", "limit"}
    assert {row["paragraph_ref"] for row in plan["proposed_answer_source_refs"]} == {"P1", "P2"}
    assert plan["orphan_judgment_ids"] == ["J3"]
    assert len(plan["judgments"]) == 3 and any(row["paragraph_ref"] == "P3" for row in plan["resolved_links"])
    assert not plan["publishable"] and not plan["answer_reachability_proves_necessity_or_inference_truth"]


@pytest.mark.parametrize("dependencies", [["J99"], ["J1", "J1"]])
def test_answer_dependencies_unknown_or_duplicate_are_rejected(dependencies):
    value = output(); value["answer_depends_on"] = dependencies
    with pytest.raises(ValueError, match="answer_unknown_or_duplicate_dependency"):
        validate_obligation_plan(QUESTION, CATALOG, value)


def test_answer_sources_reachable_through_multiple_proposed_derivations():
    value = output(); middle = value["judgments"][1]
    middle["links"] = []; middle["proposed_state"] = "supported"
    final = deepcopy(middle); final.update(judgment_id="J3", depends_on=["J2"])
    value["judgments"].append(final); value["answer_depends_on"] = ["J3"]
    plan = validate_obligation_plan(QUESTION, CATALOG, value)
    assert plan["proposed_answer_reachable_judgment_ids"] == ["J1", "J2", "J3"]
    assert len(plan["proposed_answer_source_refs"]) == 1
    assert plan["proposed_answer_source_refs"][0]["paragraph_ref"] == "P1"
    assert not plan["judgement_states_verified"] and not plan["publishable"]


def test_empty_answer_dependencies_mean_unresolved_not_irrelevant_or_success():
    plan = validate_obligation_plan(QUESTION, CATALOG, output())
    assert plan["answer_depends_on"] == [] and plan["proposed_answer_state"] == "unresolved"
    assert plan["proposed_answer_reachable_judgment_ids"] == plan["proposed_answer_source_refs"] == []
    assert plan["orphan_judgment_ids"] == ["J1", "J2"] and len(plan["resolved_links"]) == 2
    assert plan["unlinked_or_omitted_context_status"] == "unknown_not_irrelevant"
    assert not plan["judgement_states_verified"] and not plan["publishable"]
