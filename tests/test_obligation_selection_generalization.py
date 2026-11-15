"""Synthetic invariance checks; these are not competition quality scores."""
from copy import deepcopy

import pytest

from app.domain.obligation_selection import plan_obligation_selection


def scenario(actor="合成甲", year="2971", publisher="合成资料站", credibility=None):
    question = {"question": f"{actor}在{year}年的安排已实施还是仍有程序条件？"}
    catalog = [
        {"paragraph_ref": f"P{i}", "doc_id": f"raw-{i}", "para_id": "original-p",
         "title": publisher, "text": text, "paragraph_complete": True,
         "source_credibility": credibility}
        for i, text in enumerate((f"{actor}称安排已经落实。", "原记录载明先决程序尚须核实。",
                                  "另一份材料对已落实说法提出异议。", "该项安排还受附加条件约束。"), 1)
    ]
    def link(oid, ref, relation):
        return {"obligation_id": oid, "paragraph_ref": ref, "span_ref": "S1",
                "relation": relation, "usage_kind": {"support": "specific_premise",
                "conflict": "counterclaim", "limit": "limitation"}[relation],
                "usage": "仅说明这条材料对待验证命题的作用"}
    proposed = {
        "obligations": [
            {"obligation_id": "O1", "aspect_id": "A1", "role": "F", "ask": "材料是否声称安排落实？"},
            {"obligation_id": "O2", "aspect_id": "A1", "role": "P", "ask": "必要程序和附加条件是什么？"}],
        "judgments": [
            {"judgment_id": "J1", "proposition_to_test": "存在落实说法的来源记录",
             "obligation_ids": ["O1"], "proposed_state": "supported", "links": [link("O1", "P1", "support")]},
            {"judgment_id": "J2", "proposition_to_test": "是否落实仍需核对程序与边界",
             "obligation_ids": ["O2"], "depends_on": ["J1"], "proposed_state": "conflicted",
             "links": [link("O2", "P2", "support"), link("O2", "P3", "conflict"), link("O2", "P4", "limit")]}],
        "answer_depends_on": ["J2"],
    }
    return question, catalog, proposed


def plan(question, catalog, proposed):
    # An explicitly fake constant accounting function isolates structural rule
    # invariance. It is never a tokenizer, bill estimate, or paid request guard.
    return plan_obligation_selection(question, catalog, proposed,
        input_token_cost=lambda _: 100, token_limit=1000, reserved_tokens=200)


@pytest.mark.parametrize("actor,year,publisher,credibility", [
    ("合成甲", "2971", "合成资料站", None),
    ("Example Person", "1982", "Example Gazette", 0),
    ("另一位对象", "3100", "匿名转载", 1),
    ("SYNTHETIC-ID", "2000", "甲方声明", .6),
])
def test_names_dates_publishers_and_credibility_do_not_create_selection_branches(actor, year, publisher, credibility):
    question, catalog, proposed = scenario(actor, year, publisher, credibility)
    before = deepcopy((question, catalog, proposed))
    result = plan(question, catalog, proposed)
    assert result["proposed_selection_refs"] == ["P1", "P2", "P3", "P4"]
    assert result["selected_judgment_ids"] == ["J1", "J2"]
    assert not result["publishable"] and not result["semantic_entailment_verified"]
    assert (question, catalog, proposed) == before


def test_high_overlap_background_padding_does_not_steal_dependency_sources():
    question, catalog, proposed = scenario()
    for i in range(5, 75):
        catalog.append({"paragraph_ref": f"P{i}", "doc_id": f"background-{i}", "para_id": "p",
            "title": "高信源分、重复题目只是背景", "text": question["question"] * 4,
            "paragraph_complete": True, "source_credibility": 1})
    result = plan(question, catalog, proposed)
    assert result["proposed_selection_refs"] == ["P1", "P2", "P3", "P4"]
    assert all(f"P{i}" in result["pending_paragraph_refs"] for i in range(5, 75))
    assert all(row["reason"] == "unlinked_unknown" for row in result["paragraph_decisions"][4:])


def test_no_hidden_ten_paragraph_limit_and_derived_answer_keeps_all_premises():
    question = {"question": "这些合成前提共同成立时，能否作出有限判断？"}
    catalog, obligations, judgments = [], [], []
    for i in range(1, 13):
        catalog.append({"paragraph_ref": f"P{i}", "doc_id": f"source-{i}", "para_id": "p",
                        "text": f"第{i}项合成前提的原文记录。", "paragraph_complete": True})
        obligations.append({"obligation_id": f"O{i}", "aspect_id": "A1", "role": "F", "ask": f"前提{i}是否有原文？"})
        judgments.append({"judgment_id": f"J{i}", "proposition_to_test": f"第{i}项待证前提",
            "obligation_ids": [f"O{i}"], "proposed_state": "supported", "links": [{
                "obligation_id": f"O{i}", "paragraph_ref": f"P{i}", "span_ref": "S1",
                "relation": "support", "usage_kind": "specific_premise", "usage": "提供该前提的来源记录"}]})
    judgments.append({"judgment_id": "J13", "proposition_to_test": "有源前提可以组成待核验判断",
        "obligation_ids": [f"O{i}" for i in range(1, 13)],
        "depends_on": [f"J{i}" for i in range(1, 13)], "proposed_state": "supported", "links": []})
    result = plan(question, catalog, {"obligations": obligations, "judgments": judgments,
                                    "answer_depends_on": ["J13"]})
    assert len(result["proposed_selection_refs"]) == 12
    assert result["selected_judgment_ids"][-1] == "J13"
    assert result["missing_obligation_ids"] == []
    assert not result["judgement_states_verified"]


def test_identical_text_retains_each_original_source_and_local_link():
    question, catalog, proposed = scenario()
    catalog[2]["text"] = catalog[1]["text"]
    catalog[2]["title"] = "另一份原始标题及来源语境"
    result = plan(question, catalog, proposed)
    aliases = [alias for unit in result["reading_payload"]["reading_units"] for alias in unit["source_aliases"]]
    assert {row["paragraph_ref"] for row in aliases} == {"P1", "P2", "P3", "P4"}
    assert len(result["reading_payload"]["reading_units"]) == 3
    assert {link["relation"] for row in result["reading_payload"]["proposed_judgments"] for link in row["links"]} == {
        "support", "conflict", "limit"}
    assert result["independent_source_votes_created"] == 0
    assert not result["sharing_implies_fact_or_source_equivalence"]
