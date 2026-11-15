from copy import deepcopy

import pytest

from app.domain.question_recall import question_query_plan, recall_question_candidates


def para(i, text, *, doc=None, title=""):
    doc = doc or f"d{i}"
    return {"id": f"{doc}/p{i}", "raw_doc_id": doc, "raw_para_id": f"{doc}#p{i}",
            "title": title, "text": text}


def test_title_alone_never_creates_candidate_and_sources_remain_immutable():
    rows = [para(1, "苹果种植丰收。", title="燃油开关关闭导致飞机失去动力"),
            para(2, "调查报告显示燃油开关关闭后飞机失去动力。")]
    before = deepcopy(rows)
    result = recall_question_candidates("燃油开关关闭是否导致飞机失去动力？", rows)
    assert [c["paragraph_id"] for c in result["candidates"]] == [rows[1]["id"]]
    assert rows == before


def test_long_parent_recalled_from_best_overlapping_window_without_rewriting():
    long = "种植苹果需要充足阳光和水分。" * 200 + "调查记录证明燃油开关关闭导致飞机失去动力。" + "花园里的鲜花需要定期浇水。" * 200
    rows = [para(1, "飞机乘客们在开关附近观察工作人员。"), para(2, long)]
    result = recall_question_candidates("燃油开关关闭是否导致飞机失去动力？", rows, limit=1)
    assert result["candidates"][0]["paragraph_id"] == rows[1]["id"]
    assert rows[1]["text"] == long


def test_question_facets_deduplicate_aliases_and_preserve_full_conditions():
    question = "截至2026年5月，事故原因是机组操作还是设备故障？"
    plan = question_query_plan(question, [{"group": "X", "terms": [question, "设备故障"]}])
    assert [f["facet_id"] for f in plan["facets"]] == [f"F{i}" for i in range(1, len(plan["facets"])+1)]
    assert all(f["full_original_question"] == question for f in plan["facets"])
    assert sum(f["text"] == question for f in plan["facets"]) == 1
    assert any(f["text"] == "设备故障" for f in plan["facets"])


def test_opposite_literal_negation_survives_word_token_identity():
    plan = question_query_plan("是否确认事故原因？", [{"terms": ["已确认事故原因", "未确认事故原因"]}])
    assert {"已确认事故原因", "未确认事故原因"} <= {f["text"] for f in plan["facets"]}


def test_all_supplied_queries_including_last_official_group_survive():
    groups = [{"group": group, "terms": [f"{group}keyword{i}" for i in range(8)]}
              for group in ("X", "Y", "official")]
    plan = question_query_plan("事故原因是机组操作还是设备故障？", groups)
    assert {term for g in groups for term in g["terms"]} <= {f["text"] for f in plan["facets"]}
    assert len(plan["facets"]) <= 33
    with pytest.raises(ValueError, match="groups_invalid"):
        question_query_plan("事故原因", groups + [{"terms": ["extra"]}])


def test_same_document_soft_cap_and_literal_context_have_distinct_ids():
    rows = [para(i, "调查报告指出发动机燃油开关发生故障。", doc="one") for i in range(1, 13)]
    rows += [para(i, "调查报告指出发动机燃油开关发生故障。", doc=f"other{i}") for i in range(13, 23)]
    result = recall_question_candidates("调查报告指出燃油开关是否故障？", rows, limit=12, variant="window_balanced")
    candidates = result["candidates"]
    assert len(candidates) == 12
    assert len({c["paragraph_id"] for c in candidates}) == 12
    assert sum(c["raw_doc_id"] == "one" for c in candidates) <= 5
    assert any(c["raw_doc_id"] != "one" for c in candidates)


def test_exclusion_never_reenters_through_context_or_feedback():
    rows = [para(i, "燃油开关的调查报告尚未完成，具体原因仍待核实。", doc="one") for i in range(1, 8)]
    excluded = {rows[1]["id"], rows[2]["id"]}
    result = recall_question_candidates("调查报告是否已经完成？", rows, excluded_ids=excluded,
                                       variant="window_feedback")
    assert not excluded & {c["paragraph_id"] for c in result["candidates"]}


def test_zero_matches_and_empty_sources_have_no_fabricated_fallback():
    assert recall_question_candidates("燃油开关调查", [para(1, "苹果香蕉草莓")])["candidates"] == []
    assert recall_question_candidates("燃油开关调查", [])["candidates"] == []


def test_repeatable_ties_do_not_coalesce_identical_cross_document_mirrors():
    rows = [para(i, "发动机燃油开关调查结果尚未确认。") for i in range(1, 8)]
    first = recall_question_candidates("燃油开关调查结果是否确认？", rows, limit=5)
    assert first == recall_question_candidates("燃油开关调查结果是否确认？", rows, limit=5)
    assert [c["paragraph_id"] for c in first["candidates"]] == [p["id"] for p in rows[:5]]


@pytest.mark.parametrize("mutate", [lambda p: p.update(id=""), lambda p: p.update(text=None),
                                      lambda p: p.update(raw_para_id=17)])
def test_invalid_source_fails_before_any_result(mutate):
    row = para(1, "有效来源")
    mutate(row)
    with pytest.raises(ValueError, match="source_invalid"):
        recall_question_candidates("有效来源", [row])


def test_duplicate_source_unknown_exclusion_and_bad_parameters_are_rejected():
    row = para(1, "有效来源")
    with pytest.raises(ValueError, match="duplicate_source"):
        recall_question_candidates("有效来源", [row, deepcopy(row)])
    with pytest.raises(ValueError, match="unknown_exclusion"):
        recall_question_candidates("有效来源", [row], excluded_ids={"invented"})
    with pytest.raises(ValueError, match="parameters_invalid"):
        recall_question_candidates("有效来源", [row], limit=True)
    with pytest.raises(ValueError, match="groups_invalid"):
        question_query_plan("有效来源", [{"terms": ["x"]*9}])


def test_unread_reference_like_extras_cannot_change_raw_retrieval():
    rows = [para(1, "燃油开关调查尚未确认。"), para(2, "花园里的鲜花。")]
    decorated = deepcopy(rows)
    decorated[1]["reference_answer"] = "燃油开关调查"
    decorated[0]["model_score"] = 0
    assert recall_question_candidates("燃油开关调查", rows) == recall_question_candidates("燃油开关调查", decorated)
