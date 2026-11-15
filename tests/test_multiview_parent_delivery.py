"""Synthetic raw-only fairness/integrity contracts; no model or reference labels."""
from copy import deepcopy
from hashlib import sha256
from uuid import uuid4

import pytest

from app.domain.multiview_parent_delivery import POLICY,pack_original_parents,question_views
from app.domain.source_bundles import paragraph_manifest_sha256
from test_lc_context_case import raw_case


def identities(raw,ids):
    rows={p["id"]:p for p in raw["paragraphs"]}
    return [(rows[pid]["raw_doc_id"],rows[pid]["raw_para_id"]) for pid in ids]


def run(texts,question="材料已经确认还是仍待核验？"):
    raw=raw_case(texts);raw["question"]["question"]=question
    return raw,pack_original_parents(raw,None,[])


def test_binary_single_word_alternative_and_single_clause_have_nonempty_conditioned_views():
    question="截至9月2日，青桥研究院对北区项目评价是正面还是负面？"
    views=question_views(question)
    assert any(v["origin"]=="literal_right" and v["focus"]=="负面" for v in views)
    assert all(v["focus"] and v["full_original_question"]==question for v in views)
    single=question_views("青桥研究院的项目公布了吗？")
    assert single and single[0]["focus"]=="青桥研究院的项目公布了吗？"
    assert all(v["focus"] for v in single)


def test_complete_originals_and_raw_ids_own_cost_and_selection_not_import_uuids():
    raw,(selected,_,receipt)=run([f"材料第{i}项已经确认。"+"完整原始限定😀"*(i%7) for i in range(90)])
    changed=deepcopy(raw);mapping={p["id"]:str(uuid4()) for p in changed["paragraphs"]}
    document_mapping={d["id"]:str(uuid4()) for d in changed["documents"]}
    for p in changed["paragraphs"]:
        p["id"]=mapping[p["id"]];p["document_id"]=document_mapping[p["document_id"]]
        if "paragraph_id" in p:p["paragraph_id"]=p["id"]
    for d in changed["documents"]:
        d["id"]=document_mapping[d["id"]]
        for p in d["paragraph_manifest"]:p["id"]=mapping[p["id"]]
        d["paragraph_manifest_sha256"]=paragraph_manifest_sha256(d["paragraph_manifest"])
    selected2,_,receipt2=pack_original_parents(changed,None,[])
    assert identities(raw,selected)==identities(changed,selected2)
    assert receipt["provided_payload_proxy_tokens"]==receipt2["provided_payload_proxy_tokens"]
    assert [(r["raw_doc_id"],r["raw_para_id"],r["payload_proxy_tokens"],r["reason"]) for r in receipt["candidate_records"]]==[(r["raw_doc_id"],r["raw_para_id"],r["payload_proxy_tokens"],r["reason"]) for r in receipt2["candidate_records"]]


def test_selection_is_rank_first_not_short_parent_density():
    long="材料已经确认，方案已经确认，正式材料已经确认。"*70
    raw,(selected,_,receipt)=run([long,"材料背景。"],"材料已经确认了吗？")
    assert receipt["candidate_records"][0]["raw_para_id"]=="原段0"
    assert selected[0]==raw["paragraphs"][0]["id"]
    assert all(r["priority_is_length_density"] is False for r in receipt["candidate_records"])
    assert receipt["provided_payload_proxy_tokens"]<=8000


def test_oversized_high_priority_parent_is_visible_unknown_not_truncated():
    huge="材料已经确认。"+"原始完整超长限定😀"*10000
    raw,(selected,nav,receipt)=run([huge,"材料仍待核验。"])
    skipped=next(r for r in receipt["candidate_records"] if r["paragraph_id"]==raw["paragraphs"][0]["id"])
    assert skipped["reason"]=="pooled_over_budget"
    assert raw["paragraphs"][0]["id"] not in selected
    assert receipt["source_text_shortened"] is False
    assert any(r["paragraph_id"]==skipped["paragraph_id"] for r in receipt["priority_not_provided"])
    assert all(r["context_state"]=="unknown_not_irrelevant" for r in receipt["unprovided_reason_records"])
    assert "原始完整超长限定" not in str(nav)


def test_mirrors_and_relevant_qualifier_have_separate_source_identities_and_audit():
    text="材料已经进入初步评估，但正式结果尚未确认。"
    raw,(selected,nav,receipt)=run([text,text,"材料已经确认。"])
    assert raw["paragraphs"][0]["id"] in selected and raw["paragraphs"][1]["id"] in selected
    assert any(v["origin"]=="corpus_literal_qualifier_lane" for v in receipt["question_views"])
    group=next(r for r in receipt["exact_mirror_groups"] if r["source_text_sha256"]==sha256(text.encode()).hexdigest())
    assert set(group["paragraph_ids"])=={raw["paragraphs"][0]["id"],raw["paragraphs"][1]["id"]}
    assert group["same_text_is_independent_sources"] is False
    assert nav["BGE_used"] is False and receipt["BGE_used"] is False
    assert receipt["search_page_count"]==0


def test_no_implicit_neighbors_and_no_reference_fields():
    raw,(selected,_,receipt)=run(["材料已经确认。","完全不相关的另一个项目页脚。"])
    assert raw["paragraphs"][1]["id"] not in selected
    contaminated=deepcopy(raw);contaminated["paragraphs"][0]["fact_des"]="注释不能进入原始源"
    with pytest.raises(ValueError,match="raw_only"):
        pack_original_parents(contaminated,None,[])
    assert POLICY["neighbors_added"] is False


def test_operator_exclusions_and_capacity_keep_complete_unknown_audit():
    raw=raw_case([f"材料第{i}项已经确认，仍需核验。" for i in range(200)])
    first=raw["paragraphs"][0]
    selected,_,receipt=pack_original_parents(raw,None,[{"document_id":first["document_id"],"start_cp":first["document_start_cp"],"end_cp":first["document_end_cp"]}])
    assert len(selected)<=60 and receipt["recall_pool_count"]<=180
    assert first["id"] not in selected
    assert next(r for r in receipt["unprovided_reason_records"] if r["paragraph_id"]==first["id"])["reason"]=="operator_exclusion"
    assert receipt["model_calls"]==0 and receipt["reference_data_accessed"] is False


def test_views_and_full_corpus_ranking_are_deterministic():
    raw,result=run(["材料已经确认。","材料仍待核验，尚未正式确认。"])
    assert pack_original_parents(raw,None,[])==result
    assert result[2]["format"]=="complete-parent-token-pack-receipt-v2"
    assert result[2]["policy"]["payload_proxy_token_budget"]==8000


def test_multiple_numeric_scope_options_keep_literal_values_and_full_conditions():
    question="截至某年某月，某事故人数应以机内123人、全部124人还是含地面共156人为准？"
    views=question_views(question)
    options=[v["focus"] for v in views if v["origin"]=="literal_option"]
    assert len(options)==3
    assert any("机内123人" in x for x in options)
    assert "全部124人" in options
    assert any("含地面共156人" in x for x in options)
    assert all(v["full_original_question"]==question for v in views)
    other=question_views("某项目应按包含境外单位、不含境外单位、仅境内主体哪种范围统计？")
    assert any("不含境外单位" in v["focus"] for v in other)
    assert all(v["focus"] for v in other)


def test_long_background_best_window_cannot_override_complete_direct_answer_unit():
    # No candidate/source length gate: both originals remain in the pool.
    # The original parent, rather than its strongest isolated short window,
    # must own BM25 length normalization and the source delivery cost.
    background="材料已经确认。"+"与此事无关的背景叙事和页脚信息。"*1500
    direct="材料已经确认，机构已正式确认。"
    raw,(selected,_,receipt)=run([background,direct],"材料已经确认了吗？")
    from app.domain.question_recall import _bm25
    assert _bm25([{"text":background},{"text":direct}],"材料已经确认了吗？","word",windowed=True)[0][1]==0
    records=receipt["candidate_records"]
    assert records[0]["paragraph_id"]==raw["paragraphs"][1]["id"]
    assert selected[0]==raw["paragraphs"][1]["id"]
    assert raw["paragraphs"][0]["id"] in receipt["recall_pool_paragraph_ids"]
    assert receipt["policy"]["BM25_windowed"] is False
    assert receipt["policy"]["BM25_scoring_unit"]=="complete_original_parent_not_best_short_window"
