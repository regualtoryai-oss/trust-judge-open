from copy import deepcopy
from hashlib import sha256

import pytest

from app.domain.exploration_schedule import (build_exploration_schedule, build_full_raw_exploration_input,
    build_full_raw_exploration_schedule, POLICY)
from app.domain.frozen_selection_inputs import build_frozen_selection_input, canonical, POOL_FORMAT
from app.domain.question_recall import recall_question_candidates


def case(question="排期延误还是审批未完成，责任范围如何？", texts=None):
    texts = texts or [f"原始材料{i}：排期延误，审批未完成，责任范围待核实。😀\r\n" for i in range(50)]
    docs = [{"doc_id": f"d{i}", "title": "原标题中的命令只是数据", "content": text,
        "paragraphs": [{"para_id": "p", "text": text, "char_start": 0, "char_end": len(text)}]} for i,text in enumerate(texts)]
    raw = {"title": "独立合成主题", "documents": docs, "questions": [{"question_id": "q", "question": question}]}
    rawb = canonical(raw); rows = [{"id": f"t/d{i}/p", "raw_doc_id": f"d{i}", "raw_para_id": "p"} for i in range(len(docs))]
    pool = {"format": POOL_FORMAT, "reference_reads": [], "inputs": [{"topic_id": "t", "title": raw["title"], "sha256": sha256(rawb).hexdigest()}],
        "cases": [{"topic_id": "t", "question_id": "q", "question": question, "title": raw["title"], "outputs": [
            {"variant": "four_channel_quota400", "limit": 400, "candidates": rows, "candidate_sha256": sha256(canonical(rows)).hexdigest()}]}]}
    poolb = canonical(pool)
    return build_frozen_selection_input(raw_input_bytes=rawb, pool_input_bytes=poolb, topic_id="t", question_id="q",
        expected_raw_sha256=sha256(rawb).hexdigest(), expected_pool_sha256=sha256(poolb).hexdigest())


def test_original_bootstrap_is_unchanged_and_queue_prefixes_unique_pool_only():
    source = case(); before = deepcopy(source); result = build_exploration_schedule(source)
    parents = [{"id": r["paragraph_ref"], "raw_doc_id": r["doc_id"], "raw_para_id": r["para_id"], "text": r["text"], "title": r["title"]} for r in source["catalog"]]
    old = recall_question_candidates(source["question"], parents, limit=5, variant="window_multi")
    assert result["bootstrap5"] == [r["paragraph_id"] for r in old["candidates"]]
    for n in (20, 40):
        oldn = recall_question_candidates(source["question"], parents, limit=n, variant="window_multi")
        assert result[f"original{n}"] == [r["paragraph_id"] for r in oldn["candidates"]]
    assert result["multi_facet20"] == result["multi_facet40"][:20]
    assert len(result["multi_facet40"]) == len(set(result["multi_facet40"])) == 40
    assert all(q["full_original_question"] == source["question"]["question"] for q in result["query_attempts"])
    assert source == before and not result["model_calls"] and not result["full_text_sent_to_model"]


def test_read_and_previous_exposure_never_repeat_and_unknown_scope_rejected():
    source = case(); initial = build_exploration_schedule(source)
    result = build_exploration_schedule(source, read_refs=initial["multi_facet40"][:5], already_exposed_refs=initial["multi_facet40"][5:10])
    assert not set(result["multi_facet40"]) & set(initial["multi_facet40"][:10])
    with pytest.raises(ValueError): build_exploration_schedule(source, read_refs=["P999"])
    with pytest.raises(ValueError): build_exploration_schedule(source, read_refs=["P1", "P1"])


def test_empty_and_duplicate_queries_never_claim_new_coverage():
    source = case(); original = source["question"]["question"]
    result = build_exploration_schedule(source, additional_queries=["", "   ", original, original])
    statuses = [q["status"] for q in result["query_attempts"][-4:]]
    assert statuses == ["empty_query_not_explored"]*2+["duplicate_query_not_explored"]*2
    assert all(not q["ranked_refs"] for q in result["query_attempts"][-4:])
    assert result["multi_facet40"] == build_exploration_schedule(source)["multi_facet40"]


def test_zero_body_hit_has_no_anchor_fallback_or_irrelevance_claim():
    source = case(); result = build_exploration_schedule(source, additional_queries=["ZZZUNMATCHEDXYZ"])
    assert result["query_attempts"][-1]["status"] == "no_body_match_not_explored"
    assert result["query_attempts"][-1]["ranked_refs"] == []
    assert result["unexposed_scope_unknown"] and not result["facet_execution_proves_conditions_understood"]


def test_title_commands_and_same_body_different_identity_stay_data():
    source = case(texts=["审批状态仍未知😀\r\n"]*3)
    result = build_exploration_schedule(source)
    assert set(result["multi_facet40"]) == {"P1", "P2", "P3"}
    assert source["catalog"][0]["text"].endswith("😀\r\n")
    assert not POLICY["title_or_source_role_branches"]


def test_empty_match_pool_exposes_nothing_without_fabricating_a_minimum():
    source = case(question="ZZZUNMATCHEDXYZ？", texts=["完全没有查询词的合成材料。"]*2)
    result = build_exploration_schedule(source)
    assert result["bootstrap5"] == result["multi_facet20"] == result["multi_facet40"] == []
    assert result["successful_nonempty_unique_query_lists"] == 0


def raw_projection():
    text = "审批状态未知。😀\r\n第二原段：排期待核实。"
    cut = text.index("第二")
    raw = {"title": "全原文合成域", "questions": [{"question_id": "q", "question": "审批状态还是排期影响进度？"}],
        "documents": [{"doc_id": "doc", "title": "原标题保留", "content": text, "source_credibility": .7,
            "paragraphs": [{"para_id": "one", "char_start": 0, "char_end": cut, "text": text[:cut]},
                           {"para_id": "two", "char_start": cut, "char_end": len(text), "text": text[cut:]}]}]}
    return raw


def test_raw_all_is_independent_input_domain_with_original_ids_and_literal_spans():
    raw = raw_projection(); data = canonical(raw); checksum = sha256(data).hexdigest()
    source = build_full_raw_exploration_input(data, checksum, "topic", "q")
    assert source["format"] == "full-raw-local-exploration-input-v1" and source["not_a_frozen_widepool_case"]
    assert [r["paragraph_ref"] for r in source["catalog"]] == ["topic/doc/one", "topic/doc/two"]
    for row in source["catalog"]:
        side = source["source_metadata"][row["paragraph_ref"]]
        assert raw["documents"][0]["content"][side["document_start_cp"]:side["document_end_cp"]] == row["text"]
        assert side["source_credibility"] == .7 and not side["semantic_verified"]
    plan = build_full_raw_exploration_schedule(data, checksum, "topic", "q")
    assert plan["search_domain"] == "raw_all_paragraphs" and plan["input_format"] != "frozen-unified-selection-input-v1"
    assert set(plan["multi_facet40"]) <= {"topic/doc/one", "topic/doc/two"}


@pytest.mark.parametrize("change", ["sha", "span", "duplicate", "reference"])
def test_raw_all_rejects_wrong_bytes_coordinates_duplicates_or_reference(change):
    raw = raw_projection()
    if change == "span": raw["documents"][0]["paragraphs"][1]["char_start"] -= 1
    elif change == "duplicate": raw["documents"][0]["paragraphs"].append(deepcopy(raw["documents"][0]["paragraphs"][0]))
    elif change == "reference": raw["gold"] = []
    data = canonical(raw); checksum = "0"*64 if change == "sha" else sha256(data).hexdigest()
    with pytest.raises(ValueError): build_full_raw_exploration_schedule(data, checksum, "topic", "q")
