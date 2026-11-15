"""Synthetic contract fixtures, not evidence of real model correctness."""

from copy import deepcopy

import pytest

from app.domain.snapshot import build_snapshot, competition_files


def fixture():
    topic = {"id": "topic-internal"}
    question = {"id": "question-internal", "topic_id": topic["id"], "raw_question_id": "q_001", "question": "通报失联人数是多少？"}
    evidence = {"id": "evidence-internal", "paragraph_id": "paragraph-internal", "raw_doc_id": "doc_001", "raw_para_id": "doc_001#p1", "start_cp": 4, "end_cp": 8, "quote": "失联3人", "fact_des": "通报称失联3人", "argument": {"时间": None, "主体": "失联人员", "地点": None, "内容": "失联3人", "影响": None, "数值口径": "3人"}, "entities": ["失联人员"], "source_group_id": "source_1", "source_group_basis": "same_document_only_not_complete_origin_detection", "grade_score": None, "source_level": "unknown"}
    final = {"decision": "asserted", "conclusion": "通报称失联3人。", "judgments": [{"id": "judgment-1", "statement": "通报明确记载人数", "evidence_ids": [evidence["id"]], "prerequisite_judgment_ids": []}, {"id": "judgment-2", "statement": "本题按该通报口径回答", "evidence_ids": [], "prerequisite_judgment_ids": ["judgment-1"]}], "supporting_evidence_ids": [evidence["id"]], "conflicting_evidence_ids": [], "conflicts": []}
    analyses = [{"role": role, "evidence_snapshot_id": "frozen-1", **{key: deepcopy(value) for key, value in final.items() if key != "conflicts"}} for role in ("authority_analyst", "cross_validation_analyst")]
    return topic, [question], {question["id"]: {"evidence": [evidence], "analyses": analyses, "final": final, "usage": [], "evidence_snapshot_id": "frozen-1"}}


def build(topic, questions, results):
    return build_snapshot(topic, questions, results, revision_id="revision-1", graph_revision=1, run_id="run-1")


def test_complete_graph_and_four_official_files_keep_original_ids():
    topic, questions, results = fixture()
    original = deepcopy(results)
    snapshot = build(topic, questions, results)
    files = competition_files(snapshot)
    assert set(files) == {"evidence.json", "conclusions.json", "confidence.json", "conflicts.json"}
    assert len(snapshot["nodes"]) == 4
    assert {node["kind"] for node in snapshot["nodes"]} == {"evidence", "judgment", "conclusion"}
    assert {"from_node_id": "judgment-1", "to_node_id": "judgment-2", "relation": "prerequisite"} in snapshot["edges"]
    conclusion_id = snapshot["conclusions"][0]["conclusion_node_id"]
    assert {"from_node_id": "judgment-2", "to_node_id": conclusion_id, "relation": "prerequisite"} in snapshot["edges"]
    assert files["evidence.json"][0]["doc_id"] == "doc_001"
    assert files["evidence.json"][0]["para_id"] == "doc_001#p1"
    assert files["conclusions.json"][0]["question_id"] == "q_001"
    assert set(files["conclusions.json"][0]) == {"question_id", "question", "conclusion", "supporting_facts", "conflicting_facts"}
    assert snapshot["evidence"][0]["grade_score"] is None
    assert files["confidence.json"][0]["score"] == 0
    assert files["confidence.json"][0]["inputs"]["support_strength"] == 0
    assert "not been calibrated" in files["confidence.json"][0]["inputs"]["support_strength_basis"]
    assert files["confidence.json"][-1]["components"]["reasoning_completeness"] == 1
    assert files["confidence.json"][-1]["inputs"]["support_groups"][0]["grade_score"] == 0
    assert results == original
    assert len(snapshot["agent_outputs"]) == 3


def test_conclusion_id_is_stable_between_revisions():
    topic, questions, results = fixture()
    first = build(topic, questions, results)
    second = build_snapshot(topic, questions, results, revision_id="revision-2", graph_revision=2, run_id="run-2")
    assert first["conclusions"][0]["conclusion_node_id"] == second["conclusions"][0]["conclusion_node_id"]


@pytest.mark.parametrize("kind", ["support", "judgment", "prerequisite"])
def test_absent_references_are_rejected(kind):
    topic, questions, results = fixture()
    final = results[questions[0]["id"]]["final"]
    if kind == "support":
        final["supporting_evidence_ids"] = ["phantom"]
    elif kind == "judgment":
        final["judgments"][0]["evidence_ids"] = ["phantom"]
    else:
        final["judgments"][0]["prerequisite_judgment_ids"] = ["phantom"]
    with pytest.raises(ValueError, match="unknown reference"):
        build(topic, questions, results)


def test_no_evidence_can_produce_explicit_undetermined_with_zero_quality():
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    result["evidence"] = []
    result["final"].update(decision="undetermined", conclusion="缺少依据。", judgments=[], supporting_evidence_ids=[])
    for analysis in result["analyses"]:
        analysis.update(decision="undetermined", conclusion="缺少依据。", judgments=[], supporting_evidence_ids=[])
    snapshot = build(topic, questions, results)
    files = competition_files(snapshot)
    assert snapshot["conclusions"][0]["decision"] == "undetermined"
    assert snapshot["conclusions"][0]["confidence"] == 0
    assert files["evidence.json"] == []
    assert files["conflicts.json"] == []
    assert files["conclusions.json"][0]["supporting_facts"] == []
    assert files["conclusions.json"][0]["conflicting_facts"] == []
    assert len(files["confidence.json"]) == 1
    assert files["confidence.json"][0]["score"] == 0
    assert files["confidence.json"][0]["inputs"]["support_groups"] == []


def test_asserted_without_support_is_rejected_at_build_and_export_boundaries():
    topic, questions, results = fixture()
    results[questions[0]["id"]]["final"]["supporting_evidence_ids"] = []
    with pytest.raises(ValueError, match="qualified supporting"):
        build(topic, questions, results)
    topic, questions, results = fixture()
    snapshot = build(topic, questions, results)
    snapshot["conclusions"][0]["supporting_evidence_ids"] = []
    with pytest.raises(ValueError, match="qualified supporting"):
        competition_files(snapshot)


def test_dag_cycle_and_ungrounded_judgment_rejected():
    topic, questions, results = fixture()
    final = results[questions[0]["id"]]["final"]
    final["judgments"][0]["prerequisite_judgment_ids"] = ["judgment-2"]
    with pytest.raises(ValueError, match="cycle"):
        build(topic, questions, results)
    final["judgments"][0]["prerequisite_judgment_ids"] = []
    final["judgments"][0]["evidence_ids"] = []
    with pytest.raises(ValueError, match="grounded"):
        build(topic, questions, results)


def test_numeric_literal_guard_does_not_accept_substring_match():
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    result["evidence"][0].update(quote="失联30人", end_cp=9)
    with pytest.raises(ValueError, match="numeric literal"):
        build(topic, questions, results)


def test_numeric_guard_allows_identifying_a_cited_opposing_claim():
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    support = result["evidence"][0]
    support_quote = "监控记录10:52至10:59"
    support.update(quote=support_quote, end_cp=support["start_cp"] + len(support_quote), fact_des="监控记录的时间范围")
    opposing = deepcopy(support)
    opposing_quote = "官方通稿所称10:30"
    opposing.update(
        id="evidence-time-opposing", paragraph_id="paragraph-time-opposing",
        raw_doc_id="doc_time_opposing", raw_para_id="doc_time_opposing#p1",
        quote=opposing_quote, end_cp=opposing["start_cp"] + len(opposing_quote),
        fact_des="官方通稿给出的时刻", source_group_id="source_time_opposing",
    )
    result["evidence"].append(opposing)
    result["final"].update(
        conclusion="采纳监控记录10:52至10:59，官方通稿所称10:30不予采纳。",
        conflicting_evidence_ids=[opposing["id"]],
    )
    snapshot = build(topic, questions, results)
    files = competition_files(snapshot)
    assert snapshot["conclusions"][0]["decision"] == "asserted"
    support_ref = files["conclusions.json"][0]["supporting_facts"][0]
    conflict_ref = files["conclusions.json"][0]["conflicting_facts"][0]
    assert (support_ref["doc_id"], support_ref["para_id"], support_ref["quote"]) == ("doc_001", "doc_001#p1", support_quote)
    assert (conflict_ref["doc_id"], conflict_ref["para_id"], conflict_ref["quote"]) == ("doc_time_opposing", "doc_time_opposing#p1", opposing_quote)
    assert support_ref["fact_id"] == files["evidence.json"][0]["fact_id"]
    assert conflict_ref["resolution_type"] == "Other"


@pytest.mark.parametrize("uncited_source", ["candidate", "question", "absent"])
def test_numeric_guard_cannot_borrow_numbers_from_uncited_candidates_or_question(uncited_source):
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    result["final"]["conclusion"] = "通报称失联99人。"
    if uncited_source == "candidate":
        candidate = deepcopy(result["evidence"][0])
        quote = "另有说法失联99人"
        candidate.update(
            id="evidence-uncited", paragraph_id="paragraph-uncited",
            raw_doc_id="doc_uncited", raw_para_id="doc_uncited#p1",
            quote=quote, end_cp=candidate["start_cp"] + len(quote),
            fact_des="尚未引用的候选说法", source_group_id="source_uncited",
        )
        result["evidence"].append(candidate)
    elif uncited_source == "question":
        questions[0]["question"] = "通报是否称失联99人？"
    with pytest.raises(ValueError, match="absent from cited evidence"):
        build(topic, questions, results)


def add_conflict(results, question_id, status="resolved"):
    result = results[question_id]
    other = deepcopy(result["evidence"][0])
    other.update(id="evidence-opposing", paragraph_id="paragraph-opposing", raw_doc_id="doc_002", raw_para_id="doc_002#p1", quote="失联4人", fact_des="另一来源称失联4人", source_group_id="source_2")
    result["evidence"].append(other)
    result["final"]["conflicting_evidence_ids"] = [other["id"]]
    result["final"]["conflicts"] = [{"id": "conflict-1", "statement": "同口径人数存在差异", "status": status, "supporting_evidence_ids": ["evidence-internal"], "conflicting_evidence_ids": [other["id"]], "resolution_type": "source_priority", "basis": "同主体、同时间与同范围，来源等级须人工核对", "accepted_reason": "按题目指定通报记录", "rejected_reason": "另一说法缺少同等直接记录", "minority_opinion": "另一来源仍报告4人", "reopen_conditions": "出现更正通报时重新核验"}]


def test_conflict_projection_preserves_minority_and_reopen_condition():
    topic, questions, results = fixture()
    add_conflict(results, questions[0]["id"])
    snapshot = build(topic, questions, results)
    conflict = competition_files(snapshot)["conflicts.json"][0]
    assert conflict["question_id"] == "q_001"
    assert conflict["result"] == "adopt_a"
    assert conflict["minority_opinions"][0]["evidence_refs"] == [{"doc_id": "doc_002", "para_id": "doc_002#p1"}]
    assert conflict["reopen_condition"]
    assert "topic_id" not in conflict and "id" not in conflict
    assert len(snapshot["source_groups"]) == 2


def numeric_conflict_fixture(*, same_paragraph=False):
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    side_a = result["evidence"][0]
    quote_a = "甲方通报遇难14人。"
    quote_b = "乙方称可能遇难16人，尚待核实。"
    shared_quote = quote_a + quote_b
    side_a.update(quote=shared_quote if same_paragraph else quote_a,
        start_cp=0, end_cp=len(shared_quote if same_paragraph else quote_a),
        fact_des="甲方通报称遇难14人。", entities=["遇难人员"],
        argument={"时间": None, "主体": "甲方", "地点": None, "内容": quote_a, "影响": None, "数值口径": "遇难14人"})
    side_b = deepcopy(side_a)
    side_b.update(id="fact-16", quote=shared_quote if same_paragraph else quote_b,
        end_cp=len(shared_quote if same_paragraph else quote_b),
        fact_des="乙方称可能遇难16人，尚待核实。",
        argument={"时间": None, "主体": "乙方", "地点": None, "内容": quote_b, "影响": None, "数值口径": "遇难16人"})
    if not same_paragraph:
        side_b.update(paragraph_id="paragraph-16", raw_doc_id="doc_16", raw_para_id="doc_16#p1", source_group_id="source_16")
    result["evidence"].append(side_b)
    result["final"].update(conclusion="甲方通报14人，乙方16人说法尚待核实。", conflicting_evidence_ids=[side_b["id"]])
    result["final"]["conflicts"] = [{"id": "conflict-numbers", "statement": "遇难统计14人与16人的说法需核对。",
        "status": "resolved", "supporting_evidence_ids": [side_a["id"]], "conflicting_evidence_ids": [side_b["id"]],
        "resolution_type": "source_priority", "basis": "先核对各自来源与统计时点。", "accepted_reason": "按指定甲方通报口径。",
        "rejected_reason": "乙方说法仍待核实。", "minority_opinion": "乙方仍称可能16人，尚待核实。",
        "reopen_conditions": "出现对应时点的新记录或更正。"}]
    for analysis in result["analyses"]:
        analysis.update({key: deepcopy(value) for key, value in result["final"].items() if key != "conflicts"})
    return topic, questions, results


@pytest.mark.parametrize("same_paragraph", [False, True])
def test_conflict_positions_use_own_fact_ids_not_global_comparison_or_shared_quote(same_paragraph):
    topic, questions, results = numeric_conflict_fixture(same_paragraph=same_paragraph)
    result = results[questions[0]["id"]]
    original = deepcopy(results)
    snapshot = build(topic, questions, results)
    internal = snapshot["conflicts"][0]
    exported = competition_files(snapshot)["conflicts.json"][0]
    assert internal["issue"] == result["final"]["conflicts"][0]["statement"]
    assert "issue" not in exported and "statement" not in exported
    assert exported["side_a"]["position"] == "来源陈述：甲方通报称遇难14人。"
    assert exported["side_b"]["position"] == "来源陈述：乙方称可能遇难16人，尚待核实。"
    assert "16" not in exported["side_a"]["position"] and "14" not in exported["side_b"]["position"]
    assert results == original
    if same_paragraph:
        assert exported["side_a"]["evidence_refs"] == exported["side_b"]["evidence_refs"]
        assert result["evidence"][0]["quote"] == result["evidence"][1]["quote"]


def test_conflict_side_multiple_statements_keep_exact_attribution_without_other_pool_facts():
    topic, questions, results = numeric_conflict_fixture()
    result = results[questions[0]["id"]]
    extra = deepcopy(result["evidence"][0])
    quote = "补充记录提示，此数字仅为初步统计。"
    extra.update(id="a-qualifier", paragraph_id="p-qualifier", raw_para_id="doc_001#p2",
        quote=quote, start_cp=0, end_cp=len(quote), fact_des="补充记录提示：此数字仅为初步统计。")
    unused = deepcopy(extra)
    quote = "丙方另称99人。"
    unused.update(id="unused-99", paragraph_id="unused-p", raw_doc_id="unused-doc", raw_para_id="unused-doc#p1",
        quote=quote, end_cp=len(quote), fact_des="丙方另称99人，不属于这两侧。")
    result["evidence"].extend([extra, unused])
    raw = result["final"]["conflicts"][0]
    raw["statement"] += "全池另有99人说法。"
    raw["supporting_evidence_ids"] = [extra["id"], result["evidence"][0]["id"]]
    projected = competition_files(build(topic, questions, results))["conflicts.json"][0]
    assert projected["side_a"]["position"] == "来源陈述：" + extra["fact_des"] + "\n" + result["evidence"][0]["fact_des"]
    assert projected["side_b"]["position"] == "来源陈述：" + result["evidence"][1]["fact_des"]
    assert all("99" not in projected[side]["position"] for side in ("side_a", "side_b"))
    assert projected["side_a"]["evidence_refs"] == [
        {"doc_id": "doc_001", "para_id": "doc_001#p2"}, {"doc_id": "doc_001", "para_id": "doc_001#p1"}]


@pytest.mark.parametrize("resolution", ["adopt_a", "adopt_b", "unresolved", "coexist"])
def test_side_statement_projection_preserves_resolution_minority_and_unresolved_compatibility(resolution):
    topic, questions, results = numeric_conflict_fixture()
    result = results[questions[0]["id"]]
    side_a, side_b = result["evidence"]
    raw = result["final"]["conflicts"][0]
    raw["status"] = {"adopt_a": "resolved", "adopt_b": "resolved", "unresolved": "unresolved", "coexist": "coexisting"}[resolution]
    if resolution == "adopt_b":
        result["final"].update(supporting_evidence_ids=[side_b["id"]], conflicting_evidence_ids=[side_a["id"]])
        raw["minority_opinion"] = "甲方仍通报14人。"
    elif resolution == "unresolved":
        result["final"].update(decision="undetermined", conclusion="两侧说法尚未统一。", supporting_evidence_ids=[],
            conflicting_evidence_ids=[side_a["id"], side_b["id"]])
    elif resolution == "coexist":
        result["final"].update(conclusion="两个时点的14人与16人说法分别保留。", supporting_evidence_ids=[side_a["id"], side_b["id"]], conflicting_evidence_ids=[])
    snapshot = build(topic, questions, results)
    conflict = competition_files(snapshot)["conflicts.json"][0]
    assert conflict["result"] == resolution
    assert conflict["side_a"]["position"] == "来源陈述：" + side_a["fact_des"]
    assert conflict["side_b"]["position"] == "来源陈述：" + side_b["fact_des"]
    assert conflict["reopen_condition"] == raw["reopen_conditions"]
    if resolution in {"adopt_a", "adopt_b"}:
        minority = side_b if resolution == "adopt_a" else side_a
        assert conflict["minority_opinions"][0]["position"] == raw["minority_opinion"]
        assert conflict["minority_opinions"][0]["evidence_refs"] == [{"doc_id": minority["raw_doc_id"], "para_id": minority["raw_para_id"]}]
    else:
        assert conflict["minority_opinions"] == []
    if resolution == "unresolved":
        assert snapshot["conclusions"][0]["confidence"] == 0
        assert snapshot["confidence"][-1]["inputs"]["unresolved_conflict_groups"]


def test_unresolved_conflict_remains_in_quality_input():
    topic, questions, results = fixture()
    add_conflict(results, questions[0]["id"], "unresolved")
    snapshot = build(topic, questions, results)
    assert snapshot["conflicts"][0]["result"] == "unresolved"
    assert len(snapshot["confidence"][-1]["inputs"]["unresolved_conflict_groups"]) == 1


def test_undetermined_without_support_preserves_grounded_judgments_and_both_conflict_sides():
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    add_conflict(results, questions[0]["id"], "unresolved")
    result["final"].update(
        decision="undetermined",
        conclusion="通报数字存在冲突，当前证据不足以确定统一人数。",
        supporting_evidence_ids=[],
        conflicting_evidence_ids=[row["id"] for row in result["evidence"]],
    )
    for analysis in result["analyses"]:
        analysis.update(
            decision="undetermined",
            conclusion=result["final"]["conclusion"],
            supporting_evidence_ids=[],
            conflicting_evidence_ids=list(result["final"]["conflicting_evidence_ids"]),
        )
    original = deepcopy(results)
    snapshot = build(topic, questions, results)
    files = competition_files(snapshot)
    assert set(files) == {"evidence.json", "conclusions.json", "confidence.json", "conflicts.json"}
    assert results == original
    assert len([node for node in snapshot["nodes"] if node["kind"] == "judgment"]) == 2
    conclusion = snapshot["conclusions"][0]
    assert conclusion["decision"] == "undetermined"
    assert conclusion["supporting_evidence_ids"] == []
    assert conclusion["confidence"] == 0
    assert not any(edge["to_node_id"] == conclusion["conclusion_node_id"] and edge["relation"] == "supports" for edge in snapshot["edges"])
    assert files["conclusions.json"][0]["supporting_facts"] == []
    assert [(ref["doc_id"], ref["para_id"]) for ref in files["conclusions.json"][0]["conflicting_facts"]] == [
        ("doc_001", "doc_001#p1"), ("doc_002", "doc_002#p1")]
    assert all(ref["resolution_type"] == "Source_Priority" for ref in files["conclusions.json"][0]["conflicting_facts"])
    assert files["conflicts.json"][0]["result"] == "unresolved"
    assert files["conflicts.json"][0]["side_a"]["evidence_refs"]
    assert files["conflicts.json"][0]["side_b"]["evidence_refs"]
    assert files["confidence.json"][-1]["score"] == 0
    assert files["confidence.json"][-1]["inputs"]["support_groups"] == []
    assert files["confidence.json"][-1]["inputs"]["unresolved_conflict_groups"]


def test_same_evidence_across_questions_is_deduplicated_without_implying_dependency():
    topic, questions, results = fixture()
    second = {**questions[0], "id": "question-second", "raw_question_id": "q_002"}
    questions.append(second)
    results[second["id"]] = deepcopy(results[questions[0]["id"]])
    results[second["id"]]["final"]["judgments"] = []
    snapshot = build(topic, questions, results)
    assert len(snapshot["evidence"]) == 1
    assert len(snapshot["conclusions"]) == 2
    conclusions = {row["conclusion_node_id"] for row in snapshot["conclusions"]}
    assert not any(edge["from_node_id"] in conclusions and edge["to_node_id"] in conclusions for edge in snapshot["edges"])


def test_incomplete_question_set_and_nonindependent_agents_rejected():
    topic, questions, results = fixture()
    with pytest.raises(ValueError, match="every requested question"):
        build(topic, questions, {})
    results[questions[0]["id"]]["analyses"][1]["evidence_snapshot_id"] = "different-frozen-input"
    with pytest.raises(ValueError, match="same frozen"):
        build(topic, questions, results)


def test_export_revalidates_cross_file_references():
    topic, questions, results = fixture()
    snapshot = build(topic, questions, results)
    snapshot["confidence"][-1]["inputs"]["support_groups"][0]["evidence_refs"][0]["quote"] = "absent"
    with pytest.raises(ValueError, match="absent evidence"):
        competition_files(snapshot)


def test_export_cannot_silently_drop_a_question_or_change_a_score():
    topic, questions, results = fixture()
    snapshot = build(topic, questions, results)
    snapshot["confidence"][-1]["score"] = 0.99
    with pytest.raises(ValueError, match="reproduced"):
        competition_files(snapshot)
    snapshot = build(topic, questions, results)
    snapshot["conclusions"] = []
    with pytest.raises(ValueError, match="question set"):
        competition_files(snapshot)
