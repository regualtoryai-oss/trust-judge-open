"""Domain invariants using explicitly synthetic fixtures."""

from copy import deepcopy
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from app.domain.confidence import compute_conclusion_confidence, compute_evidence_confidence
from app.domain.graph import downstream_closure, validate_graph
from app.domain.retrieval import retrieve
from app.domain.spans import cp_to_utf16, validate_span


def group(identity="source_1", grade=0.9, quote="原句"):
    return {"group_id": identity, "grade_score": grade, "grade_basis": "test source grading", "evidence_refs": [{"doc_id": identity, "para_id": "p1", "quote": quote}]}


def test_codepoint_emoji_and_repeated_quote_are_position_specific():
    text = "🦊同句。中间。同句。"
    validate_span(text, "同句。", 7, 10)
    assert cp_to_utf16(text, 7) == 8
    assert cp_to_utf16(text, 0) == 0
    assert cp_to_utf16(text, len(text)) == len(text) + 1
    with pytest.raises(ValueError):
        validate_span(text, "同句。", 8, 11)
    with pytest.raises(ValueError):
        validate_span(text, "同句!", 7, 10)


@pytest.mark.parametrize("coordinates", [(-1, 1), (0, 0), (0, 5), (True, 2), (0.0, 2)])
def test_reject_invalid_spans(coordinates):
    with pytest.raises(ValueError):
        validate_span("正文", "正", *coordinates)


def test_transitive_dependency_graph_not_shared_evidence_implied_dependency():
    nodes = [{"id": "e", "kind": "evidence"}, {"id": "j", "kind": "judgment"}, {"id": "c", "kind": "conclusion"}, {"id": "other", "kind": "conclusion"}, {"id": "final", "kind": "conclusion"}]
    edges = [{"from_node_id": "e", "to_node_id": "j", "relation": "supports"}, {"from_node_id": "j", "to_node_id": "c", "relation": "prerequisite"}, {"from_node_id": "e", "to_node_id": "other", "relation": "contradicts"}, {"from_node_id": "c", "to_node_id": "final", "relation": "prerequisite"}]
    validate_graph(nodes, edges)
    assert downstream_closure(edges, ["e"]) == {"e", "j", "c", "other", "final"}
    assert downstream_closure(edges, ["c"]) == {"c", "final"}
    assert downstream_closure(edges, ["other"]) == {"other"}
    with pytest.raises(ValueError, match="cycle"):
        validate_graph(nodes, edges + [{"from_node_id": "final", "to_node_id": "j", "relation": "prerequisite"}])
    with pytest.raises(ValueError, match="endpoint"):
        validate_graph(nodes, [{"from_node_id": "missing", "to_node_id": "c", "relation": "supports"}])
    with pytest.raises(ValueError, match="multiple topics"):
        validate_graph([{"id": "e", "kind": "evidence", "topic_id": "t1"}, {"id": "c", "kind": "conclusion", "topic_id": "t2"}], [])
    with pytest.raises(ValueError, match="prerequisite"):
        validate_graph(nodes, [{"from_node_id": "j", "to_node_id": "c", "relation": "supports"}])


def test_retrieval_chinese_rank_and_no_forced_matches():
    paragraphs = [{"id": "p1", "text": "新华社发布吉隆口岸泥石流灾害通报。"}, {"id": "p2", "text": "今天的足球比赛已经结束。"}, {"id": "p3", "text": "吉隆口岸已经恢复交通。"}]
    results = retrieve("吉隆口岸泥石流灾害有哪些通报？", paragraphs, 2)
    assert results[0] is paragraphs[0]
    assert paragraphs[1] not in results
    assert retrieve("量子纠缠实验", paragraphs) == []
    assert retrieve("量子纠缠实验", []) == []


@pytest.mark.parametrize("nodes,edges", [([{"id": "e", "kind": []}], []), ([{"id": "e", "kind": "evidence", "topic_id": {}}], []), ([{"id": "e", "kind": "evidence"}, {"id": "c", "kind": "conclusion"}], [{"from_node_id": "e", "to_node_id": "c", "relation": []}])])
def test_graph_invalid_types_raise_domain_value_error(nodes, edges):
    with pytest.raises(ValueError):
        validate_graph(nodes, edges)


def test_quality_formula_deduplicates_sources_and_keeps_input_audit():
    supports = [group(grade=0.6, quote="首个引用"), group(grade=0.9, quote="另一个引用")]
    original = deepcopy(supports)
    result = compute_conclusion_confidence(supports, [], ["e", "j", "c"], ["e", "j", "c"])
    assert len(result["inputs"]["support_groups"]) == 1
    assert result["components"]["source_credibility"] == 0.9
    assert result["components"]["evidence_consistency"] == pytest.approx(1 - math.exp(-0.5))
    assert result["score"] == pytest.approx(0.4 * 0.9 + 0.35 * (1 - math.exp(-0.5)) + 0.25)
    assert result["score"] == result["calculation"]["unrounded_score"]
    assert supports == original
    assert result["interpretation"] == "quality_index"


@pytest.mark.parametrize("supports,chain,verified,valid", [([], [], [], True), ([], ["c"], ["c"], True), ([group()], [], [], True), ([group()], ["c"], ["c"], False)])
def test_empty_or_invalid_dependency_gate(supports, chain, verified, valid):
    result = compute_conclusion_confidence(supports, [], chain, verified, valid)
    assert result["score"] == 0
    assert math.isfinite(result["score"])
    assert result["components"]["gate"] == 0


def test_unresolved_competition_reduces_consistency_without_fabricating_resolution():
    result = compute_conclusion_confidence([group()], [group("opposing")], ["e", "c"], ["e", "c"])
    assert result["components"]["evidence_consistency"] == pytest.approx((1 - math.exp(-0.5)) / 2)
    assert len(result["inputs"]["unresolved_conflict_groups"]) == 1


def test_quality_rejects_nonfinite_or_duplicated_independent_evidence():
    for value in [None, True, math.nan, math.inf, -0.1, 1.1]:
        with pytest.raises(ValueError):
            compute_evidence_confidence(value, 1, True)
    with pytest.raises(ValueError, match="subset"):
        compute_conclusion_confidence([group()], [], ["c"], ["e"])
    duplicate = group("source_2")
    duplicate["evidence_refs"] = group()["evidence_refs"]
    with pytest.raises(ValueError, match="two independent"):
        compute_conclusion_confidence([group(), duplicate], [], ["c"], ["c"])


def test_confidence_outputs_validate_against_export_schema_after_caller_metadata():
    schema = json.loads((Path(__file__).resolve().parents[1] / "contracts/competition-export.schema.json").read_text(encoding="utf-8-sig"))
    conclusion = compute_conclusion_confidence([group()], [], ["e", "c"], ["e", "c"])
    conclusion["object_ref"] = {"question_id": "q_001"}
    evidence = compute_evidence_confidence(0.9, 0.5, True)
    evidence["object_ref"] = {"doc_id": "source_1", "para_id": "p1", "quote": "原句"}
    evidence["inputs"].update(source_level="test grade", source_group_id="source_1", support_strength_basis="partial support of explicit assertion")
    Draft202012Validator({"$ref": "#/$defs/ConclusionConfidenceExport", "$defs": schema["$defs"]}).validate(conclusion)
    Draft202012Validator({"$ref": "#/$defs/EvidenceConfidenceExport", "$defs": schema["$defs"]}).validate(evidence)
    assert evidence["score"] == 0.45
    assert compute_evidence_confidence(0.9, 1, False)["score"] == 0
