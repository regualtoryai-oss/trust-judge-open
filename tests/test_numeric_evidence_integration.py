"""Synthetic-only integration of numeric normalization with strict metadata."""
from copy import deepcopy

import pytest

from app.agents.strict_evidence import (
    StrictEvidenceError, annotation_time_projection, build_proposal_artifact,
    canonical_annotation_wire, validate_annotation,
)


def fixture():
    text = "\n😀公开记录：有两名人员幸存，具体身份尚未核实。\r\n"
    raw = {"question": {"question_id": "q_synthetic", "question": "公开记录的幸存人数是否明确？"}, "catalog": [
        {"paragraph_ref": "P1", "doc_id": "d1", "para_id": "p1", "title": "记录", "text": text,
         "paragraph_complete": True, "parent_complete": True}]}
    selection = {"hypotheses": [{"id": "H1", "claim": "公开记录给出明确幸存人数。"}], "decisions": [
        {"p": "P1", "relevance": 3, "stance": "supports_hypothesis", "hypothesis_id": "H1"}]}
    annotation = {"p": "P1", "fact_des": "公开记录有两名人员幸存，具体身份尚未核实。",
                  "argument": {"时间": "", "主体": "人员", "地点": "", "内容": "公开记录有两名人员幸存，具体身份尚未核实。", "影响": "", "数值口径": "2人"},
                  "entities": ["人员"]}
    return raw, selection, annotation


def test_general_chinese_count_annotation_passes_without_changing_metadata_or_literal_source():
    raw, selection, annotation = fixture(); original = deepcopy((raw, selection, annotation))
    assert validate_annotation(raw, selection, {"annotations": [annotation]}).model_dump() == {"annotations": [annotation]}
    proposed = {"annotations": [annotation], "conclusion": "公开记录有明确人数，具体身份仍未核实。", "verdict_label": "partial_support",
                "supporting": ["P1"], "conflicting": [], "basis": "依原文限定保留未知。"}
    artifact = build_proposal_artifact(raw, selection, proposed)
    assert artifact["evidence"][0]["quote"] == raw["catalog"][0]["text"]
    assert artifact["evidence"][0]["doc_id"] == "d1" and artifact["evidence"][0]["para_id"] == "p1"
    assert artifact["evidence"][0]["parent_end_cp"] == len(raw["catalog"][0]["text"])
    assert artifact["evidence"][0]["argument"]["数值口径"] == "2人"
    assert artifact["quality"]["publishable"] is False and artifact["quality"]["semantic_verified"] is False
    assert (raw, selection, annotation) == original


@pytest.mark.parametrize("field", ["fact_des", "argument", "entities"])
def test_novel_count_in_any_metadata_field_still_rejects_the_entire_proposal(field):
    raw, selection, annotation = fixture()
    if field == "fact_des": annotation["fact_des"] = "公开记录有3人幸存。"
    elif field == "argument": annotation["argument"]["数值口径"] = "3人"
    else: annotation["entities"] = ["3人"]
    with pytest.raises(StrictEvidenceError, match="annotation_number_not_in_source"):
        validate_annotation(raw, selection, {"annotations": [annotation]})


def test_time_projection_remains_independent_of_numeric_normalization_and_never_clears_other_fields():
    raw, selection, annotation = fixture()
    wire = {"annotations": {"P1": {key: value for key, value in annotation.items() if key != "p"}}}
    wire["annotations"]["P1"]["argument"]["时间"] = "明年"; original = deepcopy(wire)
    projected, changes = annotation_time_projection(raw, wire)
    assert changes and projected["annotations"]["P1"]["argument"]["时间"] == ""
    assert canonical_annotation_wire(raw, selection, wire).annotations[0].argument.数值口径 == "2人"
    assert wire == original
    wire["annotations"]["P1"]["argument"]["数值口径"] = "3人"
    with pytest.raises(StrictEvidenceError, match="annotation_number_not_in_source"):
        canonical_annotation_wire(raw, selection, wire)
