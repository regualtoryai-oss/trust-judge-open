"""Synthetic paragraph field contracts; no models, real samples or references."""

from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from app.domain.evidence_field_extraction import (
    ARGUMENT_KEYS, FORMAT, FieldExtractionError, annotation_type, compile_field_extraction,
    compile_selected_annotations, prepare_field_extraction_request,
)


def source(text="据观察报报道，研究所称截至9月2日，青桥镇有12人失联，其中3人为外籍人员；这是初步统计，尚未最终确认。", **updates):
    return {"source_id": "evidence-a", "doc_id": "doc-a", "para_id": "p-a", "text": text,
            "parent_text": text, "paragraph_complete": True, "title": "报道标题", **updates}


def span(text, start_cp=None):
    return {"text": text, "start_cp": start_cp}


def response(question, sources):
    _, messages = prepare_field_extraction_request(question, sources)
    binding = json.loads(messages[1].content)["source_catalog_id"]
    return {"source_catalog_id": binding, "annotations": {"E" + str(index + 1): {
        "fact_des": row["text"], "argument": {key: ([span(row["text"])] if key == "内容" else [])
                                              for key in ARGUMENT_KEYS}, "entities": []}
        for index, row in enumerate(sources)}}


def test_one_batch_requires_every_complete_parent_and_keeps_output_paragraph_grain():
    question = "人员失联统计是否已最终确认？"
    sources = [source(), source("另一部门表示暂无完整名单。", source_id="evidence-b", doc_id="doc-b", para_id="p-b")]
    snapshot = deepcopy(sources)
    schema, messages = prepare_field_extraction_request(question, sources)
    payload = json.loads(messages[1].content)
    assert sources == snapshot
    assert [row["id"] for row in payload["paragraphs"]] == ["E1", "E2"]
    assert payload["paragraphs"][0]["text"] == sources[0]["text"]
    assert "source_id" not in payload["paragraphs"][0]
    assert "不拆为多条原子证据" in messages[0].content
    assert "发布媒体不是事件主体" in messages[0].content
    assert "不能把报道或发布时间当作事件时间" in messages[0].content
    output = response(question, sources)
    parsed = schema.model_validate(output)
    result = compile_field_extraction(question, sources, parsed)
    assert result["format"] == FORMAT and result["all_selected_parents_annotated"] is True
    assert len(result["annotations"]) == len(sources)
    assert result["annotations"][0]["quote"] == sources[0]["text"]
    assert result["annotations"][0]["source_id"] == "evidence-a"
    assert result["semantic_verified"] is False and result["publishable"] is False
    output["annotations"].pop("E2")
    with pytest.raises(FieldExtractionError, match="output_schema"):
        compile_field_extraction(question, sources, output)


def test_literal_fields_preserve_attribution_conditions_and_count_scope_without_metadata_inference():
    question, sources = "该统计是否确定？", [source()]
    output = response(question, sources)
    annotation = output["annotations"]["E1"]
    annotation["fact_des"] = "研究所称截至9月2日青桥镇有12人失联，其中3人为外籍人员，统计初步且尚未最终确认。"
    annotation["argument"] = {
        "时间": [span("截至9月2日")], "主体": [span("研究所")], "地点": [span("青桥镇")],
        "内容": [span("研究所称截至9月2日，青桥镇有12人失联，其中3人为外籍人员；这是初步统计，尚未最终确认。")],
        "影响": [],
        "数值口径": [span("其中3人为外籍人员"), span("截至9月2日"), span("12人失联"), span("初步统计，尚未最终确认")],
    }
    annotation["entities"] = [span("研究所"), span("青桥镇")]
    result = compile_field_extraction(question, sources, output)["annotations"][0]
    assert result["argument"]["主体"] == "研究所"
    assert result["argument"]["时间"] == "截至9月2日"
    assert result["argument"]["影响"] == ""
    assert result["argument"]["数值口径"] == "截至9月2日；12人失联；其中3人为外籍人员；初步统计，尚未最终确认"
    assert result["entities"] == ["研究所", "青桥镇"]
    assert result["field_status"]["影响"] == "not_supplied_by_model"
    assert result["field_status"]["时间"] == "literal_grounded_semantics_unverified"
    for field_spans in [*result["field_spans"]["argument"].values(), result["field_spans"]["entities"]]:
        for selected in field_spans:
            assert sources[0]["text"][selected["start_cp"]:selected["end_cp"]] == selected["text"]


@pytest.mark.parametrize("mutate", [
    lambda rows: rows[0].update(text="只给了较短片段"),
    lambda rows: rows[0].update(paragraph_complete=False),
    lambda rows: rows[0].update(paragraph_complete=1),
    lambda rows: rows[0].pop("parent_text"),
    lambda rows: rows[0].update(reference_answer="禁止运行输入"),
    lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows.append(source("另一个段落", source_id="evidence-b")),
    lambda rows: rows.append(source("另一个段落", doc_id="doc-b", para_id="p-b")),
])
def test_partial_sources_reference_fields_and_repeated_identities_are_rejected(mutate):
    rows = [source()]
    mutate(rows)
    with pytest.raises(FieldExtractionError):
        prepare_field_extraction_request("完整原题", rows)


@pytest.mark.parametrize("mutate", [
    lambda out: out.update(source_catalog_id="another-binding"),
    lambda out: out["annotations"].update(E2=deepcopy(out["annotations"]["E1"])),
    lambda out: out["annotations"]["E1"].update(quote="模型不能生成引用"),
    lambda out: out["annotations"]["E1"]["argument"].pop("主体"),
    lambda out: out["annotations"]["E1"]["argument"].update(内容=[]),
    lambda out: out["annotations"]["E1"]["argument"].update(时间="今天"),
    lambda out: out["annotations"]["E1"]["argument"].update(时间=[{"text": "9月2日"}]),
    lambda out: out["annotations"]["E1"]["argument"].update(时间=[span("9月2日", True)]),
    lambda out: out["annotations"]["E1"]["argument"].update(时间=[span("9月2日", 3.0)]),
])
def test_closed_schema_rejects_missing_fields_free_quotes_and_coercion(mutate):
    question, sources = "完整原题", [source()]
    output = response(question, sources)
    mutate(output)
    with pytest.raises(FieldExtractionError, match="output_schema"):
        compile_field_extraction(question, sources, output)


def test_unsupported_literal_from_other_source_or_generated_date_cannot_fill_fields():
    question, sources = "完整原题", [source(), source("4月8日发布。", source_id="evidence-b", doc_id="doc-b", para_id="p-b")]
    output = response(question, sources)
    output["annotations"]["E1"]["argument"]["时间"] = [span("4月8日")]
    with pytest.raises(FieldExtractionError, match="literal_not_in_parent"):
        compile_field_extraction(question, sources, output)
    output["annotations"]["E1"]["argument"]["时间"] = [span("2026年9月2日")]
    with pytest.raises(FieldExtractionError, match="literal_not_in_parent"):
        compile_field_extraction(question, sources, output)


def test_source_missing_time_stays_empty_instead_of_current_or_publication_date():
    question, sources = "公开说法是否确定？", [source("研究所表示该评估尚未确认。", title="2026年10月21日发布")]
    result = compile_field_extraction(question, sources, response(question, sources))["annotations"][0]
    assert result["argument"]["时间"] == ""
    assert result["field_spans"]["argument"]["时间"] == []


def test_repeated_literal_values_use_first_exact_occurrence_with_full_audit_without_fuzzy_repairs():
    question, sources = "完整原题", [source("😀甲部称待核实。乙部也称待核实。")]
    output = response(question, sources)
    output["annotations"]["E1"]["argument"]["影响"] = [span("待核实")]
    text = sources[0]["text"]
    first = text.index("待核实")
    start = text.rindex("待核实")
    result = compile_field_extraction(question, sources, output)["annotations"][0]
    located = result["field_spans"]["argument"]["影响"][0]
    assert located["start_cp"] == first
    assert located["all_exact_occurrences"] == [
        {"start_cp": first, "end_cp": first + 3}, {"start_cp": start, "end_cp": start + 3}]
    assert located["location_policy"] == "first_exact_occurrence_for_literal_value_not_semantic_disambiguation"
    assert located["semantic_disambiguation_verified"] is False
    output["annotations"]["E1"]["argument"]["影响"] = [span("待核实", start)]
    result = compile_field_extraction(question, sources, output)["annotations"][0]
    located = result["field_spans"]["argument"]["影响"][0]
    assert located["start_cp"] == start and located["location_policy"] == "model_hint_matches_exact_occurrence"
    output["annotations"]["E1"]["argument"]["影响"] = [span("待 核实", start)]
    with pytest.raises(FieldExtractionError, match="literal_not_in_parent"):
        compile_field_extraction(question, sources, output)


@pytest.mark.parametrize("hint", [None, -1, 999999, 1])
def test_ten_repeated_organization_mentions_never_require_model_codepoint_math(hint):
    question, sources = "完整原题", [source("青桥研究院表示尚未确认。" * 10)]
    output = response(question, sources)
    output["annotations"]["E1"]["entities"] = [span("青桥研究院", hint)]
    result = compile_field_extraction(question, sources, output)["annotations"][0]
    selected = result["field_spans"]["entities"][0]
    assert result["entities"] == ["青桥研究院"]
    assert selected["start_cp"] == 0 and len(selected["all_exact_occurrences"]) == 10
    assert selected["text"] == "青桥研究院" and selected["semantic_disambiguation_verified"] is False


def test_unique_literal_is_canonicalized_from_text_instead_of_trusting_model_hint():
    question, sources = "完整原题", [source("😀原文称甲部尚未确认。")]
    output = response(question, sources)
    output["annotations"]["E1"]["entities"] = [span("甲部", 999)]
    result = compile_field_extraction(question, sources, output)["annotations"][0]
    assert result["field_spans"]["entities"][0]["start_cp"] == sources[0]["text"].index("甲部")


def test_same_field_span_and_same_entity_text_cannot_be_counted_twice():
    question, sources = "完整原题", [source("甲部称甲部尚未确认。")]
    output = response(question, sources)
    output["annotations"]["E1"]["entities"] = [span("甲部", 0), span("甲部", 0)]
    with pytest.raises(FieldExtractionError, match="duplicate_field_span"):
        compile_field_extraction(question, sources, output)
    output["annotations"]["E1"]["entities"] = [span("甲部", 0), span("甲部", 3)]
    with pytest.raises(FieldExtractionError, match="duplicate_entity_text"):
        compile_field_extraction(question, sources, output)


def test_identical_mirrors_keep_two_original_source_ids_and_annotations():
    question = "完整原题"
    sources = [source(), source(source_id="evidence-b", doc_id="doc-b", para_id="p-b")]
    result = compile_field_extraction(question, sources, response(question, sources))
    assert len(result["annotations"]) == 2
    assert [row["source_id"] for row in result["annotations"]] == ["evidence-a", "evidence-b"]


def test_changed_input_cannot_reuse_prior_annotation_binding():
    question, sources = "完整原题", [source()]
    output = response(question, sources)
    sources[0]["text"] += "新增限定。"
    sources[0]["parent_text"] = sources[0]["text"]
    with pytest.raises(FieldExtractionError, match="output_schema"):
        compile_field_extraction(question, sources, output)


def test_summary_cannot_introduce_new_numeric_claim_or_whitespace_only_annotation():
    question, sources = "完整原题", [source()]
    output = response(question, sources)
    output["annotations"]["E1"]["fact_des"] = "已确认120人失联。"
    with pytest.raises(FieldExtractionError, match="summary_number_not_in_parent"):
        compile_field_extraction(question, sources, output)
    output["annotations"]["E1"]["fact_des"] = "  "
    with pytest.raises(FieldExtractionError, match="empty_fact_description"):
        compile_field_extraction(question, sources, output)


def test_literal_success_does_not_claim_semantic_completeness_or_correct_roles():
    question, sources = "完整原题", [source()]
    output = response(question, sources)
    # Deliberately wrong role but copied literally: lexical validation alone
    # cannot establish that the reporter is the event's actual subject.
    output["annotations"]["E1"]["argument"]["主体"] = [span("观察报")]
    result = compile_field_extraction(question, sources, output)
    assert result["annotations"][0]["semantic_verified"] is False
    assert result["policy"]["literal_grounding_is_semantic_correctness"] is False


def test_empty_selection_has_empty_closed_map_and_requires_no_model_fabrication():
    question = "完整原题"
    schema, messages = prepare_field_extraction_request(question, [])
    output = response(question, [])
    assert schema.model_validate(output).annotations.model_dump() == {}
    assert json.loads(messages[1].content)["paragraphs"] == []
    assert compile_field_extraction(question, [], output)["annotations"] == []


def test_argument_literal_whitespace_does_not_pass_as_unknown_or_content():
    question, sources = "完整原题", [source("原文中有 空格。")]
    output = response(question, sources)
    output["annotations"]["E1"]["argument"]["内容"] = [span(" ")]
    with pytest.raises(FieldExtractionError, match="empty_literal_span"):
        compile_field_extraction(question, sources, output)


def test_shared_front_selection_reuses_annotation_type_and_compiles_only_retained_parents():
    question = "完整原题"
    sources = [source(), source("尚无定论。", source_id="evidence-b", doc_id="doc-b", para_id="p-b")]
    batch = response(question, sources)
    annotation = annotation_type().model_validate(batch["annotations"]["E1"])
    result = compile_selected_annotations(question, sources, ["evidence-a"], {
        "evidence-a": annotation, "evidence-b": None})
    assert result["input_parent_count"] == 2 and result["selected_parent_count"] == 1
    assert result["annotations"][0]["source_id"] == "evidence-a"
    assert result["annotations"][0]["entities"] == []  # Unknown entities do not delete retained evidence.
    assert result["annotations"][0]["argument"]["时间"] == ""
    assert result["synthesized_batch_envelope_is_provider_output"] is False
    omitted_null = compile_selected_annotations(question, sources, ["evidence-a"], {"evidence-a": annotation})
    assert result == omitted_null


@pytest.mark.parametrize("retained, annotations, message", [
    (["missing"], {}, "retained_source_identity"),
    (["evidence-a", "evidence-a"], {}, "retained_source_identity"),
    (["evidence-a"], {}, "retained_annotation_required"),
    (["evidence-a"], {"evidence-a": None}, "retained_annotation_required"),
    ([], {"missing": None}, "annotation_source_identity"),
    ([], {"evidence-a": {}}, "nonretained_annotation_must_be_null"),
])
def test_shared_front_missing_annotation_unknown_ids_or_silent_drops_are_rejected(retained, annotations, message):
    with pytest.raises(FieldExtractionError, match=message):
        compile_selected_annotations("完整原题", [source()], retained, annotations)


def test_shared_front_all_rejected_is_valid_and_does_not_fabricate_annotation():
    result = compile_selected_annotations("完整原题", [source()], [], {"evidence-a": None})
    assert result["annotations"] == []
    assert result["selected_parent_count"] == 0 and result["input_parent_count"] == 1


def test_shared_front_compilation_uses_original_order_without_mutating_decisions_or_annotations():
    question = "完整原题"
    sources = [source(), source("尚无定论。", source_id="evidence-b", doc_id="doc-b", para_id="p-b")]
    batch = response(question, sources)
    values = {"evidence-a": batch["annotations"]["E1"], "evidence-b": batch["annotations"]["E2"]}
    original = deepcopy(values)
    result = compile_selected_annotations(question, sources, ["evidence-b", "evidence-a"], values)
    assert [row["source_id"] for row in result["annotations"]] == ["evidence-a", "evidence-b"]
    assert values == original
