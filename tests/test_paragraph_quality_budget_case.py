"""Pure, synthetic selection/field contracts; no network, model or answers."""

from copy import deepcopy
import json

import pytest

from app.domain.evidence_field_extraction import ARGUMENT_KEYS, FieldExtractionError
from app.domain.paragraph_quality_budget_case import (
    WIRE_ARGUMENT_MAP, compile_paragraph_quality, paragraph_quality_schema, prepare_paragraph_quality,
)
from test_lc_context_case import raw_case


def prepare(texts=None):
    raw = raw_case(texts or ["材料尚未确认，青桥研究院仍待核验。", "材料已受理，但正式结果仍未公布。"])
    return raw, prepare_paragraph_quality(raw)


def output(prepared, retained=None):
    retained = set(retained if retained is not None else [row["source_window_id"] for row in prepared["catalog"]])
    return {
        "source_catalog_id": prepared["source_catalog_id"],
        "decisions": {row["source_window_id"]: row["source_window_id"] in retained for row in prepared["catalog"]},
        "annotations": [{"source_window_id": row["source_window_id"], "annotation": {
            "fact_des": "材料尚待核验。",
            "argument": {key: ("whole_parent" if key == "content" else []) for key in WIRE_ARGUMENT_MAP},
            "entities": [],
        }} for row in prepared["catalog"] if row["source_window_id"] in retained],
    }


def test_complete_late_qualifier_is_delivered_and_quote_offsets_remain_program_owned():
    text = "材料已开展评估。" + "原始上下文😀" * 60 + "截至9月2日，青桥研究院称该结论尚未官方确认。"
    raw, prepared = prepare([text, "材料仍待核验。"])
    selected = next(row for row in prepared["catalog"] if row["source_bundle"]["quote"] == text)
    alias = selected["source_window_id"]
    native = json.loads(prepared["messages"][1]["content"])
    delivered = next(segment for run in native["source_runs"] for segment in run["segments"]
                     if segment.get("id") == alias)
    assert delivered["text"] == text  # Not a prefix, sentence window or generated summary.
    response = output(prepared, [alias])
    annotation = response["annotations"][0]["annotation"]
    annotation["argument"]["time"] = ["截至9月2日"]
    annotation["argument"]["subject"] = ["青桥研究院"]
    annotation["entities"] = ["青桥研究院"]
    artifact = compile_paragraph_quality(raw, prepared, response)["artifact"]
    evidence = artifact["evidence"][0]
    assert evidence["quote"] == text
    assert evidence["argument"]["时间"] == "截至9月2日"
    for span in [*evidence["field_spans"]["argument"]["时间"], *evidence["field_spans"]["entities"]]:
        assert text[span["start_cp"]:span["end_cp"]] == span["text"]
    assert evidence["source_paragraph_ids"] == [evidence["paragraph_id"]]
    assert evidence["annotation_semantics_verified"] is False
    assert evidence["native_model_delivery_verified"] is False


def test_explicit_retention_above_ten_survives_missing_optional_fields_in_original_order():
    raw, prepared = prepare([f"材料第{i}条记录尚未确认，仍待核验。" for i in range(15)])
    response = output(prepared)
    assert len(prepared["catalog"]) == 15
    # Neither list ordering nor insertion ordering is a ranking / quota signal.
    response["annotations"].reverse()
    response["decisions"] = dict(reversed(list(response["decisions"].items())))
    snapshot = deepcopy(response)
    artifact = compile_paragraph_quality(raw, prepared, response)["artifact"]
    assert response == snapshot
    assert len(artifact["evidence"]) == 15
    assert [row["source_window_id"] for row in artifact["evidence"]] == [row["source_window_id"] for row in prepared["catalog"]]
    assert all(row["argument"]["时间"] == "" and row["entities"] == [] for row in artifact["evidence"])
    assert artifact["selection_audit"]["selection_count_cap_applied"] is False
    assert artifact["selection_audit"]["score_threshold_applied"] is False
    assert artifact["selection_audit"]["forced_counterclaim_quota_applied"] is False


def test_all_false_is_explicit_empty_selection_without_fabricated_fallback():
    raw, prepared = prepare()
    artifact = compile_paragraph_quality(raw, prepared, output(prepared, []))["artifact"]
    assert artifact["evidence"] == []
    assert artifact["field_extraction_audit"]["selected_parent_count"] == 0
    assert len(artifact["selection_audit"]["candidate_decisions"]) == len(prepared["catalog"])
    assert all(row["retained"] is False for row in artifact["selection_audit"]["candidate_decisions"])


@pytest.mark.parametrize("mutation,reason", [
    (lambda out: out["decisions"].pop("P1"), "output_schema"),
    (lambda out: out["decisions"].update(P1=1), "output_schema"),
    (lambda out: out["decisions"].update(P1="true"), "output_schema"),
    (lambda out: out["decisions"].update(P999=True), "output_schema"),
    (lambda out: out["annotations"].pop(), "retained_annotation_exact_coverage"),
    (lambda out: out["annotations"].append(deepcopy(out["annotations"][0])), "duplicate_annotation_source"),
    (lambda out: out["annotations"][0].update(source_window_id="P1..P2"), "output_schema"),
    (lambda out: out["annotations"][0]["annotation"].update(quote="不能生成引用"), "output_schema"),
    (lambda out: out["annotations"][0]["annotation"]["argument"].pop("time"), "output_schema"),
    (lambda out: out["annotations"][0]["annotation"]["argument"].update(content=[]), "output_schema"),
    (lambda out: out["annotations"][0]["annotation"].update(entities=None), "output_schema"),
    (lambda out: out.update(source_catalog_id="unbound"), "output_schema"),
])
def test_model_structure_failures_are_rejected_not_silently_coerced_or_dropped(mutation, reason):
    raw, prepared = prepare()
    response = output(prepared)
    mutation(response)
    with pytest.raises(ValueError, match=reason):
        compile_paragraph_quality(raw, prepared, response)


def test_annotation_for_false_parent_is_not_silently_removed():
    raw, prepared = prepare()
    response = output(prepared)
    response["decisions"][response["annotations"][0]["source_window_id"]] = False
    with pytest.raises(ValueError, match="retained_annotation_exact_coverage"):
        compile_paragraph_quality(raw, prepared, response)


def test_optional_fields_cannot_borrow_from_other_complete_parent_or_title():
    raw, prepared = prepare(["材料尚未确认。", "青桥研究院仍称材料尚待核验。"])
    first = next(row for row in prepared["catalog"] if row["source_bundle"]["quote"] == "材料尚未确认。")
    response = output(prepared, [first["source_window_id"]])
    response["annotations"][0]["annotation"]["entities"] = ["青桥研究院"]
    artifact = compile_paragraph_quality(raw, prepared, response)["artifact"]
    assert artifact["evidence"][0]["entities"] == []
    assert artifact["evidence"][0]["field_status"]["entities"] == "quarantined"


@pytest.mark.parametrize("mutation", [
    lambda prepared: prepared["catalog"][0]["source_bundle"].update(quote="修改后的原文"),
    lambda prepared: prepared["model_input"]["source_runs"][0]["segments"][0].update(text="缩短的原文"),
    lambda prepared: prepared["scope"].update(returned_complete_paragraph_ids=[]),
    lambda prepared: prepared["messages"][0].update(content="替换提示词"),
    lambda prepared: prepared.update(graded_selection=True),
])
def test_modified_preparation_cannot_become_trusted_source_or_policy(mutation):
    raw, prepared = prepare()
    response = output(prepared)
    mutation(prepared)
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_paragraph_quality(raw, prepared, response)


def test_shared_wire_annotation_schema_stays_compact_when_candidates_increase():
    _, small = prepare(["材料尚待核验。"])
    _, large = prepare([f"材料第{i}条尚待核验。" for i in range(30)])
    small_wire, large_wire = (json.dumps(p["wire_tool"], ensure_ascii=False, separators=(",", ":"))
                              for p in (small, large))
    assert large_wire.count('"fact_des"') == small_wire.count('"fact_des"')
    assert '"start_cp"' not in large_wire
    assert '"text"' not in large_wire
    assert len(large_wire) < len(small_wire) + 1600
    assert paragraph_quality_schema(large).model_json_schema() == large["schema_json"]



def test_whole_parent_content_expansion_is_explicit_and_native_response_is_not_rewritten():
    text = "材料相关事实仍待核验。" + "长原始材料😀" * 300
    raw, prepared = prepare([text])
    response = output(prepared)
    original = deepcopy(response)
    result = compile_paragraph_quality(raw, prepared, response)
    artifact = result["artifact"]
    assert response == original
    assert artifact["raw_model_selection"] == original
    assert artifact["evidence"][0]["argument"]["内容"] == text
    assert artifact["evidence"][0]["quote"] == text
    field_audit = artifact["field_extraction_audit"]
    assert field_audit["expanded_fields_are_provider_output"] is False
    expansion = field_audit["compact_expansion_audit"][0]["expansions"][0]
    assert expansion["model_selector"] == "whole_parent"
    assert expansion["expansion_is_literal_model_text"] is False
    assert expansion["expanded_codepoints"] == len(text)
    assert len(json.dumps(response, ensure_ascii=False)) < len(text) / 2


def test_literal_content_arrays_are_exact_program_located_without_coordinate_generation():
    raw, prepared = prepare(["截至9月2日，青桥研究院评估结果尚未确认，仍待核验。"])
    response = output(prepared)
    a = response["annotations"][0]["annotation"]
    a["argument"]["content"] = ["尚未确认", "仍待核验"]
    a["argument"]["subject"] = ["青桥研究院"]
    a["argument"]["time"] = ["截至9月2日"]
    a["entities"] = ["青桥研究院"]
    compiled = compile_paragraph_quality(raw, prepared, response)["artifact"]
    evidence = compiled["evidence"][0]
    assert evidence["argument"]["内容"] == "尚未确认；仍待核验"
    assert evidence["field_spans"]["argument"]["内容"][0]["start_cp"] > 0
    assert compiled["field_extraction_audit"]["compact_expansion_audit"][0]["expansions"] == []


@pytest.mark.parametrize("field,value", [
    ("content", "invalid_selector"),
    ("content", [{"text": "待核验", "start_cp": None}]),
    ("subject", [""]),
    ("time", [{"text": "9月2日", "start_cp": None}]),
])
def test_noncompact_field_forms_are_rejected(field, value):
    raw, prepared = prepare()
    response = output(prepared)
    response["annotations"][0]["annotation"]["argument"][field] = value
    with pytest.raises(ValueError, match="output_schema"):
        compile_paragraph_quality(raw, prepared, response)


def test_literal_failures_and_new_summary_numbers_remain_validation_errors():
    raw, prepared = prepare(["青桥研究院称该材料尚待核验。"])
    response = output(prepared)
    response["annotations"][0]["annotation"]["fact_des"] = "共有999人完成核验。"
    with pytest.raises(FieldExtractionError, match="summary_number_not_in_parent"):
        compile_paragraph_quality(raw, prepared, response)
    response = output(prepared)
    response["annotations"][0]["annotation"]["argument"]["impact"] = ["导致严重后果"]
    artifact = compile_paragraph_quality(raw, prepared, response)["artifact"]
    assert artifact["evidence"][0]["argument"]["影响"] == ""
    assert artifact["evidence"][0]["field_status"]["影响"] == "quarantined"


def test_optional_empty_fields_do_not_receive_guessed_dates_or_entities_from_metadata():
    raw, prepared = prepare(["材料尚待核验，结果未知。"])
    response = output(prepared)
    result = compile_paragraph_quality(raw, prepared, response)["artifact"]["evidence"][0]
    assert result["argument"]["时间"] == ""
    assert result["entities"] == []


def test_v1_contract_and_v3_contract_cannot_be_silently_reinterpreted():
    from app.domain.paragraph_quality_case import prepare_paragraph_quality as prepare_v1
    from app.domain.paragraph_quality_case import compile_paragraph_quality as compile_v1
    raw, prepared = prepare()
    previous = prepare_v1(raw)
    assert previous["version"] != prepared["version"]
    assert previous["source_catalog_id"] != prepared["source_catalog_id"]
    assert [s["text"] for r in previous["model_input"]["source_runs"] for s in r["segments"] if s["kind"]=="paragraph"] == [s["text"] for r in prepared["model_input"]["source_runs"] for s in r["segments"] if s["kind"]=="paragraph"]
    assert previous["scope"]["returned_complete_paragraph_ids"] == prepared["scope"]["returned_complete_paragraph_ids"]
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_paragraph_quality(raw, previous, output(prepared))
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_v1(raw, prepared, output(prepared))


@pytest.mark.parametrize("wire_key,standard_key", list(WIRE_ARGUMENT_MAP.items()))
def test_all_six_english_wire_keys_map_explicitly_to_standard_chinese(wire_key, standard_key):
    raw, prepared = prepare(["材料尚待核验，原文明确说明。"])
    response = output(prepared)
    response["annotations"][0]["annotation"]["argument"][wire_key] = ["尚待核验"]
    evidence = compile_paragraph_quality(raw, prepared, response)["artifact"]["evidence"][0]
    assert set(evidence["argument"]) == set(ARGUMENT_KEYS)
    assert evidence["argument"][standard_key] == "尚待核验"


@pytest.mark.parametrize("wire_key,chinese_key", list(WIRE_ARGUMENT_MAP.items()))
def test_chinese_keys_missing_or_duplicate_english_forms_are_rejected(wire_key, chinese_key):
    raw, prepared = prepare()
    response = output(prepared)
    argument = response["annotations"][0]["annotation"]["argument"]
    argument[chinese_key] = argument.pop(wire_key)
    with pytest.raises(ValueError, match="output_schema"):
        compile_paragraph_quality(raw, prepared, response)
    response = output(prepared)
    response["annotations"][0]["annotation"]["argument"][chinese_key] = []
    with pytest.raises(ValueError, match="output_schema"):
        compile_paragraph_quality(raw, prepared, response)


def test_raw_english_keys_are_preserved_while_compilation_mapping_is_audited():
    raw, prepared = prepare()
    response = output(prepared)
    artifact = compile_paragraph_quality(raw, prepared, response)["artifact"]
    assert set(artifact["raw_model_selection"]["annotations"][0]["annotation"]["argument"]) == set(WIRE_ARGUMENT_MAP)
    assert artifact["field_extraction_audit"]["wire_argument_key_mapping"] == WIRE_ARGUMENT_MAP
    assert artifact["field_extraction_audit"]["wire_key_aliases_or_repairs_applied"] is False


@pytest.mark.parametrize("texts", [
    ["9月2日青桥研究院在北区核验5项材料，暂未公布正式结果。", "9月3日南桥机构称复核正在进行。"],
    ["7月4日星河协会声明在南区完成3项审查，后续仍需确认。", "北山中心说明程序尚未结束。"],
    ["原文把设备写成甲－乙格式，材料仍待核验。", "乙机构只提出竞争说法，结果尚未确认。"],
])
def test_bad_optional_metadata_does_not_destroy_selected_evidence_across_synthetic_questions(texts):
    raw, prepared = prepare(texts)
    response = output(prepared)
    original_ids = [row["source_window_id"] for row in response["annotations"]]
    a = response["annotations"][0]["annotation"]
    a["entities"] = ["不在该段的实体"]
    for key in ("time", "subject", "location", "impact", "numeric_scope"):
        a["argument"][key] = ["原文没有的可选字段"]
    original = deepcopy(response)
    artifact = compile_paragraph_quality(raw, prepared, response)["artifact"]
    assert artifact["raw_model_selection"] == original == response
    assert [row["source_window_id"] for row in artifact["evidence"]] == original_ids
    assert all(row["quote"] == next(s["source_bundle"]["quote"] for s in prepared["catalog"]
                                   if s["source_window_id"] == row["source_window_id"]) for row in artifact["evidence"])
    first = artifact["evidence"][0]
    assert first["entities"] == []
    assert first["argument"]["内容"] == first["quote"]
    assert all(first["argument"][key] == "" for key in ARGUMENT_KEYS if key != "内容")
    assert all(first["field_status"][key] == "quarantined" for key in (*[k for k in ARGUMENT_KEYS if k != "内容"], "entities"))
    summary = artifact["optional_field_quality_summary"]
    assert summary["quarantined_value_count"] == 6
    assert summary["affected_annotation_count"] == 1
    assert summary["has_quarantined_optional_fields"] is True
    audit = artifact["field_extraction_audit"]["optional_field_quarantine"]
    assert audit["raw_model_output_modified"] is False
    assert audit["required_fields_quarantined"] is False
    assert all(set(row) == {"source_window_id", "annotation_index", "field", "path", "code", "count"}
               for row in audit["issues"])
    assert {row["code"] for row in audit["issues"]} == {"optional_literal_not_in_parent"}
    assert "原文没有的可选字段" not in json.dumps(audit, ensure_ascii=False)


def test_valid_optional_values_survive_with_exact_deduplication_and_no_fuzzy_repair():
    text = "截至9月2日，青桥研究院在北区核验5项材料。"
    raw, prepared = prepare([text])
    response = output(prepared)
    a = response["annotations"][0]["annotation"]
    a["entities"] = ["青桥研究院", "青桥研究院", "青桥研究院所"]
    a["argument"]["subject"] = ["青桥研究院", "不在原段", "青桥研究院"]
    original = deepcopy(response)
    artifact = compile_paragraph_quality(raw, prepared, response)["artifact"]
    evidence = artifact["evidence"][0]
    assert evidence["entities"] == ["青桥研究院"]
    assert evidence["argument"]["主体"] == "青桥研究院"
    assert evidence["field_status"]["主体"] == "quarantined"
    assert evidence["field_status"]["entities"] == "quarantined"
    assert artifact["optional_field_quality_summary"]["quarantined_value_count"] == 2
    assert artifact["optional_field_quality_summary"]["deduplicated_value_count"] == 2
    assert artifact["raw_model_selection"] == original
    assert len(artifact["evidence"]) == 1


@pytest.mark.parametrize("content", [["原文没有的主要内容"], ["已核验；结果确认"]])
def test_required_content_never_uses_optional_quarantine(content):
    raw, prepared = prepare(["材料尚待核验，结果尚未确认。"])
    response = output(prepared)
    response["annotations"][0]["annotation"]["argument"]["content"] = content
    with pytest.raises(FieldExtractionError, match="literal_not_in_parent"):
        compile_paragraph_quality(raw, prepared, response)


@pytest.mark.parametrize("value", [None, [None], [""], ["   "]])
def test_optional_bad_structure_is_still_rejected_before_literal_quarantine(value):
    raw, prepared = prepare()
    response = output(prepared)
    response["annotations"][0]["annotation"]["entities"] = value
    with pytest.raises(ValueError, match="output_schema"):
        compile_paragraph_quality(raw, prepared, response)


def test_repeat_literal_source_occurrences_remain_audited_and_not_semantic_disambiguation():
    raw, prepared = prepare(["青桥研究院说明材料尚待核验，青桥研究院将在稍后复核。"])
    response = output(prepared)
    response["annotations"][0]["annotation"]["entities"] = ["青桥研究院"]
    artifact = compile_paragraph_quality(raw, prepared, response)["artifact"]
    span = artifact["evidence"][0]["field_spans"]["entities"][0]
    assert len(span["all_exact_occurrences"]) == 2
    assert span["semantic_disambiguation_verified"] is False
    assert artifact["optional_field_quality_summary"]["quarantined_value_count"] == 0


def test_token_pack_recall_pool_and_native_full_read_scope_are_separate():
    raw, prepared = prepare([f"材料第{i}条记录仍待核验，尚未确认。" for i in range(200)])
    receipt = prepared["retrieval"]["parent_token_pack_receipt"]
    assert receipt["search_page_count"] == 0
    assert receipt["recall_pool_count"] == 180
    assert receipt["provided_parent_count"] <= 60
    assert receipt["provided_payload_proxy_tokens"] <= 8000
    assert len(prepared["scope"]["returned_complete_paragraph_ids"]) == receipt["provided_parent_count"]
    assert len(prepared["scope"]["recall_pool_paragraph_ids"]) == 180
    native = json.loads(prepared["messages"][1]["content"])
    assert "parent_token_pack_receipt" not in native
    assert "unprovided_reason_records" not in native
    assert "search_pages" not in native
    delivered = [segment["text"] for run in native["source_runs"] for segment in run["segments"]]
    original = {p["text"] for p in raw["paragraphs"]}
    assert set(delivered) <= original
    assert all(len(run["segments"]) == 1 and run["segments"][0]["kind"] == "paragraph" for run in native["source_runs"])
    assert len(delivered) == receipt["provided_parent_count"]
    unknown=prepared["scope"]["unprovided_reason_records"]
    assert len(unknown) == 200-len(delivered)
    assert all(row["context_state"] == "unknown_not_irrelevant" for row in unknown)


def test_oversized_parent_is_skipped_whole_never_truncated_or_inferred_irrelevant():
    huge = "材料尚待核验。" + "完整巨大原段唯一标识😀" * 8000
    raw, prepared = prepare([huge, "材料尚待核验。"])
    receipt = prepared["retrieval"]["parent_token_pack_receipt"]
    pid = next(row["id"] for row in raw["paragraphs"] if row["text"] == huge)
    decision = next(row for row in receipt["candidate_records"] if row["paragraph_id"] == pid)
    assert decision["reason"] == "pooled_over_budget"
    assert decision["provided"] is False
    assert "完整巨大原段唯一标识" not in prepared["messages"][1]["content"]
    assert receipt["source_text_shortened"] is False
    assert next(row for row in receipt["unprovided_reason_records"] if row["paragraph_id"] == pid)["context_state"] == "unknown_not_irrelevant"


def test_pack_policy_is_deterministic_and_fixed_not_model_or_reference_tuned():
    raw, prepared = prepare([f"材料第{i}条仍待核验。" for i in range(90)])
    assert prepare_paragraph_quality(raw) == prepared
    assert prepared["parent_token_pack_policy"]["parent_token_pack"]["payload_proxy_token_budget"] == 8000
    assert prepared["source_binding"]["parent_token_pack_receipt_sha256"]
    response = output(prepared)
    changed = deepcopy(prepared)
    changed["retrieval"]["parent_token_pack_receipt"]["provided_payload_proxy_tokens"] = 1
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_paragraph_quality(raw, changed, response)


def test_pack_reference_leakage_and_local_rank_mixing_are_rejected():
    raw, _ = prepare()
    contaminated = deepcopy(raw)
    contaminated["paragraphs"][0]["fact_des"] = "参考注释不能进入运行输入"
    with pytest.raises(ValueError):
        prepare_paragraph_quality(contaminated)
    with pytest.raises(ValueError, match="exclusive"):
        prepare_paragraph_quality(raw, rank_plan={}, rank_plan_sha256="f"*64)


def test_base_default_path_is_identical_and_nonboolean_pack_flag_is_rejected():
    from app.domain.lc_bundle_case import prepare_lc_case
    raw, _ = prepare()
    assert prepare_lc_case(raw) == prepare_lc_case(raw, parent_token_pack=False)
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="parent_token_pack_boolean_required"):
            prepare_lc_case(raw, parent_token_pack=invalid)
