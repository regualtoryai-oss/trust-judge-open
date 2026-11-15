"""Pure, synthetic selection/field contracts; no network, model or answers."""

from copy import deepcopy
import json

import pytest

from app.domain.evidence_field_extraction import ARGUMENT_KEYS, FieldExtractionError
from app.domain.paragraph_quality_case import (
    compile_paragraph_quality, paragraph_quality_schema, prepare_paragraph_quality,
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
            "fact_des": row["source_bundle"]["quote"],
            "argument": {key: ([{"text": row["source_bundle"]["quote"], "start_cp": None}]
                               if key == "内容" else []) for key in ARGUMENT_KEYS},
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
    annotation["argument"]["时间"] = [{"text": "截至9月2日", "start_cp": None}]
    annotation["argument"]["主体"] = [{"text": "青桥研究院", "start_cp": None}]
    annotation["entities"] = [{"text": "青桥研究院", "start_cp": None}]
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
    (lambda out: out["annotations"][0]["annotation"]["argument"].pop("时间"), "output_schema"),
    (lambda out: out["annotations"][0]["annotation"]["argument"].update(内容=[]), "output_schema"),
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


def test_fields_cannot_borrow_from_other_complete_parent_or_title():
    raw, prepared = prepare(["材料尚未确认。", "青桥研究院仍称材料尚待核验。"])
    first = next(row for row in prepared["catalog"] if row["source_bundle"]["quote"] == "材料尚未确认。")
    response = output(prepared, [first["source_window_id"]])
    response["annotations"][0]["annotation"]["entities"] = [{"text": "青桥研究院", "start_cp": None}]
    with pytest.raises(FieldExtractionError, match="literal_not_in_parent"):
        compile_paragraph_quality(raw, prepared, response)


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
    assert large_wire.count('"start_cp"') == small_wire.count('"start_cp"')
    assert len(large_wire) < len(small_wire) + 1600
    assert paragraph_quality_schema(large).model_json_schema() == large["schema_json"]
