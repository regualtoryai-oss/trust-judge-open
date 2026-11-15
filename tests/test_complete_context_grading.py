"""Synthetic closed-map grading; no model, ONNX, gold or old model outputs."""
from copy import deepcopy
import json

import pytest

from app.agents.strict_transport import strict_tool
from app.domain.complete_context_grading import (
    FORMAT, FUNCTION_NAME, ORDER_POLICY, POLICY, compile_complete_business, complete_schema,
    normalize_complete, prepare_complete, restore_complete,
)
from app.domain.lc_context_case import compile_context_business
from app.domain.context_candidates import build_context_plan
from app.domain.lc_context_case import prepare_context_case
from app.domain.local_parent_rank import fingerprint
from test_lc_context_case import prepare, raw_case, same_parent_contexts


def grades(prepared, score=0, use_type="unrelated"):
    return {"source_catalog_id": prepared["source_catalog_id"], "grades": {
        row["source_window_id"]: {"relevance_score": score, "use_type": use_type}
        for row in reversed(prepared["catalog"])}}


def test_closed_schema_requires_every_actual_context_key_without_changing_source_book():
    _, base, _ = prepare(raw_case([f"完整来源{i}尚待确认。" for i in range(32)]))
    original = deepcopy(base)
    wrapper = prepare_complete(base)
    assert base == original
    assert wrapper["format"] == FORMAT and wrapper["function_name"] == FUNCTION_NAME
    assert wrapper["complete_grading"] is True and wrapper["complete_grading_policy"] == POLICY
    assert "base_prepared" not in wrapper
    assert wrapper["catalog"] == base["catalog"] and wrapper["catalog"] is not base["catalog"]
    for field in ("model_input", "scope", "source_binding", "source_geometry", "source_bundle_protocol",
                  "graded_selection_policy", "selection_only", "graded_selection", "source_catalog_id"):
        assert wrapper[field] == base[field]
    assert wrapper["messages"][1] == base["messages"][1]
    assert "ranked_candidates" not in wrapper["messages"][0]["content"]
    assert "全部0分也合法" in wrapper["messages"][0]["content"]
    schema = complete_schema(wrapper)
    assert complete_schema(base).model_json_schema() == schema.model_json_schema() == wrapper["schema_json"]
    assert strict_tool(schema) == wrapper["wire_tool"]
    parameters = wrapper["wire_tool"]["function"]["parameters"]
    assert parameters["required"] == ["source_catalog_id", "grades"]
    assert parameters["additionalProperties"] is False
    body = parameters["properties"]["grades"]
    ids = [row["source_window_id"] for row in base["catalog"]]
    assert body["required"] == ids and list(body["properties"]) == ids
    assert body["additionalProperties"] is False
    for value in body["properties"].values():
        assert value["required"] == ["relevance_score", "use_type"]
        assert value["additionalProperties"] is False
        assert set(value["properties"]) == {"relevance_score", "use_type"}
    partial_nine = grades(base)
    partial_nine["grades"] = dict(list(partial_nine["grades"].items())[:9])
    with pytest.raises(ValueError, match="output_schema"):
        normalize_complete(wrapper, partial_nine)
    wrapper["model_input"]["source_contexts"][0]["quote"] += "篡改"
    with pytest.raises(ValueError, match="native_source_quote"):
        complete_schema(wrapper)


def test_normalization_is_complete_reversible_and_uses_catalog_order_not_json_key_order():
    _, base, _ = prepare(raw_case(["第一来源。", "第二来源。", "第三来源。"] ))
    wrapper = prepare_complete(base)
    original = grades(base)
    ids = [row["source_window_id"] for row in base["catalog"]]
    original["grades"][ids[1]] = {"relevance_score": 2, "use_type": "necessary_condition"}
    assert list(original["grades"]) == list(reversed(ids))
    normalized = normalize_complete(base, original)
    assert normalized == normalize_complete(wrapper, original)
    assert [row["source_window_id"] for row in normalized["ranked_candidates"]] == ids
    assert len(normalized["ranked_candidates"]) == len(base["catalog"])
    assert restore_complete(base, normalized) == original
    assert restore_complete(wrapper, normalized) == original
    assert normalized["ranked_candidates"][1]["use_type"] == "necessary_condition"


@pytest.mark.parametrize("mutation", [
    lambda out: out["grades"].pop(next(iter(out["grades"]))),
    lambda out: out["grades"].update(C999={"relevance_score": 0, "use_type": "unrelated"}),
    lambda out: out.update(source_catalog_id="other-catalog"),
    lambda out: out.update(unexpected="extra"),
    lambda out: out["grades"].update({next(iter(out["grades"])): {"relevance_score": True, "use_type": "direct_answer"}}),
    lambda out: out["grades"].update({next(iter(out["grades"])): {"relevance_score": 2.0, "use_type": "direct_answer"}}),
    lambda out: out["grades"].update({next(iter(out["grades"])): {"relevance_score": "2", "use_type": "direct_answer"}}),
    lambda out: out["grades"].update({next(iter(out["grades"])): {"relevance_score": 4, "use_type": "direct_answer"}}),
    lambda out: out["grades"].update({next(iter(out["grades"])): {"relevance_score": 0, "use_type": "support"}}),
    lambda out: out["grades"].update({next(iter(out["grades"])): {"relevance_score": 0}}),
    lambda out: out["grades"].update({next(iter(out["grades"])): {"relevance_score": 0, "use_type": "unrelated", "quote": "free prose"}}),
    lambda out: out.update(grades=[]),
    lambda out: out.update(grades=None),
])
def test_missing_extra_wrong_binding_and_coerced_or_untyped_values_fail_closed(mutation):
    _, base, _ = prepare(raw_case(["来源甲。", "来源乙。"] ))
    output = grades(base)
    mutation(output)
    with pytest.raises(ValueError, match="complete_context_grading_output_schema"):
        normalize_complete(base, output)


def test_old_partial_list_response_cannot_be_treated_as_complete_grading():
    _, base, _ = prepare(raw_case(["来源甲。", "来源乙。"] ))
    old_output = {"source_catalog_id": base["source_catalog_id"], "ranked_candidates": [
        {"source_window_id": base["catalog"][0]["source_window_id"], "relevance_score": 3, "use_type": "direct_answer"}]}
    with pytest.raises(ValueError, match="output_schema"):
        normalize_complete(base, old_output)
    with pytest.raises(ValueError, match="output_schema"):
        normalize_complete(prepare_complete(base), old_output)


def test_empty_catalog_requires_only_empty_closed_map_and_zero_grades_are_not_defaulted():
    raw, base, kwargs = prepare(raw_case([""]))
    wrapper = prepare_complete(base)
    assert wrapper["source_ids"] == []
    assert wrapper["wire_tool"]["function"]["parameters"]["properties"]["grades"]["properties"] == {}
    assert normalize_complete(base, grades(base)) == {"source_catalog_id": base["source_catalog_id"], "ranked_candidates": []}
    assert compile_complete_business(raw, base, grades(base), **kwargs)["artifact"]["evidence"] == []
    with pytest.raises(ValueError, match="output_schema"):
        normalize_complete(base, {"source_catalog_id": base["source_catalog_id"]})
    with pytest.raises(ValueError, match="output_schema"):
        normalize_complete(base, {"source_catalog_id": base["source_catalog_id"], "grades": {"C1": {"relevance_score": 0, "use_type": "unrelated"}}})


def test_all_zero_is_valid_complete_output_and_business_selection_algorithm_is_unchanged():
    raw, base, kwargs = prepare(raw_case(["来源甲。", "来源乙。", "来源丙。"] ))
    output = grades(base)
    result = compile_complete_business(raw, base, output, **kwargs)
    legacy = compile_context_business(raw, base, normalize_complete(base, output), **kwargs)
    assert result["artifact"]["evidence"] == legacy["artifact"]["evidence"] == []
    assert result["scope"] == legacy["scope"] == base["scope"]
    artifact = result["artifact"]
    assert artifact["raw_model_selection"] == output
    assert artifact["normalized_model_selection"] == normalize_complete(base, output)
    assert artifact["graded_selection_policy"] == legacy["artifact"]["graded_selection_policy"]
    assert artifact["normalization_audit"]["normalization_order"] == ORDER_POLICY
    assert artifact["normalization_audit"]["tie_order_is_model_priority"] is False
    assert artifact["normalization_audit"]["all_original_grades_preserved"] is True
    assert artifact["normalization_audit"]["new_model_calls"] == 0
    assert artifact["publishable"] is False and artifact["semantic_verified"] is False


def test_same_parent_tie_uses_frozen_catalog_not_model_map_insertion_order():
    raw, base, kwargs = prepare()
    first, second = same_parent_contexts(base)[:2]
    output = grades(base)
    output["grades"][first["source_window_id"]] = {"relevance_score": 3, "use_type": "direct_answer"}
    output["grades"][second["source_window_id"]] = {"relevance_score": 3, "use_type": "counterclaim"}
    assert list(output["grades"]).index(second["source_window_id"]) < list(output["grades"]).index(first["source_window_id"])
    artifact = compile_complete_business(raw, base, output, **kwargs)["artifact"]
    assert artifact["evidence"][0]["source_window_id"] == first["source_window_id"]
    assert artifact["evidence"][0]["quote"] == first["source_bundle"]["quote"]
    assert artifact["selection_audit"]["input_order_is_model_priority"] is False
    assert artifact["selection_audit"]["counterclaim_input_coverage"]["required"] is False


@pytest.mark.parametrize("mutation", [
    lambda out: out["ranked_candidates"].reverse(),
    lambda out: out["ranked_candidates"].pop(),
    lambda out: out["ranked_candidates"].append(deepcopy(out["ranked_candidates"][0])),
    lambda out: out["ranked_candidates"][0].update(quote="new"),
])
def test_inverse_does_not_fill_missing_ids_reorder_or_swallow_duplicate_list_rows(mutation):
    _, base, _ = prepare(raw_case(["来源甲。", "来源乙。"] ))
    normalized = normalize_complete(base, grades(base))
    mutation(normalized)
    with pytest.raises(ValueError, match="normalized_complete_catalog_order"):
        restore_complete(base, normalized)


def test_wrapper_binding_and_model_input_are_explicit_and_separate_from_original_prepared():
    _, base, _ = prepare(raw_case(["完整来源。"] ))
    wrapper = prepare_complete(base)
    changed = deepcopy(wrapper)
    changed["base_prepared_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="wrapper_source_binding"):
        complete_schema(changed)
    changed = deepcopy(wrapper)
    changed["model_input"]["question"]["question"] += "篡改条件"
    with pytest.raises(ValueError, match="wrapper_source_binding"):
        complete_schema(changed)
    with pytest.raises(ValueError, match="original_base_prepared_required"):
        prepare_complete(wrapper)
    assert "grades" in json.dumps(wrapper["wire_tool"], ensure_ascii=False)


def test_hybrid_plan_requires_explicit_mode_and_complete_wrapper_preserves_its_native_scope():
    raw, old, kwargs = prepare(raw_case(["来源甲仍待核验。", "来源乙也未确认。"] ))
    plan = build_context_plan(raw, kwargs["window_scores"],
        full_window_scores_digest=kwargs["full_window_scores_digest"], hybrid_retrieval=True)
    hybrid = {**kwargs, "context_plan": plan, "context_plan_sha256": fingerprint(plan), "hybrid_retrieval": True}
    base = prepare_context_case(raw, **hybrid)
    assert base["hybrid_retrieval"] is True and base["candidate_retrieval_policy"] == plan["policy"]
    assert base["scope"]["hybrid_retrieval"] is True
    assert base["source_binding"]["candidate_retrieval_policy"] == plan["policy"]
    assert "BM25正文/标题" in base["messages"][0]["content"]
    wrapper = prepare_complete(base)
    assert wrapper["model_input"] == base["model_input"] and wrapper["scope"] == base["scope"]
    assert wrapper["hybrid_retrieval"] is True and wrapper["candidate_retrieval_policy"] == plan["policy"]
    result = compile_complete_business(raw, base, grades(base), **hybrid)
    assert result["artifact"]["evidence"] == []
    assert result["scope"]["binding"] == base["source_binding"]
    with pytest.raises(ValueError, match="plan_changed"):
        prepare_context_case(raw, **{**hybrid, "hybrid_retrieval": False})
    with pytest.raises(ValueError, match="plan_changed"):
        prepare_context_case(raw, **{**kwargs, "hybrid_retrieval": True})
    assert "hybrid_retrieval" not in old and "candidate_retrieval_policy" not in old


def test_default_context_mode_is_byte_compatible_and_nonboolean_hybrid_flag_is_rejected():
    raw, base, kwargs = prepare(raw_case(["原始来源未确认。"] ))
    explicit = prepare_context_case(raw, **kwargs, hybrid_retrieval=False)
    assert explicit == base
    assert "hybrid_retrieval" not in base["source_binding"] and "candidate_retrieval_policy" not in base["scope"]
    with pytest.raises(ValueError, match="hybrid_retrieval_boolean_required"):
        prepare_context_case(raw, **kwargs, hybrid_retrieval=1)
