"""Closed, complete grading of every offered context, without new inference.

The original context catalog, source text and read geometry remain unchanged.
Every offered source ID is a required object property. No grade is defaulted,
and no positive/negative count or final conflict relation is prescribed.

Normalization is reversible at the JSON-value level. Its row order comes from
the frozen prepared catalog, never from model priority or JSON key order.
Duplicate raw JSON keys are the outer transport/parser's responsibility: a
Python dict cannot preserve or diagnose an already overwritten duplicate.
"""

from copy import deepcopy
from hashlib import sha256
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, create_model, field_validator

from app.agents.schemas import OutputModel
from app.domain.lc_bundle_case import _wire_tool
from app.domain.lc_context_case import (
    GRADED_FUNCTION_NAME as BASE_FUNCTION_NAME,
    PREPARED_FORMAT as BASE_PREPARED_FORMAT,
    VERSION as BASE_VERSION,
    compile_context_business,
    selection_schema as base_selection_schema,
)


FORMAT = "lc-complete-context-grading-prepared-v1"
VERSION = "lc-complete-context-grading-v1"
FUNCTION_NAME = "LCCompleteContextGradesV1"
PROTOCOL = "lc-complete-context-grades-v1"
ORDER_POLICY = "frozen_prepared_catalog_order_not_model_priority"
POLICY = {
    "policy": "all_offered_context_grade_keys_required_v1",
    "source_keys": "exact_frozen_catalog_once_no_missing_extra_or_defaults",
    "normalization_order": ORDER_POLICY,
    "positive_count_constraint": None,
    "counterclaim_count_constraint": None,
    "source_geometry_changed": False,
    "downstream_thresholds_caps_and_selection_algorithm_changed": False,
    "use_type_is_final_relation": False,
    "raw_json_duplicate_key_validation": "outer_parser_required",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPLACEMENTS = (
    ("只输出source_catalog_id和ranked_candidates；逐行仅source_window_id、relevance_score整数0/1/2/3、use_type。",
     "只输出source_catalog_id和grades。grades必须恰好包含本次目录每一个C键，每键恰一个对象，仅含relevance_score整数0/1/2/3和use_type；不遗漏、不加键、不用默认值。"),
    ("按最重要优先列出，同一原段只选足以保留关键限定的一项；不要借相邻C的内容、评分或立场给较短C。",
     "必须逐项评分所有已提供C，包括背景与无关项；不要借相邻C的内容、评分或立场给较短C。JSON键顺序不表示模型优先级。"),
    ("程序对同一原段只保留分数最高、并列时原始行序最早的真实单个C，不合并范围；之后仅保留2分以上最多10个原段。",
     "程序将完整评分按预先冻结的目录顺序转换为候选列表；同一原段只保留分数最高、并列时目录顺序最早的真实单个C，不合并范围；之后仍仅保留2分以上最多10个原段。"),
    ("没有合格证据可ranked_candidates:[]。程序首句预览不是模型事实摘要；下游必须阅读所选完整quote。此次只提出候选，不发布结论。",
     "没有合格证据时仍须给每个C合法低分，全部0分也合法；只有目录为空时grades:{}。不强制正分、反方或最终冲突数量。程序首句预览不是模型事实摘要；下游必须阅读所选完整quote。此次只提出候选，不发布结论。"),
)


class _CompleteGrade(OutputModel):
    relevance_score: Literal[0, 1, 2, 3]
    use_type: Literal["direct_answer", "counterclaim", "necessary_condition", "background", "unrelated"]

    @field_validator("relevance_score", mode="before")
    @classmethod
    def integer_score(cls, value):
        if type(value) is not int:
            raise ValueError("relevance_score_integer_required")
        return value


def _require(condition, reason):
    if not condition:
        raise ValueError("complete_context_grading_" + reason)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _source_ids(prepared):
    _require(type(prepared) is dict, "prepared_object_required")
    # Only the schema header is adapted for local validation of a wrapper.
    # This does not reconstruct or authenticate its original prepared object.
    base_view = {**prepared, "format": BASE_PREPARED_FORMAT, "version": BASE_VERSION,
                 "function_name": BASE_FUNCTION_NAME}
    base_selection_schema(base_view)
    ids = [row["source_window_id"] for row in prepared["catalog"]]
    native = prepared.get("model_input")
    _require(type(native) is dict and native.get("source_catalog_id") == prepared["source_catalog_id"]
             and type(native.get("source_contexts")) is list
             and len(native["source_contexts"]) == len(ids), "native_source_catalog")
    for offered, delivered in zip(prepared["catalog"], native["source_contexts"], strict=True):
        _require(type(delivered) is dict and delivered.get("id") == offered["source_window_id"]
                 and type(offered.get("source_bundle")) is dict
                 and type(offered["source_bundle"].get("quote")) is str
                 and delivered.get("quote") == offered["source_bundle"]["quote"], "native_source_quote")
    return ids


def _binding(prepared, base_sha):
    return {"format": "lc-complete-context-grading-binding-v1", "base_prepared_sha256": base_sha,
        "source_catalog_id": prepared["source_catalog_id"], "source_ids_sha256": _digest(_source_ids(prepared)),
        "catalog_sha256": _digest(prepared["catalog"]), "model_input_sha256": _digest(prepared["model_input"]),
        "source_binding_sha256": _digest(prepared["source_binding"]), "policy": deepcopy(POLICY)}


def _contract(prepared):
    _require(type(prepared) is dict, "prepared_object_required")
    if prepared.get("complete_grading") is True:
        _require(prepared.get("format") == FORMAT and prepared.get("version") == VERSION
                 and prepared.get("function_name") == FUNCTION_NAME
                 and prepared.get("grading_protocol") == PROTOCOL
                 and prepared.get("complete_grading_policy") == POLICY
                 and type(prepared.get("base_prepared_sha256")) is str
                 and bool(_SHA256.fullmatch(prepared["base_prepared_sha256"])), "wrapper_contract")
        ids = _source_ids(prepared)
        _require(prepared.get("source_ids") == ids
                 and prepared.get("complete_grading_binding") == _binding(prepared, prepared["base_prepared_sha256"]),
                 "wrapper_source_binding")
        return ids
    _require(prepared.get("format") == BASE_PREPARED_FORMAT and prepared.get("version") == BASE_VERSION
             and prepared.get("function_name") == BASE_FUNCTION_NAME
             and "complete_grading" not in prepared, "base_prepared_contract")
    return _source_ids(prepared)


def complete_schema(wrapper_or_base):
    """Require a grade object at every actual source-ID key, including empty."""
    ids = _contract(wrapper_or_base)
    grade_map = create_model("LCCompleteContextGradeMapV1", __base__=OutputModel,
        **{source_id: (_CompleteGrade, ...) for source_id in ids})
    return create_model(FUNCTION_NAME, __base__=OutputModel,
        source_catalog_id=(Literal[wrapper_or_base["source_catalog_id"]], Field(...,
            description="逐字回显本次完整上下文目录绑定ID。")),
        grades=(grade_map, Field(..., description="每个已提供来源键恰一个评分；所有来源键必需，不默认遗漏项，不强制正分或冲突数。")))


def prepare_complete(prepared):
    """Deep-copy the base; do not duplicate or alter its native source payload."""
    ids = _contract(prepared)
    _require(prepared.get("format") == BASE_PREPARED_FORMAT, "original_base_prepared_required")
    messages = prepared.get("messages")
    _require(type(messages) is list and len(messages) == 2
             and messages[0].get("role") == "system" and type(messages[0].get("content")) is str
             and messages[1] == {"role": "user", "content": _json(prepared["model_input"])}, "base_messages")
    instructions = messages[0]["content"]
    for before, after in _REPLACEMENTS:
        _require(instructions.count(before) == 1, "base_instruction_contract")
        instructions = instructions.replace(before, after)
    result = deepcopy(prepared)
    result.update(format=FORMAT, version=VERSION, function_name=FUNCTION_NAME, complete_grading=True,
        grading_protocol=PROTOCOL, complete_grading_policy=deepcopy(POLICY),
        base_prepared_sha256=_digest(prepared), source_ids=list(ids),
        complete_grading_binding=_binding(prepared, _digest(prepared)))
    result["messages"][0]["content"] = instructions
    schema = complete_schema(result)
    result.update(schema_json=schema.model_json_schema(), wire_tool=_wire_tool(schema))
    return result


def _parsed(prepared, output):
    if isinstance(output, BaseModel):
        output = output.model_dump(mode="json")
    _require(type(output) is dict, "output_object_required")
    try:
        return complete_schema(prepared).model_validate(output).model_dump(mode="json")
    except ValidationError:
        raise ValueError("complete_context_grading_output_schema") from None


def normalize_complete(wrapper_or_base, output):
    """Complete map -> legacy rows in frozen catalog order, not model order."""
    ids = _contract(wrapper_or_base)
    parsed = _parsed(wrapper_or_base, output)
    return {"source_catalog_id": parsed["source_catalog_id"], "ranked_candidates": [
        {"source_window_id": source_id, **deepcopy(parsed["grades"][source_id])} for source_id in ids]}


def restore_complete(wrapper_or_base, normalized):
    """Invert only the exact complete normalized order; never fill or reorder."""
    ids = _contract(wrapper_or_base)
    _require(type(normalized) is dict and set(normalized) == {"source_catalog_id", "ranked_candidates"}
             and normalized["source_catalog_id"] == wrapper_or_base["source_catalog_id"]
             and type(normalized["ranked_candidates"]) is list, "normalized_shape")
    rows = normalized["ranked_candidates"]
    _require(len(rows) == len(ids) and all(type(row) is dict
             and set(row) == {"source_window_id", "relevance_score", "use_type"} for row in rows)
             and [row["source_window_id"] for row in rows] == ids, "normalized_complete_catalog_order")
    restored = {"source_catalog_id": normalized["source_catalog_id"], "grades": {
        row["source_window_id"]: {"relevance_score": row["relevance_score"], "use_type": row["use_type"]} for row in rows}}
    return _parsed(wrapper_or_base, restored)


def compile_complete_business(raw_case, base_prepared, output, **context_kwargs):
    """Use unchanged context selection, retaining actual map and conversion audit."""
    _require(type(base_prepared) is dict and base_prepared.get("format") == BASE_PREPARED_FORMAT,
             "original_base_prepared_required")
    parsed = _parsed(base_prepared, output)
    normalized = normalize_complete(base_prepared, parsed)
    _require(restore_complete(base_prepared, normalized) == parsed, "internal_roundtrip")
    result = compile_context_business(raw_case, base_prepared, normalized, **context_kwargs)
    artifact = result["artifact"]
    _require(artifact["raw_model_selection"] == normalized, "legacy_normalization_drift")
    artifact["normalized_model_selection"] = artifact.pop("raw_model_selection")
    artifact["raw_model_selection"] = deepcopy(parsed)
    artifact.update(complete_grading=True, grading_protocol=PROTOCOL, complete_grading_policy=deepcopy(POLICY),
        complete_grading_binding=_binding(base_prepared, _digest(base_prepared)))
    artifact["normalization_audit"] = {"format": "complete-context-grading-normalization-audit-v1",
        "policy": deepcopy(POLICY), "base_prepared_sha256": _digest(base_prepared),
        "source_ids": _source_ids(base_prepared), "complete_source_count": len(base_prepared["catalog"]),
        "model_output_sha256": _digest(parsed), "normalized_output_sha256": _digest(normalized),
        "normalization_order": ORDER_POLICY, "tie_order_is_model_priority": False,
        "all_original_grades_preserved": True, "reversible_value_roundtrip_checked": True,
        "new_model_calls": 0, "source_geometry_changed": False, "selection_algorithm_changed": False}
    artifact["selection_audit"]["effective_input_order"] = ORDER_POLICY
    artifact["selection_audit"]["input_order_is_model_priority"] = False
    return result
