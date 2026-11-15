"""Versioned required-key codec; identical ratings with smaller integer enums."""
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BeforeValidator, create_model

from app.agents.strict_evidence import (StrictModel, _required_alias_map,
    validate_raw_case, canonical_selection_wire, SELECTION_INSTRUCTIONS,
    prepare_selection_request)

VERSION = "strict-selection-integer-codec-v2"
CODES = ("0", "1", "2_support", "2_oppose", "2_qualifier", "3_support", "3_oppose", "3_qualifier")


def _integer(value):
    if type(value) is not int:
        raise ValueError("selection_integer_code_required")
    return value


Code = Annotated[Literal[0, 1, 2, 3, 4, 5, 6, 7], BeforeValidator(_integer)]


def prepare_compact_selection_request(raw_case):
    raw = validate_raw_case(raw_case)
    required = _required_alias_map("CompactRequiredDecisions", [row["paragraph_ref"] for row in raw["catalog"]], Code)
    schema = create_model("CompactSelectionWireOutput", __base__=StrictModel,
        hypothesis=(str, ...), decisions=(required, ...))
    _old_schema, messages = prepare_selection_request(raw)
    old = "每键只填枚举编码：0无关、1背景、2_support必要支持、2_oppose必要反证、2_qualifier关键限定、3_support直接支持、3_oppose直接反证、3_qualifier直接限定。"
    new = "每键只填整数编码：0无关、1背景、2必要支持、3必要反证、4关键限定、5直接支持、6直接反证、7直接限定。整数代码不是relevance数值；2/3/4映射relevance=2，5/6/7映射relevance=3。"
    if old not in SELECTION_INSTRUCTIONS:
        raise ValueError("selection_source_instructions_changed")
    return schema, [SystemMessage(content=SELECTION_INSTRUCTIONS.replace(old, new)), messages[1]]


def canonical_compact_selection(raw_case, value):
    schema, _messages = prepare_compact_selection_request(raw_case)
    data = schema.model_validate(value).model_dump(by_alias=True)
    decoded = {"hypothesis":data["hypothesis"],
        "decisions":{ref:CODES[code] for ref,code in data["decisions"].items()}}
    return canonical_selection_wire(raw_case, decoded)
