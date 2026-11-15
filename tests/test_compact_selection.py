"""Synthetic codec tests, no external model or reference labels."""
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from app.agents.compact_selection import prepare_compact_selection_request, canonical_compact_selection
from app.agents.strict_evidence import canonical_selection_wire
from app.agents.strict_transport import strict_tool


def raw():
    return {"question":{"question_id":"q","question":"甲设备已确认还是尚未确认？"},
        "catalog":[{"paragraph_ref":f"P{i+1}","doc_id":f"d{i}","para_id":"p1",
            "text":"主管部门表示尚未确认。","title":"甲设备",
            "paragraph_complete":True,"parent_complete":True} for i in range(8)]}


def test_integer_codec_preserves_all_eight_old_rating_and_stance_values():
    r=raw();original=deepcopy(r)
    value={"hypothesis":"甲设备已确认","decisions":{f"P{i+1}":i for i in range(8)}}
    schema,messages=prepare_compact_selection_request(r)
    assert not list(Draft202012Validator(strict_tool(schema)["function"]["parameters"]).iter_errors(value))
    result=canonical_compact_selection(r,value).model_dump()
    old={"hypothesis":"甲设备已确认","decisions":dict(zip(value["decisions"],
        ["0","1","2_support","2_oppose","2_qualifier","3_support","3_oppose","3_qualifier"]))}
    assert result==canonical_selection_wire(r,old).model_dump()
    assert r==original and "只填整数编码" in messages[0].content
    assert len(result["decisions"])==8


@pytest.mark.parametrize("bad",[True,False,"2",8,-1,None,2.0])
def test_noninteger_or_unknown_codes_are_never_filled_or_coerced(bad):
    value={"hypothesis":"甲设备已确认","decisions":{f"P{i+1}":i for i in range(8)}}
    value["decisions"]["P1"]=bad
    with pytest.raises(ValueError):canonical_compact_selection(raw(),value)


@pytest.mark.parametrize("change",["missing","unknown"])
def test_complete_original_candidate_keys_remain_required(change):
    value={"hypothesis":"甲设备已确认","decisions":{f"P{i+1}":i for i in range(8)}}
    if change=="missing":value["decisions"].pop("P1")
    else:value["decisions"]["not-source"]=0
    with pytest.raises(ValueError):canonical_compact_selection(raw(),value)
