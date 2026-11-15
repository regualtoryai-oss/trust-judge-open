"""Synthetic source-address and output contracts; no model or reference scores."""
from copy import deepcopy
import json

import pytest

from app.domain.compact_semantic_selection import prepare_compact_selection, validate_compact_selection


QUESTION = {"question": "安排意愿还是程序状态造成延误，哪些前提仍待核验？", "answer": "DO_NOT_FORWARD"}
CATALOG = [{"paragraph_ref": "P1", "doc_id": "source_alpha", "para_id": "section_1", "title": "原始标题甲",
    "text": "先行说明：条件未核实。\r\n委员会已批准申请。随后说明仅为申报方说法。😀", "paragraph_complete": True,
    "source_credibility": .95, "offset_hint": "DO_NOT_FORWARD"},
    {"paragraph_ref": "P2", "doc_id": "source_alpha", "para_id": "section_2", "title": "原始标题甲",
     "text": "登记记录显示申请尚未受理。", "paragraph_complete": True},
    {"paragraph_ref": "P3", "doc_id": "source_beta", "para_id": "section_1", "title": "原始标题乙",
     "text": "仅有主题背景说明。", "paragraph_complete": True}]


def spans():
    text = CATALOG[0]["text"]; start = text.index("委员会")
    return [{"paragraph_ref": "P1", "start_cp": start, "end_cp": start + len("委员会已批准申请。")},
            {"paragraph_ref": "P2", "start_cp": 0, "end_cp": len(CATALOG[1]["text"])}]


def validate(value, catalog=CATALOG, addresses=None):
    return validate_compact_selection(QUESTION, catalog, spans() if addresses is None else addresses, value)


def test_nonprefix_source_owned_view_full_question_and_original_titles_once():
    before = deepcopy((QUESTION, CATALOG)); prepared = prepare_compact_selection(QUESTION, CATALOG, spans())
    payload = prepared["payload"]
    assert payload["question"] == QUESTION["question"]
    assert payload["titles"] == [["D1", "原始标题甲"]]
    assert payload["views"] == [["P1", "D1", "委员会已批准申请。", True, True],
                               ["P2", "D1", CATALOG[1]["text"], False, False]]
    assert prepared["context"]["views"]["P1"]["provided_text_start_cp"] > 0
    assert prepared["context"]["unshown_catalog_refs"] == ["P3"]
    wire = json.dumps(payload, ensure_ascii=False)
    assert all(secret not in wire for secret in ("DO_NOT_FORWARD", "source_credibility", "source_alpha", "section_1", "offset_hint"))
    assert (QUESTION, CATALOG) == before


def test_multiple_proposed_roles_only_alias_validation_and_every_selected_requires_full_read():
    result = validate({"selected": [{"p": "P2", "roles": ["P", "C"]}, {"p": "P1", "roles": ["L", "D", "F"]}]})
    assert result["validated_refs"] == result["required_full_read_refs"] == ["P1", "P2"]
    assert result["roles_by_ref"] == {"P1": ["D", "F", "L"], "P2": ["C", "P"]}
    assert result["validated_refs_scope"] == "program_alias_and_shape_only"
    assert result["context"]["views"]["P1"]["after_unknown"]
    assert result["full_parent_semantic_recheck_required"] and result["adjacent_context_may_still_be_required"]
    for flag in ("semantic_verified", "publishable", "formal_evidence_generated", "program_quote_compilation_allowed_now",
                 "roles_are_final_supporting_or_conflicting_relations", "model_quote_or_metadata_accepted",
                 "full_parent_reading_performed", "parent_raw_bytes_reverified"):
        assert result[flag] is False


def test_empty_selection_keeps_unselected_and_unshown_context_unknown():
    result = validate({"selected": []})
    assert result["validated_refs"] == result["required_full_read_refs"] == []
    assert result["unselected_view_refs"] == ["P1", "P2"] and result["unshown_catalog_refs"] == ["P3"]
    assert result["unselected_or_unseen_context_status"] == "unknown_not_irrelevant"


@pytest.mark.parametrize("ref", ["P3", "P999"])
def test_unknown_or_unshown_alias_refused_even_when_pattern_is_legal(ref):
    with pytest.raises(ValueError, match="unknown_program_paragraph"):
        validate({"selected": [{"p": ref, "roles": ["D"]}]})


@pytest.mark.parametrize("damage", ["duplicate_p", "duplicate_role", "unknown_role", "empty_roles", "bad_p", "p_number"])
def test_duplicate_and_invalid_output_never_silently_repaired(damage):
    value = {"selected": [{"p": "P1", "roles": ["D"]}]}
    if damage == "duplicate_p": value["selected"].append(deepcopy(value["selected"][0]))
    elif damage == "duplicate_role": value["selected"][0]["roles"] = ["D", "D"]
    elif damage == "unknown_role": value["selected"][0]["roles"] = ["B"]
    elif damage == "empty_roles": value["selected"][0]["roles"] = []
    elif damage == "bad_p": value["selected"][0]["p"] = "P01"
    else: value["selected"][0]["p"] = 1
    with pytest.raises(ValueError): validate(value)


@pytest.mark.parametrize("field,value", [("quote", "改写原文"), ("doc_id", "source_alpha"), ("para_id", "section_1"),
    ("offset", 4), ("explanation", "长理由"), ("confidence", .9), ("reasoning", "hidden text")])
def test_model_cannot_echo_or_generate_quote_source_fields_or_reasoning(field, value):
    result = {"selected": [{"p": "P1", "roles": ["D"], field: value}]}
    with pytest.raises(ValueError): validate(result)
    with pytest.raises(ValueError): validate({"selected": [], field: value})


@pytest.mark.parametrize("damage", ["unknown", "duplicate", "bool", "negative", "overrun", "reverse", "extra_text"])
def test_program_view_coordinates_are_strict_local_half_open_addresses(damage):
    addresses = spans()
    if damage == "unknown": addresses[0]["paragraph_ref"] = "P999"
    elif damage == "duplicate": addresses.append(deepcopy(addresses[0]))
    elif damage == "bool": addresses[0]["start_cp"] = True
    elif damage == "negative": addresses[0]["start_cp"] = -1
    elif damage == "overrun": addresses[0]["end_cp"] = 10000
    elif damage == "reverse": addresses[0]["start_cp"], addresses[0]["end_cp"] = addresses[0]["end_cp"], addresses[0]["start_cp"]
    else: addresses[0]["text"] = "caller rewritten excerpt"
    with pytest.raises(ValueError): prepare_compact_selection(QUESTION, CATALOG, addresses)


def test_unicode_whitespace_and_emoji_lossless_codepoints_not_bytes_or_tokens():
    catalog = deepcopy(CATALOG); text = " \r\n中 English 😀\t尾部 "
    catalog[0]["text"] = text
    addresses = [{"paragraph_ref": "P1", "start_cp": 1, "end_cp": len(text) - 1}]
    prepared = prepare_compact_selection(QUESTION, catalog, addresses)
    assert prepared["payload"]["views"][0][2] == text[1:-1]
    assert prepared["context"]["views"]["P1"]["parent_codepoints"] == len(text)
    assert prepared["budget_preflight_performed"] is False


def test_partial_supplied_parent_cannot_make_its_missing_context_known():
    catalog = deepcopy(CATALOG); catalog[0]["paragraph_complete"] = False
    address = [{"paragraph_ref": "P1", "start_cp": 0, "end_cp": len(catalog[0]["text"])}]
    prepared = prepare_compact_selection(QUESTION, catalog, address)
    assert prepared["payload"]["views"][0][3:] == [True, True]
    result = validate({"selected": [{"p": "P1", "roles": ["D"]}]}, catalog, address)
    assert result["required_full_read_refs"] == ["P1"] and not result["parent_raw_bytes_reverified"]


def test_empty_original_parent_preserves_identity_unknown_and_does_not_allow_empty_evidence():
    catalog = deepcopy(CATALOG); catalog[0]["text"] = ""
    address = [{"paragraph_ref": "P1", "start_cp": 0, "end_cp": 0}]
    prepared = prepare_compact_selection(QUESTION, catalog, address)
    assert prepared["payload"]["views"][0][2] == "" and not prepared["has_nonempty_views"]
    assert validate({"selected": []}, catalog, address)["unselected_view_refs"] == ["P1"]
    with pytest.raises(ValueError, match="empty_view_remains_unknown"):
        validate({"selected": [{"p": "P1", "roles": ["F"]}]}, catalog, address)


def test_schema_is_compact_without_enumerating_all_paragraph_aliases_and_has_no_ten_cap():
    catalog = [{"paragraph_ref": f"P{i}", "doc_id": f"s{i}", "para_id": "p", "title": f"标题{i}", "text": f"材料{i}",
                "paragraph_complete": True} for i in range(1, 22)]
    addresses = [{"paragraph_ref": row["paragraph_ref"], "start_cp": 0, "end_cp": len(row["text"])} for row in catalog]
    prepared = prepare_compact_selection(QUESTION, catalog, addresses)
    schema = prepared["response_model"].model_json_schema()
    assert schema["properties"]["selected"]["maxItems"] == 21
    assert "P21" not in json.dumps(schema)
    assert schema["$defs"]["CompactChoice"]["properties"]["p"]["pattern"] == "^P[1-9][0-9]*$"
    result = validate_compact_selection(QUESTION, catalog, addresses,
        {"selected": [{"p": row["paragraph_ref"], "roles": ["F"]} for row in catalog]})
    assert len(result["required_full_read_refs"]) == 21


def test_zero_views_schema_allows_only_zero_selections_and_keeps_all_sources_unknown():
    prepared = prepare_compact_selection(QUESTION, CATALOG, [])
    assert prepared["response_model"].model_json_schema()["properties"]["selected"]["maxItems"] == 0
    result = validate_compact_selection(QUESTION, CATALOG, [], {"selected": []})
    assert result["unshown_catalog_refs"] == ["P1", "P2", "P3"]
    with pytest.raises(ValueError): prepared["response_model"].model_validate({"selected": [{"p": "P1", "roles": ["D"]}]})


@pytest.mark.parametrize("before,visible,after", [
    ("", "委员会已批准申请。", "以上仅为申报方说法，登记记录显示尚未受理。"),
    ("本段到岗指完成报名，并非实际入职。", "已到岗四十人。", ""),
    ("", "计划月底启用。", "仅为附条件安排，须审计通过。"),
    ("以下转引旧版制度，现行版本另见附录。", "该决定已经生效。", ""),
])
def test_visible_assertion_with_unseen_negation_or_scope_never_becomes_semantic_proof(before, visible, after):
    catalog = deepcopy(CATALOG); catalog[0]["text"] = before + visible + after
    addresses = [{"paragraph_ref": "P1", "start_cp": len(before), "end_cp": len(before + visible)}]
    result = validate({"selected": [{"p": "P1", "roles": ["D"]}]}, catalog, addresses)
    assert result["context"]["views"]["P1"]["before_unknown"] is bool(before)
    assert result["context"]["views"]["P1"]["after_unknown"] is bool(after)
    assert result["required_full_read_refs"] == ["P1"] and not result["semantic_verified"]
    assert not result["program_quote_compilation_allowed_now"]


def test_same_body_different_original_sources_keeps_both_aliases_without_equating_votes():
    catalog = deepcopy(CATALOG); catalog[1]["text"] = catalog[0]["text"]
    addresses = [{"paragraph_ref": row["paragraph_ref"], "start_cp": 0, "end_cp": len(row["text"])} for row in catalog[:2]]
    result = validate({"selected": [{"p": "P1", "roles": ["D"]}, {"p": "P2", "roles": ["C"]}]}, catalog, addresses)
    assert result["validated_refs"] == ["P1", "P2"]
    assert result["context"]["views"]["P1"]["para_id"] != result["context"]["views"]["P2"]["para_id"]
    assert not result["roles_are_final_supporting_or_conflicting_relations"]


def test_full_parent_with_unresolved_pronoun_still_requires_context_and_semantic_recheck():
    catalog = deepcopy(CATALOG); catalog[0]["text"] = "此项决定尚未生效。"
    addresses = [{"paragraph_ref": "P1", "start_cp": 0, "end_cp": len(catalog[0]["text"])}]
    result = validate({"selected": [{"p": "P1", "roles": ["L"]}]}, catalog, addresses)
    assert not result["context"]["views"]["P1"]["before_unknown"]
    assert not result["context"]["views"]["P1"]["after_unknown"]
    assert result["adjacent_context_may_still_be_required"] and result["full_parent_semantic_recheck_required"]
    assert not result["semantic_verified"] and not result["program_quote_compilation_allowed_now"]


def test_typed_model_response_can_be_structurally_revalidated_without_trusting_prepared_context():
    prepared = prepare_compact_selection(QUESTION, CATALOG, spans())
    value = prepared["response_model"].model_validate({"selected": [{"p": "P1", "roles": ["D"]}]})
    prepared["context"]["views"]["P1"]["doc_id"] = "forged_external_context"
    result = validate(value)
    assert result["context"]["views"]["P1"]["doc_id"] == "source_alpha"
