"""Synthetic topology/budget tests only; fake integers are NOT token estimates."""
from copy import deepcopy

import pytest

from app.domain.obligation_selection import plan_obligation_selection


QUESTION = {"question": "安排原因还是程序原因影响进度，尚有哪些限定？", "answer": "DO_NOT_FORWARD"}


def source(index, text=None, complete=True):
    return {"paragraph_ref": f"P{index}", "doc_id": f"document_{index}", "para_id": "section_1",
            "text": text if text is not None else f"来源材料{index}提出一种待查说法。",
            "title": f"原始标题{index}", "paragraph_complete": complete,
            "source_credibility": .95, "retrieval_hint": "DO_NOT_FORWARD"}


def obligation(index):
    return {"obligation_id": f"O{index}", "aspect_id": "A1", "role": "F", "ask": f"待查前提{index}是否成立？"}


def link(p, o=1, relation="support"):
    return {"obligation_id": f"O{o}", "paragraph_ref": f"P{p}", "span_ref": "S1", "relation": relation,
            "usage_kind": {"support": "specific_premise", "conflict": "counterclaim", "limit": "limitation"}[relation],
            "usage": f"保留材料对待查命题的{relation}作用"}


def judgment(index, links, obligations=(1,), dependencies=(), state="unresolved"):
    return {"judgment_id": f"J{index}", "proposition_to_test": f"需要检验的中间判断{index}",
            "obligation_ids": [f"O{i}" for i in obligations], "depends_on": [f"J{i}" for i in dependencies],
            "proposed_state": state, "links": links}


def plan(judgments=None, obligations=1, answers=(1,)):
    return {"obligations": [obligation(i) for i in range(1, obligations + 1)],
            "judgments": judgments if judgments is not None else [judgment(1, [link(1)])],
            "answer_depends_on": [f"J{i}" for i in answers]}


def fake_cost(payload):
    # Explicit synthetic accounting units, not chars, bytes or real token claims.
    return 7 + 11 * len(payload["reading_units"]) + 3 * len(payload["proposed_judgments"])


def select(catalog=None, output=None, **kwargs):
    return plan_obligation_selection(QUESTION, catalog if catalog is not None else [source(1)],
        output if output is not None else plan(), input_token_cost=kwargs.pop("input_token_cost", fake_cost),
        token_limit=kwargs.pop("token_limit", 1000), **kwargs)


def test_dependency_closure_keeps_support_counterclaim_and_limit_atomic():
    output = plan([judgment(1, [link(1)], state="supported"),
                   judgment(2, [link(2, 2), link(3, 2, "conflict"), link(4, 3, "limit")],
                            obligations=(2, 3), dependencies=(1,))], obligations=3, answers=(2,))
    catalog = [source(i) for i in range(1, 5)]
    selected = select(catalog, output)
    assert selected["proposed_selection_refs"] == ["P1", "P2", "P3", "P4"]
    assert selected["selected_judgment_ids"] == ["J1", "J2"]
    assert {item["relation"] for row in selected["reading_payload"]["proposed_judgments"] for item in row["links"]} == {
        "support", "conflict", "limit"}
    small = select(catalog, output, token_limit=30)
    assert small["proposed_selection_refs"] == ["P1"]
    assert small["selected_judgment_ids"] == ["J1"]
    assert small["status"] == "incomplete"
    assert small["missing_obligation_ids"] == ["O2", "O3"]
    assert small["deferred_bundles"][0]["required_judgment_ids"] == ["J1", "J2"]
    assert all(row["state"] != "supported" for row in small["judgment_plans"])


def test_pure_derived_judgment_uses_reachable_sources_without_fabricated_direct_link():
    output = plan([judgment(1, [link(1)], state="supported"),
                   judgment(2, [], dependencies=(1,), state="supported")], answers=(2,))
    result = select(output=output)
    assert result["status"] == "selected_unverified"
    assert result["missing_obligation_ids"] == []
    assert result["reading_payload"]["proposed_judgments"][1]["links"] == []
    assert result["proposed_selection_refs"] == ["P1"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rename", [False, True])
def test_empty_unresolved_nodes_cannot_crowd_out_affordable_source_backed_conflict_closure(reverse, rename):
    catalog = [source(i) for i in range(1, 4)]
    judgments = [judgment(1, [link(1)]), judgment(2, [link(2, relation="conflict"), link(3, relation="limit")],
        dependencies=(1,))] + [judgment(i, []) for i in range(3, 19)]
    output = plan(judgments, answers=tuple(range(2, 19)))
    if rename:
        mapping = {f"J{i}": f"J{i + 100}" for i in range(1, 19)}
        for row in output["judgments"]:
            row["judgment_id"] = mapping[row["judgment_id"]]
            row["depends_on"] = [mapping[ref] for ref in row["depends_on"]]
            row["proposition_to_test"] = "改名后的合成待检验判断"
        output["answer_depends_on"] = [mapping[ref] for ref in output["answer_depends_on"]]
    if reverse:
        output["judgments"].reverse(); output["answer_depends_on"].reverse(); catalog.reverse()
    def budget(payload):
        linked = sum(bool(row["links"]) for row in payload["proposed_judgments"])
        empty = len(payload["proposed_judgments"]) - linked
        return 20 + 60 * len(payload["reading_units"]) + 30 * linked + 8 * empty
    result = select(catalog, output, input_token_cost=budget, token_limit=260)
    assert set(result["proposed_selection_refs"]) == {"P1", "P2", "P3"}
    assert len(result["selected_judgment_ids"]) == 2 and result["input_tokens"] == 260
    assert {row["reason"] for row in result["deferred_bundles"]} == {"source_unanchored_unknown"}
    assert len(result["deferred_bundles"]) == 16
    assert result["status"] == "incomplete" and not result["publishable"]
    assert all(row["state"] == "pending_unknown" for row in result["judgment_plans"]
               if not row["selected_in_reading_graph"])
    assert len(result["original_proposed_plan"]["judgments"]) == 18


@pytest.mark.parametrize("relation", ["support", "conflict", "limit"])
def test_source_anchoring_keeps_pure_derived_paths_for_all_three_relations(relation):
    output = plan([judgment(1, [link(1, relation=relation)]), judgment(2, [], dependencies=(1,))], answers=(2,))
    result = select(output=output)
    assert result["selected_judgment_ids"] == ["J1", "J2"]
    assert result["proposed_selection_refs"] == ["P1"]
    assert result["reading_payload"]["proposed_judgments"][0]["links"][0]["relation"] == relation
    assert not result["semantic_entailment_verified"]


def test_entirely_unanchored_answer_node_remains_pending_without_spending_reading_budget():
    result = select(output=plan([judgment(1, [])]))
    assert result["selected_judgment_ids"] == result["proposed_selection_refs"] == []
    assert result["input_tokens"] == fake_cost(result["reading_payload"]) == 7
    assert result["deferred_bundles"][0]["reason"] == "source_unanchored_unknown"
    assert result["pending_obligation_ids"] == ["O1"]
    assert result["original_proposed_plan"]["judgments"][0]["proposed_state"] == "unresolved"
    assert result["status"] == "incomplete" and result["publishable"] is False


def test_declared_but_unlinked_obligation_not_automatically_covered():
    output = plan([judgment(1, [link(1)], obligations=(1, 2))], obligations=2)
    result = select(output=output)
    assert result["missing_obligation_ids"] == ["O2"]
    assert result["status"] == "incomplete"


def test_no_hard_ten_or_role_quota():
    result = select([source(i) for i in range(1, 14)], plan([judgment(1, [link(i) for i in range(1, 14)])]))
    assert result["proposed_selection_refs"] == [f"P{i}" for i in range(1, 14)]
    assert result["status"] == "selected_unverified"


def test_exact_text_shares_body_but_preserves_every_selected_original_alias_and_link():
    body = "同一原文材料保留不同原始来源身份。🙂\r\n"
    catalog = [source(1, body), source(2, body), source(3, body)]
    output = plan([judgment(1, [link(1), link(2, relation="conflict")])])
    result = select(catalog, output)
    units = result["reading_payload"]["reading_units"]
    assert len(units) == 1
    assert "".join(row["text"] for row in units[0]["blocks"]) == body
    assert [row["paragraph_ref"] for row in units[0]["source_aliases"]] == ["P1", "P2"]
    assert [row["title"] for row in units[0]["source_aliases"]] == ["原始标题1", "原始标题2"]
    assert len(result["reading_payload"]["proposed_judgments"][0]["links"]) == 2
    assert result["pending_paragraph_refs"] == ["P3"]  # identical body is not automatic evidence
    assert len(result["paragraph_decisions"]) == 3
    assert not result["sharing_implies_fact_or_source_equivalence"]
    assert result["independent_source_votes_created"] == 0


def test_same_usage_with_different_body_does_not_deduplicate_original_paragraphs():
    result = select([source(1), source(2)], plan([judgment(1, [link(1), link(2)])]))
    assert len(result["reading_payload"]["reading_units"]) == 2


def test_orphan_minorities_and_unlinked_sources_remain_pending_unknown():
    output = plan([judgment(1, [link(1)]), judgment(2, [link(2, relation="conflict")])])
    result = select([source(1), source(2), source(3)], output)
    assert result["orphan_judgment_ids"] == ["J2"]
    assert result["pending_paragraph_refs"] == ["P2", "P3"]
    assert result["paragraph_decisions"][1]["reason"] == "outside_proposed_answer_paths_unknown"
    assert result["original_proposed_plan"]["judgments"][1]["links"][0]["relation"] == "conflict"
    assert not result["answer_path_selection_proves_counterevidence_complete"]


def test_empty_answer_dependencies_never_means_success_or_irrelevance():
    result = select(output=plan(answers=()))
    assert result["status"] == "incomplete"
    assert result["proposed_selection_refs"] == []
    assert result["orphan_judgment_ids"] == ["J1"]
    assert result["pending_paragraph_refs"] == ["P1"]
    assert result["pending_obligation_ids"] == ["O1"]


def test_cost_is_for_complete_union_payload_nonlinear_and_reservations_count():
    observed = []
    def nonlinear(payload):
        observed.append(deepcopy(payload))
        assert payload["question"] == {"question": QUESTION["question"]}
        assert "proposed_judgments" in payload and "untested_obligations" in payload
        return 10 * len(payload["reading_units"]) ** 2
    output = plan([judgment(1, [link(1)]), judgment(2, [link(2)])], answers=(1, 2))
    result = select([source(1), source(2)], output, token_limit=45, reserved_tokens=10, input_token_cost=nonlinear)
    assert len(result["proposed_selection_refs"]) == 1  # 40 union + 10 reserved exceeds 45
    assert any(len(item["reading_units"]) == 2 for item in observed)
    assert result["input_tokens"] + result["reserved_tokens"] <= result["token_limit"]
    assert result["cost_basis"].endswith("actual_final_message_tokenizer_preflight_required")


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("nan"), "10"])
def test_invalid_callback_cost_rejected(bad):
    with pytest.raises(ValueError, match="callback_cost_nonnegative_integer"):
        select(input_token_cost=lambda payload: bad)


@pytest.mark.parametrize("field,bad", [("token_limit", True), ("token_limit", -1),
                                       ("reserved_tokens", 0.0), ("reserved_tokens", None)])
def test_invalid_limit_or_reservation_rejected(field, bad):
    with pytest.raises(ValueError, match="nonnegative_integer_required"):
        select(**{field: bad})


@pytest.mark.parametrize("failure", ["none", "exception"])
def test_unknown_base_cost_cannot_create_free_or_successful_plan(failure):
    def unknown(payload):
        if failure == "exception":
            raise RuntimeError("PRIVATE_CALLBACK_DETAIL")
        return None
    result = select(input_token_cost=unknown)
    assert result["status"] == "incomplete" and result["reading_payload"] is None
    assert result["proposed_selection_refs"] == []
    assert result["planning_budget_status"] == "unknown_cost"
    assert "PRIVATE_CALLBACK_DETAIL" not in str(result)


def test_unknown_bundle_cost_deferred_not_zero_or_force_filled():
    result = select(input_token_cost=lambda payload: None if payload["reading_units"] else 0)
    assert result["status"] == "incomplete"
    assert result["deferred_bundles"][0]["reason"] == "unknown_cost"
    assert result["proposed_selection_refs"] == []


def test_even_empty_base_over_budget_is_not_a_valid_reading_request():
    result = select(input_token_cost=lambda payload: 20, token_limit=10)
    assert result["reading_payload"] is None and not result["within_planning_budget"]
    assert result["planning_budget_status"] == "exceeds_limit"
    assert result["proposed_selection_refs"] == []


def test_read_scope_and_partial_context_never_verify_truth_or_free_the_payload():
    catalog = [source(1, complete=False), source(2)]
    observed = []
    def cost(payload):
        observed.append(payload)
        return fake_cost(payload)
    result = select(catalog, read_refs=["P1"], input_token_cost=cost)
    assert result["status"] == "partial"
    assert result["unread_paragraph_refs"] == ["P2"]
    assert result["paragraph_decisions"][0]["unseen_context_unknown"]
    assert not result["paragraph_decisions"][0]["new_reading_required"]
    assert any(payload["reading_units"] for payload in observed)  # declared-read not free
    for field in ("publishable", "parent_source_mapping_verified", "judgement_states_verified",
                  "semantic_entailment_verified", "rule_applicability_verified", "formal_evidence_generated"):
        assert result[field] is False


@pytest.mark.parametrize("refs", [["P99"], ["P1", "P1"], [True]])
def test_read_refs_must_be_known_unique_catalog_members(refs):
    with pytest.raises(ValueError, match="known_unique_read_refs"):
        select(read_refs=refs)


@pytest.mark.parametrize("damage", ["unknown_span", "unknown_obligation", "cycle", "enriched_plan", "quote"])
def test_structural_model_plan_revalidated_not_trusted(damage):
    output = plan()
    if damage == "unknown_span": output["judgments"][0]["links"][0]["span_ref"] = "S99"
    elif damage == "unknown_obligation": output["judgments"][0]["links"][0]["obligation_id"] = "O99"
    elif damage == "cycle": output["judgments"][0]["depends_on"] = ["J1"]
    elif damage == "enriched_plan": output["source_addresses_resolved"] = True
    else: output["judgments"][0]["links"][0]["quote"] = "model written quote"
    with pytest.raises(ValueError):
        select(output=output)


def test_duplicate_original_identity_rejected_even_when_internal_p_refs_differ():
    catalog = [source(1), source(2)]; catalog[1].update(doc_id=catalog[0]["doc_id"], para_id=catalog[0]["para_id"])
    with pytest.raises(ValueError, match="unique_original_doc_para"):
        select(catalog)


def test_original_inputs_unchanged_and_non_allowlisted_metadata_not_forwarded():
    catalog, output = [source(1)], plan()
    before = deepcopy((QUESTION, catalog, output))
    result = select(catalog, output)
    assert (QUESTION, catalog, output) == before
    assert "DO_NOT_FORWARD" not in str(result["reading_payload"])
    assert "source_credibility" not in str(result["reading_payload"])
    assert "retrieval_hint" not in str(result["reading_payload"])


def test_cost_callback_cannot_mutate_returned_payload_or_sources():
    def destructive(payload):
        payload.clear()
        return 1
    result = select(input_token_cost=destructive)
    assert result["reading_payload"]["question"]["question"] == QUESTION["question"]
    assert result["reading_payload"]["reading_units"][0]["source_aliases"][0]["doc_id"] == "document_1"
