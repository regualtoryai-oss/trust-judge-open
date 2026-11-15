"""Synthetic local planning checks, not semantic quality measurements."""
from copy import deepcopy
from hashlib import sha256

import pytest

from app.domain.recall_recovery import RecallRecoveryInvalid, plan_recall_recovery
from app.domain.scope_batches import subset_evidence
from app.domain.strict_verification import build_fact_views, canonical


KEYS = ("时间", "主体", "地点", "内容", "影响", "数值口径")


def annotation(fact):
    return {"fact_des": fact, "argument": {key: None for key in KEYS}, "entities": ["设备"]}


def anchor(identity="a", *, quote="甲称设备已开放；乙称设备尚未开放。", facts=None,
           origin="model", paragraph=None, start=9):
    facts = ["甲称设备已开放。", "乙称设备尚未开放。"] if facts is None else facts
    row = {
        "id": identity, "paragraph_id": paragraph or "paragraph-" + identity,
        "raw_doc_id": "doc-" + identity, "raw_para_id": "para-" + identity,
        "start_cp": start, "end_cp": start + len(quote), "quote": quote,
        "interpretation_origin": origin, **annotation(facts[0]),
    }
    if len(facts) > 1:
        row["additional_interpretations"] = [annotation(fact) for fact in facts[1:]]
    return row


def view_ids(evidence):
    return [view["id"] for view in build_fact_views(evidence)]


def test_same_quote_opposing_views_selected_peer_does_not_hide_rejected_fact():
    rows = [anchor()]
    ids = view_ids(rows)
    before = deepcopy(rows)
    plan = plan_recall_recovery(rows, [ids[0]], [])
    assert plan["parent_view_ids"] == [ids[1]]
    assert plan["second_look_view_ids"] == [ids[1]]
    assert plan["audit"]["pending_automatic_view_ids"] == [ids[1]]
    assert plan["audit"]["parent_bindings"][0]["reason"] == "second_look"
    assert rows == before


def test_many_rejected_interpretations_choose_one_stable_parent_per_anchor():
    rows = [anchor("b", facts=["断言三", "断言一", "断言二"]), anchor("a")]
    views = build_fact_views(rows)
    plan = plan_recall_recovery(rows, [], [])
    expected = sorted(min(view["id"] for view in views if view["evidence_id"] == identity)
                      for identity in ("a", "b"))
    assert plan["parent_view_ids"] == expected
    assert plan["audit"]["counts"]["pending_automatic_fact_views"] == 5
    assert plan["audit"]["counts"]["pending_automatic_anchors"] == 2
    reordered = deepcopy(rows[::-1])
    reordered[1]["fact_des"], reordered[1]["additional_interpretations"][0]["fact_des"] = (
        reordered[1]["additional_interpretations"][0]["fact_des"], reordered[1]["fact_des"])
    assert plan_recall_recovery(reordered, [], []) == plan


def test_all_explicit_narrowing_peers_retained_and_no_extra_parent_for_same_anchor():
    rows = [anchor(facts=["断言一", "断言二", "断言三"]), anchor("b")]
    ids = view_ids(rows)
    requested = [ids[1], ids[0]]
    before = deepcopy(rows)
    plan = plan_recall_recovery(rows, [], requested)
    assert plan["narrowing_view_ids"] == sorted(requested)
    assert set(plan["parent_view_ids"]) >= set(requested)
    assert ids[2] not in plan["parent_view_ids"]
    assert len(plan["second_look_view_ids"]) == 1
    assert rows == before
    assert plan_recall_recovery(rows, [], requested[::-1]) == plan


def test_selected_explicit_narrowing_parent_retained_without_extra_same_quote_parent():
    rows = [anchor()]
    first, second = view_ids(rows)
    plan = plan_recall_recovery(rows, [first], [first])
    assert plan["narrowing_view_ids"] == [first]
    assert plan["parent_view_ids"] == [first]
    assert plan["second_look_view_ids"] == []
    assert plan["audit"]["retained_selected_narrowing_view_ids"] == [first]
    assert plan["audit"]["pending_automatic_view_ids"] == [second]
    assert plan["audit"]["original_narrowing_view_ids"] == [first]


def test_human_not_rewritten_even_if_original_review_requested_narrowing():
    rows = [anchor("human", origin="human"), anchor("model")]
    views = build_fact_views(rows)
    manual = [view["id"] for view in views if view["evidence_id"] == "human"]
    plan = plan_recall_recovery(rows, [], manual)
    assert not set(manual) & set(plan["parent_view_ids"])
    assert plan["audit"]["excluded_manual_narrowing_view_ids"] == sorted(manual)
    assert plan["audit"]["counts"]["second_look_parents"] == 1
    assert plan_recall_recovery([rows[0]], [], [])['parent_view_ids'] == []


def test_fixed_quote_source_binding_and_projection_never_cross_paragraphs_or_rewrite():
    quote = '同一引文🙂\n"尚未开放"。'
    rows = [anchor("a", quote=quote, start=31), anchor("b", quote=quote, start=0)]
    before = deepcopy(rows)
    plan = plan_recall_recovery(rows, [], [])
    assert len(plan["parent_view_ids"]) == 2
    projected = subset_evidence(rows, plan["parent_view_ids"])
    assert {row["paragraph_id"] for row in projected} == {"paragraph-a", "paragraph-b"}
    assert all(row["quote"] == quote for row in projected)
    by_view = {view["id"]: view for view in build_fact_views(rows)}
    for binding in plan["audit"]["parent_bindings"]:
        view = by_view[binding["parent_fact_view_id"]]
        source = {key: view[key] for key in (
            "evidence_id", "paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote")}
        assert binding["source_binding_sha256"] == sha256(canonical(source).encode()).hexdigest()
        assert binding["quote_sha256"] == sha256(quote.encode()).hexdigest()
    assert rows == before


@pytest.mark.parametrize("mutate", [
    lambda row: row.update(quote="等长改写的引文。", end_cp=row["start_cp"] + len("等长改写的引文。")),
    lambda row: row.update(raw_doc_id="new-document"),
    lambda row: row.update(raw_para_id="new-paragraph"),
    lambda row: row.update(paragraph_id="new-internal-paragraph"),
    lambda row: row.update(start_cp=10, end_cp=10 + len(row["quote"])),
    lambda row: row.update(interpretation_origin="human"),
    lambda row: row.update(fact_des="另一个解释"),
])
def test_source_or_annotation_change_invalidates_candidate_and_plan_bindings(mutate):
    rows = [anchor()]
    old = plan_recall_recovery(rows, [], [])
    changed = deepcopy(rows)
    mutate(changed[0])
    new = plan_recall_recovery(changed, [], [])
    assert old["candidate_binding_sha256"] != new["candidate_binding_sha256"]
    assert old["plan_id"] != new["plan_id"]
    if set(old["parent_view_ids"]) - set(view_ids(changed)):
        with pytest.raises(RecallRecoveryInvalid):
            plan_recall_recovery(changed, [], old["parent_view_ids"])


def test_empty_and_all_selected_plans_accept_no_fact_and_do_not_schedule_work():
    empty = plan_recall_recovery([], [], [])
    assert empty["parent_view_ids"] == []
    assert empty["audit"]["counts"]["candidate_fact_views"] == 0
    rows = [anchor()]
    plan = plan_recall_recovery(rows, view_ids(rows), [])
    assert plan["parent_view_ids"] == []
    assert plan["audit"]["pending_automatic_view_ids"] == []
    assert not {"verified_evidence", "selected_evidence_ids", "accepted_fact_view_ids"} & plan.keys()


def test_explicit_selected_incomplete_view_is_scheduled_even_without_any_rejected_peer():
    rows = [anchor(facts=["原文支持但遗漏另一个关键属性的解释"])]
    identity = view_ids(rows)[0]
    plan = plan_recall_recovery(rows, [identity], [identity])
    assert plan["parent_view_ids"] == plan["narrowing_view_ids"] == [identity]
    assert plan["second_look_view_ids"] == []
    assert plan["audit"]["pending_automatic_view_ids"] == []
    assert plan["audit"]["covered_automatic_anchor_ids"] == ["a"]
    assert plan["audit"]["counts"]["retained_selected_narrowing_parents"] == 1


def test_legacy_absent_origin_is_automatic_and_still_bound_as_model():
    rows = [anchor()]
    explicit = plan_recall_recovery(rows, [], [])
    del rows[0]["interpretation_origin"]
    assert plan_recall_recovery(rows, [], []) == explicit


@pytest.mark.parametrize("selected,narrowing", [
    (["unknown"], []), ([], ["unknown"]), (None, []), ([], None),
    ({}, []), ([], {}), (("unknown",), []), ([], "unknown"),
    ([None], []), ([], [1]), ([{}], []), ([], [[]]), ([""], []),
])
def test_invalid_id_inputs_consistently_raise_fixed_error(selected, narrowing):
    with pytest.raises(RecallRecoveryInvalid, match="^recall_recovery_input_invalid$") as caught:
        plan_recall_recovery([anchor()], selected, narrowing)
    assert caught.value.check == "recall_recovery_input_invalid"


@pytest.mark.parametrize("field", ["selected", "narrowing"])
def test_duplicate_ids_rejected_not_silently_collapsed(field):
    rows = [anchor()]
    identity = view_ids(rows)[0]
    args = ([identity, identity], []) if field == "selected" else ([], [identity, identity])
    with pytest.raises(RecallRecoveryInvalid):
        plan_recall_recovery(rows, *args)


@pytest.mark.parametrize("bad", [
    None, {}, [None], [{"id": "x"}],
    [anchor(), anchor()],
])
def test_invalid_evidence_consistently_raises_fixed_error(bad):
    with pytest.raises(RecallRecoveryInvalid, match="^recall_recovery_input_invalid$"):
        plan_recall_recovery(bad, [], [])


@pytest.mark.parametrize("change", [
    {"quote": ""}, {"start_cp": True}, {"end_cp": 1},
    {"interpretation_origin": "unknown"}, {"interpretation_origin": None},
    {"interpretation_origin": []}, {"fact_des": None}, {"entities": [None]},
    {"additional_interpretations": {}}, {"quote": "\ud800", "end_cp": 10},
])
def test_malformed_fields_do_not_leak_type_unicode_or_source_errors(change):
    row = anchor()
    row.update(change)
    with pytest.raises(RecallRecoveryInvalid, match="^recall_recovery_input_invalid$"):
        plan_recall_recovery([row], [], [])
