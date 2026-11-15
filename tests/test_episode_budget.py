"""Budget forecasts use native inputs and never masquerade as real usage."""
from copy import deepcopy
import asyncio
import json

import pytest

from app.agents.episode_budget import plan_tail, plan_source_episode, planning_sources, POLICY
from app.agents.episode_budget import PARAGRAPH_QUALITY_POLICY
from app.agents.strict_transport import LIMITS
from app.domain.lc_bundle_case import prepare_lc_case
from test_source_bundle_tool_bridge import raw_case


def options(**overrides):
    return {"known_tokens": 1000, "known_calls": 1, "threshold": 45000, "max_calls": 5,
            "model_name": "deepseek-v4-pro", "output_limits": {**LIMITS, "strict_selection": 2048},
            "compact_analyses": True, **overrides}


def evidence():
    return [{"id": "evidence-one", "paragraph_id": "paragraph-one", "quote": "原文全文。仍未官方确认。",
             "fact_des": "原文全文。", "raw_doc_id": "doc_x", "raw_para_id": "doc_x#p1",
             "grade_score": 0.9, "source_group_id": "group-x"}]


def test_tail_reserves_both_independent_calls_and_future_analyses_without_lowering_limits():
    rows = evidence()
    plan = plan_tail({"question": "程序是否完成？"}, rows, counter=lambda texts: [100, 100, 100], **options())
    assert plan["required_remaining_planning_tokens"] == 300 + 4096 + 4096 + 8192 + 1536 + 10240
    assert plan["known_tokens"] == 1000 and plan["planned_whole_tokens"] == 1000 + plan["required_remaining_planning_tokens"]
    assert plan["remaining_requests"] == 3 and plan["admitted"]
    assert plan["is_actual_usage"] is False and plan["is_guaranteed_global_upper_bound"] is False
    for body in plan["wire_bodies"]:
        assert json.loads(body["messages"][-1]["content"])["evidence"][0]["quote"] == rows[0]["quote"]
    assert [body["max_tokens"] for body in plan["wire_bodies"]] == [4096, 4096, 8192]


@pytest.mark.parametrize("overrides", [{"threshold": 20000}, {"max_calls": 3}])
def test_reject_before_start_when_whole_tail_or_required_calls_do_not_fit(overrides):
    plan = plan_tail({"question": "程序是否完成？"}, evidence(), counter=lambda texts: [100] * len(texts), **options(**overrides))
    assert plan["admitted"] is False


def test_future_analysis_envelope_requires_lossless_short_ids():
    with pytest.raises(ValueError, match="requires_compact"):
        plan_tail({}, [], counter=lambda texts: [], **options(compact_analyses=False))


def test_actual_analysis_gate_uses_complete_real_structure_and_only_reserves_arbitration():
    analyses = [{"role": role, "evidence_snapshot_id": "snapshot-x", "decision": "undetermined",
        "conclusion": "尚未完成，保留有限事实。", "judgments": [{"id": "uuid-" + role,
            "statement": "程序尚未完成。", "evidence_ids": ["evidence-one"], "prerequisite_judgment_ids": []}],
        "supporting_evidence_ids": ["evidence-one"], "conflicting_evidence_ids": []}
        for role in ("authority_analyst", "cross_validation_analyst")]
    before = deepcopy(analyses)
    plan = plan_tail({"question": "程序是否完成？"}, evidence(), analyses=analyses,
        counter=lambda texts: [1234], **options(known_tokens=24000, known_calls=3, max_calls=4))
    assert plan["remaining_requests"] == 1 and plan["future_analysis_wire_allowance"] == 0
    assert plan["planned_whole_tokens"] == 24000 + 1234 + 8192 + 512
    payload = json.loads(plan["wire_bodies"][0]["messages"][-1]["content"])
    assert [row["judgments"][0]["id"] for row in payload["analyses"]] == ["A1", "B1"]
    assert all(row["judgments"][0]["statement"] == "程序尚未完成。" for row in payload["analyses"])
    assert analyses == before


def test_before_first_paid_uses_source_schema_and_complete_native_book():
    raw = raw_case()
    prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    before = deepcopy(prepared)
    plan = plan_source_episode(raw["question"], prepared,
        counter=lambda texts: [len(t.encode("utf8")) for t in texts], **options(threshold=1))
    assert not plan["admitted"] and plan["remaining_requests"] == 4
    assert plan["front_request"]["reserved_output_tokens"] == 2048
    assert plan["wire_bodies"][0]["tools"][0]["function"]["name"] == "LCSourceBundleRankedIDsV1"
    assert json.loads(plan["wire_bodies"][0]["messages"][-1]["content"]) == prepared["model_input"]
    assert prepared == before
    selected = planning_sources(prepared, counter=lambda texts: [len(t) for t in texts])
    original_quotes = {row["source_bundle"]["quote"] for row in prepared["catalog"]}
    assert all(row["quote"] in original_quotes for row in selected)
    assert len({row["paragraph_id"] for row in selected}) == len(selected) <= 10


def test_long_source_is_retained_in_size_envelope_instead_of_silently_shortened():
    catalog = {"catalog": [{"source_window_id": "P1", "source_bundle": {
        "paragraph_id": "one", "quote": "整段。" + "很长的完整原文" * 2000,
        "raw_doc_id": "doc_x", "raw_para_id": "doc_x#p1", "document_id": "doc-one"}}]}
    selected = planning_sources(catalog, counter=lambda texts: [len(t) for t in texts])
    assert selected[0]["quote"] == catalog["catalog"][0]["source_bundle"]["quote"]


def test_paragraph_quality_source_envelope_keeps_every_selectable_anchor_not_top_ten():
    catalog = {"paragraph_quality": True, "catalog": [{"source_window_id": f"P{i}", "source_bundle": {
        "paragraph_id": f"parent-{i}", "quote": "完整原段。" + "上下文" * i,
        "raw_doc_id": "doc_x", "raw_para_id": f"opaque-{i}", "document_id": "doc-one"}}
        for i in range(17)]}
    original = deepcopy(catalog)
    selected = planning_sources(catalog, counter=lambda texts: [len(t) for t in texts])
    assert len(selected) == 17 and {r["id"] for r in selected} == {f"P{i}" for i in range(17)}
    assert catalog == original
    plan = plan_tail({}, selected, counter=lambda texts: [100] * len(texts),
        planning_policy=PARAGRAPH_QUALITY_POLICY, **options(threshold=80000))
    assert plan["policy"]["selected_anchor_cap"] is None
    assert plan["source_count"] == 17 and plan["is_guaranteed_global_upper_bound"] is False
    assert plan["policy"]["source_quote_envelope_is_not_an_annotation_or_future_analysis_bound"] is True
    assert all(len(json.loads(body["messages"][-1]["content"])["evidence"]) == 17
               for body in plan["wire_bodies"])


def test_production_whole_chain_gate_blocks_before_any_provider_call(monkeypatch, tmp_path):
    from app.agents import episode_budget
    from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
    from app.agents.provider import ModelOutputInvalid
    from test_lc_bundle_transport import config, inputs
    adapter = LCBundleWorkbenchProvider(config(tmp_path, source_bundle_selection_only=True,
        source_bundle_graded_selection=True, source_bundle_compact_analyses_enabled=True))
    raw, question = inputs()
    original = episode_budget.plan_source_episode
    monkeypatch.setattr(episode_budget, "plan_source_episode", lambda *args, **kwargs:
        original(*args, **kwargs, counter=lambda texts: [100000] * len(texts)))
    async def forbidden(*args, **kwargs):
        raise AssertionError("forecast rejection must happen before provider entry")
    monkeypatch.setattr(adapter, "invoke_strict", forbidden)
    with pytest.raises(ModelOutputInvalid) as error:
        asyncio.run(adapter.select_source_bundles(question, raw["documents"], raw["paragraphs"]))
    assert error.value.code == "experiment_usage_stop"
    assert adapter.front_provider is None and adapter.attempt_index == 0
    receipt = json.loads((adapter.directory / "budget-before-first-paid.json").read_text("utf8"))
    assert not receipt["admitted"] and receipt["known_tokens"] == 0


def test_resumed_budget_forecast_counts_paid_ancestors_before_first_tail_request(tmp_path):
    from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
    from test_lc_bundle_transport import config, parents
    adapter = LCBundleWorkbenchProvider(config(tmp_path, source_bundle_selection_only=True,
        source_bundle_graded_selection=True, source_bundle_compact_analyses_enabled=True))
    records = parents()
    adapter.set_previous_usage(records)
    assert adapter.prior_known_tokens == 0
    arguments = adapter._tail_budget_arguments()
    assert arguments["known_tokens"] == sum(row["total_tokens"] for row in records)
    assert arguments["known_calls"] == len(records)
    assert adapter._tail_seeded and adapter.attempt_index == 0
