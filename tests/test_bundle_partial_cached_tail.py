"""Exact paid A/B may be reused, but never invent a cached arbitration."""
from copy import deepcopy
import json

import pytest

from scripts.capture_bundle_workbench_real import save
from scripts.score_bundle_workbench_real import _remaining_usage_records, known_usage
import scripts.score_bundle_workbench_real as scorer
from scripts.score_workbench_real_episode import audit_cached_requests


def cached_pair(tmp_path):
    parents, cached = [], []
    for number, operation in enumerate(("authority_analyst", "cross_validation_analyst"), 1):
        wire = {"messages": [{"role": "user", "content": operation}], "tools": [{"function": {"name": "Shape"}}]}
        args = {"value": operation}
        call = {"id": "call", "type": "function", "function": {"name": "Shape", "arguments": json.dumps(args)}}
        response = {"tool_calls_wire": [call], "tool_calls": [{"id": "call", "name": "Shape", "args": args, "type": "tool_call"}],
            "finish_reason": "tool_calls"}
        parent = {"wire": save(tmp_path / f"request-{number:02}-wire.json", wire),
            "response": save(tmp_path / f"request-{number:02}-response.json", response),
            "preflight": save(tmp_path / f"request-{number:02}-preflight.json", {"operation": operation, "admitted": True}),
            "finished": save(tmp_path / f"request-{number:02}-finished.json", {"request_started": True})}
        parents.append(parent)
        cached.append({"operation": operation, "wire": save(tmp_path / f"cache-{number:02}-wire.json", wire),
            "response": save(tmp_path / f"cache-{number:02}-response.json", response),
            "finished": save(tmp_path / f"cache-{number:02}-finished.json", {"operation": operation, "request_started": False,
                "cache_hit": True, "outcome": "success", "unknown_usage_requests": 0}),
            "cache_hit": save(tmp_path / f"cache-{number:02}-hit.json", {"parent_run_id": "paid", "parent_request_ordinal": number,
                "no_provider_request_started": True, "exact_request_JSON_values_equal": True,
                "parent_wire_sha256": parent["wire"]["sha256"], "parent_response_sha256": parent["response"]["sha256"]})})
    return cached, {"paid": {"requests": parents}}


@pytest.mark.parametrize("tamper", [None, "unapproved_partial", "missing_B", "duplicate_A", "changed_wire", "wrong_parent"])
def test_partial_cache_is_explicit_exact_AB_and_no_cached_arb(tmp_path, tamper):
    rows, parents = cached_pair(tmp_path)
    if tamper == "missing_B":
        rows.pop()
    elif tamper == "duplicate_A":
        rows[1] = deepcopy(rows[0])
    elif tamper == "changed_wire":
        rows[0]["wire"] = save(tmp_path / "changed-wire.json", {"messages": ["changed"]})
    elif tamper == "wrong_parent":
        parents = {"unrelated": parents["paid"]}
    kwargs = {"allow_bundle_tail": True, "allow_bundle_partial_analyses": tamper != "unapproved_partial"}
    if tamper:
        with pytest.raises(ValueError):
            audit_cached_requests(rows, parents, [], **kwargs)
    else:
        _, _, operations, _, audit = audit_cached_requests(rows, parents, [], **kwargs)
        assert operations == ["authority_analyst", "cross_validation_analyst"]
        assert all(row["new_cost"] is False and row["new_provider_request"] is False for row in audit)
        assert "arbitrate" not in operations


def test_source_owner_may_have_paid_AB_without_dropping_their_cost():
    def row(operation, count):
        return {"operation": operation, "input_tokens": count - 1, "output_tokens": 1,
            "total_tokens": count, "usage_source": "provider"}
    source, a, b = row("strict_selection", 8721), row("authority_analyst", 6845), row("cross_validation_analyst", 6824)
    rest = _remaining_usage_records([source, a, b], [source])
    assert rest == [a, b]
    assert known_usage([source, *rest])["total_tokens"] == 22390
    assert len([source, *rest]) == 3
    with pytest.raises(ValueError, match="receipt_not_in_actual"):
        _remaining_usage_records([source, a, b], [source, source])


@pytest.mark.parametrize("tamper", [None, "historical_actual", "actual_wire", "prior_reset", "missing_phase", "preflight_estimate"])
def test_cached_AB_still_requires_current_exact_arb_forecast_and_two_historical_phases(tmp_path, tamper):
    from app.agents.episode_budget import POLICY, plan_source_episode, plan_tail, planning_sources
    from app.agents.strict_transport import LIMITS
    from app.domain.lc_bundle_case import prepare_lc_case
    from test_source_bundle_tool_bridge import raw_case
    raw = raw_case()
    prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    common = {"threshold": 45000, "max_calls": 4, "model_name": "deepseek-v4-pro",
        "output_limits": {**LIMITS, "strict_selection": 2048}, "compact_analyses": True,
        "counter": lambda texts: [10] * len(texts)}
    evidence = planning_sources(prepared, counter=common["counter"])
    analyses = [{"role": role, "evidence_snapshot_id": "mock", "decision": "undetermined", "conclusion": "待确认。",
        "judgments": [], "supporting_evidence_ids": [], "conflicting_evidence_ids": []}
        for role in ("authority_analyst", "cross_validation_analyst")]
    source = save(tmp_path / "old/budget-before-first-paid.json", plan_source_episode(raw["question"], prepared,
        known_tokens=0, known_calls=0, **common))
    selected = save(tmp_path / "old/budget-selected-sources.json", plan_tail(raw["question"], evidence,
        known_tokens=7, known_calls=1, **common))
    actual_value = plan_tail(raw["question"], evidence, analyses=analyses, known_tokens=21, known_calls=3, **common)
    actual_plan = save(tmp_path / "current/budget-actual-analyses.json", actual_value)
    wire = actual_value["wire_bodies"][0]
    item = {"wire": save(tmp_path / "current/request-03-wire.json", wire),
        "actual-wire": save(tmp_path / "current/request-03-actual-wire.json", wire)}
    pre = {"operation": "arbitrate", "admitted": True, "wire": item["wire"], "actual_wire": item["actual-wire"],
        "known_tokens_before": 21, "experimental_stop_threshold": 45000,
        "serialized_json_token_estimate": 10, "reserved_output_tokens": LIMITS["arbitrate"],
        "safety_tokens": POLICY["safety_per_request"]}
    if tamper == "preflight_estimate":
        pre["serialized_json_token_estimate"] += 1
    item["preflight"] = save(tmp_path / "current/request-03-preflight.json", pre)
    owner = save(tmp_path / "old/source-bundle-episode.json", {"whole_token_stop": 45000, "max_total_requests": 4,
        "source_bundle_config": {"whole_chain_planning": POLICY}})
    native = {"front": {}, "tail": {"budget_forecasts": [actual_plan], "requests": [item],
        "cached_requests": [{"operation": "authority_analyst"}, {"operation": "cross_validation_analyst"}]},
        "ancestor_budget_forecasts": [source, selected],
        "ancestor_budget_forecast_metadata": {source["file"]: owner, selected["file"]: owner}}
    metadata = {"source_bundle_config": {"whole_chain_planning": POLICY, "sealed_front_recovery": {"origin_run_id": "old"}}}
    freeze = {"engineering_episode_stop": 45000, "max_total_requests": 4,
        "prior_known_tokens": 21, "prior_actual_requests": 3}
    if tamper == "historical_actual":
        native["tail"]["budget_forecasts"] = []
        native["ancestor_budget_forecasts"].append(actual_plan)
        native["ancestor_budget_forecast_metadata"][actual_plan["file"]] = owner
    elif tamper == "actual_wire":
        item["actual-wire"] = save(tmp_path / "current/wrong-actual-wire.json", {"changed": True})
    elif tamper == "prior_reset":
        freeze["prior_known_tokens"] = 0
    elif tamper == "missing_phase":
        native["ancestor_budget_forecasts"].remove(selected)
    if tamper:
        with pytest.raises(ValueError):
            scorer._budget_forecasts(native, metadata, freeze, [])
    else:
        audit = scorer._budget_forecasts(native, metadata, freeze, [])
        assert len(audit) == 3
        assert next(row for row in audit if row["phase"] == "actual_analyses")["current_actual_arbitration_wire_verified"] is True
