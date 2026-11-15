"""No provider/network: budget admissions, cancellation, recovery and exact bytes."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest

from app.domain.model_cost_guard import CostGuardPolicy, CostGuardStopped, ModelCostGuard, message_size_proxy


def messages(text="原始题目和证据。"):
    return [{"role": "system", "content": "边界规则+JSON Schema"}, {"role": "human", "content": text}]


def admit(guard, text="a", operation="paragraph_rank", capacity=8192, **kwargs):
    return guard.prepare(operation, messages(text), capacity, **kwargs)


def finish(guard, ticket, total=10, **kwargs):
    guard.mark_invoked(ticket)
    return guard.settle(ticket, {"input_tokens": total - 1, "output_tokens": 1, "total_tokens": total}, **kwargs)


def test_final_message_bytes_include_system_schema_unicode_and_do_not_claim_tokens():
    value = messages("🦊真实原文")
    result = message_size_proxy(value)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert result["message_content_utf8_bytes"] == len(raw)
    assert result["input_tokens_estimate"] is None
    assert "not_wire_tokens_or_price" in result["input_byte_proxy_scope"]
    assert "真实原文" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("invalid", [None, [], [{"role": "tool", "content": "x"}], [{"role": "human", "content": []}]])
def test_invalid_messages_fail_before_actual_request(invalid):
    guard = ModelCostGuard()
    with pytest.raises(CostGuardStopped, match="messages_invalid"):
        guard.prepare("paragraph_rank", invalid, 8192)
    assert guard.report()["actual_requests_started"] == 0


def test_actual_byte_ceiling_rejects_before_request_and_preserves_full_capacity():
    guard = ModelCostGuard()
    with pytest.raises(CostGuardStopped, match="message_bytes_limit"):
        admit(guard, "中" * 16000)
    assert guard.report()["actual_requests_started"] == 0
    assert guard.policy.initial_output_limits["paragraph_rank"] == 8192


def test_request_count_counts_actual_invocations_not_preflight_and_stops_eleven():
    guard = ModelCostGuard()
    for index in range(10):
        ticket = admit(guard, str(index))
        assert guard.report()["actual_requests_started"] == index
        finish(guard, ticket)
    with pytest.raises(CostGuardStopped, match="request_limit"):
        admit(guard, "eleven")
    assert guard.report()["actual_requests_started"] == 10


@pytest.mark.parametrize("usage,known", [
    (None, 0), ({"input_tokens": 5, "output_tokens": None, "total_tokens": 12}, 12),
    ({"input_tokens": True, "output_tokens": 1, "total_tokens": 2}, 2),
    ({"input_tokens": 5, "output_tokens": 4, "total_tokens": -1}, 0),
    ({"input_tokens": 5, "output_tokens": 4, "total_tokens": 10}, 0),
])
def test_unknown_partial_bool_negative_or_inconsistent_usage_latches_without_inventing_zero(usage, known):
    guard = ModelCostGuard();ticket = admit(guard)
    guard.mark_invoked(ticket)
    row = guard.settle(ticket, usage)
    assert row["usage_known"] is False
    assert guard.report()["known_total_tokens"] == known
    assert guard.report()["unknown_usage_requests"] == 1
    with pytest.raises(CostGuardStopped, match="already_stopped"):
        admit(guard, "queued-AA", operation="authority_analyst")
    assert guard.report()["actual_requests_started"] == 1


def test_contradictory_full_usage_preserves_reported_values_but_does_not_credit_claimed_total():
    guard = ModelCostGuard();ticket = admit(guard);guard.mark_invoked(ticket)
    row = guard.settle(ticket, {"input_tokens": 5, "output_tokens": 4, "total_tokens": 10})
    assert (row["input_tokens"], row["output_tokens"], row["total_tokens"]) == (5, 4, 10)
    assert row["credited_total_tokens"] is None
    assert row["usage_total_basis"] == "inconsistent_full_usage_not_credited"
    assert guard.report()["known_total_tokens"] == 0
    assert guard.report()["stop_reason"] == "cost_guard_usage_unknown"
    other = ModelCostGuard();ticket = admit(other);other.mark_invoked(ticket)
    partial = other.settle(ticket, {"input_tokens": None, "output_tokens": 4, "total_tokens": 10})
    assert partial["usage_total_basis"] == "provider_reported_partial_total"
    assert other.report()["partial_provider_reported_total_tokens"] == 10
    with pytest.raises(CostGuardStopped):admit(other, "never-after-unknown")


@pytest.mark.parametrize("status", [402, 429, 504, None])
def test_transport_timeout_or_rate_failure_stops_later_queued_request_even_when_usage_known(status):
    guard = ModelCostGuard();ticket = admit(guard)
    finish(guard, ticket, outcome="transport_failure", provider_http_status=status)
    with pytest.raises(CostGuardStopped, match="already_stopped"):
        admit(guard, "queued-AA", operation="cross_validation_analyst")
    assert guard.report()["actual_requests_started"] == 1


def test_cancelled_inflight_keeps_paid_slot_and_unknown_usage_but_preflight_cancel_has_no_slot():
    guard = ModelCostGuard();ticket = admit(guard)
    guard.abort_before_invocation(ticket, "cost_guard_preflight_cancelled")
    assert guard.report()["actual_requests_started"] == 0
    other = ModelCostGuard();ticket = admit(other);other.mark_invoked(ticket)
    other.settle(ticket, None, outcome="cancelled")
    assert other.report()["actual_requests_started"] == other.report()["unknown_usage_requests"] == 1


def test_preflight_or_settlement_journal_failure_blocks_further_requests():
    guard = ModelCostGuard();ticket = admit(guard)
    guard.abort_before_invocation(ticket)
    assert guard.report()["actual_requests_started"] == 0
    with pytest.raises(CostGuardStopped, match="already_stopped"):
        admit(guard, "later")
    other = ModelCostGuard();finish(other, admit(other));other.stop_after_audit_failure()
    assert other.report()["known_total_tokens"] == 10
    with pytest.raises(CostGuardStopped, match="already_stopped"):
        admit(other, "later")


def test_stop_latched_during_preflight_cannot_be_bypassed_by_an_already_prepared_ticket():
    guard = ModelCostGuard();ticket = admit(guard)
    guard.stop_after_audit_failure()
    with pytest.raises(CostGuardStopped, match="already_stopped"):
        guard.mark_invoked(ticket)
    assert guard.report()["actual_requests_started"] == 0
    other = ModelCostGuard();ticket = admit(other);tampered = deepcopy(ticket)
    tampered["requested_max_output_tokens"] = 1
    with pytest.raises(CostGuardStopped, match="ticket_mismatch"):
        other.mark_invoked(tampered)
    with pytest.raises(CostGuardStopped, match="already_stopped"):
        other.mark_invoked(ticket)
    assert other.report()["actual_requests_started"] == 0


def test_successful_http_status_does_not_stop_known_complete_usage():
    guard = ModelCostGuard();finish(guard, admit(guard), provider_http_status=200)
    assert guard.report()["stop_reason"] is None
    finish(guard, admit(guard, "next"))


def test_output_reservation_refuses_remainder_instead_of_cutting_response_capacity():
    guard = ModelCostGuard();finish(guard, admit(guard), 52000)
    before = deepcopy(guard.policy.initial_output_limits)
    with pytest.raises(CostGuardStopped, match="reservation_exceeds"):
        admit(guard, "later")
    assert guard.policy.initial_output_limits == before
    assert guard.report()["actual_requests_started"] == 1


def test_soft_threshold_can_be_crossed_by_returned_input_and_is_not_a_hard_cost_claim():
    guard = ModelCostGuard();row = finish(guard, admit(guard), 61000)
    assert row["known_total_tokens_after"] == 61000
    assert guard.report()["stop_reason"] == "cost_guard_known_token_soft_threshold_reached"
    assert guard.report()["soft_threshold_is_not_a_hard_total_or_cost_cap"]
    with pytest.raises(CostGuardStopped):
        admit(guard, "later")


def test_one_same_input_confirmed_length_recovery_and_its_learned_floor_are_allowed():
    guard = ModelCostGuard();finish(guard, admit(guard), outcome="length", finish_reason="length")
    retry = admit(guard, capacity=16384)
    assert retry["length_recovery"] is True
    finish(guard, retry)
    next_batch = admit(guard, "new-original-batch", capacity=16384)
    assert next_batch["capacity_origin"] == "learned_stage_floor"
    finish(guard, next_batch)
    assert guard.report()["length_recoveries"] == 1


@pytest.mark.parametrize("change", ["different_input", "no_confirmed_length", "no_remaining_budget", "second_global_expansion"])
def test_capacity_recovery_cannot_use_unrelated_input_unproved_reason_missing_budget_or_second_retry(change):
    guard = ModelCostGuard()
    if change == "no_confirmed_length":
        finish(guard, admit(guard))
        with pytest.raises(CostGuardStopped):admit(guard, capacity=16384)
    elif change == "different_input":
        finish(guard, admit(guard), outcome="length", finish_reason="length")
        with pytest.raises(CostGuardStopped):admit(guard, "changed", capacity=16384)
    elif change == "no_remaining_budget":
        finish(guard, admit(guard), 50000, outcome="length", finish_reason="length")
        with pytest.raises(CostGuardStopped, match="reservation_exceeds"):admit(guard, capacity=16384)
    else:
        finish(guard, admit(guard), outcome="length", finish_reason="length")
        finish(guard, admit(guard, capacity=16384), outcome="length", finish_reason="length")
        with pytest.raises(CostGuardStopped, match="recovery_exhausted"):admit(guard, capacity=32768)


def test_structure_repair_is_bound_to_previously_called_context_and_global_once():
    guard = ModelCostGuard();binding = "a" * 64
    finish(guard, admit(guard, context_sha256=binding))
    repair = admit(guard, "same-data-plus-safe-diagnostics", structure_repair=True, context_sha256=binding)
    finish(guard, repair)
    with pytest.raises(CostGuardStopped, match="structure_repair_exhausted"):
        admit(guard, "second-diagnostics", structure_repair=True, context_sha256=binding)
    assert guard.report()["structure_repairs"] == 1


def test_known_schema_failure_remains_failed_output_and_can_receive_only_one_bound_repair():
    guard = ModelCostGuard();binding = "b" * 64
    failed = finish(guard, admit(guard, context_sha256=binding), outcome="invalid_output")
    assert failed["outcome"] == "invalid_output" and failed["usage_known"]
    assert guard.report()["stop_reason"] is None
    finish(guard, admit(guard, "safe-structure-diagnostics", context_sha256=binding, structure_repair=True))
    assert guard.report()["structure_repairs"] == 1


def test_unbound_structure_repair_and_duplicate_request_fail_before_sdk():
    guard = ModelCostGuard()
    with pytest.raises(CostGuardStopped, match="context_unproved"):
        admit(guard, structure_repair=True)
    other = ModelCostGuard();finish(other, admit(other))
    with pytest.raises(CostGuardStopped, match="duplicate_request"):
        admit(other)
    assert other.report()["actual_requests_started"] == 1


def test_concurrent_admissions_cannot_reserve_two_actual_calls_and_ticket_tamper_fails():
    guard = ModelCostGuard()
    def prepare(index):
        try:return admit(guard, str(index))
        except CostGuardStopped:return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        tickets = list(pool.map(prepare, range(2)))
    assert sum(ticket is not None for ticket in tickets) == 1
    ticket = next(ticket for ticket in tickets if ticket is not None)
    ticket["requested_max_output_tokens"] = 1
    with pytest.raises(CostGuardStopped, match="ticket_mismatch"):
        guard.mark_invoked(ticket)
    assert guard.report()["actual_requests_started"] == 0


def test_mutating_the_original_policy_map_does_not_authorize_a_new_operation():
    policy = CostGuardPolicy();guard = ModelCostGuard(policy)
    policy.initial_output_limits["unapproved"] = 1
    with pytest.raises(CostGuardStopped, match="operation_not_authorized"):
        admit(guard, operation="unapproved", capacity=1)
