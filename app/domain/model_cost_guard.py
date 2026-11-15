"""Local admission ledger for one deliberately small model experiment.

The threshold is a soft *known-token* threshold. Input tokens are unknown before
the response: UTF-8 message bytes are a size proxy, never token or price units.
The caller must serialize admission through response settlement, persist the
preflight receipt, and only then mark the actual SDK invocation.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from threading import RLock


INITIAL_OUTPUT_LIMITS = {
    "paragraph_queries": 8192, "paragraph_rank": 8192,
    "paragraph_annotate": 8192, "paragraph_fields": 8192,
    "authority_analyst": 8192, "cross_validation_analyst": 8192,
    "arbitrate": 16384,
}
BYTE_PROXY_SCOPE = "canonical_final_message_roles_and_text_utf8_not_wire_tokens_or_price"


class CostGuardStopped(RuntimeError):
    def __init__(self, code, details=None):
        self.code = code
        self.details = deepcopy(details or {})
        super().__init__(code)


@dataclass(frozen=True)
class CostGuardPolicy:
    soft_total_tokens: int = 60000
    max_requests: int = 10
    max_message_bytes: int = 48000
    max_output_tokens: int = 32768
    max_length_recoveries: int = 1
    max_structure_repairs: int = 1
    initial_output_limits: dict = field(default_factory=lambda: dict(INITIAL_OUTPUT_LIMITS))

    def __post_init__(self):
        for name in ("soft_total_tokens", "max_requests", "max_message_bytes", "max_output_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("invalid_cost_guard_policy")
        for name in ("max_length_recoveries", "max_structure_repairs"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 1:
                raise ValueError("invalid_cost_guard_recovery_policy")
        if (not isinstance(self.initial_output_limits, dict) or not self.initial_output_limits
                or any(not isinstance(k, str) or not k or type(v) is not int or not 1 <= v <= self.max_output_tokens
                       for k, v in self.initial_output_limits.items())):
            raise ValueError("invalid_cost_guard_output_limits")
        object.__setattr__(self, "initial_output_limits", deepcopy(self.initial_output_limits))


def message_size_proxy(messages):
    """Measure the actual final message text, including instructions and schema.

    No text is returned or logged. This excludes SDK/network framing and is not
    an upper bound for provider input tokens or monetary cost.
    """
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("invalid_final_messages")
    rows = []
    for item in messages:
        if isinstance(item, dict):
            role, content = item.get("role", item.get("type")), item.get("content")
        else:
            role, content = getattr(item, "type", None), getattr(item, "content", None)
        if role not in {"system", "human", "user", "assistant", "ai"} or not isinstance(content, str):
            raise ValueError("only_final_text_messages_are_supported")
        rows.append({"role": role, "content": content})
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"message_content_utf8_bytes": len(raw), "message_content_sha256": hashlib.sha256(raw).hexdigest(),
            "input_byte_proxy_scope": BYTE_PROXY_SCOPE, "input_tokens_estimate": None}


class ModelCostGuard:
    """Fail-closed synchronous ledger; no model, I/O or implicit retry here."""
    def __init__(self, policy=None):
        self.policy = policy or CostGuardPolicy()
        self._initial_output_limits = deepcopy(self.policy.initial_output_limits)
        self._lock = RLock()
        self._pending = None
        self._sequence = 0
        self._known_tokens = 0
        self._started = 0
        self._length_recoveries = 0
        self._structure_repairs = 0
        self._unknown_usage_requests = 0
        self._partial_reported_total_tokens = 0
        self._stop_reason = None
        self._records = []
        self._eligible_length = None
        self._stage_floors = {}
        self._seen_inputs = set()

    def _refuse(self, code, *, latch=True):
        if latch and self._stop_reason is None:
            self._stop_reason = code
        raise CostGuardStopped(code, {"known_total_tokens": self._known_tokens, "actual_requests_started": self._started,
                                     "soft_threshold_is_not_a_hard_total_or_cost_cap": True})

    def prepare(self, operation, messages, requested_max_output_tokens, *, structure_repair=False, context_sha256=None):
        with self._lock:
            if self._stop_reason:
                self._refuse("cost_guard_already_stopped")
            if self._pending is not None:
                self._refuse("cost_guard_single_inflight_required", latch=False)
            if type(structure_repair) is not bool or operation not in self._initial_output_limits:
                self._refuse("cost_guard_operation_not_authorized")
            try:
                proxy = message_size_proxy(messages)
            except (TypeError, ValueError):
                self._refuse("cost_guard_final_messages_invalid")
            if proxy["message_content_utf8_bytes"] > self.policy.max_message_bytes:
                self._refuse("cost_guard_message_bytes_limit")
            if context_sha256 is None:
                context_sha256 = proxy["message_content_sha256"]
            if (not isinstance(context_sha256, str) or len(context_sha256) != 64
                    or any(char not in "0123456789abcdef" for char in context_sha256)):
                self._refuse("cost_guard_context_binding_invalid")
            initial = self._initial_output_limits[operation]
            limit = requested_max_output_tokens
            if type(limit) is not int or not initial <= limit <= self.policy.max_output_tokens:
                self._refuse("cost_guard_output_capacity_not_authorized")
            key = (operation, proxy["message_content_sha256"])
            prior = self._eligible_length
            length_recovery = bool(prior and prior["key"] == key and limit == min(prior["max_output_tokens"] * 2, self.policy.max_output_tokens))
            floor = self._stage_floors.get(operation, initial)
            if limit > initial and not length_recovery and limit != floor:
                self._refuse("cost_guard_unproved_capacity_increase")
            if key in self._seen_inputs and not length_recovery:
                self._refuse("cost_guard_duplicate_request_not_authorized")
            if length_recovery and self._length_recoveries >= self.policy.max_length_recoveries:
                self._refuse("cost_guard_length_recovery_exhausted")
            if structure_repair and (length_recovery or self._structure_repairs >= self.policy.max_structure_repairs):
                self._refuse("cost_guard_structure_repair_exhausted")
            if structure_repair and not any(record["operation"] == operation and record.get("context_sha256") == context_sha256
                    and record.get("usage_known") is True and record.get("outcome") in {"success", "invalid_output"}
                    for record in self._records):
                self._refuse("cost_guard_structure_repair_context_unproved")
            if self._started >= self.policy.max_requests:
                self._refuse("cost_guard_request_limit")
            # Never shrink a complete requested response to fit the remainder.
            # Provider input tokens remain unknown until this request returns.
            if self._known_tokens + limit > self.policy.soft_total_tokens:
                self._refuse("cost_guard_output_reservation_exceeds_remaining_soft_budget")
            self._sequence += 1
            ticket = {"ticket_id": self._sequence, "operation": operation, **proxy,
                "context_sha256": context_sha256,
                "requested_max_output_tokens": limit, "known_tokens_before": self._known_tokens,
                "reserved_max_output_tokens": limit, "known_plus_reserved_output_tokens": self._known_tokens + limit,
                "soft_total_token_threshold": self.policy.soft_total_tokens,
                "soft_threshold_is_not_a_hard_total_or_cost_cap": True,
                "structure_repair": structure_repair, "length_recovery": length_recovery,
                "capacity_origin": "length_recovery" if length_recovery else ("learned_stage_floor" if limit > initial else "initial_stage_capacity"),
                "actual_request_started": False, "is_usage": False}
            self._pending = deepcopy(ticket)
            return deepcopy(ticket)

    def _current(self, ticket):
        if not isinstance(ticket, dict) or self._pending is None or ticket != self._pending:
            self._refuse("cost_guard_ticket_mismatch")
        return self._pending

    def mark_invoked(self, ticket):
        with self._lock:
            if self._stop_reason:
                self._refuse("cost_guard_already_stopped")
            row = self._current(ticket)
            if row["actual_request_started"]:
                self._refuse("cost_guard_ticket_already_invoked")
            row["actual_request_started"] = True
            ticket["actual_request_started"] = True
            self._started += 1
            self._length_recoveries += int(row["length_recovery"])
            self._structure_repairs += int(row["structure_repair"])
            self._seen_inputs.add((row["operation"], row["message_content_sha256"]))
            self._eligible_length = None
            return deepcopy(row)

    def abort_before_invocation(self, ticket, reason="cost_guard_preflight_journal_failed"):
        with self._lock:
            row = self._current(ticket)
            if row["actual_request_started"]:
                self._refuse("cost_guard_cannot_abort_invoked_request")
            self._records.append({**deepcopy(row), "outcome": "not_invoked", "known_total_tokens": None, "usage_known": None})
            self._pending = None
            self._stop_reason = reason if reason in {"cost_guard_preflight_journal_failed", "cost_guard_preflight_cancelled"} else "cost_guard_preflight_failed"

    def settle(self, ticket, usage, *, outcome="success", finish_reason=None, provider_http_status=None):
        with self._lock:
            row = self._current(ticket)
            if not row["actual_request_started"]:
                self._refuse("cost_guard_cannot_settle_uninvoked_request")
            if outcome not in {"success", "invalid_output", "length", "transport_failure", "cancelled", "filtered"}:
                self._refuse("cost_guard_invalid_outcome")
            valid = isinstance(usage, dict)
            known = {key: (usage.get(key) if valid and type(usage.get(key)) is int and usage[key] >= 0 else None)
                     for key in ("input_tokens", "output_tokens", "total_tokens")}
            all_numeric = all(value is not None for value in known.values())
            inconsistent = all_numeric and known["input_tokens"] + known["output_tokens"] != known["total_tokens"]
            complete = all_numeric and not inconsistent
            total_basis = ("inconsistent_full_usage_not_credited" if inconsistent else
                ("provider_reported_consistent_full_total" if complete else
                 ("provider_reported_partial_total" if known["total_tokens"] is not None else "unknown_total")))
            credited_total = known["total_tokens"] if not inconsistent else None
            if credited_total is not None:
                self._known_tokens += known["total_tokens"]
                if not complete:
                    self._partial_reported_total_tokens += known["total_tokens"]
            status = provider_http_status if type(provider_http_status) is int and 100 <= provider_http_status <= 599 else None
            if not complete:
                self._unknown_usage_requests += 1
                self._stop_reason = "cost_guard_usage_unknown"
            elif outcome in {"transport_failure", "cancelled", "filtered"} or (status is not None and status >= 400):
                self._stop_reason = "cost_guard_provider_or_inflight_failure"
            elif outcome == "length":
                if finish_reason != "length":
                    self._stop_reason = "cost_guard_length_not_confirmed"
                else:
                    self._eligible_length = {"key": (row["operation"], row["message_content_sha256"]),
                                             "max_output_tokens": row["requested_max_output_tokens"]}
            elif row["length_recovery"]:
                self._stage_floors[row["operation"]] = row["requested_max_output_tokens"]
            if self._known_tokens >= self.policy.soft_total_tokens and self._stop_reason is None:
                self._stop_reason = "cost_guard_known_token_soft_threshold_reached"
            record = {**deepcopy(row), **known, "usage_known": complete, "outcome": outcome,
                "credited_total_tokens": credited_total, "usage_total_basis": total_basis,
                "provider_http_status": status, "known_total_tokens_after": self._known_tokens,
                "stop_reason": self._stop_reason, "is_usage": True}
            self._records.append(record)
            self._pending = None
            return deepcopy(record)

    def stop_after_audit_failure(self):
        with self._lock:
            self._stop_reason = "cost_guard_settlement_journal_failed"

    def report(self):
        with self._lock:
            return {"known_total_tokens": self._known_tokens, "actual_requests_started": self._started,
                "length_recoveries": self._length_recoveries, "structure_repairs": self._structure_repairs,
                "unknown_usage_requests": self._unknown_usage_requests, "stop_reason": self._stop_reason,
                "partial_provider_reported_total_tokens": self._partial_reported_total_tokens,
                "pending_request": deepcopy(self._pending), "records": deepcopy(self._records),
                "soft_threshold_is_not_a_hard_total_or_cost_cap": True, "input_bytes_are_not_tokens_or_price": True}
