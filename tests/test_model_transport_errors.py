"""Typed request-error contracts: synthetic exceptions/MockTransport, no fees."""
import inspect
import json
import socket

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError
import pytest

from app.agents.provider import ModelInvocationFailed, capture_usage
from app.domain.model_transport import classify_model_transport, public_transport_failure
from app.worker import Worker
from tests.test_deepseek_provider import intercepted_adapter
from tests.test_repository import create, setup_repo


PRIVATE = "MOCK_SECRET_BODY_AUTH_REASONING_NOT_PUBLIC"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    original = socket.socket.connect
    def deny(sock, address):
        if isinstance(address, tuple) and address[0] == "127.0.0.1" and any(frame.function == "_fallback_socketpair" for frame in inspect.stack()):
            return original(sock, address)
        raise AssertionError("transport-error tests prohibit real network")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


def status_error(status, headers=None):
    request = httpx.Request("POST", "https://synthetic.invalid/chat/completions", headers={"Authorization": "Bearer " + PRIVATE})
    response = httpx.Response(status, request=request, headers=httpx.Headers(headers or {}, encoding="utf-8"),
        json={"error": {"message": PRIVATE, "reasoning_content": PRIVATE}})
    return APIStatusError(PRIVATE, response=response, body={"secret": PRIVATE})


@pytest.mark.parametrize("status,code,retryable", [
    (400, "model_request_invalid", False), (401, "model_authentication_failed", False),
    (402, "model_insufficient_balance", False), (403, "model_access_denied", False),
    (404, "model_endpoint_not_found", False), (408, "model_request_timeout", True),
    (409, "model_request_conflict", True), (413, "model_request_too_large", False),
    (422, "model_request_invalid", False), (429, "model_rate_limited", True),
    (418, "model_request_rejected", False), (500, "model_service_unavailable", True),
    (503, "model_service_unavailable", True), (599, "model_service_unavailable", True),
])
def test_all_typed_http_statuses_have_stable_safe_codes_and_advice(status, code, retryable):
    error = status_error(status, {"Retry-After": "12", "Authorization": PRIVATE, "X-Reasoning": PRIVATE})
    wrapper = RuntimeError(PRIVATE); wrapper.__cause__ = error
    failure = classify_model_transport(wrapper)
    assert failure["code"] == code
    assert failure["details"] == {"provider_http_status": status, "provider_error_type": "APIStatusError", "retryable": retryable, "retry_after_seconds": 12, "usage_known": False}
    assert PRIVATE not in json.dumps(failure, ensure_ascii=False)
    assert "显式重试" in failure["message"] or status == 404
    assert error.body == {"secret": PRIVATE}  # No in-place redaction/repair of SDK input.


@pytest.mark.parametrize("error,code,label", [
    (APITimeoutError(request=httpx.Request("POST", "https://synthetic.invalid")), "model_transport_timeout", "APITimeoutError"),
    (httpx.ReadTimeout(PRIVATE), "model_transport_timeout", "TimeoutException"),
    (TimeoutError(PRIVATE), "model_transport_timeout", "TimeoutError"),
    (APIConnectionError(message=PRIVATE, request=httpx.Request("POST", "https://synthetic.invalid")), "model_connection_failed", "APIConnectionError"),
    (httpx.ConnectError(PRIVATE), "model_connection_failed", "TransportError"),
])
def test_typed_timeout_and_connection_failures_do_not_copy_error_strings(error, code, label):
    failure = classify_model_transport(error)
    assert failure["code"] == code
    assert failure["details"] == {"retryable": True, "provider_error_type": label, "usage_known": False}
    assert PRIVATE not in json.dumps(failure)


def test_arbitrary_exception_status_and_unbounded_context_do_not_become_trusted_http_metadata():
    error = ValueError(PRIVATE); error.status_code = 402
    error.response = {"headers": {"Retry-After": "12"}}
    error.__context__ = error
    failure = classify_model_transport(error)
    assert failure["code"] == "model_request_failed"
    assert failure["details"] == {"retryable": False, "provider_error_type": "unknown", "usage_known": False}


@pytest.mark.parametrize("retry", ["-1", "1.5", "nan", "inf", "86401", "10000000000", "１２", "12, 13", PRIVATE])
def test_nonstandard_or_unbounded_retry_after_is_not_public_metadata(retry):
    failure = classify_model_transport(status_error(429, {"retry-after": retry}))
    assert "retry_after_seconds" not in failure["details"] and "retry_after_utc" not in failure["details"]


def test_http_date_and_remaining_counts_only_project_strict_bounded_values():
    headers = {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT", "x-ratelimit-remaining-requests": "0", "x-ratelimit-remaining-tokens": "999999999999", "set-cookie": PRIVATE}
    details = classify_model_transport(status_error(429, headers))["details"]
    assert details["retry_after_utc"] == "2015-10-21T07:28:00+00:00"
    assert details["remaining_requests"] == 0 and details["remaining_tokens"] == 999999999999
    for value in ("-1", "+1", "1.5", "１２", "1000000000000", PRIVATE):
        bad = classify_model_transport(status_error(429, {"x-ratelimit-remaining-requests": value}))["details"]
        assert "remaining_requests" not in bad


def test_public_error_sanitizer_does_not_accept_arbitrary_headers_or_flags():
    failure = public_transport_failure("model_insufficient_balance", {
        "provider_http_status": 429, "provider_error_type": PRIVATE, "retryable": True,
        "retry_after_seconds": True, "retry_after_utc": "2026-99-99T00:00:00+00:00",
        "remaining_requests": -1, "remaining_tokens": True, "headers": {"Authorization": PRIVATE},
        "body": PRIVATE, "reasoning_content": PRIVATE, "user_action": PRIVATE, "usage_known": True})
    assert failure["details"] == {"retryable": False, "provider_error_type": "unknown", "usage_known": False}
    assert PRIVATE not in json.dumps(failure)


@pytest.mark.parametrize("status,code", [(402, "model_insufficient_balance"), (429, "model_rate_limited")])
async def test_actual_langchain_sdk_adapter_retains_unknown_usage_and_does_not_add_retry(status, code):
    async with intercepted_adapter(content=PRIVATE, status_code=status) as (adapter, requests):
        with capture_usage() as usage:
            with pytest.raises(ModelInvocationFailed) as caught:
                await adapter.analyze("authority_analyst", {"id": "synthetic-question", "question": "合成问题"}, [], "frozen")
    error = caught.value
    assert len(requests) == 1  # Only MockTransport; no SDK/application extra attempts.
    assert error.code == code and error.stage == "authority_analyst"
    assert error.details["provider_http_status"] == status
    assert error.details["retryable"] is (status == 429)
    assert len(usage) == 1 and usage[0]["usage_source"] == "unavailable"
    assert all(usage[0][key] is None for key in ("input_tokens", "output_tokens", "total_tokens"))
    assert PRIVATE not in str(error) and PRIVATE not in json.dumps(error.__dict__)


@pytest.mark.parametrize("status,code", [(402, "model_insufficient_balance"), (429, "model_rate_limited")])
async def test_worker_repository_and_activity_preserve_safe_reason_stage_and_unknown_usage(tmp_path, monkeypatch, status, code):
    repo, topic = setup_repo(tmp_path)
    queued = create(repo, topic); run = repo.claim(); assert run["id"] == queued["id"]
    question_id = repo.questions(topic)[0]["id"]
    failure = classify_model_transport(status_error(status, {"retry-after": "12"}))
    async def analyze(_run):
        repo.record_usage(run["id"], question_id, [{"operation": "paragraph_rank", "input_tokens": None, "output_tokens": None, "total_tokens": None}])
        raise ModelInvocationFailed(PRIVATE, stage="paragraph_rank", code=failure["code"], details={**failure["details"], "body": PRIVATE})
    worker = Worker(repo, provider=object()); monkeypatch.setattr(worker, "analyze", analyze)
    await worker.execute(run)
    actual = repo.run(run["id"])
    assert actual["status"] == "failed" and actual["error"]["code"] == code
    assert actual["error"]["stage"] == "paragraph_rank"
    assert actual["error"]["details"]["provider_http_status"] == status
    assert actual["error"]["details"]["retryable"] is (status == 429)
    record = actual["usage"][question_id][0]
    assert record["usage_source"] == "unavailable" and record["total_tokens"] is None
    activity = repo.activity(run["id"])
    failed = activity["progress"]["failed_stages"]
    assert len(failed) == 1 and failed[0]["details"] == actual["error"]["details"]
    assert PRIVATE not in json.dumps(activity, ensure_ascii=False)
    assert repo.topic(topic)["published_revision_id"] is None
