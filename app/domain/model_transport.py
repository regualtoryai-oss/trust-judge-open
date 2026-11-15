"""Stable public request failures from typed transport metadata, never bodies.

Retryability is advice for an explicit future attempt. This module neither
retries requests nor claims that a failed/timeout request was free.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError


# code -> (fixed message, retryable, matching HTTP statuses)
_FAILURES = {
    "model_authentication_failed": ("模型服务鉴权失败（401）；请检查 API Key 后显式重试。", False, {401}),
    "model_insufficient_balance": ("模型服务余额不足（402）；请充值或更换可用账号后显式重试。", False, {402}),
    "model_access_denied": ("模型服务拒绝访问（403）；请检查账号和模型权限后显式重试。", False, {403}),
    "model_endpoint_not_found": ("模型服务地址或模型不存在（404）；请检查服务地址和模型名称。", False, {404}),
    "model_request_timeout": ("模型服务请求超时（408）；用量可能未返回，请稍后显式重试。", True, {408}),
    "model_request_conflict": ("模型服务请求冲突（409）；请稍后显式重试。", True, {409}),
    "model_request_too_large": ("模型请求过大（413）；请调整单次输入或上下文配置后显式重试。", False, {413}),
    "model_request_invalid": ("模型请求参数不被接受；请检查模型及请求参数后显式重试。", False, {400, 422}),
    "model_rate_limited": ("模型服务限流（429）；请按等待时间或服务限额调整后显式重试。", True, {429}),
    "model_service_unavailable": ("模型服务暂时不可用；请稍后显式重试。", True, set(range(500, 600))),
    "model_request_rejected": ("模型服务拒绝该请求；请检查服务配置后显式重试。", False, set(range(400, 500)) - {400, 401, 402, 403, 404, 408, 409, 413, 422, 429}),
    "model_transport_timeout": ("连接模型服务超时；用量可能未返回，请稍后显式重试。", True, set()),
    "model_connection_failed": ("无法连接模型服务；请检查网络及服务地址后显式重试。", True, set()),
    "model_request_failed": ("模型请求失败；未生成替代结果，请检查服务配置后显式重试。", False, set()),
}
MODEL_TRANSPORT_CODES = frozenset(_FAILURES)
_ERROR_TYPES = frozenset({"APIStatusError", "APIConnectionError", "APITimeoutError", "HTTPStatusError", "TimeoutException", "TransportError", "TimeoutError", "unknown"})
_HTTP_DATE = re.compile(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4} \d{2}:\d{2}:\d{2} GMT\Z", re.ASCII)


def _rate_headers(headers):
    output = {}
    if not isinstance(headers, httpx.Headers):
        return output
    retry = headers.get("retry-after")
    if isinstance(retry, str) and re.fullmatch(r"[0-9]{1,5}", retry) and int(retry) <= 86400:
        output["retry_after_seconds"] = int(retry)
    elif isinstance(retry, str) and _HTTP_DATE.fullmatch(retry):
        try:
            parsed = parsedate_to_datetime(retry)
            if parsed.tzinfo is not None:
                output["retry_after_utc"] = parsed.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
    for name, field in (("x-ratelimit-remaining-requests", "remaining_requests"), ("x-ratelimit-remaining-tokens", "remaining_tokens")):
        value = headers.get(name)
        if isinstance(value, str) and re.fullmatch(r"[0-9]{1,12}", value):
            output[field] = int(value)
    return output


def public_transport_failure(code="model_request_failed", details=None):
    """Rebuild messages/flags and retain only bounded, independently safe values."""
    code = code if isinstance(code, str) and code in _FAILURES else "model_request_failed"
    message, retryable, statuses = _FAILURES[code]
    safe = {"retryable": retryable, "provider_error_type": "unknown", "usage_known": False}
    if isinstance(details, dict):
        status = details.get("provider_http_status")
        if type(status) is int and status in statuses:
            safe["provider_http_status"] = status
        error_type = details.get("provider_error_type")
        if isinstance(error_type, str) and error_type in _ERROR_TYPES:
            safe["provider_error_type"] = error_type
        seconds = details.get("retry_after_seconds")
        if type(seconds) is int and 0 <= seconds <= 86400:
            safe["retry_after_seconds"] = seconds
        utc = details.get("retry_after_utc")
        if isinstance(utc, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", utc, re.ASCII):
            try:
                datetime.fromisoformat(utc)
                safe["retry_after_utc"] = utc
            except ValueError:
                pass
        for field in ("remaining_requests", "remaining_tokens"):
            value = details.get(field)
            if type(value) is int and 0 <= value <= 999999999999:
                safe[field] = value
    return {"code": code, "message": message, "details": safe}


def classify_model_transport(error):
    """Inspect typed exceptions and their bounded cause/context chain only."""
    seen = set()
    for _ in range(5):
        if not isinstance(error, BaseException) or id(error) in seen:
            break
        seen.add(id(error))
        if isinstance(error, (APIStatusError, httpx.HTTPStatusError)):
            response = error.response
            status = response.status_code
            code = next((key for key, (_, _, statuses) in _FAILURES.items() if type(status) is int and status in statuses), "model_request_failed")
            details = {"provider_http_status": status,
                "provider_error_type": "APIStatusError" if isinstance(error, APIStatusError) else "HTTPStatusError",
                **_rate_headers(response.headers)}
            return public_transport_failure(code, details)
        for kind, code, label in (
            (APITimeoutError, "model_transport_timeout", "APITimeoutError"),
            (httpx.TimeoutException, "model_transport_timeout", "TimeoutException"),
            (TimeoutError, "model_transport_timeout", "TimeoutError"),
            (APIConnectionError, "model_connection_failed", "APIConnectionError"),
            (httpx.TransportError, "model_connection_failed", "TransportError"),
        ):
            if isinstance(error, kind):
                return public_transport_failure(code, {"provider_error_type": label})
        error = error.__cause__ or error.__context__
    return public_transport_failure()
