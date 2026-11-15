"""Actual business diagnostics have a fixed public vocabulary, never raw bodies."""
import json
import pytest
from app.agents.bundle_transport import BundleWorkbenchProvider, PUBLIC_TOOL_FAILURES
from app.agents.provider import ModelOutputInvalid, capture_usage
from test_bundle_transport import settings, raw_case


@pytest.mark.parametrize("code", [*PUBLIC_TOOL_FAILURES, "private_provider_body", "Bearer private", None])
def test_tool_error_cards_keep_known_business_cause_and_remove_unknown_payload(tmp_path, code):
    provider = BundleWorkbenchProvider(settings(tmp_path))
    events = []
    provider.workbench_event = lambda kind, data: events.append((kind, data))
    provider._event("tool_started", {"call_id": "one", "tool": "read_source_bundles"})
    provider._event("tool_completed", {"call_id": "one", "failed": True, "deferred": False, "failure_code": code})
    kind, card = events[-1]
    assert kind == "tool_failed"
    if code in PUBLIC_TOOL_FAILURES:
        assert card["code"] == code
        assert card["message"] == PUBLIC_TOOL_FAILURES[code]
        assert card["summary"] == PUBLIC_TOOL_FAILURES[code]
    else:
        assert "code" not in card and "message" not in card
        assert all("failure_code" not in data for event, data in events)
        assert "private" not in str(events)


@pytest.mark.asyncio
async def test_terminal_budget_failure_retains_prior_business_cause_and_bill(tmp_path, monkeypatch):
    provider = BundleWorkbenchProvider(settings(tmp_path))
    case = raw_case()
    async def fail(manifest, descriptor):
        native = provider.bundle_directory / "native"
        native.mkdir()
        rows = [{"status": "request_started", "request_sequence": 1},
            {"status": "usage_settled", "request_sequence": 1, "unknown_usage_requests": 0,
                "provider_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}]
        (native / "budget.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
        (native / "closed-receipt.json").write_text(json.dumps({"status": "budget_blocked",
            "known_usage_complete": True, "failure_code": "whole_episode_budget_exceeded",
            "last_business_failure_code": "bundle_bridge_duplicate_selected_start_anchor"}))
    monkeypatch.setattr(provider, "_launch", fail)
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid) as failure:
            await provider.select_source_bundles(case["question"], case["documents"], case["paragraphs"])
    assert failure.value.code == "whole_episode_budget_exceeded"
    assert "重复起点" in str(failure.value)
    assert sum(row["total_tokens"] for row in usage) == 120
    assert not provider._tail_seeded
