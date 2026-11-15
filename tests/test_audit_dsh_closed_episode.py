from hashlib import sha256
import json

import pytest

from scripts.audit_dsh_closed_episode import audit_closed_episode


def write(path, value):
    data = (json.dumps(value) + "\n").encode()
    path.write_bytes(data)
    return sha256(data).hexdigest()


def episode(tmp_path, *, mutation=None):
    case = tmp_path / "raw-case.json"
    case_sha = write(case, {"raw_data": "纯合成材料"})
    manifest = {"casePath": "raw-case.json", "priorKnownTokens": 0, "episodeId": "synthetic", "wholeLimit": 20000}
    manifest_path = tmp_path / "manifest.json"
    manifest_sha = write(manifest_path, manifest)
    common = {"episode_id": "synthetic", "wire_sha256": "b" * 64, "estimated_input_tokens": 50,
              "known_episode_tokens_before": 0, "reserved_output_tokens": 100, "safety_tokens": 10, "preflight_total_tokens": 160}
    rows = [{**common, "sequence": 1, "request_sequence": 1, "status": "admitted"},
            {**common, "sequence": 2, "request_sequence": 1, "status": "request_started"},
            {**common, "sequence": 3, "request_sequence": 1, "status": "usage_settled", "provider_usage": {"input_tokens": 50, "output_tokens": 5, "total_tokens": 55}, "known_episode_tokens_after": 55, "unknown_usage_requests": 0},
            {**common, "sequence": 4, "request_sequence": 2, "status": "blocked", "estimated_input_tokens": 20000,
             "known_episode_tokens_before": 55, "preflight_total_tokens": 20165, "code": "whole_episode_budget_exceeded"}]
    if mutation == "pending": rows.pop(2); rows[-1]["sequence"] = 3
    if mutation == "drift": rows[-1]["known_episode_tokens_before"] = 0
    if mutation == "wrong_usage": rows[2]["provider_usage"]["total_tokens"] = 100
    journal = tmp_path / "budget.jsonl"
    journal.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    receipt = {"format": "dsh-comparison-closed-v1", "closed": True, "fixture": False, "initializeOnly": False,
               "manifest_sha256": manifest_sha, "case_sha256": case_sha, "budget_audit_sha256": sha256(journal.read_bytes()).hexdigest(),
               "status": "budget_blocked", "model_requests": 1, "known_tokens": 55, "unknown_usage_requests": 0,
               "submission_sha256": None, "elapsed_seconds": 1}
    receipt_path = tmp_path / "closed-receipt.json"
    expected = write(receipt_path, receipt)
    return receipt_path, expected, manifest_path


def test_blocked_unsent_request_is_not_a_call_and_absence_is_not_quality_zero(tmp_path):
    args = episode(tmp_path)
    result = audit_closed_episode(*args, root=tmp_path)
    assert result["known_total_tokens"] == 55
    assert len(result["actual_usage_by_request"]) == 1
    assert result["quality_metrics"] is None
    assert result["no_final_is_quality_zero"] is False


@pytest.mark.parametrize("mutation", ["pending", "drift", "wrong_usage"])
def test_invalid_usage_journal_fails_closed(tmp_path, mutation):
    args = episode(tmp_path, mutation=mutation)
    with pytest.raises(ValueError):
        audit_closed_episode(*args, root=tmp_path)


def test_unbound_submission_is_not_accepted_as_final(tmp_path):
    args = episode(tmp_path)
    write(tmp_path / "submission.json", {"some": "artifact"})
    with pytest.raises(ValueError, match="unbound_final"):
        audit_closed_episode(*args, root=tmp_path)
