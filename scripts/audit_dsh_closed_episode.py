"""Read-only usage/closure audit. No profile, credential, reference or model IO."""
import argparse
from hashlib import sha256
import json
from pathlib import Path


def bound(path, expected=None):
    path = Path(path).resolve()
    data = path.read_bytes()
    actual = sha256(data).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("closed_episode_bound_bytes_changed")
    return data, {"file": str(path), "sha256": actual, "bytes": len(data)}


def audit_closed_episode(receipt_path, expected_sha256, manifest_path, *, root):
    """Verify exact native-request journal; never equate no final with zero P/R."""
    path = Path(receipt_path).resolve()
    receipt_data, receipt_d = bound(path, expected_sha256)
    receipt = json.loads(receipt_data)
    if (receipt.get("format") != "dsh-comparison-closed-v1" or receipt.get("closed") is not True
            or receipt.get("fixture") is not False or receipt.get("initializeOnly") is not False):
        raise ValueError("closed_actual_episode_required")
    manifest_data, manifest_d = bound(manifest_path, receipt["manifest_sha256"])
    manifest = json.loads(manifest_data)
    _, case_d = bound(Path(root) / manifest["casePath"], receipt["case_sha256"])
    journal_data, journal_d = bound(path.parent / "budget.jsonl", receipt["budget_audit_sha256"])
    rows = [json.loads(line) for line in journal_data.decode("utf-8").splitlines() if line]
    if [row["sequence"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("ordered_journal_required")
    known, started, settled, usage, blocked = manifest["priorKnownTokens"], {}, set(), [], []
    for row in rows:
        if row.get("episode_id") != manifest["episodeId"]:
            raise ValueError("same_episode_journal_required")
        number, status = row["request_sequence"], row["status"]
        if status in {"admitted", "request_started", "blocked"}:
            expected = known + row["estimated_input_tokens"] + row["reserved_output_tokens"] + row["safety_tokens"]
            if row["known_episode_tokens_before"] != known or row["preflight_total_tokens"] != expected:
                raise ValueError("whole_episode_accounting_changed")
            if status in {"admitted", "request_started"} and expected > manifest["wholeLimit"]:
                raise ValueError("over_budget_request_was_admitted")
            if status == "request_started":
                if number in started:
                    raise ValueError("duplicate_request_start")
                started[number] = row["wire_sha256"]
            if status == "blocked":
                if number in started:
                    raise ValueError("blocked_request_was_already_started")
                if row["code"] == "whole_episode_budget_exceeded" and expected <= manifest["wholeLimit"]:
                    raise ValueError("budget_block_arithmetic_does_not_exceed_limit")
                blocked.append({"request_sequence": number, "code": row["code"], "preflight_total_tokens": expected})
        elif status == "usage_settled":
            if number not in started or number in settled or row["wire_sha256"] != started[number]:
                raise ValueError("one_settlement_per_started_request_required")
            counters = row["provider_usage"]
            if (any(type(counters.get(key)) is not int or counters[key] < 0 for key in ("input_tokens", "output_tokens", "total_tokens"))
                    or counters["input_tokens"] + counters["output_tokens"] != counters["total_tokens"]):
                raise ValueError("known_actual_provider_usage_required")
            known += counters["total_tokens"]
            if row["known_episode_tokens_after"] != known or row["unknown_usage_requests"] != 0:
                raise ValueError("actual_usage_total_changed")
            settled.add(number)
            usage.append({"request_sequence": number, **counters})
        elif status == "usage_unknown":
            raise ValueError("episode_has_unknown_usage")
    if (set(started) != settled or len(started) != receipt["model_requests"]
            or known != receipt["known_tokens"] or receipt["unknown_usage_requests"] != 0):
        raise ValueError("episode_pending_or_total_changed")
    artifact = path.parent / "submission.json"
    has_final = receipt["submission_sha256"] is not None
    if has_final:
        bound(artifact, receipt["submission_sha256"])
    elif artifact.exists():
        raise ValueError("unbound_final_artifact_exists")
    for desc in (receipt_d, manifest_d, case_d, journal_d):
        bound(desc["file"], desc["sha256"])
    return {"format": "independent-dsh-closed-usage-audit-v1", "status": receipt["status"],
            "input_guards": [receipt_d, manifest_d, case_d, journal_d], "actual_usage_by_request": usage,
            "known_total_tokens": known, "known_input_tokens": sum(row["input_tokens"] for row in usage),
            "known_output_tokens": sum(row["output_tokens"] for row in usage), "unknown_usage_requests": 0,
            "pending_requests": 0, "blocked_before_request": blocked,
            "has_final_artifact": has_final, "quality_metrics": None if not has_final else "requires_separate_offline_evaluation",
            "no_final_is_quality_zero": False, "invoice_cost_known": False,
            "original_full_raw_bytes_verified_here": False, "references_read": False,
            "new_model_calls": 0, "elapsed_seconds_from_closed_receipt": receipt["elapsed_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_closed_episode(args.receipt, args.expected_sha256, args.manifest, root=args.root)
    with Path(args.output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"known_tokens": result["known_total_tokens"], "has_final_artifact": result["has_final_artifact"],
                      "quality_metrics": result["quality_metrics"], "new_model_calls": 0}))
