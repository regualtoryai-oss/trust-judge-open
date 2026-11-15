"""Raw-only, zero-network preparation for a shared-budget framework comparison.

This is deliberately not a launcher or model client.  Native request estimates
must come from each framework's own serializer/fixture; this module never guesses
token counts from characters and cannot authorize a paid request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIXED_CASE = ("04", "q_002")
FIXED_VARIANT = "title_bge_hybrid_rrf40"
FRAMEWORK_ORDER = ("langgraph", "dsh")
WHOLE_LIMIT = 20_000
SAFETY = 512


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    forbidden = {".env", "credentials", "gold", "evidence_samples.json", "conclusions.json"}
    if any(part.lower() in forbidden or part.lower().startswith(".env.") for part in path.parts):
        raise ValueError("raw_only_path_required")
    return path


def read_bound(file: str | Path, expected_sha256: str) -> bytes:
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("sha256_required")
    data = safe_path(file).read_bytes()
    if digest(data) != expected_sha256:
        raise ValueError("frozen_file_changed")
    return data


def validate_raw_case(case: dict[str, Any]) -> None:
    if set(case) - {"format", "question", "catalog", "case_id", "topic_id", "topic_title", "source",
                    "source_metadata", "binding", "unprovided_context_state", "publishable"}:
        raise ValueError("unexpected_raw_case_fields")
    if case.get("publishable", False) is not False:
        raise ValueError("unpublished_raw_case_required")
    question = case.get("question", {})
    if question.get("question_id") != FIXED_CASE[1] or not isinstance(question.get("question"), str):
        raise ValueError("fixed_question_required")
    paragraphs = case.get("catalog")
    if not isinstance(paragraphs, list) or len(paragraphs) != 40:
        raise ValueError("forty_complete_parents_required")
    refs: set[str] = set()
    addresses: set[tuple[str, str]] = set()
    title_by_doc: dict[str, str] = {}
    fields = {"paragraph_ref", "doc_id", "para_id", "title", "text", "parent_complete", "paragraph_complete"}
    for paragraph in paragraphs:
        if set(paragraph) != fields or paragraph.get("parent_complete") is not True or paragraph.get("paragraph_complete") is not True:
            raise ValueError("complete_raw_parent_required")
        for key in ("paragraph_ref", "doc_id", "para_id", "title", "text"):
            if not isinstance(paragraph[key], str) or not paragraph[key]:
                raise ValueError("source_identity_and_text_required")
        ref = paragraph["paragraph_ref"]
        address = (paragraph["doc_id"], paragraph["para_id"])
        if ref in refs or address in addresses:
            raise ValueError("duplicate_parent_identity")
        refs.add(ref)
        addresses.add(address)
        old_title = title_by_doc.setdefault(paragraph["doc_id"], paragraph["title"])
        if old_title != paragraph["title"]:
            raise ValueError("inconsistent_document_title")


def select_fixed_case(index: dict[str, Any]) -> dict[str, str]:
    if index.get("format") != "frozen-local-rerank-native-fit-inputs-v1":
        raise ValueError("frozen_native_fit_index_required")
    rows = [r for r in index.get("cases", []) if (r.get("topic_id"), r.get("question_id")) == FIXED_CASE]
    if len(rows) != 1:
        raise ValueError("fixed_case_unique_required")
    variants = [v for v in rows[0].get("variants", []) if v.get("label") == FIXED_VARIANT]
    if len(variants) != 1:
        raise ValueError("preregistered_primary_variant_required")
    return variants[0]


def paired_budget_projection(requests: list[dict[str, Any]], prior_known_tokens: int = 0) -> dict[str, Any]:
    """Project every request at its reserved output, with one shared episode ledger.

    This conservative preparation assumes *all* planned outputs hit their reserve.
    Live guards must instead carry actual settled input/cache/output and stop on
    unknown usage; passing this projection never substitutes for live admission.
    """
    if isinstance(prior_known_tokens, bool) or not isinstance(prior_known_tokens, int) or prior_known_tokens < 0:
        raise ValueError("nonnegative_prior_required")
    cumulative = prior_known_tokens
    rows = []
    previous_arm = 0
    arm_steps = {arm: 0 for arm in FRAMEWORK_ORDER}
    for request in requests:
        arm = request.get("framework")
        if arm not in FRAMEWORK_ORDER:
            raise ValueError("unknown_framework")
        arm_index = FRAMEWORK_ORDER.index(arm)
        if arm_index < previous_arm:
            raise ValueError("fixed_framework_order_required")
        previous_arm = arm_index
        if request.get("estimate_basis") != "official_tokenizer_native_request_estimate":
            raise ValueError("native_serialized_estimate_required")
        for key in ("estimated_input_tokens", "reserved_output_tokens"):
            if isinstance(request.get(key), bool) or not isinstance(request.get(key), int) or request[key] < 0:
                raise ValueError("nonnegative_token_count_required")
        if request["reserved_output_tokens"] == 0:
            raise ValueError("positive_output_reservation_required")
        wire_sha = request.get("native_wire_sha256", "")
        if len(wire_sha) != 64 or any(c not in "0123456789abcdef" for c in wire_sha):
            raise ValueError("native_wire_binding_required")
        arm_steps[arm] += 1
        cumulative += request["estimated_input_tokens"] + request["reserved_output_tokens"]
        rows.append({**request, "shared_reserved_total_after": cumulative,
                     "preflight_with_safety": cumulative + SAFETY,
                     "fits_shared_limit": cumulative + SAFETY <= WHOLE_LIMIT})
    complete = bool(requests) and all(count == 2 for count in arm_steps.values())
    return {"whole_limit": WHOLE_LIMIT, "safety_tokens": SAFETY, "prior_known_tokens": prior_known_tokens,
            "fixed_framework_order": list(FRAMEWORK_ORDER), "requests": rows,
            "complete_two_step_pair": complete,
            "reserved_pair_upper_bound": cumulative + SAFETY,
            "all_native_fixtures_fit": complete and all(r["fits_shared_limit"] for r in rows),
            "paid_execution_authorized": False, "live_guard_still_required": True,
            "projection_is_billing_guarantee": False}


def prepare(*, index_file: Path, index_sha: str, protocol_file: Path, protocol_sha: str,
            policy_file: Path, policy_sha: str, output_directory: Path) -> dict[str, Any]:
    index_bytes = read_bound(index_file, index_sha)
    read_bound(protocol_file, protocol_sha)
    read_bound(policy_file, policy_sha)
    entry = select_fixed_case(json.loads(index_bytes))
    case_bytes = read_bound(entry["file"], entry["sha256"])
    case = json.loads(case_bytes)
    validate_raw_case(case)
    output = safe_path(output_directory)
    repo_runtime = Path(__file__).resolve().parents[1] / ".runtime"
    if not output.is_relative_to(repo_runtime.resolve()):
        raise ValueError("new_runtime_output_required")
    output.mkdir()  # Exclusive, never reuse a paid or fixture result directory.
    case_file = output / "shared-raw-case.json"
    with case_file.open("xb") as handle:
        handle.write(case_bytes)
    descriptor = lambda file, sha: {"file": str(safe_path(file)), "sha256": sha}
    preparation = {
        "format": "framework-paired-raw-preparation-v1", "topic_id": FIXED_CASE[0], "question_id": FIXED_CASE[1],
        "candidate_variant": FIXED_VARIANT, "candidate_count": 40,
        "model": "deepseek-v4-pro", "thinking": "disabled", "framework_order": list(FRAMEWORK_ORDER),
        "shared_episode_limit_tokens": WHOLE_LIMIT, "safety_tokens": SAFETY,
        "shared_prior_known_tokens": 0, "prior_usage_must_be_rechecked_before_paid_run": True,
        "source_index": descriptor(index_file, index_sha), "shared_case": descriptor(case_file, entry["sha256"]),
        "protocol": descriptor(protocol_file, protocol_sha), "shared_policy_source": descriptor(policy_file, policy_sha),
        "native_fixtures": "required_for_both_frameworks_before_paid_admission",
        "real_api_requests": 0, "reference_reads": [], "execution_allowed": False,
        "quality_metrics": None, "unknown_sources_remain_unknown": True,
    }
    out_file = output / "paired-preparation.json"
    with out_file.open("xb") as handle:
        handle.write((json.dumps(preparation, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return descriptor(out_file, digest(out_file.read_bytes()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("index", "index-sha", "protocol", "protocol-sha", "policy-source", "policy-sha", "output"):
        parser.add_argument(f"--{flag}", required=True)
    args = parser.parse_args()
    result = prepare(index_file=Path(args.index), index_sha=args.index_sha, protocol_file=Path(args.protocol),
                     protocol_sha=args.protocol_sha, policy_file=Path(args.policy_source), policy_sha=args.policy_sha,
                     output_directory=Path(args.output))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
