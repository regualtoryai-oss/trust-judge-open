"""Offline exact-message preflight. No provider, settings, model or download.

``count_message_tokens`` counts final message CONTENTS (no chat wrapper). The
observed wrapper allowance is separate; every result remains a cost estimate.
The current unbounded proposal schema is reported, never silently rewritten.
"""
import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TOKENIZER = ROOT / ".runtime/v32-deepseek-tokenizer-r2/tokenizer.json"
DEFAULT_TOKENIZER_SHA = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
DEFAULT_PROVIDER_SHA = "506ec377247104923e14c0d3657a9fee5469a3a921ec5aaccf2e048e63bc116b"
ANSWER_FIELDS = {"gold", "answer", "answers", "standard_answer", "reference_answer",
                 "reference_labels", "evidence_samples", "conclusions",
                 "supporting_facts", "conflicting_facts", "fact_des"}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return sha256(_canonical(value)).hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid_" + name)
    return value


def _reject_answers(value):
    if isinstance(value, dict):
        if ANSWER_FIELDS.intersection(value):
            raise ValueError("answer_bearing_input_forbidden")
        for child in value.values():
            _reject_answers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_answers(child)


def _capacity_risk(schema):
    strings, arrays = [], []
    def walk(value, path):
        if isinstance(value, dict):
            if value.get("type") == "string" and not any(k in value for k in ("maxLength", "enum", "const")):
                strings.append(path)
            if value.get("type") == "array" and "maxItems" not in value:
                arrays.append(path)
            for key, child in value.items():
                walk(child, path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + "/" + str(index))
    walk(schema, "#")
    return {"risk_present": bool(strings or arrays), "unbounded_string_paths": strings,
            "unbounded_array_paths": arrays, "formal_maximum_output_tokens_known": False,
            "output_reservation_guarantees_fit": False, "bounded_wire_applied": False,
            "limitation": "No new fact/list quotas imposed. Review a bounded wire contract before paid execution; truncation must stop without automatic repair."}


def prepare_obligation_selection(question, catalog, *, count_message_tokens,
                                prior_known_tokens=0, output_reservation=4096,
                                safety_reservation=512, whole_limit=20000,
                                system_boundary=None, chat_overhead_tokens=25):
    """Build the exact native JSON-mode messages without importing its provider.

    The caller must supply the reviewed provider SYSTEM_BOUNDARY literally.
    A strict int callback result counts every final content with no truncation.
    None/errors/invalid counts block, as does unknown prior usage. No paragraph,
    title, question or source block is selected, dropped, shortened or reordered.
    """
    # The standalone count CLI runs in the isolated tokenizers-only environment.
    # Counting final strings must not import Pydantic or the application domain.
    from app.domain.evidence_obligations import (
        OBLIGATION_INSTRUCTIONS, ObligationPlanOutput, obligation_payload,
    )
    if not isinstance(system_boundary, str) or not system_boundary.strip():
        raise ValueError("explicit_reviewed_system_boundary_required")
    if not callable(count_message_tokens):
        raise ValueError("message_token_counter_required")
    for name, value, minimum in (("output_reservation", output_reservation, 1),
                                  ("safety_reservation", safety_reservation, 0),
                                  ("whole_limit", whole_limit, 1),
                                  ("chat_overhead_tokens", chat_overhead_tokens, 0)):
        _integer(value, name, minimum)
    if prior_known_tokens is not None:
        _integer(prior_known_tokens, "prior_known_tokens")
    _reject_answers(question); _reject_answers(catalog)
    if not isinstance(catalog, list) or any(not isinstance(row, dict) for row in catalog):
        raise ValueError("program_source_catalog_list_required")
    wire_question = {"question": question} if isinstance(question, str) else question
    payload = obligation_payload(wire_question, catalog)
    schema = ObligationPlanOutput.model_json_schema()
    messages = [
        {"role": "system", "content": system_boundary + OBLIGATION_INSTRUCTIONS + "\nJSON Schema:\n" + json.dumps(schema, ensure_ascii=False)},
        {"role": "human", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    reasons = []
    if prior_known_tokens is None:
        reasons.append("prior_usage_unknown")
    count = None
    try:
        value = count_message_tokens(deepcopy(messages))
        if value is None:
            reasons.append("message_token_cost_unknown")
        elif type(value) is not int or value < 0:
            reasons.append("message_token_counter_invalid")
        else:
            count = value
    except Exception:
        reasons.append("message_token_counter_failed")  # Never persist arbitrary exception text.
    readable_blocks = sum(len(row["blocks"]) for row in payload["paragraphs"])
    if not readable_blocks:
        reasons.append("no_readable_source_blocks")
    estimated_input = count + chat_overhead_tokens if count is not None else None
    estimate = (prior_known_tokens + estimated_input + output_reservation + safety_reservation
                if prior_known_tokens is not None and estimated_input is not None else None)
    if estimate is not None and estimate > whole_limit:
        reasons.append("estimated_whole_episode_exceeds_limit")
    return {
        "format": "obligation-selection-offline-preflight-v1",
        "status": "blocked" if reasons else "prepared_for_review",
        "blocked_reasons": reasons, "messages": messages, "schema": schema, "payload": payload,
        "message_content_sha256": _digest(messages), "message_content_utf8_bytes": len(_canonical(messages)),
        "schema_sha256": _digest(schema), "payload_sha256": _digest(payload),
        "question_input_sha256": _digest(question), "catalog_input_sha256": _digest(catalog),
        "system_boundary_sha256": sha256(system_boundary.encode()).hexdigest(),
        "instructions_sha256": sha256(OBLIGATION_INSTRUCTIONS.encode()).hexdigest(),
        "preflight": {"prior_known_tokens": prior_known_tokens, "final_message_content_tokens": count,
            "observed_chat_overhead_tokens": chat_overhead_tokens, "estimated_new_input_tokens": estimated_input,
            "output_reservation": output_reservation, "safety_reservation": safety_reservation,
            "estimated_whole_episode_tokens": estimate, "whole_episode_soft_limit": whole_limit,
            "counter_scope": "all_exact_final_message_contents_without_server_chat_wrapper",
            "is_actual_bill_or_hard_provider_cap": False},
        "schema_capacity_risk": _capacity_risk(schema),
        "parent_count": len(payload["paragraphs"]), "provided_source_block_count": readable_blocks,
        "source_parent_order_unchanged": True, "automatic_truncation_or_topic_selection": False,
        "model_output_observed": False, "actual_input_tokens": None, "actual_total_tokens": None,
        "parent_source_mapping_verified": False, "semantic_entailment_verified": False,
        "publishable": False, "paid_execution_authorized": False, "model_calls": 0,
    }


def offline_content_counter(tokenizer_json=DEFAULT_TOKENIZER, expected_sha256=DEFAULT_TOKENIZER_SHA):
    """Read one fixed local tokenizer JSON; never download/import remote code."""
    path = Path(tokenizer_json).resolve()
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("exact_local_tokenizer_sha256_required")
    if sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("local_tokenizer_sha256_changed")
    from tokenizers import Tokenizer  # Optional isolated environment, not a main dependency.
    tokenizer = Tokenizer.from_file(str(path)); tokenizer.no_padding(); tokenizer.no_truncation()
    def count(messages):
        if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("final_text_messages_required")
        return sum(len(tokenizer.encode(m["content"], add_special_tokens=False).ids) for m in messages)
    count.provenance = {"tokenizer_file": str(path), "tokenizer_sha256": expected_sha256,
                        "add_special_tokens": False, "download_or_remote_code": False}
    return count


def literal_system_boundary(provider_source, expected_sha256):
    """Read only the reviewed string literal via AST; never import/exec source."""
    path = Path(provider_source).resolve(); raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("reviewed_provider_source_changed")
    values = [node.value for node in ast.parse(raw.decode("utf-8-sig")).body
              if isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SYSTEM_BOUNDARY"]
    if len(values) != 1 or not isinstance(values[0], ast.Constant) or not isinstance(values[0].value, str) or not values[0].value.strip():
        raise ValueError("one_nonempty_literal_system_boundary_required")
    return values[0].value


def load_case_bundles(input_file, expected_sha256=None):
    """Verify frozen case/index bytes before parsing; retain every indexed case."""
    path = Path(input_file).resolve(); raw = path.read_bytes()
    if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("frozen_selection_input_sha256_changed")
    value = json.loads(raw.decode("utf-8-sig")); _reject_answers(value)
    def case(row):
        if row.get("format") != "frozen-unified-selection-input-v1":
            raise ValueError("frozen_selection_case_format_required")
        if row.get("publishable") is not False or row.get("v19_formal_proof_created") is not False:
            raise ValueError("raw_only_unpublished_input_required")
        if not isinstance(row.get("question"), dict) or set(row["question"]) != {"question_id", "question"} or row["question_id"] != row["question"]["question_id"]:
            raise ValueError("projected_original_question_required")
        if not isinstance(row.get("catalog"), list) or any(not isinstance(r, dict) or any(not isinstance(r.get(k), str) for k in ("doc_id", "para_id", "paragraph_ref", "text")) for r in row["catalog"]):
            raise ValueError("program_original_doc_paragraph_catalog_required")
        return row
    if value.get("format") == "frozen-unified-selection-input-index-v1":
        if expected_sha256 is None:
            raise ValueError("frozen_index_expected_sha256_required")
        if value.get("model_calls") != 0 or value.get("reference_reads") or value.get("network_attempts"):
            raise ValueError("raw_only_zero_call_input_index_required")
        descriptors = value["cases"]
        if not descriptors or len(descriptors) != value["case_count"]:
            raise ValueError("complete_index_scope_required")
        rows, seen = [], set()
        for ordinal, descriptor in enumerate(descriptors, 1):
            current = Path(descriptor["file"]).resolve()
            if not current.is_relative_to(path.parent) or descriptor["ordinal"] != ordinal:
                raise ValueError("frozen_case_path_or_order_invalid")
            data = current.read_bytes()
            if sha256(data).hexdigest() != descriptor["sha256"]:
                raise ValueError("frozen_case_sha256_changed")
            row = json.loads(data.decode("utf-8-sig")); _reject_answers(row); row = case(row)
            identity = row["topic_id"], row["question_id"]
            if identity in seen or identity != (descriptor["topic_id"], descriptor["question_id"]) or row["binding"] != descriptor["binding"]:
                raise ValueError("frozen_case_identity_or_binding_changed")
            rows.append(row); seen.add(identity)
        return rows
    if value.get("format") == "frozen-unified-selection-input-v1":
        if expected_sha256 is None:
            raise ValueError("frozen_case_expected_sha256_required")
        return [case(value)]
    if set(value) == {"question", "catalog"}:
        return [value]  # Explicit small local fixture; no raw binding claim.
    raise ValueError("question_catalog_bundle_or_frozen_index_required")


def _offline_audit(event, args):
    if event.startswith("socket."):
        raise PermissionError("offline_preflight_network_forbidden")
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        name = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).name.lower()
        if name in {".env", "token", "stored_tokens", "evidence_samples.json", "conclusions.json", "expected.json"}:
            raise PermissionError("preflight_credentials_or_answers_forbidden")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("count", "prepare"))
    parser.add_argument("--input", type=Path, required=True, help="count: final-messages JSON; prepare: {question,catalog} JSON")
    parser.add_argument("--input-sha256")
    parser.add_argument("--system-boundary-file", type=Path)
    parser.add_argument("--provider-source", type=Path, default=ROOT / "app/agents/provider.py")
    parser.add_argument("--provider-sha256", default=DEFAULT_PROVIDER_SHA)
    parser.add_argument("--tokenizer-json", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-sha256", default=DEFAULT_TOKENIZER_SHA)
    parser.add_argument("--prior-known-tokens", type=int, default=0)
    parser.add_argument("--output-reservation", type=int, default=4096)
    parser.add_argument("--safety-reservation", type=int, default=512)
    parser.add_argument("--whole-limit", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); sys.addaudithook(_offline_audit)
    counter = offline_content_counter(args.tokenizer_json, args.tokenizer_sha256)
    if args.mode == "count":
        raw = args.input.read_bytes()
        if args.input_sha256 is not None and sha256(raw).hexdigest() != args.input_sha256:
            raise ValueError("final_messages_file_changed")
        value = json.loads(raw.decode("utf-8-sig")); _reject_answers(value)
        result = {"status": "offline_content_count_only", "message_content_tokens": counter(value),
                  "message_content_sha256": _digest(value), "is_actual_bill": False, "model_calls": 0}
    else:
        boundary = (args.system_boundary_file.read_text("utf-8") if args.system_boundary_file else
                    literal_system_boundary(args.provider_source, args.provider_sha256))
        rows = load_case_bundles(args.input, args.input_sha256); plans = []
        for row in rows:
            plan = prepare_obligation_selection(row["question"], row["catalog"], count_message_tokens=counter,
                prior_known_tokens=args.prior_known_tokens, output_reservation=args.output_reservation,
                safety_reservation=args.safety_reservation, whole_limit=args.whole_limit, system_boundary=boundary)
            plans.append({"topic_id": row.get("topic_id"), "question_id": row.get("question_id"), **plan})
        result = {"status": "closed_offline_preflight", "cases": plans, "case_count": len(plans),
                  "blocked_case_count": sum(p["status"] == "blocked" for p in plans), "model_calls": 0,
                  "automatic_scope_compression_or_question_selection": False,
                  "paid_execution_authorized": False, "raw_parent_binding_not_reverified_here": True}
    result["counter_provenance"] = counter.provenance
    result["input_file_sha256"] = sha256(args.input.read_bytes()).hexdigest()
    encoded = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"status": result["status"], "file": str(args.output.resolve()),
                      "sha256": sha256(encoded).hexdigest(), "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
