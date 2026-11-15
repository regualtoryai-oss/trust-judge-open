"""Bound raw-only JSON business bridge. No model, network, shell or publication.

Each invocation reconstructs local tool state by verifying a bounded transcript.
Only the launcher supplies case/script paths; operation arguments cannot name a
file. Result hashes use canonical JSON UTF-8, independent of native wire bytes.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.frozen_selection_inputs import canonical  # noqa: E402
from app.domain.source_bundle_tools import SourceBundleTools  # noqa: E402
from app.domain.numeric_literals import unsupported_numeric_claims  # noqa: E402


FORMAT = "frozen-source-bundle-case-v1"
ARGUMENT_KEYS = {"时间", "主体", "地点", "内容", "影响", "数值口径"}
LIMIT_KEYS = {"max_search_calls", "max_read_calls", "max_paragraphs", "max_read_anchors",
              "max_read_codepoints", "max_read_payload_bytes"}


def _closed(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or not set(required) <= set(value):
        raise ValueError("bundle_bridge_unknown_or_missing_fields")


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("bundle_bridge_duplicate_json_field")
        value[key] = item
    return value


def decode(data):
    return json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("bundle_bridge_nonfinite_json")))


def _annotation(row):
    _closed(row, {"start_paragraph_id", "end_paragraph_id", "fact_des", "argument", "entities", "proposed_roles"})
    if any(not isinstance(row[key], str) or not row[key].strip()
           for key in ("start_paragraph_id", "end_paragraph_id", "fact_des")):
        raise ValueError("bundle_bridge_literal_ids_and_description_required")
    if (not isinstance(row["argument"], dict) or set(row["argument"]) != ARGUMENT_KEYS
            or any(not isinstance(value, str) for value in row["argument"].values())):
        raise ValueError("bundle_bridge_argument_shape_required")
    if (not isinstance(row["entities"], list) or any(not isinstance(v, str) or not v.strip() for v in row["entities"])
            or len(set(row["entities"])) != len(row["entities"])):
        raise ValueError("bundle_bridge_unique_literal_entities_required")
    roles = row["proposed_roles"]
    if (not isinstance(roles, list) or not roles or any(v not in ("S", "C", "L") for v in roles)
            or len(set(roles)) != len(roles)):
        raise ValueError("bundle_bridge_unverified_role_shape_required")


def _operate(tools, operation, arguments):
    if operation == "inspect":
        _closed(arguments, set())
        return tools.scope()
    if operation == "search":
        _closed(arguments, {"query"}, {"limit", "exclude_paragraph_ids"})
        return tools.search(**arguments)
    if operation == "read":
        _closed(arguments, {"anchor_paragraph_ids", "before", "after"})
        return tools.read(**arguments)
    if operation == "plan":
        _closed(arguments, {"anchor_paragraph_ids", "before", "after"})
        return tools.plan(**arguments)
    if operation == "cite":
        _closed(arguments, {"start_paragraph_id", "end_paragraph_id"})
        return tools.cite(**arguments)
    if operation == "submit":
        _closed(arguments, {"selected"})
        selected = arguments["selected"]
        if not isinstance(selected, list) or len(selected) > 30:
            raise ValueError("bundle_bridge_bounded_selected_list_required")
        ranges, anchors, validated = set(), set(), []
        for row in selected:
            _annotation(row)
            pair = row["start_paragraph_id"], row["end_paragraph_id"]
            if pair in ranges:
                raise ValueError("bundle_bridge_duplicate_selected_range")
            if pair[0] in anchors:
                raise ValueError("bundle_bridge_duplicate_selected_start_anchor")
            ranges.add(pair)
            anchors.add(pair[0])
            validated.append(row)
        evidence = []
        for row in validated:
            source = tools.cite(row["start_paragraph_id"], row["end_paragraph_id"])
            # Same basic literal guards as strict_evidence.validate_annotation,
            # applied to the real document bundle, without forging a P catalog.
            # These prove neither entity attribution nor semantic truth.
            if not row["argument"]["内容"].strip():
                raise ValueError("bundle_bridge_nonempty_argument_content_required")
            if row["argument"]["时间"] and row["argument"]["时间"] not in source["quote"]:
                raise ValueError("bundle_bridge_time_must_be_literal_or_empty")
            fields = json.dumps([row["fact_des"], row["argument"], row["entities"]], ensure_ascii=False)
            if unsupported_numeric_claims(fields, source["quote"]):
                raise ValueError("bundle_bridge_annotation_number_not_in_source")
            evidence.append({**source, **deepcopy({key: row[key] for key in ("fact_des", "argument", "entities", "proposed_roles")})})
        scope = tools.scope()
        return {"format": "proposed-source-bundle-evidence-v1", "status": "proposed_unverified",
                "question": scope["original_question"], "evidence": evidence, "source_binding": scope["binding"],
                "source_only_validation": True, "basic_literal_grounding_checked": True, "annotation_semantics_verified": False,
                "native_model_delivery_verified": False, "publishable": False}
    raise ValueError("bundle_bridge_unknown_operation")


def _entry(tools, operation, arguments):
    if not isinstance(operation, str) or not isinstance(arguments, dict):
        raise ValueError("bundle_bridge_operation_arguments_required")
    try:
        value = _operate(tools, operation, arguments)
        return {"operation": operation, "arguments": deepcopy(arguments), "status": "ok",
                "result_sha256": sha256(canonical(value)).hexdigest()}, value
    except (ValueError, TypeError) as error:
        code = str(error)
        if not re.fullmatch(r"(?:source_bundle(?:_tools)?|bundle_bridge)_[a-z0-9_]+", code):
            code = "bundle_bridge_invalid_tool_arguments"
        return {"operation": operation, "arguments": deepcopy(arguments), "status": "rejected", "error_code": code}, None


def execute(case_bytes, case_sha256, request):
    if (not isinstance(case_bytes, bytes) or not re.fullmatch(r"[0-9a-f]{64}", case_sha256 or "")
            or sha256(case_bytes).hexdigest() != case_sha256):
        raise ValueError("bundle_bridge_case_sha256_mismatch")
    case = decode(case_bytes)
    _closed(case, {"format", "question", "documents", "paragraphs", "limits"}, {"topic_id"})
    if case["format"] != FORMAT or not isinstance(case["limits"], dict) or set(case["limits"]) - LIMIT_KEYS:
        raise ValueError("bundle_bridge_complete_raw_case_required")
    _closed(request, {"operation", "arguments", "history"})
    history = request["history"]
    if not isinstance(history, list) or len(history) > 64:
        raise ValueError("bundle_bridge_bounded_history_required")
    tools = SourceBundleTools(case["documents"], case["paragraphs"], case["question"], **case["limits"])
    for supplied in history:
        if not isinstance(supplied, dict) or supplied.get("operation") in ("inspect", "submit"):
            raise ValueError("bundle_bridge_nonterminal_tool_history_required")
        actual, _ = _entry(tools, supplied.get("operation"), supplied.get("arguments"))
        if actual != supplied:
            raise ValueError("bundle_bridge_history_replay_mismatch")
    entry, result = _entry(tools, request["operation"], request["arguments"])
    return {"format": "source-bundle-tool-response-v1", "case_sha256": case_sha256, "status": entry["status"],
            "history_entry": entry, **({"result": result} if result is not None else {"error_code": entry["error_code"]}),
            "scope": tools.scope(), "trace": tools.trace(), "model_calls": 0, "publishable": False}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(argv) != 2 or not Path(argv[0]).is_absolute():
            raise ValueError("bundle_bridge_absolute_bound_case_required")
        path = Path(argv[0])
        if path.name.lower().startswith(".env") or re.search(r"(?:evidence_samples|conclusions|expected|gold)(?:\.|[\\/])", str(path), re.I):
            raise ValueError("bundle_bridge_raw_case_path_required")
        case_bytes = path.read_bytes()
        if len(case_bytes) > 50_000_000:
            raise ValueError("bundle_bridge_case_too_large")
        request_bytes = sys.stdin.buffer.read(2_000_001)
        if len(request_bytes) > 2_000_000:
            raise ValueError("bundle_bridge_request_too_large")
        response = execute(case_bytes, argv[1], decode(request_bytes))
        sys.stdout.buffer.write(canonical(response) + b"\n")
        return 0
    except Exception as error:
        code = str(error)
        if not re.fullmatch(r"(?:source_bundle(?:_tools)?|bundle_bridge)_[a-z0-9_]+", code):
            code = "bundle_bridge_invalid_bound_request"
        sys.stdout.buffer.write(canonical({"status": "rejected", "error_code": code, "model_calls": 0, "publishable": False}) + b"\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
