"""Read-only CLOSED workbench/native audit followed by offline reference scoring.

Never constructs Repository (which can migrate/write SQLite), reads credentials,
calls a provider, repairs outputs or writes reference material to runtime paths.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sys
from uuid import NAMESPACE_URL, uuid5

from scripts.audit_dsh_closed_episode import bound
from scripts.final_evidence_scoring import normalize_legacy, score_final
from scripts.offline_reference_tools import load_reference_tools
from scripts.score_real_quality_episode import (brief_category, original_sources,
    read_bound, score_normalized, union_rows)
from scripts.two_stage_evidence_matching import location
from scripts.workbench_source_archive import SourceArchive, active_archived_module, relative_source

ROOT = Path(__file__).resolve().parents[1]


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("native_tool_arguments_duplicate_key")
        result[key] = value
    return result


def verify_visible_tool_wire(response):
    """Check raw provider tool JSON against the SDK view without interpreting it."""
    if "tool_calls_wire" not in response:
        return False  # Historical parent logs may predate raw argument capture.
    raw, parsed = response["tool_calls_wire"], response.get("tool_calls")
    if not isinstance(raw, list) or not isinstance(parsed, list) or len(raw) != 1 or len(parsed) != 1:
        raise ValueError("one_raw_and_SDK_visible_tool_call_required")
    function = raw[0].get("function", {})
    if (not isinstance(function.get("arguments"), str)
            or function.get("name") != parsed[0].get("name")
            or json.loads(function["arguments"], object_pairs_hook=unique_json_object) != parsed[0].get("args")):
        raise ValueError("native_raw_tool_arguments_differ_from_SDK_view")
    return True


def bound_code_module(relative_path, expected_sha256):
    if expected_sha256 is None:
        raise ValueError("explicit_source_bound_module_required")
    archived = active_archived_module(relative_path, expected_sha256)
    if archived is not None:
        return archived
    path = ROOT / relative_path
    bound(path, expected_sha256)
    name = "offline_bound_" + path.stem + "_" + expected_sha256
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def archived_empty_predicate_projection(schema):
    """Name the old closed empty-predicate bug; retain all actual field rules."""
    result = deepcopy(schema)
    changes = []
    def walk(node, path):
        if isinstance(node, list):
            for i, child in enumerate(node): walk(child, path + [i])
        elif isinstance(node, dict):
            predicates = node.get("anyOf")
            required = set(node.get("required", []))
            if isinstance(predicates, list) and predicates and required:
                empty = all(isinstance(branch, dict)
                    and not (set(branch) - {"properties", "required", "additionalProperties", "type"})
                    and branch.get("additionalProperties") is False and branch.get("type", "object") == "object"
                    and isinstance(branch.get("properties"), dict) and branch["properties"]
                    and all(value == {} for value in branch["properties"].values())
                    and set(branch.get("required", [])) == set(branch["properties"]) <= required
                    for branch in predicates)
                if empty:
                    changes.append({"path": path + ["anyOf"], "reason": "archived_closed_empty_predicate_after_minItems_removal",
                        "original_predicates": deepcopy(predicates)})
                    node.pop("anyOf")
            for key, child in list(node.items()): walk(child, path + [key])
    walk(result, [])
    return result, changes


def wire_payload(wire):
    rows = []
    for message in wire.get("messages", []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            try:
                value = json.loads(message["content"], object_pairs_hook=unique_json_object)
            except ValueError:
                continue
            if isinstance(value, dict) and isinstance(value.get("evidence"), list): rows.append(value)
    if len(rows) != 1:
        raise ValueError("one_exact_native_evidence_payload_required")
    return rows[0]


def verify_tail_schema(decoded, wire, *, operation, local_schema_sha256=None, allow_archived_predicates=False,
                       scoped_schema_sha256=None):
    from jsonschema import Draft202012Validator, ValidationError
    schema = wire["tools"][0]["function"]["parameters"]
    original_valid, changes = True, []
    try:
        Draft202012Validator(schema).validate(decoded)
    except ValidationError:
        if not allow_archived_predicates or local_schema_sha256 is None:
            raise
        projected, changes = archived_empty_predicate_projection(schema)
        if not changes:
            raise
        Draft202012Validator(projected).validate(decoded)
        original_valid = False
    local_checked = False
    if local_schema_sha256 is not None:
        module = bound_code_module("app/agents/schemas.py", local_schema_sha256)
        if operation == "arbitrate" and wire["tools"][0]["function"]["name"] == "ScopedArbitrationWireOutput":
            scope_module = bound_code_module("app/agents/scoped_arbitration.py", scoped_schema_sha256)
            scope_module.ArbitrationOutput = module.ArbitrationOutput
            aliases = wire_payload(wire)["evidence"]
            cls, _instructions = scope_module.prepare_scoped_arbitration(module.ArbitrationOutput, aliases,
                **({"question": wire_payload(wire)["question"], "issue_frame": True}
                   if "issue_frame" in schema.get("properties", {}) else {}))
        else:
            cls = module.ArbitrationOutput if operation == "arbitrate" else module.AnalysisOutput
        if cls.model_validate(decoded).model_dump() != decoded:
            raise ValueError("local_schema_cannot_rewrite_or_complete_original_model_fields")
        local_checked = True
    return {"wire_schema_independently_validated": original_valid,
        "archived_wire_schema_anomaly": changes, "authoritative_source_bound_local_schema_validated": local_checked,
        "original_wire_preserved": True, "all_model_field_values_unchanged": True}


def verify_syntax_recovery(response, wire, audit, *, operation, response_sha256, decoder_sha256,
                           local_schema_sha256=None, allow_archived_predicates=False, scoped_schema_sha256=None):
    """Recompute quote-only insertions from immutable raw arguments, with no AI."""
    if decoder_sha256 is None or operation not in {"authority_analyst", "cross_validation_analyst", "arbitrate"}:
        raise ValueError("syntax_recovery_requires_source_bound_tail_decoder")
    module = bound_code_module("app/agents/strict_json_syntax.py", decoder_sha256)
    calls = response.get("tool_calls_wire")
    if (not isinstance(calls, list) or len(calls) != 1 or calls[0].get("type") != "function"
            or calls[0].get("function", {}).get("name") != wire["tools"][0]["function"]["name"]):
        raise ValueError("syntax_recovery_one_original_named_function_required")
    original = calls[0]["function"]["arguments"]
    decoded, recomputed = module.decode_visible_arguments(original)
    if recomputed is None or any(audit.get(key) != value for key, value in recomputed.items()):
        raise ValueError("syntax_recovery_audit_differs_from_source_bound_decoder")
    closer = recomputed.get("format") == "strict-json-single-unmatched-closer-v1"
    expected_scope = "single_unmatched_closing_delimiter_deletion_only" if closer else "quote_escape_insertions_only"
    if (audit.get("operation") != operation or audit.get("original_response_sha256") != response_sha256
            or audit.get("original_response_preserved") is not True or audit.get("schema_validation_passed") is not True
            or audit.get("new_model_calls") != 0 or audit.get("scope") != expected_scope):
        raise ValueError("syntax_recovery_original_response_or_zero_call_scope_changed")
    if closer:
        deletions = recomputed.get("deletions", [])
        if len(deletions) != 1 or recomputed.get("insertions") != []:
            raise ValueError("syntax_recovery_exact_single_closer_deletion_required")
        change = deletions[0]; offset, char = change.get("offset"), change.get("delete")
        if (type(offset) is not int or not 0 <= offset < len(original) or char not in "]}" or original[offset] != char
                or change != {"offset": offset, "delete": char}):
            raise ValueError("syntax_recovery_original_unmatched_closer_offset_required")
        in_string, escaped, stack = False, False, []
        for index, current in enumerate(original[:offset + 1]):
            if in_string:
                if escaped: escaped = False
                elif current == "\\": escaped = True
                elif current == '"': in_string = False
            elif current == '"': in_string = True
            elif current in "{[": stack.append(current)
            elif current in "}]":
                matched = bool(stack) and stack[-1] == {"}": "{", "]": "["}[current]
                if index == offset:
                    if matched: raise ValueError("syntax_recovery_may_not_delete_matched_container")
                elif not matched:
                    raise ValueError("syntax_recovery_more_than_one_unmatched_closer")
                else: stack.pop()
        if in_string:
            raise ValueError("syntax_recovery_may_not_delete_string_content")
        repaired = original[:offset] + original[offset + 1:]
        if (sha256(original.encode()).hexdigest() != audit["original_arguments_sha256"]
                or sha256(repaired.encode()).hexdigest() != audit["repaired_arguments_sha256"]
                or json.loads(repaired, object_pairs_hook=unique_json_object) != decoded):
            raise ValueError("syntax_recovery_independent_deletion_or_hash_disagree")
        schema_audit = verify_tail_schema(decoded, wire, operation=operation, local_schema_sha256=local_schema_sha256,
            allow_archived_predicates=allow_archived_predicates, scoped_schema_sha256=scoped_schema_sha256)
        return {"operation": operation, **recomputed, "original_response_preserved": True, **schema_audit,
            "new_model_calls": 0, "only_quote_escape_insertions": False, "only_single_unmatched_closer_deletion": True}
    offsets = [row["offset"] for row in recomputed["insertions"]]
    if (not offsets or any(type(offset) is not int or not 0 <= offset < len(original) for offset in offsets)
            or sorted(set(offsets)) != offsets):
        raise ValueError("syntax_recovery_unique_original_codepoint_offsets_required")
    repaired = original
    for row in reversed(recomputed["insertions"]):
        if row != {"offset": row["offset"], "insert": "\\", "before": '"'} or original[row["offset"]] != '"':
            raise ValueError("syntax_recovery_may_only_insert_backslash_before_original_quote")
        repaired = repaired[:row["offset"]] + "\\" + repaired[row["offset"]:]
    if (sha256(original.encode()).hexdigest() != audit["original_arguments_sha256"]
            or sha256(repaired.encode()).hexdigest() != audit["repaired_arguments_sha256"]
            or json.loads(repaired, object_pairs_hook=unique_json_object) != decoded):
        raise ValueError("syntax_recovery_independent_insertions_or_hash_disagree")
    schema_audit = verify_tail_schema(decoded, wire, operation=operation, local_schema_sha256=local_schema_sha256,
        allow_archived_predicates=allow_archived_predicates, scoped_schema_sha256=scoped_schema_sha256)
    return {"operation": operation, "insertions": recomputed["insertions"],
        "original_arguments_sha256": recomputed["original_arguments_sha256"],
        "repaired_arguments_sha256": recomputed["repaired_arguments_sha256"],
        "library": recomputed["library"], "library_version": recomputed["library_version"],
        "original_response_preserved": True, **schema_audit,
        "new_model_calls": 0, "only_quote_escape_insertions": True}


def audit_cached_requests(rows, parent_manifests, guards, *, syntax_decoder_sha256=None, local_schema_sha256=None,
                          allow_bundle_tail=False, scoped_schema_sha256=None, allow_bundle_partial_analyses=False):
    """Audit exact, pre-paid raw responses separately from new HTTP requests."""
    if allow_bundle_partial_analyses and not allow_bundle_tail:
        raise ValueError("partial_cached_analyses_require_explicit_bundle_tail")
    wires, responses, operations, projections, bindings = [], [], [], [], []
    allowed = (["authority_analyst", "cross_validation_analyst", "arbitrate"] if allow_bundle_tail else
        ["strict_selection", "strict_annotation", "authority_analyst", "cross_validation_analyst"])
    for row in rows:
        operation = row.get("operation")
        if operation not in allowed or operation in operations:
            raise ValueError("cache_reuse_limited_to_unique_pre_arbitration_stages")
        hit = descriptor(row["cache_hit"], guards)
        delivered = descriptor(row["finished"], guards)
        if (delivered.get("operation") != operation or delivered.get("request_started") is not False
                or delivered.get("cache_hit") is not True or delivered.get("outcome") != "success"
                or delivered.get("unknown_usage_requests") != 0):
            raise ValueError("cache_prime_alone_is_not_successful_structured_invoke_delivery")
        parent = parent_manifests.get(hit.get("parent_run_id"))
        ordinal = hit.get("parent_request_ordinal")
        if (parent is None or type(ordinal) is not int or ordinal < 1
                or hit.get("no_provider_request_started") is not True
                or hit.get("exact_request_JSON_values_equal") is not True):
            raise ValueError("cache_hit_must_bind_explicit_paid_cost_parent_request")
        matches = []
        for index, candidate in enumerate(parent["requests"], 1):
            match = re.match(r"request-(\d+)-", Path(candidate["wire"]["file"]).name)
            if (int(match.group(1)) if match else index) == ordinal:
                matches.append(candidate)
        if len(matches) != 1:
            raise ValueError("cache_hit_parent_native_request_ordinal_not_unique")
        original = matches[0]
        pre = descriptor(original["preflight"], guards)
        finish = descriptor(original["finished"], guards)
        if (pre.get("operation") != operation or pre.get("admitted") is not True
                or finish.get("request_started") is not True
                or hit.get("parent_wire_sha256") != original["wire"]["sha256"]
                or hit.get("parent_response_sha256") != original["response"]["sha256"]):
            raise ValueError("cache_hit_parent_operation_response_or_wire_identity_changed")
        original_wire, original_response = (descriptor(original[key], guards) for key in ("wire", "response"))
        wire, response = (descriptor(row[key], guards) for key in ("wire", "response"))
        if wire != original_wire or response != original_response:
            raise ValueError("cache_reuse_requires_identical_whole_request_and_original_visible_response")
        syntax = None
        if "syntax-recovery" in row:
            original_settlement = descriptor(original["settlement"], guards)
            if (operation not in ({"authority_analyst", "cross_validation_analyst", "arbitrate"} if allow_bundle_tail else {"authority_analyst", "cross_validation_analyst"})
                    or original_settlement.get("outcome") != "invalid_output" or finish.get("outcome") != "invalid_output"
                    or response.get("tool_calls") != []):
                raise ValueError("cached_syntax_recovery_requires_preserved_original_invalid_output")
            syntax = verify_syntax_recovery(response, wire, descriptor(row["syntax-recovery"], guards),
                operation=operation, response_sha256=row["response"]["sha256"], decoder_sha256=syntax_decoder_sha256,
                local_schema_sha256=local_schema_sha256, allow_archived_predicates=True, scoped_schema_sha256=scoped_schema_sha256)
        else:
            verify_visible_tool_wire(response)
        schema_audit = None
        if operation in ({"authority_analyst", "cross_validation_analyst", "arbitrate"} if allow_bundle_tail else {"authority_analyst", "cross_validation_analyst"}) and local_schema_sha256 is not None:
            if syntax is not None:
                schema_audit = {key: syntax[key] for key in ("wire_schema_independently_validated", "archived_wire_schema_anomaly",
                    "authoritative_source_bound_local_schema_validated", "original_wire_preserved", "all_model_field_values_unchanged")}
            else:
                decoded = json.loads(response["tool_calls_wire"][0]["function"]["arguments"], object_pairs_hook=unique_json_object)
                schema_audit = verify_tail_schema(decoded, wire, operation=operation, local_schema_sha256=local_schema_sha256,
                    allow_archived_predicates=True, scoped_schema_sha256=scoped_schema_sha256)
        projection = descriptor(row["field-projection"], guards) if "field-projection" in row else None
        if projection is not None and (operation != "strict_annotation" or projection.get("operation") != operation):
            raise ValueError("cache_projection_bound_to_reused_annotation_required")
        wires.append(wire); responses.append(response); operations.append(operation); projections.append(projection)
        bindings.append({"operation": operation, "parent_run_id": hit["parent_run_id"],
            "parent_request_ordinal": ordinal, "original_wire_sha256": original["wire"]["sha256"],
            "original_response_sha256": original["response"]["sha256"], "new_provider_request": False,
            "new_cost": False, "entire_request_JSON_values_equal": True, "syntax_recovery": syntax,
            "model_schema_audit": schema_audit})
    invalid = ((len(operations) != 2 or set(operations) != {"authority_analyst", "cross_validation_analyst"})
        if allow_bundle_partial_analyses else
        (len(operations) != 3 or set(operations) != set(allowed) or operations[-1] != "arbitrate")
        if allow_bundle_tail else operations not in (allowed[:2], allowed))
    if operations and invalid:
        raise ValueError("cache_reuse_must_cover_exact_two_or_four_logical_stages_in_order")
    return wires, responses, operations, projections, bindings


def descriptor(row, guards):
    name = Path(row["file"]).name.lower()
    if name in {"evidence_samples.json", "conclusions.json", ".env"} or name.startswith("gold_"):
        raise ValueError("runtime_binding_cannot_open_reference_or_credential_file")
    return read_bound(row["file"], guards, row["sha256"])


def known_usage(rows):
    if not isinstance(rows, list):
        raise ValueError("actual_provider_usage_records_required")
    total = {key: 0 for key in ("input_tokens", "output_tokens", "total_tokens")}
    for row in rows:
        if (row.get("usage_source") != "provider"
                or any(type(row.get(k)) is not int or row[k] < 0 for k in total)
                or row["input_tokens"] + row["output_tokens"] != row["total_tokens"]):
            raise ValueError("complete_consistent_actual_provider_usage_required")
        for key in total:
            total[key] += row[key]
    return total


def native_paid_identity(manifest, guards):
    """Stable provenance despite different closure paths or API-derived fields."""
    identity = {"episode": manifest["episode"]["sha256"], "known_usage": manifest["known_usage"]}
    for kind in ("requests", "cached_requests"):
        identity[kind] = []
        for row in manifest.get(kind, []):
            fields = {}
            for key, value in row.items():
                if isinstance(value, dict) and "file" in value and "sha256" in value:
                    descriptor(value, guards)
                    fields[key] = value["sha256"]
                elif key == "operation":
                    fields[key] = value
            identity[kind].append(fields)
    identity["parent_run_ids"] = sorted(descriptor(row["run"], guards)["id"] for row in manifest.get("costparents", []))
    if manifest.get("resume_prior_usage"):
        descriptor(manifest["resume_prior_usage"], guards)
        identity["resume_prior_usage"] = manifest["resume_prior_usage"]["sha256"]
    return identity


def usage_signature(rows):
    known_usage(rows)
    return sorted((x["operation"], x["input_tokens"], x["output_tokens"], x["total_tokens"]) for x in rows)


def audit_native(manifest, guards, *, run_id, run_usage, _seen=None, _memo=None, _check_sources=True):
    episode = descriptor(manifest["episode"], guards)
    archive = {relative_source(key): value for key, value in manifest.get("source_archive", episode.get("source_archive", {})).items()}
    expected = {relative_source(key): value for key, value in episode.get("sources", {}).items()}
    if archive and set(archive) != set(expected):
        raise ValueError("source_archive_must_cover_exact_all_native_runtime_sources")
    for key, row in archive.items():
        if row.get("sha256") != expected[key]:
            raise ValueError("archive_sha_does_not_match_native_runtime_source")
        guards.append(bound(row["file"], expected[key])[1])
    with SourceArchive(archive):
        return _audit_native(manifest, guards, run_id=run_id, run_usage=run_usage, _seen=_seen,
            _memo=_memo, _check_sources=_check_sources, _source_archive=archive)


def _audit_native(manifest, guards, *, run_id, run_usage, _seen=None, _memo=None, _check_sources=True, _source_archive=None):
    if (manifest.get("closed") is not True or manifest.get("real_provider") is not True
            or manifest.get("fixture") is not False or manifest.get("run_id") != run_id):
        raise ValueError("closed_actual_native_workbench_manifest_required")
    seen = set() if _seen is None else _seen
    if run_id in seen:
        raise ValueError("cost_parent_cycle")
    memo = {} if _memo is None else _memo
    identity = native_paid_identity(manifest, guards)
    if run_id in memo:
        previous = memo[run_id]
        if identity != previous["identity"] or usage_signature(run_usage) != previous["usage_signature"]:
            raise ValueError("same_cost_parent_run_has_inconsistent_native_or_usage_bindings")
        return previous["value"]
    seen.add(run_id)
    episode = descriptor(manifest["episode"], guards)
    if (episode.get("strict") is not True or episode.get("thinking") != "disabled"
            or episode.get("temperature") != 0 or episode.get("automatic_retries") != 0
            or episode.get("model") != "deepseek-v4-pro"):
        raise ValueError("frozen_strict_actual_model_required")
    runtime_sources = {}
    for file, expected in episode.get("sources", {}).items():
        if not file.replace("\\", "/").startswith("app/") or not file.endswith(".py") or ".." in Path(file).parts:
            raise ValueError("native_runtime_source_must_be_project_python_file")
        runtime_sources[file.replace("\\", "/")] = expected
        if _check_sources and not _source_archive:
            guards.append(bound(ROOT / file, expected)[1])
    prior_by_run = {}
    for parent in manifest.get("costparents", []):
        parent_run = descriptor(parent["run"], guards)
        parent_manifest = descriptor(parent["native_manifest"], guards)
        if parent_run.get("id") != parent_manifest.get("run_id") or len(parent_run.get("usage", {})) != 1:
            raise ValueError("bound_single_question_cost_parent_required")
        parent_audit, _w, _r = audit_native(parent_manifest, guards, run_id=parent_run["id"],
            run_usage=next(iter(parent_run["usage"].values())), _seen=seen, _memo=memo, _check_sources=False)
        for identity_id, usage_rows in parent_audit["usage_records_by_run_id"].items():
            if identity_id in prior_by_run and prior_by_run[identity_id] != usage_rows:
                raise ValueError("same_cost_parent_run_has_inconsistent_native_or_usage_bindings")
            prior_by_run[identity_id] = usage_rows
    prior_records = [record for rows in prior_by_run.values() for record in rows]
    prior_requests = len(prior_records)
    prior = known_usage(prior_records)
    if prior_records:
        seed = descriptor(manifest["resume_prior_usage"], guards)
        if (known_usage(seed["records"]) != prior or len(seed["records"]) != prior_requests
                or seed["prior_known_tokens"] != prior["total_tokens"] or seed["prior_calls"] != prior_requests
                or seed.get("not_new_usage") is not True):
            raise ValueError("runtime_prior_seed_must_equal_all_closed_failed_ancestor_usage")
    known, records, wires, responses, operations, projections, invalid_visible, syntax_recoveries, actual_schemas, relation_projections, response_shas = 0, [], [], [], [], [], [], [], [], [], []
    for request in manifest["requests"]:
        wire = descriptor(request["wire"], guards)
        pre = descriptor(request["preflight"], guards)
        finish = descriptor(request["finished"], guards)
        if (pre["wire"]["sha256"] != request["wire"]["sha256"]
                or pre["known_tokens_before"] != prior["total_tokens"] + known):
            raise ValueError("native_preflight_bound_wire_and_whole_cost_required")
        expected = prior["total_tokens"] + known + pre["serialized_json_token_estimate"] + pre["reserved_output_tokens"] + pre["safety_tokens"]
        if pre.get("admitted") is not True or not finish.get("request_started"):
            if pre.get("admitted") is not False or finish.get("request_started") is not False:
                raise ValueError("unadmitted_request_must_not_start")
            if request.get("settlement") or request.get("response"):
                raise ValueError("blocked_request_cannot_have_paid_response")
            continue
        if expected > pre["experimental_stop_threshold"] or pre["experimental_stop_threshold"] != episode["experimental_stop_threshold"]:
            raise ValueError("over_experimental_stop_request_admitted")
        tools = wire.get("tools", [])
        if (wire.get("model") != episode["model"] or wire.get("thinking") != {"type": "disabled"}
                or len(tools) != 1 or tools[0].get("function", {}).get("strict") is not True
                or wire.get("tool_choice", {}).get("function", {}).get("name") != tools[0]["function"]["name"]
                or wire.get("max_tokens") != pre["reserved_output_tokens"]):
            raise ValueError("actual_native_strict_tool_or_output_contract_changed")
        settlement = descriptor(request["settlement"], guards)
        response = descriptor(request["response"], guards)
        if "syntax-recovery" in request:
            syntax_recoveries.append(verify_syntax_recovery(response, wire, descriptor(request["syntax-recovery"], guards),
                operation=pre["operation"], response_sha256=request["response"]["sha256"],
                decoder_sha256=runtime_sources.get("app/agents/strict_json_syntax.py"),
                local_schema_sha256=runtime_sources.get("app/agents/schemas.py"),
                scoped_schema_sha256=runtime_sources.get("app/agents/scoped_arbitration.py")))
        else:
            try:
                verify_visible_tool_wire(response)
            except ValueError as error:
                if finish.get("outcome") not in {"invalid_output", "length", "filtered", "transport_failure"}:
                    raise
                invalid_visible.append({"operation": pre["operation"], "outcome": finish["outcome"],
                    "raw_response_preserved": True, "tool_parse_error": str(error), "paid_cost_retained": True})
        if (_check_sources and finish.get("outcome") == "success"
                and pre["operation"] in {"authority_analyst", "cross_validation_analyst", "arbitrate"}
                and runtime_sources.get("app/agents/schemas.py")):
            raw_arguments = response["tool_calls_wire"][0]["function"]["arguments"]
            if "syntax-recovery" in request:
                decoded, _recovery = bound_code_module("app/agents/strict_json_syntax.py",
                    runtime_sources["app/agents/strict_json_syntax.py"]).decode_visible_arguments(raw_arguments)
            else:
                decoded = json.loads(raw_arguments, object_pairs_hook=unique_json_object)
            actual_schemas.append({"operation": pre["operation"], **verify_tail_schema(decoded, wire,
                operation=pre["operation"], local_schema_sha256=runtime_sources["app/agents/schemas.py"],
                allow_archived_predicates=False, scoped_schema_sha256=runtime_sources.get("app/agents/scoped_arbitration.py"))})
        projection = descriptor(request["field-projection"], guards) if "field-projection" in request else None
        if projection is not None and (projection.get("operation") != pre["operation"]
                or pre["operation"] != "strict_annotation"):
            raise ValueError("field_projection_bound_to_annotation_operation_required")
        if (settlement.get("actual_request_started") is not True or settlement.get("usage_known") is not True
                or settlement.get("operation") != pre["operation"] or settlement.get("known_tokens_before") != known
                or finish.get("operation") != pre["operation"] or finish.get("unknown_usage_requests") != 0):
            raise ValueError("settled_known_actual_native_request_required")
        counts = {key: settlement.get(key) for key in ("input_tokens", "output_tokens", "total_tokens")}
        known_usage([{**counts, "usage_source": "provider"}])
        if any(response.get("usage", {}).get(key) != counts[key] for key in counts):
            raise ValueError("visible_provider_usage_and_settlement_disagree")
        known += counts["total_tokens"]
        if settlement["known_total_tokens_after"] != known or finish["known_total_tokens"] != known:
            raise ValueError("cumulative_native_usage_changed")
        if prior_records and finish.get("known_episode_tokens_with_ancestors") != prior["total_tokens"] + known:
            raise ValueError("finished_whole_episode_usage_dropped_ancestors")
        records.append({**counts, "operation": pre["operation"], "usage_source": "provider"})
        wires.append(wire)
        responses.append(response)
        operations.append(pre["operation"])
        projections.append(projection)
        relation_projections.append(descriptor(request["relation-projection"], guards) if "relation-projection" in request else None)
        response_shas.append(request["response"]["sha256"])
    if (prior_requests + len(records) > episode.get("max_requests", 0)
            or known_usage(records) != manifest["known_usage"] or known_usage(records) != known_usage(run_usage)
            or sorted((x["operation"], x["input_tokens"], x["output_tokens"], x["total_tokens"]) for x in records)
               != sorted((x["operation"], x["input_tokens"], x["output_tokens"], x["total_tokens"]) for x in run_usage)):
        raise ValueError("native_run_usage_or_request_order_disagrees")
    reused_wires, reused_responses, reused_operations, reused_projections, cached_bindings = audit_cached_requests(
        manifest.get("cached_requests", []), {key: value["manifest"] for key, value in memo.items()}, guards,
        syntax_decoder_sha256=runtime_sources.get("app/agents/strict_json_syntax.py"),
        local_schema_sha256=runtime_sources.get("app/agents/schemas.py"))
    if set(reused_operations) & set(operations):
        raise ValueError("cached_and_actual_stage_cannot_both_claim_same_logical_delivery")
    logical_operations = reused_operations + operations
    if "logical_operations" in manifest and manifest["logical_operations"] != logical_operations:
        raise ValueError("native_logical_operation_order_disagrees_with_bound_deliveries")
    result = {"actual_requests": len(records), "known_usage": known_usage(records),
        "prior_actual_requests": prior_requests, "prior_known_usage": prior,
        "whole_episode_actual_requests": prior_requests + len(records),
        "whole_episode_known_usage": known_usage(prior_records + records),
        "all_episode_usage_records": prior_records + records,
        "usage_records_by_run_id": {**prior_by_run, run_id: records},
        "cost_ancestor_run_ids": list(prior_by_run), "repeated_parent_run_ids_charged_once": True,
        "invalid_visible_paid_responses": invalid_visible,
        "actual_request_syntax_recoveries": syntax_recoveries,
        "current_actual_tail_schema_audits": actual_schemas,
        "operations": operations, "logical_operations": logical_operations,
        "cached_requests": len(cached_bindings), "cached_request_bindings": cached_bindings,
        "field_projections": reused_projections + projections, "recorded_native_JSON_values_verified": True,
        "relation_projections": [None] * len(reused_operations) + relation_projections,
        "response_file_sha256": [None] * len(reused_operations) + response_shas,
        "runtime_sources_sha256": runtime_sources, "runtime_source_archive_descriptors": _source_archive or {},
        "archived_runtime_source_bytes_checked": bool(_source_archive),
        "current_runtime_source_bytes_checked": _check_sources and not _source_archive,
        "balanced_scoped_enabled": episode.get("balanced_scoped_enabled", False),
        "balance_policy": episode.get("balance_policy"), "recall_strategy": episode.get("recall_strategy"),
        "experimental_stop_threshold": episode["experimental_stop_threshold"],
        "remote_service_signed_receipt": False, "hidden_reasoning_read": False}, reused_wires + wires, reused_responses + responses
    seen.remove(run_id)
    memo[run_id] = {"identity": identity, "usage_signature": usage_signature(run_usage),
        "manifest": manifest, "value": result}
    return result


def verify_source_rows(raw, paragraphs):
    contexts, documents = original_sources(raw)
    mapping = {}
    for row in paragraphs:
        key = (row["raw_doc_id"], row["raw_para_id"])
        if (key in mapping or row["text"] != contexts.get(key)
                or row.get("title") != documents.get(key[0], {}).get("title")
                or row["id"] in {p["id"] for p in mapping.values()}):
            raise ValueError("same_complete_original_workbench_source_rows_required")
        mapping[key] = row
    if set(mapping) != set(contexts):
        raise ValueError("workbench_full_original_corpus_binding_required")
    return contexts, mapping


def verify_candidates(raw_case, question, source_rows, first_wire):
    if raw_case.get("question") != question:
        raise ValueError("same_original_candidate_question_required")
    seen, seen_keys, expected = set(), set(), []
    for parent in raw_case["catalog"]:
        key, ref = location(parent), parent["paragraph_ref"]
        original = source_rows.get(key)
        if (original is None or ref in seen or key in seen_keys or parent.get("paragraph_complete") is not True
                or parent["text"] != original["text"] or parent["title"] != original["title"]
                or raw_case["source_metadata"][ref]["paragraph_id"] != original["id"]):
            raise ValueError("candidate_full_original_identity_binding_required")
        seen.add(ref)
        seen_keys.add(key)
        expected.append({"p": ref, "title": parent["title"], "text": parent["text"]})
    payloads = []
    for message in first_wire.get("messages", []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            try:
                payloads.append(json.loads(message["content"]))
            except ValueError:
                pass
    wanted = {"question": question["question"], "paragraphs": expected,
        "complete_parents": True, "outside_and_adjacent_context": "unknown"}
    if payloads.count(wanted) != 1:
        raise ValueError("first_actual_wire_must_contain_exact_complete_candidate_payload")
    return {location(row) for row in raw_case["catalog"]}


def verify_annotation_projection(raw_case, raw_annotations, canonical, projection_audit):
    """Independently permit only a nonliteral optional time becoming unknown."""
    originals = raw_annotations.get("annotations")
    rows = canonical.get("annotations")
    if not isinstance(originals, dict) or not isinstance(rows, list):
        raise ValueError("keyed_raw_annotation_and_canonical_list_required")
    sources = {row["paragraph_ref"]: row["text"] for row in raw_case["catalog"]}
    by_ref = {row["p"]: {k: v for k, v in row.items() if k != "p"} for row in rows}
    if len(by_ref) != len(rows) or set(originals) != set(by_ref):
        raise ValueError("annotation_projection_cannot_add_drop_or_duplicate_parents")
    changes = []
    for p, annotation in originals.items():
        expected = deepcopy(annotation)
        time = expected.get("argument", {}).get("时间")
        if isinstance(time, str) and time and time not in sources[p]:
            expected["argument"]["时间"] = ""
            changes.append({"p": p, "field": "argument.时间", "before": time,
                "after": "", "reason": "time_must_be_literal_or_empty"})
        if expected != by_ref[p]:
            raise ValueError("annotation_projection_changed_other_or_literal_model_fields")
    if projection_audit is None:
        if changes:
            raise ValueError("actual_time_projection_requires_bound_native_audit")
    elif (projection_audit.get("operation") != "strict_annotation"
            or projection_audit.get("original_model_response_preserved") is not True
            or projection_audit.get("allowed_projection") != "nonliteral_argument_time_to_empty_only"
            or not isinstance(projection_audit.get("changes"), list)
            or sorted(projection_audit["changes"], key=lambda row: row["p"]) != sorted(changes, key=lambda row: row["p"])):
        raise ValueError("native_projection_audit_differs_from_independent_raw_field_diff")
    return {"changes": changes, "native_audit_present": projection_audit is not None,
        "only_nonliteral_optional_time_to_empty": True, "all_other_model_fields_unchanged": True}


def canonical_digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_annotations(result, raw_case, responses, *, decoder_sha256=None, projection_audit=None,
                       source_hashes=None, expected_balance_policy=None, expected_recall_strategy=None):
    proof = result.get("strict_evidence_proof", {})
    try:
        selection = responses[0]["tool_calls"][0]["args"]
        annotations = responses[1]["tool_calls"][0]["args"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("visible_actual_selection_and_annotation_tool_results_required") from None
    source_hashes = source_hashes or {}
    balanced = result.get("verification_protocol") == "strict-workbench-evidence-balanced-v2"
    used_keyed_decoder = isinstance(selection.get("decisions"), dict)
    projection = {"changes": [], "native_audit_present": False, "all_other_model_fields_unchanged": True}
    annotation_raw, annotation_selection, allocation = raw_case, None, None
    if used_keyed_decoder:
        if decoder_sha256 is None:
            raise ValueError("keyed_wire_requires_source_bound_fixed_decoder")
        module = bound_code_module("app/agents/strict_evidence.py", decoder_sha256)
        if balanced:
            compact = bound_code_module("app/agents/compact_selection.py",
                source_hashes.get("app/agents/compact_selection.py"))
            selection = compact.canonical_compact_selection(raw_case, selection).model_dump()
        else:
            selection = module.canonical_selection_wire(raw_case, selection).model_dump()
        if balanced:
            workflow = bound_code_module("app/workflows/strict_workbench.py",
                source_hashes.get("app/workflows/strict_workbench.py"))
            if (proof.get("balance_policy") != expected_balance_policy
                    or proof.get("recall_strategy") != expected_recall_strategy
                    or expected_balance_policy is None or expected_recall_strategy != "balanced"):
                raise ValueError("published_balance_policy_and_recall_must_match_before_launch_freeze")
            annotation_raw, annotation_selection, allocation = workflow.annotation_inputs(
                raw_case, selection, expected_balance_policy)
            if proof.get("balance_ledger") != allocation:
                raise ValueError("published_balance_ledger_differs_from_full_model_selection_projection")
            hashes = {"raw_case_sha256": raw_case, "annotation_raw_case_sha256": annotation_raw,
                "annotation_selection_sha256": annotation_selection,
                "source_metadata_sha256": raw_case["source_metadata"], "retrieval_sha256": raw_case["retrieval"]}
            if any(proof.get(key) != canonical_digest(value) for key, value in hashes.items()):
                raise ValueError("published_balanced_projection_input_hashes_disagree")
            copied_proof = deepcopy(proof)
            proof_hash = copied_proof.pop("proof_sha256", None)
            if proof_hash != canonical_digest(copied_proof):
                raise ValueError("published_balanced_proof_hash_changed")
        raw_annotations = deepcopy(annotations)
        annotations = module.canonical_annotation_wire(annotation_raw,
            annotation_selection or selection, annotations).model_dump()
        projection = verify_annotation_projection(annotation_raw, raw_annotations, annotations, projection_audit)
    elif balanced:
        raise ValueError("balanced_execution_requires_complete_integer_keyed_model_wire")
    if proof.get("selection") != selection or proof.get("annotation") != annotations:
        raise ValueError("published_selection_annotations_differ_from_actual_model_outputs")
    refs = {row["paragraph_ref"] for row in raw_case["catalog"]}
    decisions = selection["decisions"]
    eligible = {row["p"] for row in decisions if row["relevance"] >= 2}
    selected = set(allocation["retained_refs"]) if balanced else eligible
    if len(decisions) != len(refs) or {row["p"] for row in decisions} != refs:
        raise ValueError("actual_selection_must_cover_every_candidate_once")
    rows = annotations["annotations"]
    if len(rows) != len(selected) or {row["p"] for row in rows} != selected:
        raise ValueError("actual_metadata_must_cover_every_selected_parent")
    by_ref = {row["p"]: row for row in rows}
    pid_ref = {meta["paragraph_id"]: ref for ref, meta in raw_case["source_metadata"].items()}
    for row in rows:
        if (not isinstance(row.get("fact_des"), str) or not row["fact_des"].strip()
                or not isinstance(row.get("argument", {}).get("内容"), str) or not row["argument"]["内容"].strip()
                or not isinstance(row.get("entities"), list)):
            raise ValueError("missing_metadata_cannot_be_fabricated_as_completed")
    for fact in result["evidence"]:
        ref = pid_ref.get(fact["paragraph_id"])
        annotation = by_ref.get(ref)
        if annotation is None or any(fact.get(key) != annotation[key] for key in ("fact_des", "argument", "entities")):
            raise ValueError("final_annotations_not_identical_to_visible_actual_model_fields")
        if balanced:
            decision = next(row for row in decisions if row["p"] == ref)
            if any(fact.get("selection_" + key) != decision[key]
                   for key in ("relevance", "stance", "hypothesis_id")):
                raise ValueError("final_selection_metadata_differs_from_complete_model_score_map")
    if balanced:
        final_refs = [pid_ref.get(fact["paragraph_id"]) for fact in result["evidence"]]
        if (len(final_refs) != len(selected) or set(final_refs) != selected
                or proof.get("qualified_evidence_sha256") != canonical_digest(result["evidence"])):
            raise ValueError("published_balanced_evidence_must_cover_allocated_parents_exactly_once")
    return {"actual_selected_parents": len(selected), "actual_annotated_parents": len(rows),
        "actual_model_eligible_parents": len(eligible), "complete_candidate_decisions": len(decisions),
        "balanced_subset_projection": balanced, "balance_ledger": allocation,
        "final_parents": len(result["evidence"]), "metadata_fabricated": False,
        "fixed_source_bound_keyed_decoder_used": used_keyed_decoder, "optional_time_projection": projection}


def remap_aliases(value, mapping):
    if isinstance(value, dict):
        return {mapping.get(key, key): remap_aliases(child, mapping) for key, child in value.items()}
    if isinstance(value, list):
        return [remap_aliases(child, mapping) for child in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def staged_source_errors(normalized, sides, raw_case, proof, contexts):
    """Exhaustive source-ID proxy errors; no semantic false-fact judgment."""
    from collections import Counter
    candidates = {location(row): row["paragraph_ref"] for row in raw_case["catalog"]}
    decisions = {row["p"]: row for row in proof["selection"]["decisions"]}
    eligible = {ref for ref, row in decisions.items() if row["relevance"] >= 2}
    ledger = proof.get("balance_ledger")
    retained = set(ledger["retained_refs"]) if ledger else eligible
    selected = {location(row) for row in normalized["evidence"]}
    targets = {"evidence": union_rows(*sides.values()), **sides}
    categories = {}
    def detail(key):
        ref = candidates.get(key)
        decision = decisions.get(ref)
        text = contexts.get(key, "")
        return {"doc_id": key[0] if key else None, "para_id": key[1] if key else None,
            "candidate_ref": ref, "model_decision": deepcopy(decision),
            "allocated": ref in retained if ref else False,
            "source_text_sha256_utf8": sha256(text.encode()).hexdigest() if text else None,
            "source_excerpt": text[:240], "excerpt_is_literal_prefix": True}
    for kind, target in targets.items():
        target_keys = {location(row) for row in target}
        submitted = union_rows(normalized[kind])
        submitted_keys = {location(row) for row in submitted}
        missed, extra = [], []
        for row in union_rows(target):
            key = location(row)
            if key in submitted_keys:
                continue
            ref = candidates.get(key)
            stage = ("unlocated_reference" if key is None else "candidate_recall_loss" if ref is None else
                "model_selection_below_2" if ref not in eligible else "balance_allocation_loss" if ref not in retained else
                "annotation_or_publication_loss" if key not in selected else "conclusion_relation_loss")
            missed.append({**detail(key), "stage": stage, "semantic_match_status": "unknown"})
        for row in submitted:
            key = location(row)
            if key not in target_keys:
                extra.append({**detail(key), "stage": "final_source_id_not_in_reference_category",
                    "semantic_false_fact_asserted": False})
        categories[kind] = {"source_id_proxy_only": True, "false_negative_count": len(missed),
            "false_positive_count": len(extra), "false_negatives": missed, "false_positives": extra,
            "false_negative_stage_counts": dict(Counter(row["stage"] for row in missed))}
    return categories


def verify_scoped_relations(result, wires, responses, native, *, question_internal_id=None):
    """Bind final roles to exact visible model arguments, never selection stances."""
    if result.get("verification_protocol") != "strict-workbench-evidence-balanced-v2":
        return {"enabled": False}
    operations = native["logical_operations"]
    if operations.count("arbitrate") != 1:
        raise ValueError("one_actual_scoped_arbitration_required")
    index = operations.index("arbitrate")
    wire, response = wires[index], responses[index]
    projection = native["relation_projections"][index]
    if any(row is not None for i, row in enumerate(native["relation_projections"]) if i != index):
        raise ValueError("scoped_relation_projection_only_belongs_to_arbitration")
    if wire["tools"][0]["function"]["name"] != "ScopedArbitrationWireOutput":
        raise ValueError("scoped_actual_arbitration_wire_required")
    original_arguments = response["tool_calls_wire"][0]["function"]["arguments"]
    recoveries = [row for row in native.get("actual_request_syntax_recoveries", []) if row["operation"] == "arbitrate"]
    if recoveries:
        if len(recoveries) != 1:
            raise ValueError("one_source_bound_arbitration_syntax_recovery_required")
        decoder = bound_code_module("app/agents/strict_json_syntax.py",
            native["runtime_sources_sha256"].get("app/agents/strict_json_syntax.py"))
        raw, _audit = decoder.decode_visible_arguments(original_arguments)
    else:
        verify_visible_tool_wire(response)
        raw = json.loads(original_arguments, object_pairs_hook=unique_json_object)
    evidence = wire_payload(wire)["evidence"]
    original = {(row["raw_doc_id"], row["raw_para_id"], row["quote"]): row for row in result["evidence"]}
    if len(original) != len(result["evidence"]) or len(evidence) != len(original):
        raise ValueError("scoped_arbitration_requires_exact_unique_final_source_inventory")
    aliases, used = {}, set()
    for row in evidence:
        fact = original.get((row.get("raw_doc_id"), row.get("raw_para_id"), row.get("quote")))
        alias = row.get("id")
        if (fact is None or not isinstance(alias, str) or not re.fullmatch(r"E[1-9]\d*", alias)
                or alias in aliases or fact["id"] in used
                or row.get("fact_des") != fact["fact_des"]):
            raise ValueError("scoped_E_alias_must_bind_exact_complete_final_original_parent")
        aliases[alias] = fact["id"]
        used.add(fact["id"])
    hashes = native["runtime_sources_sha256"]
    module = bound_code_module("app/agents/scoped_arbitration.py", hashes.get("app/agents/scoped_arbitration.py"))
    schema = bound_code_module("app/agents/schemas.py", hashes.get("app/agents/schemas.py"))
    module.ArbitrationOutput = schema.ArbitrationOutput
    canonical, scope = module.canonical_scoped_arbitration(raw, evidence,
        **({"question": wire_payload(wire)["question"], "issue_frame": True}
           if "issue_frame" in wire["tools"][0]["function"].get("parameters", {}).get("properties", {}) else {}))
    canonical, scope = remap_aliases(canonical, aliases), remap_aliases(scope, aliases)
    if scope.get("format") in {"issue-frame-evidence-relations-v2", "conclusion-aligned-evidence-relations-v3"}:
        scope["evidence_order_sha256"] = canonical_digest(scope["evidence_order"])
    if (projection is None or projection.get("operation") != "arbitrate"
            or projection.get("source_response_sha256") != native["response_file_sha256"][index]
            or projection.get("canonical_output_sha256") != canonical_digest(canonical)
            or projection.get("scope") != scope or projection.get("role_assignment_origin") != "model"
            or projection.get("new_model_calls") != 0 or result.get("evidence_relation_scope") != scope):
        raise ValueError("published_scoped_roles_or_claim_differ_from_actual_model_projection")
    final = result["final"]
    for key in ("conclusion", "decision", "supporting_evidence_ids", "conflicting_evidence_ids"):
        # Service IDs of judgments/conflicts may change, but prose and final relations may not.
        if final.get(key) != canonical[key]:
            raise ValueError("published_final_claim_or_relations_changed_after_actual_arbitration")
    if question_internal_id is not None:
        judgments = deepcopy(canonical["judgments"])
        ids = {row["id"]: str(uuid5(NAMESPACE_URL,
            f"trust-judge:judgment:{question_internal_id}:arbitrate:{row['id']}")) for row in judgments}
        for row in judgments:
            row["id"] = ids[row["id"]]
            row["prerequisite_judgment_ids"] = [ids[ref] for ref in row["prerequisite_judgment_ids"]]
        conflicts = deepcopy(canonical["conflicts"])
        for row in conflicts:
            row["id"] = str(uuid5(NAMESPACE_URL,
                f"trust-judge:conflict:{question_internal_id}:arbitrate:{row['id']}"))
        if final.get("judgments") != judgments or final.get("conflicts") != conflicts:
            raise ValueError("published_nested_model_judgments_or_conflicts_changed_after_arbitration")
    module.validate_scoped_result(scope, final, list(aliases.values()),
        **({"question": wire_payload(wire)["question"]} if "issue_frame" in scope else {}))
    return {"enabled": True, "visible_actual_model_wire_bound": True,
        "all_final_evidence_explicitly_classified": len(aliases),
        "principal_claim": scope["principal_claim"], "evidence_roles": scope["evidence_roles"],
        "semantic_target_verified": False, "selection_stances_copied_to_final_relations": False,
        "program_fabricated_roles_or_claim": False, "new_model_calls": 0}


def verify_legacy_sources(result, contexts, source_rows=None):
    for row in result["evidence"]:
        key = (row["raw_doc_id"], row["raw_para_id"])
        text = contexts.get(key)
        if (text is None or type(row.get("start_cp")) is not int or type(row.get("end_cp")) is not int
                or not 0 <= row["start_cp"] < row["end_cp"] <= len(text)
                or text[row["start_cp"]:row["end_cp"]] != row["quote"]
                or (source_rows is not None and row["paragraph_id"] != source_rows[key]["id"])):
            raise ValueError("final_original_quote_coordinates_or_identity_changed")


def prepare_closed(closure_path, expected_sha256):
    """All source, execution and baseline checks occur before reference IO."""
    guards = []
    closure = read_bound(closure_path, guards, expected_sha256)
    if (closure.get("format") != "workbench-real-quality-closed-v1" or closure.get("closed") is not True
            or closure.get("fixture") is not False or closure.get("real_provider") is not True):
        raise ValueError("actual_closed_workbench_quality_inputs_required")
    required = {"run", "revision", "snapshot", "question_result", "source_paragraphs", "questions",
        "activity", "native_episode_manifest", "candidate_catalog"}
    if not required <= set(closure["artifacts"]) or set(closure["artifacts"]) - required - {"export_readiness", "ui_verification", "run_options", "topic"}:
        raise ValueError("explicit_known_closed_workbench_artifact_bindings_required")
    artifact = {name: descriptor(value, guards) for name, value in closure["artifacts"].items()}
    run, revision, snapshot, result = (artifact[k] for k in ("run", "revision", "snapshot", "question_result"))
    qid, rid, tid = closure["question_internal_id"], closure["run_id"], closure["topic_id"]
    if (run.get("status") != "succeeded" or run.get("id") != rid or not run.get("completed_at")
            or run.get("cancel_requested") is not False or run.get("topic_id") != tid
            or run.get("question_ids") != [qid] or revision.get("state") != "ready"
            or revision.get("id") != run.get("target_revision_id") or revision.get("topic_id") != tid
            or any(snapshot.get(k) != v for k, v in {"run_id": rid, "topic_id": tid, "revision_id": revision["id"]}.items())
            or snapshot.get("question_results", {}).get(qid) != result or result.get("origin_run_id") != rid):
        raise ValueError("actual_current_run_ready_snapshot_result_binding_required")
    freeze = descriptor(closure["input_freeze"], guards)
    matches = [c for c in freeze["cases"] if c["question"]["question_id"] == closure["raw_question_id"]
               and c["topic_id"] == closure["raw_topic_id"]]
    if len(matches) != 1:
        raise ValueError("one_frozen_original_question_required")
    case = matches[0]
    question, baseline = case["question"], case["baseline"]
    raw = descriptor(case["raw_input"], guards)
    contexts, sources = verify_source_rows(raw, artifact["source_paragraphs"])
    qmatches = [q for q in artifact["questions"] if q.get("id") == qid]
    if (len(qmatches) != 1 or qmatches[0].get("raw_question_id") != question["question_id"]
            or qmatches[0].get("question") != question["question"]):
        raise ValueError("workbench_internal_question_must_bind_original_raw_question")
    usage = run["usage"][qid]
    if (result.get("usage") != usage or artifact["activity"].get("usage") != run["usage"]
            or artifact["activity"].get("run", {}).get("id") != rid):
        raise ValueError("workbench_result_run_activity_usage_disagree")
    known_usage(usage)
    native, wires, responses = audit_native(artifact["native_episode_manifest"], guards, run_id=rid, run_usage=usage)
    if native["experimental_stop_threshold"] != freeze["engineering_episode_stop"]:
        raise ValueError("native_protection_threshold_must_match_before_launch_freeze")
    if freeze.get("source_bound_authoritative_local_schemas"):
        expected = freeze["source_bound_authoritative_local_schemas"]["sha256"]
        if native["runtime_sources_sha256"].get("app/agents/schemas.py") != expected:
            raise ValueError("native_local_schema_must_match_before_launch_frozen_source")
    required = {"strict_selection", "strict_annotation", "authority_analyst", "cross_validation_analyst", "arbitrate"}
    if (set(native["logical_operations"]) != required or native["logical_operations"][0] != "strict_selection"
            or native["logical_operations"][-1] != "arbitrate" or len(result.get("analyses", [])) != 2):
        raise ValueError("actual_selection_annotation_A_B_arbitration_required")
    candidates = verify_candidates(artifact["candidate_catalog"], question, sources, wires[0])
    balanced = result.get("verification_protocol") == "strict-workbench-evidence-balanced-v2"
    if balanced and (not native["archived_runtime_source_bytes_checked"]
            or native["balanced_scoped_enabled"] is not True or freeze.get("balanced_scoped_enabled") is not True
            or native["balance_policy"] != freeze.get("balance_policy")
            or native["recall_strategy"] != freeze.get("recall_strategy")):
        raise ValueError("balanced_execution_requires_archived_sources_and_before_launch_policy")
    if balanced:
        for key, relative in (("compact_codec_source_guard", "app/agents/compact_selection.py"),
                              ("scoped_codec_source_guard", "app/agents/scoped_arbitration.py")):
            if (key not in freeze or freeze[key]["sha256"] != native["runtime_sources_sha256"].get(relative)):
                raise ValueError("balanced_codec_must_match_before_launch_frozen_source")
        if (case.get("actual_candidate_catalog") is not None
                and descriptor(case["actual_candidate_catalog"], guards) != artifact["candidate_catalog"]):
            raise ValueError("balanced_candidates_must_equal_before_launch_frozen_catalog")
    with SourceArchive(native["runtime_source_archive_descriptors"]):
        annotation_audit = verify_annotations(result, artifact["candidate_catalog"], responses,
            decoder_sha256=native["runtime_sources_sha256"].get("app/agents/strict_evidence.py"),
            projection_audit=native["field_projections"][1], source_hashes=native["runtime_sources_sha256"],
            expected_balance_policy=freeze.get("balance_policy"), expected_recall_strategy=freeze.get("recall_strategy"))
        if balanced:
            workflow = bound_code_module("app/workflows/strict_workbench.py",
                native["runtime_sources_sha256"].get("app/workflows/strict_workbench.py"))
            workflow.validate_result(result, qmatches[0], artifact["source_paragraphs"],
                candidate_limit=freeze["candidate_limit"])
            annotation_audit["complete_published_projection_replayed_from_archived_sources"] = True
        relation_audit = verify_scoped_relations(result, wires, responses, native, question_internal_id=qid)
    verify_legacy_sources(result, contexts, sources)
    normalized = normalize_legacy(result, question)
    if not {location(row) for row in normalized["evidence"]} <= candidates:
        raise ValueError("final_sources_must_have_been_sent_in_complete_candidate_pool")
    old_terminal = descriptor(baseline["terminal"], guards)
    old_result = descriptor(baseline["result"], guards)
    old_raw = descriptor(baseline["raw_input"], guards)
    if (old_terminal.get("status") != "succeeded" or old_result.get("origin_run_id") != old_terminal.get("run_id")
            or old_raw != raw or baseline["same_original_raw_sha256"] != case["raw_input"]["sha256"]):
        raise ValueError("same_original_raw_valid_closed_historical_baseline_required")
    old_normalized = normalize_legacy(old_result, question)
    verify_legacy_sources(old_result, contexts)
    if known_usage(old_result["usage"]) != baseline["usage_summary"]:
        # The frozen baseline includes provider_records as well as token totals.
        if any(known_usage(old_result["usage"])[key] != baseline["usage_summary"][key]
               for key in ("input_tokens", "output_tokens", "total_tokens")):
            raise ValueError("frozen_historical_usage_changed")
    return {"closure": closure, "run": run, "snapshot": snapshot, "result": result,
        "question": question, "raw": raw, "contexts": contexts, "candidates": candidates,
        "normalized": normalized, "baseline": baseline, "old_normalized": old_normalized,
        "native": native, "guards": guards, "raw_case": artifact["candidate_catalog"], "responses": responses,
        "topic_title": Path(case["raw_input"]["file"]).parent.name, "annotation_audit": annotation_audit,
        "relation_audit": relation_audit}


def bind_ui_export(prepared, proof_path, expected_sha256):
    """Bind a real browser download to this published revision and exact fields."""
    guards = prepared["guards"]
    proof = read_bound(proof_path, guards, expected_sha256)
    if (proof.get("actual_browser_UI_download") is not True
            or proof.get("run_id") != prepared["run"]["id"]
            or proof.get("revision_id") != prepared["run"]["target_revision_id"]
            or proof.get("whole_question_tokens") != prepared["native"]["whole_episode_known_usage"]["total_tokens"]
            or proof.get("exact_API_preview_bytes_equal") is not True
            or proof.get("exact_API_preview_sha256") != proof["download_file"]["sha256"]):
        raise ValueError("actual_UI_export_must_bind_published_revision_and_whole_usage")
    download = descriptor(proof["download_file"], guards)
    if (download.get("revision_id") != prepared["run"]["target_revision_id"]
            or download.get("topic_id") != prepared["run"]["topic_id"]):
        raise ValueError("downloaded_preview_revision_or_topic_changed")
    files, normalized = download["files"], prepared["normalized"]
    evidence = files["evidence.json"]
    rows = {location(row): row for row in normalized["evidence"]}
    aliases = {}
    for row in evidence:
        original = rows.get(location(row))
        if (original is None or not isinstance(row.get("fact_id"), str) or not row["fact_id"]
                or row["fact_id"] in aliases
                or {k: v for k, v in row.items() if k != "fact_id"}
                    != {k: v for k, v in original.items() if k != "fact_id"}):
            raise ValueError("downloaded_evidence_fields_differ_from_published_snapshot")
        aliases[row["fact_id"]] = original["fact_id"]
    if len(evidence) != len(rows) or len(set(aliases.values())) != len(rows):
        raise ValueError("downloaded_preview_must_cover_all_published_evidence_once")
    conclusions = [row for row in files["conclusions.json"] if row.get("question_id") == prepared["question"]["question_id"]]
    if (len(conclusions) != 1 or conclusions[0].get("question") != prepared["question"]["question"]
            or conclusions[0].get("conclusion") != normalized["conclusion"]):
        raise ValueError("downloaded_conclusion_differs_from_published_snapshot")
    for kind in ("supporting_facts", "conflicting_facts"):
        exported = conclusions[0][kind]
        expected_ids = [row["fact_id"] for row in normalized[kind]]
        if ([aliases.get(row.get("fact_id")) for row in exported] != expected_ids
                or any(any(row.get(key) != original.get(key) for key in ("doc_id", "para_id", "quote", "fact_des"))
                    for row, original in zip(exported, normalized[kind]))):
            raise ValueError("downloaded_relations_differ_from_published_snapshot")
    return {"proof": bound(proof_path, expected_sha256)[1], "download_file": proof["download_file"],
        "all_source_and_annotation_fields_equal_published_snapshot": True,
        "conclusion_and_S_C_relations_equal_published_snapshot": True,
        "formal_submission": download.get("formal_submission"), "partial": download.get("partial"),
        "new_model_calls": 0}


def score_episode(closure_path, expected_sha256, sample_root, *, ui_export_proof=None, ui_export_sha256=None):
    prepared = prepare_closed(closure_path, expected_sha256)
    if bool(ui_export_proof) != bool(ui_export_sha256):
        raise ValueError("UI_export_proof_and_explicit_sha256_required_together")
    export_audit = bind_ui_export(prepared, ui_export_proof, ui_export_sha256) if ui_export_proof else None
    question, guards = prepared["question"], prepared["guards"]
    topic_title = prepared["topic_title"]
    folder = Path(sample_root) / topic_title
    # FIRST reference access: generation, application publication and all costs bound.
    evidence = read_bound(folder / "evidence_samples.json", guards)
    conclusions = read_bound(folder / "conclusions.json", guards)
    reference = [q for q in conclusions if q["question_id"] == question["question_id"]]
    if len(reference) != 1 or reference[0]["question"] != question["question"]:
        raise ValueError("one_same_original_reference_question_required")
    _, engine, mapper_guards = load_reference_tools()
    guards.extend(mapper_guards)
    current = score_normalized(question, prepared["normalized"], evidence, reference[0], prepared["contexts"],
        prepared["candidates"], prepared["candidates"], engine=engine)
    issues = []
    sides = {kind: engine._relation_rows(reference[0][kind], evidence, submitted=False, issues=issues,
        context=[question["question_id"], kind]) for kind in ("supporting_facts", "conflicting_facts")}
    old = score_final(question, prepared["old_normalized"], union_rows(*sides.values()),
        sides["supporting_facts"], sides["conflicting_facts"], prepared["contexts"], engine=engine)
    baseline_quality = {label: brief_category(old["categories"][key]) for label, key in (
        ("question_selection", "evidence"), ("supporting", "supporting_facts"), ("conflicting", "conflicting_facts"))}
    stage_errors = staged_source_errors(prepared["normalized"], sides, prepared["raw_case"],
        prepared["result"]["strict_evidence_proof"], prepared["contexts"])
    for kind, label in (("evidence", "question_selection"), ("supporting_facts", "supporting"),
                        ("conflicting_facts", "conflicting")):
        measured, errors = current[label]["A_id_only_proxy"], stage_errors[kind]
        if (errors["false_negative_count"] != measured["reference_records"] - measured["matches"]
                or errors["false_positive_count"] != measured["submitted_records"] - measured["matches"]):
            raise ValueError("exhaustive_source_id_error_audit_must_reconcile_with_measured_counts")
    from scripts.score_real_quality_episode import source_errors
    before_errors = {kind: source_errors(prepared["old_normalized"][kind], target,
        set(prepared["contexts"]), set(prepared["contexts"]), prepared["contexts"])
        for kind, target in {"evidence": union_rows(*sides.values()), **sides}.items()}
    guards.append(bound(__file__)[1])
    for record in guards:
        bound(record["file"], record["sha256"])
    return {"format": "closed-real-workbench-quality-score-v1", "question": question,
        "topic_title": topic_title, "after": current, "before": baseline_quality,
        "after_usage": prepared["native"], "before_usage": prepared["baseline"]["usage_summary"],
        "actual_model_annotation_audit": prepared["annotation_audit"],
        "actual_model_relation_audit": prepared["relation_audit"],
        "after_source_id_error_audit": stage_errors, "before_source_id_error_audit": before_errors,
        "actual_browser_export_audit": export_audit,
        "before_usage_is_new_cost": False, "real_application_run_id": prepared["run"]["id"],
        "workbench_ready_revision_id": prepared["run"]["target_revision_id"],
        "scope": prepared["snapshot"].get("scope"), "historical_before_actual_after_comparison": True,
        "same_input_framework_causal_pair": False, "official_score": False,
        "semantic_B_point_score": None, "new_model_calls_by_scorer": 0,
        "reference_written_to_runtime": False, "sample_is_unseen_blind_test": False,
        "input_guards": guards, "input_guards_unchanged": True, "reference_issues": issues}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("closed-inputs", "expected-sha256", "sample-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--ui-export-proof")
    parser.add_argument("--ui-export-sha256")
    args = parser.parse_args()
    result = score_episode(args.closed_inputs, args.expected_sha256, args.sample_root,
        ui_export_proof=args.ui_export_proof, ui_export_sha256=args.ui_export_sha256)
    with Path(args.output).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"after": result["after"]["question_selection"]["A_id_only_proxy"],
        "before": result["before"]["question_selection"]["A_id_only_proxy"]}, ensure_ascii=False))
