"""v16 single-request wire; the canonical v15 ledger/resolver stays unchanged.

Literal clause hints are deterministic locations, not semantically reviewed
atomic targets. Finite IDs eliminate three cross-field composition mistakes;
they do not establish entailment or relevance and never override a bridge.
"""
from copy import deepcopy
from hashlib import sha256
import json
import re
from typing import Literal

from pydantic import Field, ValidationError, create_model

from app.agents.schemas import OutputModel, ScopeFieldChecks
from app.domain.strict_verification import (
    ScopeVerificationInvalid, build_scope_request, canonical, resolve_scope_verification,
)


WIRE_VERSION = "compiled-scope-v16-literal-target-bindings"
DIMENSIONS = ("subject", "time", "scope", "value", "attribution", "certainty", "causality")
SOURCE_PROTOCOL = {"version": "immutable-fixed-source-fact-view-v1",
    "span_unit": "unicode_codepoint", "span_interval": "half_open",
    "source_fields": ["paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote"],
    "source_geometry": "unchanged_from_preverified_candidate"}
SOURCE_PROTOCOL_SHA256 = sha256(canonical(SOURCE_PROTOCOL).encode("utf-8")).hexdigest()
_CLAUSE_BOUNDARIES = re.compile(r"[，,、；;。.!?！？：:\r\n]+")


class CompiledScopeDecision(OutputModel):
    fact_entailment: Literal["entailed", "needs_narrowing", "unsupported", "contradicted"]
    target_binding_id: str
    quote_requires_unstated_bridge: bool
    target_requires_unstated_bridge: bool
    field_checks: ScopeFieldChecks
    target_fact_present_but_annotation_incomplete: bool
    reason: str = Field(min_length=1)


class CompiledScopeVerificationOutput(OutputModel):
    candidate_catalog_id: str
    target_catalog_id: str
    decisions: dict[str, CompiledScopeDecision]


def _digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def literal_target_catalog(question_text):
    """Always include T0; punctuation slices never invent or join target text."""
    if not isinstance(question_text, str) or not question_text.strip():
        raise ScopeVerificationInvalid("scope_question_invalid")
    result = [{"id": "T0", "start_cp": 0, "end_cp": len(question_text),
        "text": question_text, "kind": "full_question"}]
    start = 0
    ends = [match.end() for match in _CLAUSE_BOUNDARIES.finditer(question_text)]
    if not ends or ends[-1] != len(question_text):
        ends.append(len(question_text))
    for end in ends:
        text = question_text[start:end]
        if (start, end) != (0, len(question_text)) and text.strip() and _CLAUSE_BOUNDARIES.sub("", text).strip():
            result.append({"id": f"T{len(result)}", "start_cp": start, "end_cp": end,
                "text": text, "kind": "punctuation_clause_hint"})
        start = end
    return result


def _bindings(targets):
    result = []
    for target in targets:
        roles = [("direct", "direct_answer", None), ("counter", "explicit_counterclaim", None),
                 *[("qualifier." + dimension, "necessary_qualifier", dimension) for dimension in DIMENSIONS]]
        for suffix, relation, dimension in roles:
            result.append({"id": target["id"] + "." + suffix, "target_id": target["id"],
                "target_relation": relation, "qualifier_dimension": dimension})
    result.extend({"id": relation, "target_id": None, "target_relation": relation,
                   "qualifier_dimension": None} for relation in ("outside_target", "uncertain"))
    return result


def build_compiled_scope_request(question, evidence):
    old, _schema, views = build_scope_request(question, evidence, detailed=True)
    targets = literal_target_catalog(old["question"]["question"])
    bindings = _bindings(targets)
    body = {"scope_wire_version": WIRE_VERSION, "question": old["question"],
        "candidate_catalog_id": old["candidate_catalog_id"], "candidate_catalog": old["candidate_catalog"],
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "target_catalog": targets, "target_binding_catalog": bindings}
    binding = "G" + _digest(body)
    decision_schema = create_model("BoundCompiledScopeDecision", __base__=CompiledScopeDecision,
        target_binding_id=(Literal[tuple(row["id"] for row in bindings)], Field(...)))
    required = create_model("RequiredCompiledScopeDecisions", __base__=OutputModel,
        **{row["id"]: (decision_schema, Field(...)) for row in old["candidate_catalog"]})
    schema = create_model("BoundCompiledScopeVerification", __base__=CompiledScopeVerificationOutput,
        candidate_catalog_id=(Literal[old["candidate_catalog_id"]], Field(...)),
        target_catalog_id=(Literal[binding], Field(...)), decisions=(required, Field(...)))
    return {**body, "target_catalog_id": binding}, schema, views


def _parse_raw_json(value):
    if not isinstance(value, str):
        return value  # Internal typed output from the provider's strict raw boundary.
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ScopeVerificationInvalid("scope_verification_duplicate_json_key")
            result[key] = item
        return result
    try:
        return json.loads(value, object_pairs_hook=unique)
    except (ValueError, TypeError) as error:
        if isinstance(error, ScopeVerificationInvalid):
            raise
        raise ScopeVerificationInvalid("scope_compiled_contract") from None


def decode_compiled_scope_output(output, question, evidence):
    """Map only valid NEW wire to old canonical fields; never fix old replies."""
    payload, schema, _views = build_compiled_scope_request(question, evidence)
    try:
        typed = schema.model_validate(_parse_raw_json(output)).model_dump()
    except (ValidationError, TypeError, ValueError) as error:
        if isinstance(error, ScopeVerificationInvalid):
            raise
        raise ScopeVerificationInvalid("scope_compiled_contract") from None
    targets = {row["id"]: row for row in payload["target_catalog"]}
    bindings = {row["id"]: row for row in payload["target_binding_catalog"]}
    decisions = {}
    for alias, row in typed["decisions"].items():
        choice = bindings[row["target_binding_id"]]
        target = targets[choice["target_id"]] if choice["target_id"] is not None else None
        decisions[alias] = {key: deepcopy(row[key]) for key in (
            "fact_entailment", "quote_requires_unstated_bridge", "target_requires_unstated_bridge",
            "field_checks", "target_fact_present_but_annotation_incomplete", "reason")}
        decisions[alias].update(target_relation=choice["target_relation"],
            target_question_text=target["text"] if target else None,
            qualifier_dimension=choice["qualifier_dimension"],
            requires_unstated_bridge=row["quote_requires_unstated_bridge"] or row["target_requires_unstated_bridge"])
    canonical_output = {"candidate_catalog_id": payload["candidate_catalog_id"], "decisions": decisions}
    _payload, old_detailed_schema, _views = build_scope_request(question, evidence, detailed=True)
    old_detailed_schema.model_validate(canonical_output)
    return canonical_output


def resolve_compiled_scope(output, question, evidence):
    return resolve_scope_verification(decode_compiled_scope_output(output, question, evidence), question, evidence)


def compiled_repair_context(payload, visible_output):
    """Ephemeral NEW-wire error descriptor; never carry the provider body."""
    visible_digest = sha256((visible_output if isinstance(visible_output, str) else canonical(visible_output)).encode("utf-8", errors="surrogatepass")).hexdigest()
    return {"scope_wire_version": WIRE_VERSION, "candidate_catalog_id": payload["candidate_catalog_id"],
        "target_catalog_id": payload["target_catalog_id"], "invalid_response_sha256": visible_digest}


def validate_compiled_repair_context(payload, context):
    keys = {"scope_wire_version", "candidate_catalog_id", "target_catalog_id", "invalid_response_sha256"}
    if (not isinstance(context, dict) or set(context) != keys
            or any(context.get(key) != payload[key] for key in keys - {"invalid_response_sha256"})
            or not isinstance(context["invalid_response_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", context["invalid_response_sha256"])):
        raise ScopeVerificationInvalid("scope_compiled_repair_binding")
    return deepcopy(context)


def safe_compiled_scope_diagnostics(diagnostics):
    """Repair input/error detail whitelist, excluding old bodies and free text."""
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    known_checks = {"scope_compiled_contract", "scope_compiled_repair_binding", "scope_decision_contract",
        "scope_target_contract", "scope_verification_duplicate_json_key"}
    safe = {}
    check = diagnostics.get("check")
    if isinstance(check, str) and check in known_checks:
        safe["check"] = check
    for name in ("invalid_candidate_count", "duplicate_key_count"):
        value = diagnostics.get(name)
        if type(value) is int and value >= 0:
            safe[name] = value
    finish_reason = diagnostics.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason in {"length", "content_filter", "insufficient_system_resource", "aborted", "unknown"}:
        safe["finish_reason"] = finish_reason
    indexes = diagnostics.get("candidate_indexes")
    if isinstance(indexes, list):
        safe["candidate_indexes"] = [value for value in indexes if type(value) is int and value >= 0]
    fields = {"candidate_catalog_id", "target_catalog_id", "decisions", "target_binding_id", "fact_entailment",
        "quote_requires_unstated_bridge", "target_requires_unstated_bridge", "field_checks", "fact_des",
        "argument", "entities", "target_fact_present_but_annotation_incomplete", "reason", "<field>",
        "时间", "主体", "地点", "内容", "影响", "数值口径"}
    types = {"missing", "extra_forbidden", "literal_error", "string_type", "string_too_short", "dict_type",
        "model_type", "bool_type", "value_error", "json_invalid", "json_type"}
    errors = diagnostics.get("schema_errors")
    if isinstance(errors, list):
        safe["schema_errors"] = []
        for row in errors:
            if not isinstance(row, dict):
                continue
            loc = row.get("loc", [])
            kind = row.get("type")
            safe["schema_errors"].append({"loc": [value if (type(value) is int and value >= 0)
                or (isinstance(value, str) and value in fields) else "<field>" for value in loc] if isinstance(loc, list) else [],
                "type": kind if isinstance(kind, str) and kind in types else "value_error"})
    return safe or {"check": "scope_compiled_contract"}
