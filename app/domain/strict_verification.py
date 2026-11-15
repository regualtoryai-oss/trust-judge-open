"""A strict second opinion on existing atomic interpretations, never geometry.

This boundary cannot prove model semantics. It enforces complete decisions,
source/version binding, explicit rejection of inferred bridges and lossless
separation of rejected annotations from the immutable internal candidate pool.
"""
from copy import deepcopy
from hashlib import sha256
import json
from typing import Literal

from pydantic import Field, ValidationError, create_model

from app.agents.schemas import (OutputModel, ScopeVerificationDecision, ScopeVerificationOutput,
                                ScopeFieldChecks, AnnotationCorrection, AnnotationNarrowingOutput)

ARGUMENT_KEYS = ("时间", "主体", "地点", "内容", "影响", "数值口径")
TARGET_RELATIONS = {"direct_answer", "explicit_counterclaim", "necessary_qualifier"}


class ScopeVerificationInvalid(ValueError):
    def __init__(self, check, *, indexes=()):
        super().__init__(check)  # Fixed codes only; never source/model content.
        self.details = {"check": check, "invalid_candidate_count": len(indexes),
                        "candidate_indexes": list(indexes)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def project_question(question):
    if not isinstance(question, dict) or not isinstance(question.get("question"), str) or not question["question"].strip():
        raise ScopeVerificationInvalid("scope_question_invalid")
    return {key: deepcopy(question[key]) for key in ("id", "topic_id", "raw_question_id", "question_id", "question") if key in question}


def build_fact_views(evidence):
    """A view identity denotes an annotation version, not proven fact equality."""
    if not isinstance(evidence, list):
        raise ScopeVerificationInvalid("scope_catalog_invalid")
    views, seen_sources, seen_views = [], set(), set()
    source_fields = ("paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote")
    for source_index, source in enumerate(evidence):
        if not isinstance(source, dict) or not isinstance(source.get("id"), str) or not source["id"] or source["id"] in seen_sources:
            raise ScopeVerificationInvalid("scope_catalog_invalid", indexes=[source_index])
        seen_sources.add(source["id"])
        quote = source.get("quote")
        if (not isinstance(quote, str) or not quote or any(not isinstance(source.get(key), str) or not source[key] for key in source_fields[:3])
                or type(source.get("start_cp")) is not int or type(source.get("end_cp")) is not int
                or source["start_cp"] < 0 or source["end_cp"] - source["start_cp"] != len(quote)):
            raise ScopeVerificationInvalid("scope_catalog_invalid", indexes=[source_index])
        additional = source.get("additional_interpretations", [])
        if not isinstance(additional, list):
            raise ScopeVerificationInvalid("scope_catalog_invalid", indexes=[source_index])
        for annotation in [source, *additional]:
            if not isinstance(annotation, dict):
                raise ScopeVerificationInvalid("scope_catalog_invalid", indexes=[source_index])
            args, entities = annotation.get("argument"), annotation.get("entities")
            if (not isinstance(annotation.get("fact_des"), str) or not annotation["fact_des"].strip()
                    or not isinstance(args, dict) or set(args) - set(ARGUMENT_KEYS)
                    or any(value is not None and not isinstance(value, str) for value in args.values())
                    or not isinstance(entities, list) or any(not isinstance(value, str) for value in entities)):
                raise ScopeVerificationInvalid("scope_catalog_invalid", indexes=[source_index])
            fact = {"fact_des": annotation["fact_des"], "argument": {key: args.get(key) for key in ARGUMENT_KEYS},
                    "entities": sorted(set(entities))}
            material = {"evidence_id": source["id"], **{key: source[key] for key in source_fields}, **fact}
            view_id = "F" + sha256(canonical(material).encode("utf-8")).hexdigest()
            if view_id not in seen_views:
                seen_views.add(view_id)
                views.append({"id": view_id, **deepcopy(material)})
    return views


def build_scope_request(question, evidence, *, detailed=False):
    question = project_question(question)
    views = build_fact_views(evidence)
    aliases = [f"C{index + 1}" for index in range(len(views))]
    binding = "V" + sha256(canonical({"question": question, "fact_views": views}).encode("utf-8")).hexdigest()
    decision_schema = ScopeVerificationDecision
    if detailed:
        decision_schema = create_model("DetailedScopeDecision", __base__=ScopeVerificationDecision,
            quote_requires_unstated_bridge=(bool, Field(..., description=ScopeVerificationDecision.model_fields["quote_requires_unstated_bridge"].description)),
            target_requires_unstated_bridge=(bool, Field(..., description=ScopeVerificationDecision.model_fields["target_requires_unstated_bridge"].description)),
            field_checks=(ScopeFieldChecks, Field(..., description=ScopeVerificationDecision.model_fields["field_checks"].description)))
    required = create_model("RequiredScopeDecisions", __base__=OutputModel,
                            **{alias: (decision_schema, Field(...)) for alias in aliases})
    schema = create_model("BoundScopeVerification", __base__=ScopeVerificationOutput,
                          candidate_catalog_id=(Literal[binding], Field(...)), decisions=(required, Field(...)))
    payload = {"question": question, "candidate_catalog_id": binding,
               "candidate_catalog": [{**deepcopy(view), "id": alias} for alias, view in zip(aliases, views)]}
    return payload, schema, views


def resolve_scope_verification(output, question, evidence):
    """Map one exact required-key decision per atomic view; never amend quotes."""
    payload, schema, views = build_scope_request(question, evidence)
    try:
        output = schema.model_validate(output).model_dump()
    except (ValidationError, ValueError, TypeError):
        raise ScopeVerificationInvalid("scope_decision_contract") from None
    decisions = output["decisions"]
    issues, assessments, selected, narrowing = [], [], [], []
    for index, view in enumerate(views):
        decision = decisions[f"C{index + 1}"]
        target = decision["target_question_text"]
        target_inside = decision["target_relation"] in TARGET_RELATIONS
        if ((target_inside and (not isinstance(target, str) or not target or target not in payload["question"]["question"]))
                or (decision["target_relation"] == "necessary_qualifier" and decision["qualifier_dimension"] is None)
                or (decision["target_relation"] != "necessary_qualifier" and decision["qualifier_dimension"] is not None)):
            issues.append(index)
        quote_bridge, target_bridge = decision["quote_requires_unstated_bridge"], decision["target_requires_unstated_bridge"]
        detailed = quote_bridge is not None or target_bridge is not None or decision["field_checks"] is not None
        if detailed and (quote_bridge is None or target_bridge is None or decision["field_checks"] is None
                         or decision["requires_unstated_bridge"] != (quote_bridge or target_bridge)):
            issues.append(index)
        field_issues = []
        if decision["field_checks"] is not None:
            checks = decision["field_checks"]
            checked_fields = [("fact_des", view["fact_des"], checks["fact_des"]),
                              ("entities", view["entities"], checks["entities"]),
                              *[("argument." + key, view["argument"][key], checks["argument"][key]) for key in ARGUMENT_KEYS]]
            field_issues = [name for name, value, verdict in checked_fields
                            if verdict not in ({"entailed"} if value else {"entailed", "not_present"})]
        bridge = (quote_bridge or target_bridge) if detailed else decision["requires_unstated_bridge"]
        retain = decision["fact_entailment"] == "entailed" and not field_issues and target_inside and not bridge
        visible_decision = deepcopy(decision)
        if not detailed:
            for key in ("quote_requires_unstated_bridge", "target_requires_unstated_bridge", "field_checks"):
                visible_decision.pop(key)
        assessments.append({"fact_view_id": view["id"], "evidence_id": view["evidence_id"],
                            **visible_decision, "selected": retain,
                            **({"field_issues": field_issues} if detailed else {})})
        if retain:
            selected.append(view)
        if (((decision["fact_entailment"] in {"needs_narrowing", "contradicted"} or field_issues) and target_inside)
                or decision["target_fact_present_but_annotation_incomplete"]):
            narrowing.append(view["id"])
    if issues:
        raise ScopeVerificationInvalid("scope_target_contract", indexes=sorted(set(issues)))
    # Copy only accepted annotation versions into the new frozen projection.
    # The caller must persist its complete original candidate_evidence unchanged.
    by_source = {}
    for view in selected:
        by_source.setdefault(view["evidence_id"], []).append(view)
    verified = []
    for source in evidence:
        accepted = by_source.get(source["id"], [])
        if not accepted:
            continue
        row = deepcopy(source)
        for key in ("fact_des", "argument", "entities"):
            row[key] = deepcopy(accepted[0][key])
        row.pop("additional_interpretations", None)
        if len(accepted) > 1:
            row["additional_interpretations"] = [{key: deepcopy(view[key]) for key in ("fact_des", "argument", "entities")} for view in accepted[1:]]
        verified.append(row)
    return {"candidate_catalog_id": payload["candidate_catalog_id"], "assessments": assessments,
            "selected_fact_view_ids": [view["id"] for view in selected],
            "selected_evidence_ids": [source["id"] for source in verified],
            "narrowing_view_ids": narrowing, "verified_fact_views": deepcopy(selected), "verified_evidence": verified}


def validate_scope_ledger(ledger, question, evidence):
    """Rebuild every accepted projection instead of trusting provider flags."""
    payload, _, views = build_scope_request(question, evidence)
    try:
        rows = ledger["assessments"]
        if not isinstance(rows, list) or len(rows) != len(views):
            raise ValueError
        by_id = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("fact_view_id") in by_id:
                raise ValueError
            by_id[row["fact_view_id"]] = row
        if set(by_id) != {view["id"] for view in views}:
            raise ValueError
        decisions = {}
        for index, view in enumerate(views):
            row = by_id[view["id"]]
            if row["evidence_id"] != view["evidence_id"]:
                raise ValueError
            decisions[f"C{index + 1}"] = {field: row.get(field) if field in {
                "quote_requires_unstated_bridge", "target_requires_unstated_bridge", "field_checks"} else row[field]
                for field in ScopeVerificationDecision.model_fields}
        checked = resolve_scope_verification({"candidate_catalog_id": ledger["candidate_catalog_id"], "decisions": decisions}, question, evidence)
        if canonical(checked) != canonical(ledger):
            raise ValueError
        return checked
    except (KeyError, TypeError, ValueError):
        raise ScopeVerificationInvalid("scope_ledger_invalid") from None


def build_narrowing_request(question, evidence, fact_view_ids):
    """Select existing interpretation versions, not caller/model quotations."""
    question = project_question(question)
    all_views = build_fact_views(evidence)
    by_id = {view["id"]: view for view in all_views}
    if (not isinstance(fact_view_ids, list) or any(not isinstance(value, str) for value in fact_view_ids)
            or len(set(fact_view_ids)) != len(fact_view_ids) or any(value not in by_id for value in fact_view_ids)):
        raise ScopeVerificationInvalid("annotation_refocus_catalog_invalid")
    sources = {source["id"]: source for source in evidence}
    selected = [by_id[value] for value in fact_view_ids]
    if any(sources[view["evidence_id"]].get("interpretation_origin") == "human" for view in selected):
        raise ScopeVerificationInvalid("annotation_refocus_manual_protected")
    binding = "N" + sha256(canonical({"question": question, "fact_views": selected,
                                     "operation": "fixed-source-annotation-refocus-v1"}).encode("utf-8")).hexdigest()
    required = create_model("RequiredAnnotationCorrections", __base__=OutputModel,
                            **{f"C{index + 1}": (AnnotationCorrection, Field(...)) for index in range(len(selected))})
    schema = create_model("BoundAnnotationNarrowing", __base__=AnnotationNarrowingOutput,
                          candidate_catalog_id=(Literal[binding], Field(...)), corrections=(required, Field(...)))
    payload = {"question": question, "candidate_catalog_id": binding,
               "candidate_catalog": [{**deepcopy(view), "id": f"C{index + 1}"} for index, view in enumerate(selected)]}
    return payload, schema, selected


def resolve_annotation_narrowing(output, question, evidence, fact_view_ids):
    payload, schema, views = build_narrowing_request(question, evidence, fact_view_ids)
    try:
        output = schema.model_validate(output).model_dump()
    except (ValidationError, ValueError, TypeError):
        raise ScopeVerificationInvalid("annotation_correction_contract") from None
    by_source = {source["id"]: source for source in evidence}
    projected, correction_rows, versions = {}, [], []
    for index, parent in enumerate(views):
        decision = output["corrections"][f"C{index + 1}"]
        annotations = decision["annotations"]
        if decision["status"] == "corrected" and not annotations:
            raise ScopeVerificationInvalid("annotation_correction_empty", indexes=[index])
        if decision["status"] == "no_supported_annotation" and annotations:
            raise ScopeVerificationInvalid("annotation_correction_no_supported_nonempty", indexes=[index])
        source = by_source[parent["evidence_id"]]
        local_ids, version_ids, seen_annotations, duplicate_count = [], [], set(), 0
        for raw_annotation in annotations:
            annotation = {"fact_des": raw_annotation["fact_des"],
                          "argument": {key: raw_annotation["argument"].get(key) for key in ARGUMENT_KEYS},
                          "entities": sorted(set(raw_annotation["entities"]))}
            key = canonical(annotation)
            if key in seen_annotations:
                duplicate_count += 1
                continue
            seen_annotations.add(key)
            narrowed = deepcopy(source)
            narrowed.pop("additional_interpretations", None)
            narrowed.update(deepcopy(annotation))
            new_view = build_fact_views([narrowed])[0]
            identity = new_view["id"]
            version_id = "R" + sha256(canonical({"catalog_id": payload["candidate_catalog_id"],
                "parent_fact_view_id": parent["id"], "fact_view_id": identity}).encode("utf-8")).hexdigest()
            version = {"id": version_id, "parent_fact_view_id": parent["id"], "fact_view_id": identity,
                       "evidence_id": source["id"], "operation": "bounded_annotation_refocus", "semantic_attempt": 1,
                       "annotation_changed": canonical({field: parent[field] for field in ("fact_des", "argument", "entities")}) != key}
            versions.append(version)
            local_ids.append(identity)
            version_ids.append(version_id)
            group = projected.setdefault(source["id"], {"source": source, "annotations": {}})
            group["annotations"].setdefault(key, annotation)
        correction_rows.append({"source_fact_view_id": parent["id"], "evidence_id": source["id"],
            "status": decision["status"], "annotations": deepcopy(annotations),
            "corrected_fact_view_ids": local_ids, "annotation_version_ids": version_ids,
            "duplicate_annotation_count": duplicate_count})
    corrected_evidence = []
    for identity, group in projected.items():
        annotations = list(group["annotations"].values())
        row = deepcopy(group["source"])
        row.update(deepcopy(annotations[0]))
        row.pop("additional_interpretations", None)
        if len(annotations) > 1:
            row["additional_interpretations"] = deepcopy(annotations[1:])
        prior_versions = row.get("annotation_versions", [])
        if not isinstance(prior_versions, list):
            raise ScopeVerificationInvalid("annotation_refocus_catalog_invalid")
        row["annotation_versions"] = [*deepcopy(prior_versions),
            *[deepcopy(version) for version in versions if version["evidence_id"] == identity]]
        corrected_evidence.append(row)
    return {"candidate_catalog_id": payload["candidate_catalog_id"], "corrections": correction_rows,
            "corrected_evidence": corrected_evidence, "corrected_fact_views": build_fact_views(corrected_evidence),
            "annotation_versions": versions}


def validate_narrowing_ledger(ledger, question, evidence, fact_view_ids):
    payload, _, views = build_narrowing_request(question, evidence, fact_view_ids)
    try:
        rows = ledger["corrections"]
        if not isinstance(rows, list) or len(rows) != len(views):
            raise ValueError
        by_id = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("source_fact_view_id") in by_id:
                raise ValueError
            by_id[row["source_fact_view_id"]] = row
        if set(by_id) != set(fact_view_ids):
            raise ValueError
        corrections = {f"C{index + 1}": {"status": by_id[view["id"]]["status"],
                        "annotations": by_id[view["id"]]["annotations"]} for index, view in enumerate(views)}
        checked = resolve_annotation_narrowing({"candidate_catalog_id": ledger["candidate_catalog_id"], "corrections": corrections}, question, evidence, fact_view_ids)
        if canonical(checked) != canonical(ledger):
            raise ValueError
        return checked
    except (KeyError, TypeError, ValueError):
        raise ScopeVerificationInvalid("annotation_refocus_ledger_invalid") from None


def verified_fact_export_rows(ledger, question, evidence):
    """Only independently verified fact versions; never expand unchecked views."""
    checked = validate_scope_ledger(ledger, question, evidence)
    rows, seen = [], set()
    for view in checked["verified_fact_views"]:
        row = {"doc_id": view["raw_doc_id"], "para_id": view["raw_para_id"], "quote": view["quote"],
               "fact_des": view["fact_des"], "argument": {key: view["argument"].get(key) or "" for key in ARGUMENT_KEYS},
               "entities": list(view["entities"])}
        key = canonical(row)
        if key not in seen:
            seen.add(key)
            rows.append(row)
    return rows


def safe_annotation_diagnostics(diagnostics):
    """Refocus never receives a prior verdict/reason or previous model body."""
    if not isinstance(diagnostics, dict):
        return {"check": "annotation_correction_contract"}
    known_checks = {"annotation_correction_contract", "annotation_correction_empty",
                    "annotation_correction_no_supported_nonempty", "annotation_refocus_duplicate_json_key"}
    check = diagnostics.get("check")
    safe = {"check": check if isinstance(check, str) and check in known_checks else "annotation_correction_contract"}
    for field in ("invalid_candidate_count", "duplicate_key_count"):
        value = diagnostics.get(field)
        if type(value) is int and value >= 0:
            safe[field] = value
    indexes = diagnostics.get("candidate_indexes")
    if isinstance(indexes, list):
        safe["candidate_indexes"] = [value for value in indexes if type(value) is int and value >= 0]
    fields = {"candidate_catalog_id", "corrections", "status", "annotations", "fact_des", "argument", "entities", "<field>", *ARGUMENT_KEYS}
    types = {"missing", "extra_forbidden", "literal_error", "string_type", "list_type", "dict_type", "model_type", "bool_type", "value_error", "json_invalid"}
    schema_errors = diagnostics.get("schema_errors")
    if isinstance(schema_errors, list):
        safe["schema_errors"] = []
        for row in schema_errors:
            if not isinstance(row, dict):
                continue
            loc = row.get("loc", [])
            error_type = row.get("type")
            safe["schema_errors"].append({"loc": [value if (type(value) is int and value >= 0) or (isinstance(value, str) and value in fields) else "<field>" for value in loc] if isinstance(loc, list) else [],
                "type": error_type if isinstance(error_type, str) and error_type in types else "value_error"})
    return safe
