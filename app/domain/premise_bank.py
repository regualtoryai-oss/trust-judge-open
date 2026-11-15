"""Quote-faithful facts and independent question-purpose permissions.

An old target rejection is neither reversed nor a fact-entailment rejection.
The bank replays detailed source judgments only; a second complete review must
authorize each formal fact's use. Both ledgers remain independently replayable.
"""
from copy import deepcopy
from hashlib import sha256
import json
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, ValidationError, create_model

from app.agents.schemas import OutputModel, ScopeVerificationDecision
from app.domain.compiled_scope import literal_target_catalog
from app.domain.spans import validate_span
from app.domain.strict_verification import (
    ARGUMENT_KEYS, build_fact_views, build_scope_request, canonical, project_question,
    resolve_scope_verification, validate_narrowing_ledger,
)


VERSION = "quote-faithful-bank-independent-purpose-v18"
RETAINED_PURPOSES = {"direct_answer", "counterclaim", "necessary_premise", "qualifier"}
_SOURCE_FIELDS = ("paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote")
_ANNOTATION_FIELDS = ("fact_des", "argument", "entities")


class PremiseBankInvalid(ValueError):
    def __init__(self, check):
        super().__init__(check)
        self.check = check
        self.details = {"check": check}


class FactPurposeOutput(OutputModel):
    bank_id: str
    purpose_catalog_id: str
    decisions: dict


class PurposeDecision(OutputModel):
    purpose_binding_id: str
    introduces_unstated_source_fact: bool
    equivalent_to: str | None
    reason: str = Field(min_length=1, max_length=500)


def digest(value):
    try:
        return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    except (TypeError, ValueError, UnicodeError):
        raise PremiseBankInvalid("premise_binding_material") from None


def _source_manifest(paragraphs):
    from app.domain.material_eligibility import _source_manifest as source_manifest
    return source_manifest(paragraphs)


def _question_identity(question):
    q = project_question(question)
    return {"id": q.get("id"), "raw_question_id": q.get("raw_question_id", q.get("question_id")),
            "question": q["question"]}


def _check_sources(evidence, paragraphs):
    by_id = {row["id"]: row for row in paragraphs}
    if len(by_id) != len(paragraphs):
        raise PremiseBankInvalid("premise_source_identity")
    for row in evidence:
        if row.get("interpretation_origin", "model") not in {"model", "human"}:
            raise PremiseBankInvalid("premise_source_identity")
        source = by_id.get(row.get("paragraph_id"))
        if source is None or any(row.get(k) != source.get(k) for k in ("raw_doc_id", "raw_para_id")):
            raise PremiseBankInvalid("premise_source_identity")
        try:
            validate_span(source["text"], row["quote"], row["start_cp"], row["end_cp"])
        except (KeyError, TypeError, ValueError):
            raise PremiseBankInvalid("premise_source_span") from None


def _replay_scope(question, candidates, scope):
    """Rebuild original and corrected views without trusting selected flags."""
    from app.domain.scope_batches import subset_evidence, merge_corrections
    views = build_fact_views(candidates)
    excluded = set(scope.get("operator_excluded_fact_view_ids", []))
    if excluded - {row["id"] for row in views}:
        raise PremiseBankInvalid("premise_scope_exclusions")
    original = []
    for row in candidates:
        own = {view["id"] for view in build_fact_views([row])}
        if own & excluded and not own <= excluded:
            raise PremiseBankInvalid("premise_scope_exclusions")
        if not own <= excluded:
            original.append(deepcopy(row))
    working, corrected = deepcopy(original), set()
    for ledger in scope.get("narrowing_ledger", []):
        if ledger.get("status") in {"invalid", "interrupted_unknown"}:
            if ledger.get("corrections") != []:
                raise PremiseBankInvalid("premise_scope_corrections")
            continue
        ids = [row["source_fact_view_id"] for row in ledger["corrections"]]
        checked = validate_narrowing_ledger(ledger, question, subset_evidence(working, ids), ids)
        working = merge_corrections(working, checked["corrected_evidence"])
        corrected.update(fid for row in checked["corrections"] for fid in row["corrected_fact_view_ids"])
    assessments = scope.get("assessments")
    if not isinstance(assessments, list) or any(row.get("phase") not in {"original", "reverify"} for row in assessments):
        raise PremiseBankInvalid("premise_scope_assessments")
    latest, selected = {}, set()
    for phase, pool in (("original", original), ("reverify", subset_evidence(working, sorted(corrected)))):
        phase_views = build_fact_views(pool)
        records = [row for row in assessments if row["phase"] == phase]
        by_id = {row["fact_view_id"]: row for row in records}
        if len(by_id) != len(records) or set(by_id) != {row["id"] for row in phase_views}:
            raise PremiseBankInvalid("premise_scope_coverage")
        payload, _, _ = build_scope_request(question, pool, detailed=True)
        decisions = {}
        for index, view in enumerate(phase_views):
            row = by_id[view["id"]]
            if (row.get("field_checks") is None or type(row.get("quote_requires_unstated_bridge")) is not bool
                    or type(row.get("target_requires_unstated_bridge")) is not bool):
                raise PremiseBankInvalid("premise_detailed_quote_review_required")
            decisions[f"C{index + 1}"] = {key: row[key] for key in ScopeVerificationDecision.model_fields}
        rebuilt = resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"],
                                              "decisions": decisions}, question, pool)
        for row in rebuilt["assessments"]:
            observed = {key: value for key, value in by_id[row["fact_view_id"]].items() if key != "phase"}
            if canonical(row) != canonical(observed):
                raise PremiseBankInvalid("premise_scope_assessment_replay")
            latest[row["fact_view_id"]] = {"phase": phase, **deepcopy(row)}
        if phase == "reverify":
            selected.difference_update(corrected)
        selected.update(rebuilt["selected_fact_view_ids"])
    if scope.get("selected_fact_view_ids") != sorted(selected) or scope.get("complete") is not True:
        raise PremiseBankInvalid("premise_original_scope_selection_replay")
    return original, working, latest


def build_quote_faithful_bank(question, candidates, strict_scope_review, paragraphs):
    """Keep literal-faithful views, regardless of their OLD target verdict.

    Detailed quote/field judgments are reused to avoid another paid fact check.
    The original full scope ledger remains untouched, including its rejections.
    """
    try:
        question = project_question(question)
        _check_sources(candidates, paragraphs)
        original, working, latest = _replay_scope(question, candidates, strict_scope_review)
        _check_sources(working, paragraphs)
        by_source = {row["id"]: row for row in working}
        primary_views = {row["id"]: build_fact_views([row])[0]["id"] for row in working}
        rows, decisions = [], []
        for view in build_fact_views(working):
            review = latest.get(view["id"])
            if review is None:
                raise PremiseBankInvalid("premise_missing_quote_review")
            checks = review["field_checks"]
            values = [("fact_des", view["fact_des"], checks["fact_des"]),
                      ("entities", view["entities"], checks["entities"]),
                      *[("argument." + key, view["argument"][key], checks["argument"][key]) for key in ARGUMENT_KEYS]]
            failures = [name for name, value, label in values
                        if label not in ({"entailed"} if value else {"entailed", "not_present"})]
            faithful = review["fact_entailment"] == "entailed" and not review["quote_requires_unstated_bridge"] and not failures
            decisions.append({"fact_view_id": view["id"], "quote_faithful": faithful,
                "review_phase": review["phase"], "field_issues": failures,
                "quote_requires_unstated_bridge": review["quote_requires_unstated_bridge"],
                "original_scope_selected": review["selected"]})
            if not faithful:
                continue
            row = deepcopy(by_source[view["evidence_id"]])
            row.pop("additional_interpretations", None)
            row.update({key: deepcopy(view[key]) for key in _ANNOTATION_FIELDS})
            manual = row.get("interpretation_origin") == "human"
            manual_primary = manual and primary_views[view["evidence_id"]] == view["id"]
            row.update(id=row["id"] if manual_primary else str(uuid5(NAMESPACE_URL, "trust-judge:premise-bank:" + view["id"])),
                source_evidence_id=view["evidence_id"], fact_view_id=view["id"],
                bank_fact_verified=True, quote_verified=True, purpose_verified=False, scope_verified=False)
            rows.append(row)
        body = {"version": VERSION, "question": question, "question_sha256": digest(_question_identity(question)),
            "source_manifest_sha256": digest(_source_manifest(paragraphs)),
            "original_candidate_sha256": digest(candidates), "working_evidence_sha256": digest(working),
            "original_scope_review_sha256": digest(strict_scope_review),
            "rows": sorted(rows, key=lambda row: row["id"]), "decisions": decisions,
            "manual_candidate_ids": sorted(row["id"] for row in original if row.get("interpretation_origin") == "human"),
            "complete": True, "formal_evidence": False,
            "meaning": "quote-faithful source assertions only; independent purpose permission and material qualification still required"}
        return {"bank_id": "QB" + digest(body), **body}
    except PremiseBankInvalid:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise PremiseBankInvalid("premise_bank_replay") from None


def _check_bank(bank):
    if (not isinstance(bank, dict) or bank.get("version") != VERSION or bank.get("complete") is not True
            or bank.get("formal_evidence") is not False
            or bank.get("bank_id") != "QB" + digest({k: v for k, v in bank.items() if k != "bank_id"})):
        raise PremiseBankInvalid("premise_bank_binding")


def _same_annotation(left, right):
    return all(left.get(key) == right.get(key) for key in (
        "paragraph_id", "raw_doc_id", "raw_para_id", *_ANNOTATION_FIELDS))


def build_fact_purpose_request(question, bank, fact_ids):
    question = project_question(question)
    _check_bank(bank)
    if digest(_question_identity(question)) != bank["question_sha256"]:
        raise PremiseBankInvalid("premise_question_binding")
    # A stored bank's hash is a binding, not permission for unknown question
    # fields to enter prompts (including synthetic component fixtures).
    question = project_question(bank["question"])
    by_id = {row["id"]: row for row in bank["rows"]}
    if (not isinstance(fact_ids, list) or any(not isinstance(fid, str) for fid in fact_ids)
            or len(fact_ids) != len(set(fact_ids)) or set(fact_ids) - set(by_id)):
        raise PremiseBankInvalid("premise_purpose_batch_identity")
    rows = [by_id[fid] for fid in fact_ids]
    targets = literal_target_catalog(question["question"])
    bindings = [{"id": f"{target['id']}.{role}", "target_id": target["id"], "role": role}
                for target in targets for role in sorted(RETAINED_PURPOSES)]
    bindings.extend({"id": role, "target_id": None, "role": role} for role in ("background", "uncertain", "duplicate"))
    visible = [{"id": f"C{index + 1}", **{key: deepcopy(row[key]) for key in (*_SOURCE_FIELDS, *_ANNOTATION_FIELDS)},
                **{key: deepcopy(row[key]) for key in ("source_group_id", "source_group_basis", "source_level") if key in row}}
               for index, row in enumerate(rows)]
    payload = {"version": VERSION, "question": question, "bank_id": bank["bank_id"],
        "target_catalog": targets, "purpose_binding_catalog": bindings, "candidate_catalog": visible}
    binding = "PU" + digest(payload)
    decisions = {}
    for index, row in enumerate(rows):
        alias = f"C{index + 1}"
        equivalents = [f"C{j + 1}" for j, other in enumerate(rows) if j != index and _same_annotation(row, other)]
        choices = [choice["id"] for choice in bindings if choice["role"] != "duplicate" or equivalents]
        decision = create_model("BoundPurposeDecision" + alias, __base__=PurposeDecision,
            purpose_binding_id=(Literal[tuple(choices)], Field(...)),
            equivalent_to=(Literal[tuple(equivalents)] | None if equivalents else type(None), Field(...)))
        decisions[alias] = (decision, Field(...))
    required = create_model("RequiredFactPurposes", __base__=OutputModel, **decisions)
    schema = create_model("BoundFactPurpose", __base__=FactPurposeOutput,
        bank_id=(Literal[bank["bank_id"]], Field(...)), purpose_catalog_id=(Literal[binding], Field(...)),
        decisions=(required, Field(...)))
    return {**payload, "purpose_catalog_id": binding}, schema, rows


def _parse_unique(output):
    if not isinstance(output, str):
        return output
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PremiseBankInvalid("premise_purpose_duplicate_json_key")
            result[key] = value
        return result
    return json.loads(output, object_pairs_hook=unique)


def resolve_fact_purpose(output, question, bank, fact_ids):
    payload, schema, rows = build_fact_purpose_request(question, bank, fact_ids)
    try:
        value = schema.model_validate(_parse_unique(output)).model_dump()
    except (ValueError, TypeError, ValidationError) as error:
        if isinstance(error, PremiseBankInvalid):
            raise
        raise PremiseBankInvalid("premise_purpose_contract") from None
    bindings = {row["id"]: row for row in payload["purpose_binding_catalog"]}
    targets = {row["id"]: row for row in payload["target_catalog"]}
    aliases = {f"C{index + 1}": row["id"] for index, row in enumerate(rows)}
    selected, assessments = [], []
    for alias, fid in aliases.items():
        decision = value["decisions"][alias]
        choice = bindings[decision["purpose_binding_id"]]
        retained = choice["role"] in RETAINED_PURPOSES and not decision["introduces_unstated_source_fact"]
        if retained:
            selected.append(fid)
        target = decision["equivalent_to"]
        if choice["role"] != "duplicate" and target is not None:
            raise PremiseBankInvalid("premise_purpose_equivalence")
        if choice["role"] == "duplicate":
            if target is None or target == alias:
                raise PremiseBankInvalid("premise_purpose_equivalence")
            peer = value["decisions"][target]
            if bindings[peer["purpose_binding_id"]]["role"] not in RETAINED_PURPOSES or peer["introduces_unstated_source_fact"]:
                raise PremiseBankInvalid("premise_purpose_equivalence")
        assessments.append({"evidence_id": fid, **deepcopy(decision),
            "equivalent_to": aliases[target] if target else None, "role": choice["role"],
            "target_question_text": targets[choice["target_id"]]["text"] if choice["target_id"] else None,
            "selected": retained})
    return {"version": VERSION, "bank_id": bank["bank_id"], "purpose_catalog_id": payload["purpose_catalog_id"],
        "fact_ids": list(fact_ids), "assessments": assessments, "selected_evidence_ids": selected}


def validate_fact_purpose_ledger(ledger, question, bank):
    try:
        ids = ledger["fact_ids"]
        aliases = {fid: f"C{index + 1}" for index, fid in enumerate(ids)}
        assessments = ledger["assessments"]
        if len(assessments) != len(ids) or [row["evidence_id"] for row in assessments] != ids:
            raise PremiseBankInvalid("premise_purpose_coverage")
        wire = {"bank_id": ledger["bank_id"], "purpose_catalog_id": ledger["purpose_catalog_id"], "decisions": {}}
        for row in assessments:
            wire["decisions"][aliases[row["evidence_id"]]] = {
                key: aliases[row[key]] if key == "equivalent_to" and row[key] is not None else row[key]
                for key in PurposeDecision.model_fields}
        rebuilt = resolve_fact_purpose(wire, question, bank, ids)
        if canonical(rebuilt) != canonical(ledger):
            raise PremiseBankInvalid("premise_purpose_ledger_replay")
        return rebuilt
    except PremiseBankInvalid:
        raise
    except (KeyError, TypeError, ValueError):
        raise PremiseBankInvalid("premise_purpose_ledger_replay") from None


def plan_fact_purpose_batches(question, bank, *, max_rows=48, max_codepoints=40_000):
    _check_bank(bank)
    if type(max_rows) is not int or max_rows < 1 or type(max_codepoints) is not int or max_codepoints < 1:
        raise PremiseBankInvalid("premise_purpose_batch_identity")
    batches, pending = [], []
    for row in bank["rows"]:
        prospective = [*pending, row["id"]]
        payload, schema, _ = build_fact_purpose_request(question, bank, prospective)
        size = len(json.dumps(payload, ensure_ascii=False)) + len(json.dumps(schema.model_json_schema(), ensure_ascii=False))
        if pending and (len(prospective) > max_rows or size > max_codepoints):
            batches.append(pending)
            pending = []
        pending.append(row["id"])
    if pending:
        batches.append(pending)
    return batches


def compile_purpose_projection(question, candidates, scope, bank, ledgers, paragraphs):
    from app.domain.material_eligibility import apply_workflow_material_eligibility
    from app.domain.scope_batches import deduplicate_verified_facts
    if digest(_question_identity(question)) != bank["question_sha256"]:
        raise PremiseBankInvalid("premise_question_binding")
    rebuilt = build_quote_faithful_bank(bank["question"], candidates, scope, paragraphs)
    if canonical(rebuilt) != canonical(bank):
        raise PremiseBankInvalid("premise_bank_replay")
    checked = [validate_fact_purpose_ledger(ledger, question, bank) for ledger in ledgers]
    ids = [fid for ledger in checked for fid in ledger["fact_ids"]]
    if len(ids) != len(set(ids)) or set(ids) != {row["id"] for row in bank["rows"]}:
        raise PremiseBankInvalid("premise_purpose_full_coverage")
    selected = {fid for ledger in checked for fid in ledger["selected_evidence_ids"]}
    manual_pending = sorted(set(bank["manual_candidate_ids"]) - selected)
    authorization = {"version": VERSION, "bank_id": bank["bank_id"], "purpose_ledgers_sha256": digest(checked)}
    rows = []
    for row in bank["rows"]:
        if row["id"] in selected:
            rows.append({**deepcopy(row), "scope_verified": True, "purpose_verified": True,
                "manual_review_required": False, "purpose_authorization": deepcopy(authorization)})
    qualified, material = apply_workflow_material_eligibility(question, paragraphs, rows)
    formal, duplicate_map = deduplicate_verified_facts(qualified)
    purpose = {"version": VERSION, "bank_id": bank["bank_id"], "complete": True,
        "ledgers": deepcopy(checked), "selected_evidence_ids": sorted(selected),
        "formal_evidence_ids": [row["id"] for row in formal], "manual_pending_ids": manual_pending,
        "formal_export_ready": not manual_pending, "exact_annotation_duplicate_map": duplicate_map,
        "authorization": authorization,
        "meaning": "formal pool requires quote fidelity AND independent question-purpose permission AND material qualification"}
    material["purpose_authorization"] = deepcopy(authorization)
    material["audit_sha256"] = digest({k: v for k, v in material.items() if k != "audit_sha256"})
    return formal, purpose, material


def validate_premise_projection(result, question, paragraphs):
    try:
        bank, purpose = result["quote_faithful_bank"], result["purpose_review"]
        expected, rebuilt, material = compile_purpose_projection(question, result["candidate_evidence"],
            result["strict_scope_review"], bank, purpose["ledgers"], paragraphs)
        if canonical(rebuilt) != canonical(purpose) or canonical(material) != canonical(result["material_eligibility"]):
            raise PremiseBankInvalid("premise_formal_authorization_replay")
        if purpose["manual_pending_ids"]:
            raise PremiseBankInvalid("premise_manual_purpose_unverified")
        actual = result["evidence"]
        by_id = {row["id"]: row for row in actual}
        if len(actual) != len(by_id) or set(by_id) != {row["id"] for row in expected}:
            raise PremiseBankInvalid("premise_formal_projection_replay")
        for row in expected:
            if any(by_id[row["id"]].get(key) != value for key, value in row.items()):
                raise PremiseBankInvalid("premise_formal_projection_replay")
    except PremiseBankInvalid:
        raise
    except (KeyError, TypeError, ValueError):
        raise PremiseBankInvalid("premise_formal_authorization_replay") from None


def safe_purpose_diagnostics(value):
    value = value if isinstance(value, dict) else {}
    safe = {"check": "premise_purpose_contract"}
    if isinstance(value.get("check"), str) and value["check"] in {"premise_purpose_duplicate_json_key", "premise_purpose_equivalence"}:
        safe["check"] = value["check"]
    fields = {"bank_id", "purpose_catalog_id", "decisions", "purpose_binding_id", "introduces_unstated_source_fact", "equivalent_to", "reason", "<field>"}
    kinds = {"missing", "extra_forbidden", "literal_error", "string_type", "bool_type", "model_type", "dict_type", "json_invalid", "value_error"}
    if isinstance(value.get("schema_errors"), list):
        safe["schema_errors"] = [{"loc": [part if (type(part) is int and part >= 0) or (isinstance(part, str) and part in fields) else "<field>" for part in row.get("loc", [])],
            "type": row.get("type") if isinstance(row.get("type"), str) and row["type"] in kinds else "value_error"}
            for row in value["schema_errors"] if isinstance(row, dict) and isinstance(row.get("loc"), list)]
    return safe
