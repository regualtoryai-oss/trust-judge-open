"""Pure OFFLINE final-output adapters, not an agent or runtime evaluator.

Call only after a separate verifier has checked CLOSED generation, source bytes,
usage and artifacts. Reference data is passed explicitly; this module opens no
reference files, makes no model calls and does not certify that caller's closure.
"""
from copy import deepcopy

from scripts.offline_reference_tools import load_reference_tools
from scripts.two_stage_evidence_matching import FACT_FIELDS, compare_pairs, prepare_pairs


def _question(question):
    if (not isinstance(question, dict) or not isinstance(question.get("question_id"), str)
            or not question["question_id"] or not isinstance(question.get("question"), str)
            or not question["question"].strip()):
        raise ValueError("original_question_identity_required")
    return question["question_id"]


def _rows(rows, id_key, doc_key="doc_id", para_key="para_id"):
    if not isinstance(rows, list):
        raise ValueError("final_evidence_list_required")
    result, by_id = [], {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("final_evidence_object_required")
        identifier = row.get(id_key)
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("unique_final_fact_identity_required")
        normalized = {key: deepcopy(row[key]) for key in FACT_FIELDS if key in row}
        normalized.update(doc_id=row.get(doc_key), para_id=row.get(para_key))
        for key in ("doc_id", "para_id", "quote", "fact_des"):
            if not isinstance(normalized.get(key), str) or not normalized[key]:
                raise ValueError("final_original_identity_quote_and_claim_required")
        normalized["fact_id"] = identifier
        result.append(normalized)
        by_id[identifier] = normalized
    return result, by_id


def _relations(refs, by_id, *, expanded):
    if not isinstance(refs, list):
        raise ValueError("final_relation_list_required")
    result, seen = [], set()
    for ref in refs:
        identifier = ref.get("fact_id") if expanded and isinstance(ref, dict) else ref
        if not isinstance(identifier, str) or identifier not in by_id or identifier in seen:
            raise ValueError("relation_must_reference_unique_exported_fact")
        row = by_id[identifier]
        if expanded and any(ref.get(key) != row[key] for key in ("doc_id", "para_id", "quote")):
            raise ValueError("expanded_relation_source_identity_changed")
        seen.add(identifier)
        result.append(deepcopy(row))
    return result


def normalize_legacy(result, question):
    """Adapt an already CLOSED legacy result without re-running or repairing it."""
    qid = _question(question)
    if not isinstance(result, dict) or not isinstance(result.get("final"), dict):
        raise ValueError("complete_legacy_final_required")
    evidence, by_id = _rows(result.get("evidence"), "id", "raw_doc_id", "raw_para_id")
    # Match the archived public export projection (six fields, null -> "").
    # This does not generate or repair a claim; original result stays untouched.
    for row in evidence:
        for key in ("start_cp", "end_cp", "char_start", "char_end"):
            row.pop(key, None)  # internal positions are not part of the archived six-field export
        if isinstance(row.get("argument"), dict):
            row["argument"] = {key: "" if value is None else value for key, value in row["argument"].items()}
    final = result["final"]
    return {"question_id": qid, "conclusion": final.get("conclusion"), "evidence": evidence,
            "supporting_facts": _relations(final.get("supporting_evidence_ids"), by_id, expanded=False),
            "conflicting_facts": _relations(final.get("conflicting_evidence_ids"), by_id, expanded=False),
            "normalization": "legacy-final-no-repair-v1", "contains_reference_material": False}


def normalize_dsh(artifact, question):
    """Adapt DSH business-tool submission; proposed roles never become S/C."""
    qid = _question(question)
    if (not isinstance(artifact, dict) or artifact.get("format") != "dsh-proposed-evidence-v1"
            or artifact.get("question_id") != qid or not isinstance(artifact.get("conclusion"), dict)
            or artifact["conclusion"].get("question_id") != qid):
        raise ValueError("same_question_dsh_submission_required")
    evidence, by_id = _rows(artifact.get("evidence"), "fact_id")
    final = artifact["conclusion"]
    return {"question_id": qid, "conclusion": final.get("conclusion"), "evidence": evidence,
            "supporting_facts": _relations(final.get("supporting_facts"), by_id, expanded=True),
            "conflicting_facts": _relations(final.get("conflicting_facts"), by_id, expanded=True),
            "normalization": "dsh-submission-no-repair-v1", "contains_reference_material": False}


def score_final(question, normalized, reference_evidence, reference_supporting,
                reference_conflicting, source_contexts, *, closed_decisions=(), engine=None):
    """Compare final facts and each actual S/C relation, raw and paragraph-unique.

    References must be mapped by the pinned offline mapper first. Keep unknown
    reference rows in the denominator. Cached decisions are admitted by exact
    pair payload only; the caller must verify their CLOSED files independently.
    This is not conclusion truth scoring, official points or a source-ID proxy
    for semantic fact correctness.
    """
    if normalized.get("question_id") != _question(question):
        raise ValueError("same_question_final_required")
    if engine is None:
        _, engine, _ = load_reference_tools()
    categories = {}
    for kind, reference in (("evidence", reference_evidence),
                            ("supporting_facts", reference_supporting),
                            ("conflicting_facts", reference_conflicting)):
        submitted = normalized[kind]
        measured = engine.compare_rows(question["question_id"], submitted, reference)
        semantics = {}
        for policy in ("raw_records", "unique_paragraphs"):
            pairs = prepare_pairs(question, submitted, reference, source_contexts, record_policy=policy)
            hashes = {value for pair in pairs["pairs"] for value in
                      (pair["payload_sha256"], pair["legacy_payload_sha256"]) if value}
            exact = [row for row in closed_decisions if row.get("payload_sha256") in hashes]
            semantics[policy] = compare_pairs(pairs, exact)
        categories[kind] = {"A_source_id_proxy": measured, "two_interpretation_bounds": semantics}
    literal = []
    for i, row in enumerate(normalized["evidence"]):
        context = source_contexts.get((row["doc_id"], row["para_id"]))
        if not isinstance(context, str) or row["quote"] not in context:
            literal.append({"row_index": i, "reason": "missing_source_or_quote_not_literal_in_bound_context"})
    return {"format": "offline-final-evidence-comparison-v1", "question_id": question["question_id"],
            "categories": categories, "literal_quote_issues": literal,
            "raw_records_preserved": True, "same_text_different_source_ids_preserved": True,
            "relation_roles_inferred_from_reading_roles": False,
            "conclusion_semantic_correctness": "not_judged", "official_score": False,
            "closed_file_provenance_verified_here": False, "permitted_runtime_use": False,
            "contains_reference_material": True, "new_model_calls": 0}


def score_reference_conclusion(question, normalized, reference_evidence,
                               reference_conclusion, source_contexts, *, closed_decisions=()):
    """Use the existing relation mapper, preserving unknowns and multiplicity."""
    if (reference_conclusion.get("question_id") != _question(question)
            or reference_conclusion.get("question") != question["question"]):
        raise ValueError("same_original_reference_question_required")
    _, engine, guards = load_reference_tools()
    issues = []
    sides = [engine._relation_rows(reference_conclusion[side], reference_evidence,
              submitted=False, issues=issues, context=[question["question_id"], side, "reference"])
             for side in ("supporting_facts", "conflicting_facts")]
    result = score_final(question, normalized, reference_evidence, *sides, source_contexts,
                         closed_decisions=closed_decisions, engine=engine)
    result.update(reference_issues=issues, reference_mapper_guards=guards)
    return result
