"""Pure raw-only retrieval and replayable strict front-half projection.

This module owns no agent/model loop. The existing analysis StateGraph runs its
two paid nodes and the real independent A/B/arbitration tail.
"""
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from uuid import NAMESPACE_URL, uuid5

from app.agents.strict_evidence import validate_raw_case, validate_selection, validate_annotation
from app.domain.material_eligibility import apply_workflow_material_eligibility
from app.domain.paragraph_ranked import source_manifest, source_material_policy
from app.domain.retrieval import retrieve
from app.domain.source_metadata import source_credibility_value
from app.domain.question_recall import recall_question_candidates
from app.domain.evidence_balance import balance_selection

VERSION = "strict-workbench-evidence-v1"
BALANCED_VERSION = "strict-workbench-evidence-balanced-v2"
PROMPT_VERSION = "trusted-analysis-strict-workbench-v1"


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare_case(question, paragraphs, *, candidate_limit=60, excluded_spans=(), recall_strategy=None):
    if type(candidate_limit) is not int or not 1 <= candidate_limit <= 60:
        raise ValueError("strict_candidate_limit_invalid")
    if recall_strategy not in (None, "balanced", "window_balanced"):
        raise ValueError("strict_recall_strategy_invalid")
    manifest = source_manifest(paragraphs)
    byid = {row["id"]: row for row in manifest}
    excluded = set()
    for entry in excluded_spans:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            raise ValueError("strict_exclusion_invalid")
        pid, start, end, quote = entry
        if (pid not in byid or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(byid[pid]["text"]) or byid[pid]["text"][start:end] != quote):
            raise ValueError("strict_exclusion_source_changed")
        excluded.add(pid)
    material = source_material_policy(question, manifest)
    eligible_ids = set(material["eligible_paragraph_ids"])
    eligible = [p for p in manifest if p["id"] not in excluded and p["id"] in eligible_ids and p["text"].strip()]
    if recall_strategy is None:
        # Keep the original branch byte-for-byte equivalent for v1 replay.
        scores = defaultdict(float)
        for field in ("title", "text"):
            projected = [{"id": p["id"], "text": p.get(field) or ""} for p in eligible]
            for rank, hit in enumerate(retrieve(question["question"], projected, limit=max(1, len(projected))), 1):
                scores[hit["id"]] += 1 / (60 + rank)
        order = {p["id"]: i for i, p in enumerate(manifest)}
        selected_ids = sorted(scores, key=lambda pid: (-scores[pid], order[pid]))[:candidate_limit]
        recall = None
    else:
        recall = recall_question_candidates(question, eligible, limit=candidate_limit, variant=recall_strategy)
        selected_ids = [row["paragraph_id"] for row in recall["candidates"]]
    catalog, metadata = [], {}
    source_rows = {p["id"]: p for p in paragraphs}
    for number, pid in enumerate(selected_ids, 1):
        p = byid[pid]; original = source_rows[pid]; ref = f"P{number}"
        catalog.append({"paragraph_ref": ref, "doc_id": p["raw_doc_id"], "para_id": p["raw_para_id"],
                        "text": p["text"], "title": p.get("title") or "", "paragraph_complete": True, "parent_complete": True})
        grade, _basis = source_credibility_value(p)
        meta = {"paragraph_id": pid, "doc_id": p["raw_doc_id"], "para_id": p["raw_para_id"],
                "source_credibility": grade, "raw_publish_time": p.get("raw_publish_time"),
                "source_text_sha256_utf8": sha256(p["text"].encode()).hexdigest(),
                "parent_start_cp": 0, "parent_end_cp": len(p["text"])}
        start, end = original.get("document_start_cp"), original.get("document_end_cp")
        if type(start) is int and type(end) is int and end - start == len(p["text"]):
            meta.update(document_start_cp=start, document_end_cp=end)
        metadata[ref] = meta
    raw = {"format": "strict-workbench-raw-case-v1", "question": {"question_id": question["raw_question_id"], "question": question["question"]},
           "catalog": catalog, "source_metadata": metadata, "publishable": False,
           "retrieval": {"strategy": "title_body_bm25_rrf60_full_parents", "candidate_limit": candidate_limit,
                         "source_manifest_sha256": digest(manifest), "material_policy": material,
                         "operator_exclusions": sorted(map(list, excluded_spans)), "all_corpus_read_by_model": False}}
    if recall is not None:
        raw["retrieval"].update(strategy=f"question_{recall_strategy}_full_parents", recall_strategy=recall_strategy,
                                recall=recall)
    return validate_raw_case(raw)


def _balance_policy(value):
    if not isinstance(value, dict) or set(value) - {"target", "per_doc_soft_cap"}:
        raise ValueError("strict_balance_policy_invalid")
    result = {"target": value.get("target", 10), "per_doc_soft_cap": value.get("per_doc_soft_cap", 2)}
    if any(type(number) is not int or number < 1 for number in result.values()):
        raise ValueError("strict_balance_policy_invalid")
    return result


def annotation_inputs(raw_case, full_selection, balance_policy=None):
    """Detach only allocated full parents; keep the original P aliases unchanged.

    The caller retains the complete model selection. This projection never
    lowers rejected ratings or makes the allocated subset a semantic verdict.
    """
    raw = validate_raw_case(raw_case)
    selection = validate_selection(raw, full_selection).model_dump()
    if balance_policy is None:
        return raw, selection, None
    allocation = balance_selection(raw, selection, **_balance_policy(balance_policy))
    retained = set(allocation["retained_refs"])
    subset = deepcopy(raw)
    subset["catalog"] = [row for row in subset["catalog"] if row["paragraph_ref"] in retained]
    subset["source_metadata"] = {ref: deepcopy(meta) for ref, meta in raw["source_metadata"].items() if ref in retained}
    subset["annotation_scope"] = {"full_raw_case_sha256": digest(raw), "full_selection_sha256": digest(selection),
                                  "balance_ledger_sha256": digest(allocation), "original_aliases_preserved": True}
    chosen = {"hypotheses": deepcopy(selection["hypotheses"]),
              "decisions": [row for row in selection["decisions"] if row["p"] in retained]}
    return validate_raw_case(subset), validate_selection(subset, chosen).model_dump(), allocation


def _groups(paragraphs, selected_ids):
    parents = {p["document_id"]: p["document_id"] for p in paragraphs}
    def root(doc):
        while parents[doc] != doc:
            parents[doc] = parents[parents[doc]]; doc = parents[doc]
        return doc
    signals = {}
    for p in paragraphs:
        # Correlated exact copies constrain voting, never delete evidence rows.
        for signal in (*(("text:" + sha256(p["text"].encode()).hexdigest(),) if p["id"] in selected_ids and p["text"].strip() else ()),
                       *(f"{key}:{p[key]}" for key in ("content_sha256", "content_url") if p.get(key))):
            if signal in signals:
                a, b = sorted((root(p["document_id"]), root(signals[signal]))); parents[b] = a
            else:
                signals[signal] = p["document_id"]
    members = defaultdict(list)
    for doc in sorted(parents):
        members[root(doc)].append(doc)
    return {doc: "SG" + digest(members[root(doc)]) for doc in parents}


def compile_result(question, paragraphs, raw_case, selection, annotation, *, candidate_limit=60, excluded_spans=(),
                   balance_policy=None, recall_strategy=None):
    expected = prepare_case(question, paragraphs, candidate_limit=candidate_limit, excluded_spans=excluded_spans,
                            recall_strategy=recall_strategy)
    if raw_case != expected:
        raise ValueError("strict_raw_case_replay_changed")
    selected = validate_selection(raw_case, selection)
    balanced = balance_policy is not None or recall_strategy is not None
    annotation_raw, annotation_selection, balance_ledger = annotation_inputs(raw_case, selected, balance_policy)
    annotations = validate_annotation(annotation_raw, annotation_selection, annotation)
    selection_data, annotation_data = selected.model_dump(), annotations.model_dump()
    decisions = {row["p"]: row for row in selection_data["decisions"]}
    byid = {p["id"]: p for p in paragraphs}
    selected_ids = {raw_case["source_metadata"][row["p"]]["paragraph_id"] for row in annotation_data["annotations"]}
    groups = _groups(paragraphs, selected_ids)
    canonical = []
    for row in annotation_data["annotations"]:
        ref = row["p"]; p = byid[raw_case["source_metadata"][ref]["paragraph_id"]]
        grade, grade_basis = source_credibility_value(p)
        identity = str(uuid5(NAMESPACE_URL, "trust-judge:strict:" + digest({"question_id": question["id"], "paragraph_id": p["id"], "annotation": row})))
        canonical.append({"id": identity, "paragraph_id": p["id"], "raw_doc_id": p["raw_doc_id"], "raw_para_id": p["raw_para_id"],
                          "quote": p["text"], "start_cp": 0, "end_cp": len(p["text"]), "fact_des": row["fact_des"],
                          "argument": deepcopy(row["argument"]), "entities": list(row["entities"]),
                          "source_group_id": groups[p["document_id"]],
                          "source_group_basis": "known_doc_identity_or_exact_paragraph_correlation_not_proven_independent",
                          "grade_score": grade, "grade_basis": grade_basis, "source_level": "unknown", "quote_verified": True,
                          "interpretation_origin": "model", "annotation_semantics_verified": False,
                          "span_location_method": "program_strict_whole_original_paragraph", "model_span_corrected": False})
        if balanced:
            decision = decisions[ref]
            canonical[-1].update(selection_relevance=decision["relevance"], selection_stance=decision["stance"],
                                 selection_hypothesis_id=decision["hypothesis_id"],
                                 selection_interpretation_origin="model", selection_semantics_verified=False)
    canonical.sort(key=lambda row: row["id"])
    evidence, material = apply_workflow_material_eligibility(question, paragraphs, canonical)
    version = BALANCED_VERSION if balanced else VERSION
    proof = {"version": version, "question_sha256": digest(question), "source_manifest_sha256": digest(source_manifest(paragraphs)),
             "candidate_limit": candidate_limit, "raw_case_sha256": digest(raw_case), "selection": selection_data,
             "annotation": annotation_data, "operator_exclusions": sorted(map(list, excluded_spans)),
             "qualified_evidence_sha256": digest(evidence), "source_quote_span_verified": True, "semantic_verified": False}
    if balanced:
        proof.update(balance_policy=None if balance_policy is None else _balance_policy(balance_policy),
                     balance_ledger=balance_ledger, recall_strategy=recall_strategy,
                     annotation_raw_case_sha256=digest(annotation_raw),
                     annotation_selection_sha256=digest(annotation_selection),
                     source_metadata_sha256=digest(raw_case["source_metadata"]),
                     retrieval_sha256=digest(raw_case["retrieval"]))
    proof["proof_sha256"] = digest(proof)
    return {"evidence": evidence, "candidate_evidence": deepcopy(canonical), "material_eligibility": material,
            "verification_protocol": version, "strict_evidence_proof": proof,
            "evidence_selection": {"strategy": version, "selected_evidence_ids": [row["id"] for row in evidence], "semantic_verified": False,
                                   **({"hypotheses": deepcopy(selection_data["hypotheses"]), "interpretation_origin": "model",
                                       "balance_ledger": deepcopy(balance_ledger)} if balanced else {})},
            "source_coverage": {"strategy": version, "complete": False, "total_paragraphs": len(paragraphs),
                                "provided_full_paragraphs": len(raw_case["catalog"]), "selected_paragraphs": len(evidence),
                                "outside_candidate_scope": "unknown"}}


def validate_result(result, question, paragraphs, *, candidate_limit=None, excluded_spans=None):
    proof = result.get("strict_evidence_proof")
    version = result.get("verification_protocol")
    if (version not in (VERSION, BALANCED_VERSION) or not isinstance(proof, dict)
            or proof.get("version") != version):
        raise ValueError("strict_evidence_proof_required")
    limit = proof["candidate_limit"] if candidate_limit is None else candidate_limit
    excluded = proof["operator_exclusions"] if excluded_spans is None else excluded_spans
    options = ({"balance_policy": proof["balance_policy"], "recall_strategy": proof["recall_strategy"]}
               if version == BALANCED_VERSION else {})
    raw = prepare_case(question, paragraphs, candidate_limit=limit, excluded_spans=excluded,
                       recall_strategy=options.get("recall_strategy"))
    rebuilt = compile_result(question, paragraphs, raw, proof["selection"], proof["annotation"], candidate_limit=limit,
                             excluded_spans=excluded, **options)
    if any(result.get(key) != value for key, value in rebuilt.items()):
        raise ValueError("strict_evidence_projection_replay_failed")
    return rebuilt
