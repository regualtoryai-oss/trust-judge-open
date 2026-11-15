"""Deterministic paragraph allocation after a model's unchanged 0--3 ratings.

This is an offline policy primitive, not a semantic judge. It reads candidate
identities and canonical ratings only, keeps counterpositions/qualifications,
and does not infer independent sources or deduplicate mirrored paragraphs.
"""
from collections import Counter
from hashlib import sha256
from itertools import product
import json


VERSION = "paragraph-role-document-balance-v1"
STANCES = ("supports_hypothesis", "opposes_hypothesis", "qualification")


def _digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _inputs(raw_case, canonical_selection):
    if not isinstance(raw_case, dict) or not isinstance(raw_case.get("catalog"), list):
        raise ValueError("balance_catalog_required")
    catalog = raw_case["catalog"]
    refs, identities = set(), set()
    for item in catalog:
        if (not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key]
                for key in ("paragraph_ref", "doc_id", "para_id"))):
            raise ValueError("balance_original_identity_required")
        ref, identity = item["paragraph_ref"], (item["doc_id"], item["para_id"])
        if ref in refs or identity in identities:
            raise ValueError("balance_duplicate_original_identity")
        refs.add(ref); identities.add(identity)
    if not isinstance(canonical_selection, dict) or not isinstance(canonical_selection.get("decisions"), list):
        raise ValueError("balance_canonical_decisions_required")
    decisions = {}
    for item in canonical_selection["decisions"]:
        if (not isinstance(item, dict) or item.get("p") not in refs or item["p"] in decisions
                or type(item.get("relevance")) is not int or not 0 <= item["relevance"] <= 3
                or item.get("stance") not in STANCES):
            raise ValueError("balance_invalid_or_duplicate_decision")
        decisions[item["p"]] = item
    if set(decisions) != refs:
        raise ValueError("balance_decisions_must_cover_original_catalog")
    # Selection order is never allowed to replace the raw retrieval order.
    return [{"p": item["paragraph_ref"], "doc_id": item["doc_id"], "para_id": item["para_id"],
             "rank": rank, "relevance": decisions[item["paragraph_ref"]]["relevance"],
             "stance": decisions[item["paragraph_ref"]]["stance"]}
            for rank, item in enumerate(catalog)]


def balance_selection(raw_case, canonical_selection, target=10, per_doc_soft_cap=2):
    """Return an auditable subset without rewriting any model/source fields.

    Existing 2+/3-rated stance lanes receive one slot each. Document caps may
    be exceeded only when necessary to cover those existing lanes. Remaining
    slots use score first, then new-document coverage, then retrieval order;
    background rows never pad the result. ``target`` is a desired size, and is
    raised explicitly if it is smaller than the number of existing lanes.
    """
    if type(target) is not int or target < 1:
        raise ValueError("balance_positive_target_required")
    if type(per_doc_soft_cap) is not int or per_doc_soft_cap < 1:
        raise ValueError("balance_positive_document_cap_required")
    rows = _inputs(raw_case, canonical_selection)
    qualified = [row for row in rows if row["relevance"] >= 2]
    lanes = {stance: [row for row in qualified if row["stance"] == stance] for stance in STANCES}
    present = [stance for stance in STANCES if lanes[stance]]
    effective_target = min(len(qualified), max(target, len(present)))
    chosen, reasons, relaxations = {}, {}, []
    allocations = Counter()

    if present:
        # Three disjoint stance lists from the 60-row catalog need at most
        # 20*20*20 combinations. Full coverage is checked before relaxing caps,
        # so an early greedy choice cannot invent a need to exceed the cap.
        def coverage_key(combination):
            counts = Counter(row["doc_id"] for row in combination)
            return (sum(max(0, n - per_doc_soft_cap) for n in counts.values()),
                    -sum(row["relevance"] == 3 for row in combination),
                    -len(counts), tuple(sorted(row["rank"] for row in combination)))
        coverage = min(product(*(lanes[stance] for stance in present)), key=coverage_key)
        for row in sorted(coverage, key=lambda row: row["rank"]):
            previous = allocations[row["doc_id"]]
            chosen[row["p"]] = row
            reasons[row["p"]] = "existing_stance_coverage"
            allocations[row["doc_id"]] += 1
            if previous >= per_doc_soft_cap:
                relaxations.append({"p": row["p"], "doc_id": row["doc_id"], "stance": row["stance"],
                    "previous_allocation": previous, "reason": "required_existing_stance_coverage",
                    "no_full_lane_cover_with_lower_total_cap_excess": True})

    while len(chosen) < effective_target:
        candidates = [row for row in qualified if row["p"] not in chosen
                      and allocations[row["doc_id"]] < per_doc_soft_cap]
        if not candidates:
            break
        score = max(row["relevance"] for row in candidates)
        tier = [row for row in candidates if row["relevance"] == score]
        new_documents = [row for row in tier if allocations[row["doc_id"]] == 0]
        row = (new_documents or tier)[0]
        chosen[row["p"]] = row
        reasons[row["p"]] = "score_then_new_document_then_retrieval_order"
        allocations[row["doc_id"]] += 1

    retained = [row for row in rows if row["p"] in chosen]
    retained_refs = [row["p"] for row in retained]
    truncated = [row for row in qualified if row["p"] not in chosen]
    policy = {"version": VERSION, "minimum_relevance": 2, "target": target,
        "per_doc_soft_cap": per_doc_soft_cap, "existing_stance_minimum": 1,
        "stance_coverage_overrides_target": True,
        "coverage_order": ["minimum_necessary_document_cap_excess", "score_3_before_2",
                           "more_document_ids", "original_retrieval_order"],
        "fill_order": ["score_3_before_2", "new_document_within_same_score", "original_retrieval_order"],
        "cap_relaxation": "existing_stance_coverage_only", "output_order": "original_retrieval_order",
        "mirrors_deduplicated": False, "independent_source_votes_computed": False,
        "model_scores_roles_and_source_metadata_rewritten": False}
    document_order = list(dict.fromkeys(row["doc_id"] for row in rows))
    documents = [{"doc_id": doc,
        "qualified_refs": [row["p"] for row in qualified if row["doc_id"] == doc],
        "retained_refs": [row["p"] for row in retained if row["doc_id"] == doc],
        "allocation": allocations[doc], "over_soft_cap": max(0, allocations[doc] - per_doc_soft_cap)}
        for doc in document_order]
    audit = {"format": VERSION, "raw_case_sha256": _digest(raw_case),
        "canonical_selection_sha256": _digest(canonical_selection),
        "effective_target": effective_target, "target_raised_for_lane_coverage": len(present) > target,
        "candidate_count": len(rows), "qualified_count": len(qualified), "retained_count": len(retained),
        "source_counts": {"basis": "document_ids_not_independent_source_votes",
            "candidate_document_ids": len(document_order),
            "qualified_document_ids": len({row["doc_id"] for row in qualified}),
            "retained_document_ids": len({row["doc_id"] for row in retained})},
        "lanes": {stance: {"qualified_refs": [row["p"] for row in lanes[stance]],
                            "retained_refs": [row["p"] for row in retained if row["stance"] == stance]}
                  for stance in STANCES},
        "document_allocations": documents, "cap_relaxations": relaxations,
        "retention_reasons": {ref: reasons[ref] for ref in retained_refs},
        "truncated_refs": [row["p"] for row in truncated],
        "truncated": [{"p": row["p"], "doc_id": row["doc_id"], "para_id": row["para_id"],
            "relevance": row["relevance"], "stance": row["stance"],
            "reason": ("document_soft_cap" if allocations[row["doc_id"]] >= per_doc_soft_cap
                       else "effective_target_reached")} for row in truncated],
        "below_threshold_refs": [row["p"] for row in rows if row["relevance"] < 2],
        "decision": "undetermined" if not retained else "requires_independent_analysis",
        "semantic_quality_verified": False, "reference_data_used": False}
    return {"retained_refs": retained_refs, "policy": policy, "audit": audit}
