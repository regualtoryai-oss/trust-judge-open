"""Restricted source-inventory facts, never an arbitrary-number allowlist."""
from copy import deepcopy

from app.domain.source_lineage import COUNT_BASIS, get_program_source_counts


def inventory_statements(counts):
    if (not isinstance(counts, dict) or counts.get("basis") != COUNT_BASIS
            or any(type(counts.get(key)) is not int or not 0 <= counts[key] <= 1_000_000
                   for key in ("active_source_documents", "known_origin_groups"))):
        raise ValueError("invalid_program_source_inventory")
    return [f"已核验来源目录含{counts['active_source_documents']}篇原始文档",
            f"已核验来源目录含{counts['known_origin_groups']}个已知同源组"]


def inventory_proof(cited_evidence, candidates, paragraphs, *, excluded_spans=()):
    counts = get_program_source_counts(cited_evidence, candidates, paragraphs, excluded_spans=excluded_spans)
    return {"kind": "program_source_inventory", **counts,
            "cited_evidence_ids": sorted(row["id"] for row in cited_evidence),
            "allowed_statements": inventory_statements(counts)}


def strip_verified_inventory_statements(statement, proof, cited_ids):
    """Only fixed count-noun statements may bypass quote numeric checks.

    IDs, URLs, timestamps and arbitrary metadata numbers never become known
    event numbers. Source authenticity is rechecked by repository publication.
    """
    if proof is None:
        return statement
    if (not isinstance(proof, dict) or set(proof) != {"kind", "active_source_documents", "known_origin_groups",
            "basis", "cited_evidence_ids", "allowed_statements"}
            or proof.get("kind") != "program_source_inventory"
            or proof.get("cited_evidence_ids") != sorted(cited_ids)):
        raise ValueError("invalid_program_source_inventory")
    expected = inventory_statements(proof)
    if proof.get("allowed_statements") != expected:
        raise ValueError("invalid_program_source_inventory")
    remaining = statement
    for item in expected:
        remaining = remaining.replace(item, "[程序来源盘点]")
    return remaining


def validate_published_inventory(result, paragraphs, *, excluded_spans=()):
    final = result.get("final", {})
    proof = final.get("program_source_inventory")
    if proof is None:
        return
    ids = set(final.get("supporting_evidence_ids", [])) | set(final.get("conflicting_evidence_ids", []))
    rows = [row for row in result["evidence"] if row["id"] in ids]
    if {row["id"] for row in rows} != ids:
        raise ValueError("invalid_program_source_inventory")
    expected = inventory_proof(rows, result.get("candidate_evidence", []), paragraphs, excluded_spans=excluded_spans)
    if proof != expected:
        raise ValueError("invalid_program_source_inventory")
    strip_verified_inventory_statements(final.get("conclusion", ""), proof, ids)


def attach_inventory_basis(rows, candidates, paragraphs, *, excluded_spans=()):
    # This total is only an available snapshot inventory. Final numbers still
    # require exactly the groups of the final cited rows, recomputed separately.
    proof = inventory_proof(rows, candidates, paragraphs, excluded_spans=excluded_spans)
    result = deepcopy(rows)
    if result:
        result[0]["source_inventory_basis"] = proof
    return result
