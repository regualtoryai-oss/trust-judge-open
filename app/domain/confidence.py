"""Reproducible quality indices, never calibrated truth probabilities.

Conclusion output is ConclusionConfidenceExport without ``object_ref``. Evidence
output also needs caller-supplied inputs.source_level/source_group_id/
support_strength_basis to satisfy EvidenceConfidenceExport. Scores are unrounded
for persistence; calculation includes a separate rounded display value. A caller
must explicitly map an unknown source grade to 0 and preserve its unknown basis.
"""

from copy import deepcopy
import math

PARAMETER_VERSION = "parameters-initial-v1"
_BASIS = "v0.2 initial engineering parameter; not calibrated as factual probability"


def _unit(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _groups(groups, label):
    if not isinstance(groups, list):
        raise ValueError(f"{label} must be a list")
    by_id, ref_groups = {}, {}
    for group in groups:
        if not isinstance(group, dict) or set(group) - {"group_id", "grade_score", "grade_basis", "evidence_refs"}:
            raise ValueError("invalid independent source group fields")
        identity = group.get("group_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("source group requires a nonempty group_id")
        grade = _unit(group.get("grade_score"), "grade_score")
        basis = group.get("grade_basis", "caller-supplied source grade; not calibrated")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("source grade requires a nonempty grade_basis")
        refs = group.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            raise ValueError("source group requires evidence_refs")
        existing = by_id.setdefault(identity, {"group_id": identity, "grade_score": grade, "grade_basis": basis, "evidence_refs": []})
        if grade > existing["grade_score"]:
            existing["grade_score"], existing["grade_basis"] = grade, basis
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"doc_id", "para_id", "quote"} or not all(isinstance(v, str) and v.strip() for v in ref.values()):
                raise ValueError("evidence ref requires nonempty doc_id/para_id/quote only")
            key = (ref["doc_id"], ref["para_id"], ref["quote"])
            if key in ref_groups and ref_groups[key] != identity:
                raise ValueError("one evidence cannot count as two independent source groups")
            ref_groups[key] = identity
            if ref not in existing["evidence_refs"]:
                existing["evidence_refs"].append(deepcopy(ref))
    return list(by_id.values())


def _ids(values, label):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label} must be a list of nonempty node IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate IDs")
    return list(values)


def _calculation(formula, score, substitution, parameters):
    return {"formula": formula, "parameters": [{"name": name, "value": value, "basis": _BASIS} for name, value in parameters], "substitution": substitution, "unrounded_score": score, "rounded_score": round(score, 6), "decimal_places": 6}


def compute_conclusion_confidence(support_groups: list[dict], unresolved_conflict_groups: list[dict], chain_node_ids: list[str], verified_chain_node_ids: list[str], all_dependencies_valid: bool = True) -> dict:
    supports = _groups(support_groups, "support_groups")
    conflicts = _groups(unresolved_conflict_groups, "unresolved_conflict_groups")
    chain = _ids(chain_node_ids, "chain_node_ids")
    verified = _ids(verified_chain_node_ids, "verified_chain_node_ids")
    valid = _bool(all_dependencies_valid, "all_dependencies_valid")
    if not set(verified).issubset(chain):
        raise ValueError("verified chain node IDs must be a subset of chain node IDs")
    n_support, n_conflict = len(supports), len(conflicts)
    credibility = sum(g["grade_score"] for g in supports) / n_support if n_support else 0.0
    consistency = (1 - math.exp(-n_support / 2)) * n_support / (n_support + n_conflict) if n_support + n_conflict else 0.0
    completeness = len(verified) / len(chain) if chain else 0.0
    gate = int(bool(n_support and chain and valid))
    score = gate * (0.4 * credibility + 0.35 * consistency + 0.25 * completeness)
    return {"object_type": "conclusion", "score": score, "formula_version": "confidence-v0.2", "parameter_version": PARAMETER_VERSION, "interpretation": "quality_index", "inputs": {"support_groups": supports, "unresolved_conflict_groups": conflicts, "chain_node_ids": chain, "verified_chain_node_ids": verified, "all_dependencies_valid": valid}, "components": {"gate": gate, "source_credibility": credibility, "evidence_consistency": consistency, "reasoning_completeness": completeness}, "calculation": _calculation("C=gate*(0.40*S+0.35*((1-exp(-n_support/2))*n_support/(n_support+n_conflict))+0.25*R)", score, f"{gate}*(0.40*{credibility!r}+0.35*{consistency!r}+0.25*{completeness!r})={score!r}; n_support={n_support}; n_conflict={n_conflict}; R={len(verified)}/{len(chain)} (empty chain => 0)", [("source_weight", 0.4), ("consistency_weight", 0.35), ("reasoning_weight", 0.25), ("support_scale", 2.0)])}


def compute_evidence_confidence(grade_score: float, support_strength: float, quote_verified: bool) -> dict:
    grade, strength = _unit(grade_score, "grade_score"), _unit(support_strength, "support_strength")
    verified = _bool(quote_verified, "quote_verified")
    gate = int(verified)
    score = gate * grade * strength
    return {"object_type": "evidence", "score": score, "formula_version": "evidence-confidence-v0.2", "parameter_version": PARAMETER_VERSION, "interpretation": "quality_index", "inputs": {"grade_score": grade, "support_strength": strength, "quote_verified": verified}, "components": {"gate": gate, "grade_score": grade, "support_strength": strength}, "calculation": _calculation("C=quote_gate*grade_score*support_strength", score, f"{gate}*{grade!r}*{strength!r}={score!r}", [])}
