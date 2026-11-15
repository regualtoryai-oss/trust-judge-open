"""Replayable quarantine of ambiguous provisional A/B relation labels.

This does not edit a model response or decide evidence direction. Only a
derived A/B view changes; final arbitration relations remain a hard contract.
"""
from copy import deepcopy
from hashlib import sha256
import json


FORMAT = "provisional-evidence-relation-quarantine-v1"
ROLES = frozenset({"authority_analyst", "cross_validation_analyst"})
RELATION_KEYS = ("supporting_evidence_ids", "conflicting_evidence_ids")


def _digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def _require(condition):
    if not condition:
        raise ValueError("provisional_relation_quality_contract")


def quarantine_provisional_relations(output, role):
    """Return a derived output and optional exact relation-only audit.

    The caller must validate original references against its frozen evidence
    inventory before calling. Unknown or duplicate references are not repaired.
    All other output fields are preserved, including the decision and DAG.
    """
    _require(role in ROLES and isinstance(output, dict))
    original = {key: deepcopy(output.get(key)) for key in RELATION_KEYS}
    for refs in original.values():
        _require(isinstance(refs, list) and all(isinstance(ref, str) and ref for ref in refs)
            and len(refs) == len(set(refs)))
    overlap = set(original[RELATION_KEYS[0]]) & set(original[RELATION_KEYS[1]])
    projected = deepcopy(output)
    if not overlap:
        return projected, None
    for key in RELATION_KEYS:
        projected[key] = [ref for ref in original[key] if ref not in overlap]
    projected_relations = {key: projected[key] for key in RELATION_KEYS}
    quarantined = [ref for ref in original[RELATION_KEYS[0]] if ref in overlap]
    audit = {"format": FORMAT, "role": role, "original_relations": original,
        "original_relations_sha256": _digest(original),
        "quarantined_evidence_ids": quarantined, "quarantined_count": len(quarantined),
        "projected_relations_sha256": _digest(projected_relations),
        "raw_model_response_modified": False, "evidence_removed": False,
        "decision_changed": False, "semantic_verified": False}
    return projected, audit


def validate_relation_quality(analysis, *, role=None):
    """Replay the local relation projection, not the model's semantic truth.

    Native-response provenance remains separately bound by transport/scorer.
    A self-declared audit cannot establish that native response provenance.
    """
    _require(isinstance(analysis, dict))
    if "relation_quality" not in analysis:
        return None
    audit = analysis["relation_quality"]
    role = role if role is not None else analysis.get("role")
    _require(role in ROLES and isinstance(audit, dict) and audit.get("role") == role)
    _require("role" not in analysis or analysis["role"] == role)
    original = audit.get("original_relations")
    _require(isinstance(original, dict) and set(original) == set(RELATION_KEYS))
    replay_input = {key: deepcopy(value) for key, value in analysis.items() if key != "relation_quality"}
    replay_input.update(deepcopy(original))
    projected, expected = quarantine_provisional_relations(replay_input, role)
    _require(expected is not None and audit == expected
        and all(analysis.get(key) == projected[key] for key in RELATION_KEYS)
        and (analysis.get("decision") != "asserted" or bool(projected[RELATION_KEYS[0]])))
    return deepcopy(audit)
