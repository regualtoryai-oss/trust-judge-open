"""Lossless, role-scoped judgment aliases for arbitration input only.

This codec does not summarize, merge stances, select evidence or change any
statement. It shortens only judgment identity/reference fields. Snapshot IDs
stay in the trusted local audit instead of being repeated in model input.
"""
from copy import deepcopy
from hashlib import sha256
import json
import re


FORMAT = "analysis-wire-judgment-aliases-v1"
ROLE_PREFIXES = {"authority_analyst": "A", "cross_validation_analyst": "B"}
ANALYSIS_KEYS = frozenset({"role", "evidence_snapshot_id", "decision", "conclusion", "judgments",
    "supporting_evidence_ids", "conflicting_evidence_ids"})
JUDGMENT_KEYS = frozenset({"id", "statement", "evidence_ids", "prerequisite_judgment_ids"})
_ALIAS = re.compile(r"[AB][1-9][0-9]*\Z")


def _require(condition, reason):
    if not condition:
        raise ValueError("analysis_wire_projection_" + reason)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def _fingerprint(value):
    return sha256(_canonical(value)).hexdigest()


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _refs(value):
    _require(isinstance(value, list) and all(_text(item) for item in value)
        and len(value) == len(set(value)), "reference_list")


def _validate_original(analyses):
    _require(isinstance(analyses, list) and len(analyses) == 2
        and all(isinstance(row, dict) for row in analyses), "two_independent_analyses_required")
    _require(all(isinstance(row.get("role"), str) for row in analyses)
        and {row["role"] for row in analyses} == set(ROLE_PREFIXES), "role_binding")
    all_ids, all_evidence, snapshots = set(), set(), set()
    for analysis in analyses:
        # Unknown fields cannot accidentally carry credentials, future ledgers
        # or internal metadata through this explicitly closed model view.
        _require(ANALYSIS_KEYS <= set(analysis) <= ANALYSIS_KEYS | {"summary", "relation_quality"}, "analysis_keys")
        if "relation_quality" in analysis:
            from app.domain.provisional_relations import validate_relation_quality
            quality = validate_relation_quality(analysis)
            all_evidence.update(quality["quarantined_evidence_ids"])
        _require(_text(analysis["evidence_snapshot_id"]), "snapshot_binding")
        snapshots.add(analysis["evidence_snapshot_id"])
        _require(isinstance(analysis["decision"], str) and analysis["decision"] in {"asserted", "undetermined"}
            and _text(analysis["conclusion"]), "decision_or_conclusion")
        if "summary" in analysis:
            _require(isinstance(analysis["summary"], str), "summary_shape")
        supporting, conflicting = analysis["supporting_evidence_ids"], analysis["conflicting_evidence_ids"]
        _refs(supporting)
        _refs(conflicting)
        _require(not set(supporting) & set(conflicting)
            and (analysis["decision"] != "asserted" or bool(supporting)), "decision_relations")
        all_evidence.update(supporting + conflicting)
        judgments = analysis["judgments"]
        _require(isinstance(judgments, list), "judgment_list")
        own_ids = set()
        for judgment in judgments:
            _require(isinstance(judgment, dict) and set(judgment) == JUDGMENT_KEYS, "judgment_keys")
            identifier = judgment["id"]
            _require(_text(identifier) and not _ALIAS.fullmatch(identifier), "alias_collision")
            _require(identifier not in all_ids, "duplicate_judgment_identity")
            _require(_text(judgment["statement"]), "statement_shape")
            _refs(judgment["evidence_ids"])
            _refs(judgment["prerequisite_judgment_ids"])
            _require(bool(judgment["evidence_ids"] or judgment["prerequisite_judgment_ids"]), "ungrounded_judgment")
            own_ids.add(identifier)
            all_ids.add(identifier)
            all_evidence.update(judgment["evidence_ids"])
        # Kahn traversal preserves the complete DAG contract without a
        # recursion limit or an arbitrary judgment-count truncation.
        pending, dependants = {}, {identifier: [] for identifier in own_ids}
        for judgment in judgments:
            refs = judgment["prerequisite_judgment_ids"]
            _require(all(ref in own_ids for ref in refs), "foreign_prerequisite")
            pending[judgment["id"]] = len(refs)
            for ref in refs:
                dependants[ref].append(judgment["id"])
        ready, visited = [identifier for identifier, count in pending.items() if count == 0], 0
        while ready:
            identifier = ready.pop()
            visited += 1
            for dependant in dependants[identifier]:
                pending[dependant] -= 1
                if pending[dependant] == 0:
                    ready.append(dependant)
        _require(visited == len(judgments), "prerequisite_cycle")
    _require(len(snapshots) == 1, "snapshot_binding")
    _require(not all_ids & all_evidence, "judgment_evidence_identity_collision")
    _require(not any(_ALIAS.fullmatch(ref) for ref in all_evidence), "alias_collision")


def project_analyses(analyses):
    """Return ``{projected, audit}``; caller keeps audit out of the prompt.

    A means authority_analyst and B means cross_validation_analyst regardless
    of list order. Original evidence references and all prose stay untouched.
    """
    _validate_original(analyses)
    projected, bindings = [], []
    for analysis in analyses:
        prefix = ROLE_PREFIXES[analysis["role"]]
        mapping = {row["id"]: f"{prefix}{index + 1}" for index, row in enumerate(analysis["judgments"])}
        row = deepcopy(analysis)
        del row["evidence_snapshot_id"]
        for judgment in row["judgments"]:
            judgment["id"] = mapping[judgment["id"]]
            judgment["prerequisite_judgment_ids"] = [mapping[ref] for ref in judgment["prerequisite_judgment_ids"]]
        projected.append(row)
        bindings.append({"role": analysis["role"], "alias_prefix": prefix,
            "judgment_ids": [{"alias": alias, "original_id": identifier} for identifier, alias in mapping.items()],
            "omitted_metadata": {"evidence_snapshot_id": analysis["evidence_snapshot_id"]}})
    return {"projected": projected, "audit": {"format": FORMAT,
        "original_sha256": _fingerprint(analyses), "projected_sha256": _fingerprint(projected),
        "role_bindings": bindings}}


def restore_analyses(projected, audit, *, trusted_original):
    """Rebuild original identities/metadata, requiring the trusted original.

    An audit received from a model is never authority. Exact re-projection of
    the trusted input binds its complete prose, references, ordering and DAG.
    """
    expected = project_analyses(trusted_original)
    _require(projected == expected["projected"] and audit == expected["audit"], "trusted_binding_changed")
    restored = deepcopy(projected)
    for row, binding in zip(restored, audit["role_bindings"]):
        inverse = {item["alias"]: item["original_id"] for item in binding["judgment_ids"]}
        row.update(binding["omitted_metadata"])
        for judgment in row["judgments"]:
            judgment["id"] = inverse[judgment["id"]]
            judgment["prerequisite_judgment_ids"] = [inverse[ref] for ref in judgment["prerequisite_judgment_ids"]]
    _require(restored == trusted_original, "reconstruction_changed")
    return restored


def validate_projection(trusted_original, projected, audit):
    """Verify reversible projection without IO, model calls or prose inference."""
    restored = restore_analyses(projected, audit, trusted_original=trusted_original)
    return {"format": FORMAT, "original_sha256": audit["original_sha256"],
        "projected_sha256": audit["projected_sha256"], "original_bytes": len(_canonical(restored)),
        "projected_bytes": len(_canonical(projected)), "roles": [{"role": row["role"],
            "judgment_count": len(row["judgments"]),
            "dependency_count": sum(len(j["prerequisite_judgment_ids"]) for j in row["judgments"])} for row in restored],
        "all_prose_and_evidence_refs_preserved": True, "dag_preserved": True, "semantic_verified": False}
