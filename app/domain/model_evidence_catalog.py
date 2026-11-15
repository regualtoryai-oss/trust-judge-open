"""Lossless fact projection and request-local aliases for independent analysts.

This is an opt-in v19+ model boundary, not a source verifier or a replacement for
the frozen evidence. The caller retains the original snapshot and validates all
resolved results against it. No semantic text is shortened or synthesized here.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json


class ModelEvidenceCatalogInvalid(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compact_schema(schema):
    """Drop display-only JSON Schema titles; retain every constraint/description."""
    if not isinstance(schema, dict):
        return deepcopy(schema)
    # Only schema positions are recursive. A default/const/example object with
    # a literal `title` member is data and must not be changed.
    maps = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
    lists = {"allOf", "anyOf", "oneOf", "prefixItems"}
    single = {"items", "contains", "additionalProperties", "unevaluatedProperties", "unevaluatedItems",
              "propertyNames", "not", "if", "then", "else", "contentSchema", "additionalItems"}
    result = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key in maps and isinstance(value, dict):
            result[key] = {name: compact_schema(item) for name, item in value.items()}
        elif (key in lists or key == "items") and isinstance(value, list):
            result[key] = [compact_schema(item) for item in value]
        elif key in single:
            result[key] = compact_schema(value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class ModelEvidenceCatalog:
    """Immutable map bound to exactly one question/snapshot/content combination."""
    binding_sha256: str
    _question_json: str
    _evidence_json: str
    _snapshot_id: str
    _aliases: tuple[tuple[str, str], ...]

    @property
    def evidence(self):
        return json.loads(self._evidence_json)

    @property
    def aliases(self):
        return dict(self._aliases)

    def assert_scope(self, question, snapshot_id):
        if _canonical(question) != self._question_json or snapshot_id != self._snapshot_id:
            raise ModelEvidenceCatalogInvalid("model_evidence_catalog_scope_mismatch")

    def _translate(self, output, mapping):
        """Only declared evidence-reference slots change, never quote/prose/J IDs."""
        result = deepcopy(output)
        if not isinstance(result, dict):
            raise ModelEvidenceCatalogInvalid("model_evidence_output_not_object")

        def references(row, key):
            if key not in row:
                return
            values = row[key]
            if (not isinstance(values, list) or any(not isinstance(value, str) or value not in mapping for value in values)
                    or len(set(values)) != len(values)):
                raise ModelEvidenceCatalogInvalid("model_evidence_unknown_or_duplicate_reference")
            row[key] = [mapping[value] for value in values]

        references(result, "supporting_evidence_ids")
        references(result, "conflicting_evidence_ids")
        for key in ("judgments", "conflicts"):
            values = result.get(key, [])
            if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
                raise ModelEvidenceCatalogInvalid("model_evidence_invalid_nested_references")
            for row in values:
                references(row, "evidence_ids")
                references(row, "supporting_evidence_ids")
                references(row, "conflicting_evidence_ids")
        return result

    def resolve(self, output, *, question, snapshot_id):
        self.assert_scope(question, snapshot_id)
        return self._translate(output, self.aliases)

    def project_analyses(self, analyses, *, question, snapshot_id):
        """For arbitration only: preserve visible judgments, drop runtime stamps.

        Independent A/B callers must never be given this list. A/B generation
        keeps the original isolated-role API; this function does not call models.
        """
        self.assert_scope(question, snapshot_id)
        if not isinstance(analyses, list):
            raise ModelEvidenceCatalogInvalid("model_evidence_analyses_not_list")
        reverse = {original: alias for alias, original in self._aliases}
        fields = {"decision", "conclusion", "judgments", "supporting_evidence_ids", "conflicting_evidence_ids", "conflicts", "role"}
        output = []
        for analysis in analyses:
            if (not isinstance(analysis, dict)
                    or analysis.get("evidence_snapshot_id", snapshot_id) != snapshot_id):
                raise ModelEvidenceCatalogInvalid("model_evidence_analysis_scope_mismatch")
            output.append(self._translate({key: value for key, value in analysis.items() if key in fields}, reverse))
        return output


def build_evidence_catalog(question, evidence, snapshot_id):
    """Remove bookkeeping, retain every original quote and fact annotation.

    Legacy expanded lineage is deliberately unsupported: silently discarding its
    member windows/counts could change an independence question's answer. Keep
    legacy requests unchanged until a separate lineage-preserving adapter exists.
    """
    if (not isinstance(question, dict) or not isinstance(snapshot_id, str) or not snapshot_id
            or not isinstance(evidence, list)):
        raise ModelEvidenceCatalogInvalid("model_evidence_invalid_scope")
    aliases, groups, originals, rows = [], {}, set(), []
    fields = {"quote", "fact_des", "argument", "entities", "source_level", "grade_score",
              "source_group_basis", "interpretation_origin", "stance"}
    for index, row in enumerate(evidence, 1):
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                or row["id"] in originals or not isinstance(row.get("quote"), str) or not row["quote"]
                or row.get("quote_verified") is False or row.get("paragraph_fields_verified") is False):
            raise ModelEvidenceCatalogInvalid("model_evidence_invalid_or_unverified_row")
        if "source_lineage" in row or "source_group_members" in row or "source_counts" in row:
            raise ModelEvidenceCatalogInvalid("model_evidence_expanded_lineage_requires_adapter")
        if "source_inventory_basis" in row:
            # Inventory counts are not just decorative metadata. Until their
            # allowlisted cited-ID proof is projected losslessly, fail closed.
            raise ModelEvidenceCatalogInvalid("model_evidence_inventory_requires_adapter")
        originals.add(row["id"])
        alias = f"E{index}"
        aliases.append((alias, row["id"]))
        projected = {key: deepcopy(value) for key, value in row.items() if key in fields}
        projected["id"] = alias
        group = row.get("source_group_id")
        if group is not None:
            if not isinstance(group, str) or not group:
                raise ModelEvidenceCatalogInvalid("model_evidence_invalid_group")
            projected["source_group_id"] = groups.setdefault(group, f"G{len(groups) + 1}")
        rows.append(projected)
    # Full original content is committed to the private binding, although no
    # internal UUID, source hash, coordinates or verification flag is sent.
    binding = sha256(_canonical({"question": question, "snapshot_id": snapshot_id, "evidence": evidence}).encode()).hexdigest()
    return ModelEvidenceCatalog(binding, _canonical(question), _canonical(rows), snapshot_id, tuple(aliases))
