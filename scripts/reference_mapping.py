"""Generic source-ID diagnostics from the project's original offline evaluator.

No historical answers, model responses or process-wide audit hooks are included.
"""
import hashlib
import importlib.util
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    if not isinstance(row, dict):
        return None
    result = row.get("doc_id"), row.get("para_id")
    return result if all(isinstance(v, str) and v for v in result) else None


def question_union_reference(evidence, conclusion):
    """Union by fact identity; preserve ambiguous references in denominators."""
    facts = {}
    for row in evidence:
        facts.setdefault(row.get("fact_id"), []).append(row)
    rows, seen, issues = [], set(), []
    for side in ("supporting_facts", "conflicting_facts"):
        for ix, ref in enumerate(conclusion[side]):
            identity = ("fact_id", ref["fact_id"]) if isinstance(ref, dict) and ref.get("fact_id") else ("source", key(ref)) if key(ref) else ("unresolved", side, ix)
            if identity in seen:
                continue
            seen.add(identity)
            matches = facts.get(ref.get("fact_id"), []) if isinstance(ref, dict) and ref.get("fact_id") else []
            if len(matches) == 1:
                row = {k: matches[0].get(k) for k in ("doc_id", "para_id")}
                if any(k in ref and ref[k] != row[k] for k in ("doc_id", "para_id")):
                    issues.append({"check": "reference_relation_disagrees_with_fact_source", "side": side, "index": ix})
            elif not ref.get("fact_id") and key(ref):
                row = {k: ref[k] for k in ("doc_id", "para_id")}
            else:
                row = {"_unresolved_reference": identity}
                issues.append({"check": "unknown_or_ambiguous_reference_fact_id_retained", "side": side, "index": ix})
            rows.append(row)
    return rows, issues


def evaluator():
    path = MAIN / "scripts/evaluate_paragraphs.py"
    spec = importlib.util.spec_from_file_location("selected_source_id_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"file": str(path), "sha256": sha(path)}
