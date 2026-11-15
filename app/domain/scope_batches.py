"""Lossless bounded atomic-view batches and immutable fact projections."""
from copy import deepcopy
import json
from uuid import NAMESPACE_URL, uuid5

from app.domain.strict_verification import build_fact_views, canonical, project_question
from app.domain.compiled_scope import build_compiled_scope_request


def subset_evidence(evidence, view_ids):
    wanted = set(view_ids)
    views = build_fact_views(evidence)
    if len(wanted) != len(view_ids) or wanted - {view["id"] for view in views}:
        raise ValueError("scope_batch_view_identity")
    by_anchor = {}
    for view in views:
        if view["id"] in wanted:
            by_anchor.setdefault(view["evidence_id"], []).append(view)
    result = []
    for anchor in evidence:
        selected = by_anchor.get(anchor["id"], [])
        if not selected:
            continue
        row = deepcopy(anchor)
        row.pop("additional_interpretations", None)
        for key in ("fact_des", "argument", "entities"):
            row[key] = deepcopy(selected[0][key])
        if len(selected) > 1:
            row["additional_interpretations"] = [
                {key: deepcopy(view[key]) for key in ("fact_des", "argument", "entities")}
                for view in selected[1:]]
        result.append(row)
    if {view["id"] for view in build_fact_views(result)} != wanted:
        raise ValueError("scope_batch_view_projection")
    return result


def scope_payload_codepoints(question, evidence):
    """Codepoints in the provider's actual base HumanMessage JSON projection.

    Includes the projected question, catalog binding, aliases, all per-view
    source fields, complete repeated quotes, annotations and JSON separators/
    escapes. This matches provider._request's json.dumps(ensure_ascii=False).
    System instructions, dynamic output schema, HTTP envelope and any repair
    diagnostics/body are separate overhead; this is not a provider token cap.
    """
    payload, _, _ = build_compiled_scope_request(question, evidence)
    return len(json.dumps(payload, ensure_ascii=False))


def plan_scope_batches(evidence, *, question, max_views=48, max_codepoints=40_000, view_ids=None):
    """Soft-limit real base request payloads without dropping/cutting any view.

    A single view exceeding the target stays whole in its own batch. The same
    projection is used for original review and reverify. Narrowing's unchanged
    base payload is smaller; it lacks the compiled target catalog. Its output
    schema and instructions are different and are not counted in this target.
    """
    if type(max_views) is not int or max_views < 1 or type(max_codepoints) is not int or max_codepoints < 1:
        raise ValueError("scope_batch_limits")
    question = project_question(question)
    views = build_fact_views(evidence)
    wanted = set(view_ids) if view_ids is not None else {view["id"] for view in views}
    if view_ids is not None and len(wanted) != len(view_ids):
        raise ValueError("scope_batch_view_identity")
    if wanted - {view["id"] for view in views}:
        raise ValueError("scope_batch_view_identity")
    batches, pending = [], []
    for view in views:
        if view["id"] not in wanted:
            continue
        prospective = [*pending, view["id"]]
        # Quote windows remain whole in every batch. One oversized view alone
        # is retained; no suffix, condition, source or view is silently cut.
        if pending and (len(prospective) > max_views
                        or scope_payload_codepoints(question, subset_evidence(evidence, prospective)) > max_codepoints):
            batches.append(pending)
            pending = []
        pending.append(view["id"])
    if pending:
        batches.append(pending)
    flat = [identity for batch in batches for identity in batch]
    if len(flat) != len(wanted) or set(flat) != wanted:
        raise ValueError("scope_batch_incomplete")
    return batches


def merge_corrections(original, corrected):
    """A new working projection; the original candidate ledger stays unchanged."""
    result = deepcopy(original)
    by_id = {row["id"]: row for row in result}
    for row in corrected:
        old = by_id.get(row["id"])
        if old is None or any(old[key] != row[key] for key in (
                "paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote")):
            raise ValueError("scope_correction_geometry")
        existing = {view["id"] for view in build_fact_views([old])}
        for view in build_fact_views([row]):
            if view["id"] not in existing:
                old.setdefault("additional_interpretations", []).append({
                    key: deepcopy(view[key]) for key in ("fact_des", "argument", "entities")})
                existing.add(view["id"])
    return result


def verified_fact_rows(evidence, selected_view_ids, assessments):
    """Each accepted automatic view is an independently visible canonical row.

    Manual source/annotation IDs retain user intent. Failed manual opinions are
    explicit; they are never silently removed or stamped as model-verified.
    """
    selected = set(selected_view_ids)
    views = build_fact_views(evidence)
    by_id = {row["id"]: row for row in evidence}
    unknown = selected - {view["id"] for view in views}
    if unknown:
        raise ValueError("scope_unknown_accepted_view")
    result = []
    replaced_views = {row.get("manual_identity", {}).get("replaces_fact_view_id") for row in evidence if row.get("interpretation_origin") == "human"}
    human = {row["id"] for row in evidence if row.get("interpretation_origin") == "human"}
    for identity in human:
        row = deepcopy(by_id[identity])
        own = [view for view in views if view["evidence_id"] == identity]
        row.pop("additional_interpretations", None)
        row["fact_view_id"] = own[0]["id"]
        row["scope_verified"] = own[0]["id"] in selected
        row["manual_review_required"] = not row["scope_verified"]
        result.append(row)
    for view in views:
        if view["id"] not in selected:
            continue
        if view["evidence_id"] in human and view["id"] == next(
                candidate["id"] for candidate in views if candidate["evidence_id"] == view["evidence_id"]):
            continue  # Preserve the explicitly edited primary identity.
        if view["evidence_id"] not in human and view["id"] in replaced_views:
            continue  # Explicit user edit replaces only this fact, never its peer views.
        row = deepcopy(by_id[view["evidence_id"]])
        row.pop("additional_interpretations", None)
        row.update({key: deepcopy(view[key]) for key in ("fact_des", "argument", "entities")})
        row.update({"id": str(uuid5(NAMESPACE_URL, "trust-judge:verified-fact:" + view["id"])),
                    "source_evidence_id": view["evidence_id"], "fact_view_id": view["id"],
                    "scope_verified": True})
        result.append(row)
    return sorted(result, key=lambda row: row["id"])


def deduplicate_verified_facts(rows):
    """Exact annotation equality within one paragraph only; no semantic guess.

    Keep one already verified quote, choosing shortest then stable ID. Manual
    identities are never silently removed. The full view/anchor audit stays
    separate; different documents retain their own source facts.
    """
    groups, manual = {}, []
    for row in rows:
        if row.get("interpretation_origin") == "human" or row.get("scope_verified") is not True:
            manual.append(row)
            continue
        key = canonical({"paragraph_id": row["paragraph_id"], "raw_doc_id": row["raw_doc_id"],
            "fact_des": row["fact_des"], "argument": {key: row["argument"].get(key) for key in
                ("时间", "主体", "地点", "内容", "影响", "数值口径")},
            "entities": sorted(set(row["entities"]))})
        groups.setdefault(key, []).append(row)
    result, audit = [deepcopy(row) for row in manual], []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: (len(row["quote"]), row["id"]))
        representative = ordered[0]
        result.append(deepcopy(representative))
        audit.extend({"fact_evidence_id": row["id"], "fact_view_id": row["fact_view_id"],
                      "representative_evidence_id": representative["id"],
                      "basis": "same_paragraph_exact_annotation_shortest_independently_verified_quote"}
                     for row in ordered[1:])
    return sorted(result, key=lambda row: row["id"]), audit
