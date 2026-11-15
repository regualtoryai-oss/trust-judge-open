"""Plan one quote-first second look without approving or changing evidence.

The plan covers rejected automatic anchors and explicit incomplete-annotation
requests, not proven facts. A selected peer interpretation does not cover a
rejected interpretation of the same quote.
Every output still needs the existing annotation narrowing and independent scope
verification. This module neither calls a model nor decides semantic relevance.
"""
from hashlib import sha256

from app.domain.strict_verification import (
    ScopeVerificationInvalid, build_fact_views, canonical,
)


VERSION = "quote-first-recall-recovery-v1"
_SOURCE_KEYS = (
    "evidence_id", "paragraph_id", "raw_doc_id", "raw_para_id",
    "start_cp", "end_cp", "quote",
)


class RecallRecoveryInvalid(ValueError):
    """A fixed, source-free error for invalid planning inputs."""

    def __init__(self):
        super().__init__("recall_recovery_input_invalid")
        self.check = "recall_recovery_input_invalid"


def _digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def _ids(value, known):
    if (not isinstance(value, list)
            or any(not isinstance(identity, str) or not identity for identity in value)
            or len(value) != len(set(value)) or set(value) - known):
        raise RecallRecoveryInvalid()
    return set(value)


def plan_recall_recovery(evidence, scope_selected_view_ids, scope_narrowing_view_ids):
    """Return deterministic parent view IDs plus a source-bound audit.

    Preserve every explicitly requested automatic narrowing view, including an
    already selected but incomplete interpretation. Passing one interpretation
    must not suppress the verifier's request to recover another omitted fact.
    For each other automatic anchor with an unselected view, choose its smallest
    view ID as a second-look parent. Existing narrowing already inspects that
    fixed quote, so it does not need an additional second-look parent. Multiple
    explicit narrowing views of one anchor remain distinct.

    Selected views are excluded only from newly added second-look work. Human
    anchors are never scheduled for automatic rewriting; explicit manual
    narrowing requests remain visible in the exclusion list.
    An absent interpretation_origin follows the existing domain convention of
    an automatic candidate; an explicitly unknown origin is rejected.

    All ID lists and bindings use canonical sorting, independent of input list
    order. Bindings cover the complete candidate view catalog and source origin,
    not just the chosen parent. They do not certify a quote against a paragraph
    absent from these inputs; the workflow retains that separate hard check.
    """
    try:
        views = build_fact_views(evidence)
        by_view = {view["id"]: view for view in views}
        known = set(by_view)
        selected = _ids(scope_selected_view_ids, known)
        requested = _ids(scope_narrowing_view_ids, known)
        origins = {}
        for source in evidence:
            origin = source.get("interpretation_origin", "model")
            if origin not in {"model", "human"}:
                raise RecallRecoveryInvalid()
            origins[source["id"]] = origin

        automatic = {
            view["id"] for view in views
            if origins[view["evidence_id"]] != "human"
        }
        pending = automatic - selected
        narrowing = requested & automatic
        by_anchor = {}
        for identity in sorted(pending):
            by_anchor.setdefault(by_view[identity]["evidence_id"], []).append(identity)
        narrowing_anchors = {by_view[identity]["evidence_id"] for identity in narrowing}
        second_look = {
            identities[0] for anchor, identities in by_anchor.items()
            if anchor not in narrowing_anchors
        }
        parent_ids = sorted(narrowing | second_look)
        parent_rows = []
        for identity in parent_ids:
            view = by_view[identity]
            source = {key: view[key] for key in _SOURCE_KEYS}
            parent_rows.append({
                "parent_fact_view_id": identity,
                "source_evidence_id": view["evidence_id"],
                "reason": "original_narrowing" if identity in narrowing else "second_look",
                "source_binding_sha256": _digest(source),
                "quote_sha256": sha256(view["quote"].encode("utf-8")).hexdigest(),
            })
        source_rows = {}
        for view in views:
            source_rows[view["evidence_id"]] = {
                **{key: view[key] for key in _SOURCE_KEYS},
                "interpretation_origin": origins[view["evidence_id"]],
            }
        source_binding = _digest([source_rows[key] for key in sorted(source_rows)])
        candidate_binding = _digest({
            "version": VERSION,
            "source_binding_sha256": source_binding,
            "fact_views": [by_view[key] for key in sorted(by_view)],
        })
        audit = {
            "original_narrowing_view_ids": sorted(requested),
            "retained_narrowing_view_ids": sorted(narrowing),
            "second_look_view_ids": sorted(second_look),
            "retained_selected_narrowing_view_ids": sorted(narrowing & selected),
            "excluded_manual_narrowing_view_ids": sorted(requested - automatic),
            "pending_automatic_view_ids": sorted(pending),
            "covered_automatic_anchor_ids": sorted({by_view[identity]["evidence_id"] for identity in parent_ids}),
            "parent_bindings": parent_rows,
            "counts": {
                "candidate_fact_views": len(views),
                "selected_fact_views": len(selected),
                "pending_automatic_fact_views": len(pending),
                "pending_automatic_anchors": len(by_anchor),
                "retained_narrowing_parents": len(narrowing),
                "retained_selected_narrowing_parents": len(narrowing & selected),
                "second_look_parents": len(second_look),
                "planned_parents": len(parent_ids),
            },
            "meaning": (
                "One bounded quote-first review plan only; no fact is accepted, "
                "no rejection is reversed, and no source or annotation is changed. "
                "Corrected annotations require independent scope verification."
            ),
        }
        body = {
            "version": VERSION,
            "source_binding_sha256": source_binding,
            "candidate_binding_sha256": candidate_binding,
            "parent_view_ids": parent_ids,
            "narrowing_view_ids": sorted(narrowing),
            "second_look_view_ids": sorted(second_look),
            "audit": audit,
        }
        return {"plan_id": "RR" + _digest(body), **body}
    except RecallRecoveryInvalid:
        raise
    except (ScopeVerificationInvalid, TypeError, ValueError, KeyError, UnicodeError):
        raise RecallRecoveryInvalid() from None
