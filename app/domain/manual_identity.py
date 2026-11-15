"""Stable explicit manual annotation identities, separate from quote anchors."""
from copy import deepcopy
import hashlib
import re
from uuid import NAMESPACE_URL, uuid5

from app.domain.strict_verification import build_fact_views, canonical


def _uuid(text):
    return str(uuid5(NAMESPACE_URL, "trust-judge:" + text))


def manual_identity(raw, canonical_row):
    anchor = canonical_row["id"]
    previous = raw.get("manual_identity")
    if previous is not None:
        if (not isinstance(previous, dict) or previous.get("source_anchor_id") != anchor
                or raw.get("source_evidence_id") != anchor or previous.get("evidence_id") != raw.get("id")):
            raise ValueError("invalid_manual_identity")
        kind = previous.get("kind")
        if kind == "verified_fact":
            if set(previous) != {"kind", "source_anchor_id", "evidence_id", "original_fact_view_id", "replaces_fact_view_id"}:
                raise ValueError("invalid_manual_identity")
            identity = previous.get("original_fact_view_id")
            if not isinstance(identity, str) or not re.fullmatch(r"F[0-9a-f]{64}", identity):
                raise ValueError("invalid_manual_identity")
            expected = _uuid("verified-fact:" + identity)
            if previous.get("replaces_fact_view_id") != identity:
                raise ValueError("invalid_manual_identity")
        elif kind == "new_manual_annotation":
            if set(previous) != {"kind", "source_anchor_id", "evidence_id", "original_annotation_sha256", "replaces_fact_view_id"}:
                raise ValueError("invalid_manual_identity")
            identity = previous.get("original_annotation_sha256")
            if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
                raise ValueError("invalid_manual_identity")
            expected = _uuid("manual-fact:" + anchor + ":" + identity)
        else:
            raise ValueError("invalid_manual_identity")
        if previous["evidence_id"] != expected:
            raise ValueError("invalid_manual_identity")
        target = previous.get("replaces_fact_view_id")
        if target is not None and (not isinstance(target, str) or not re.fullmatch(r"F[0-9a-f]{64}", target)):
            raise ValueError("invalid_manual_identity")
        return {"id": expected, "source_evidence_id": anchor, "manual_identity": deepcopy(previous)}
    original = build_fact_views([canonical_row])[0]["id"]
    if raw.get("source_evidence_id") is not None or raw.get("fact_view_id") is not None:
        if (raw.get("source_evidence_id") != anchor or raw.get("fact_view_id") != original
                or raw.get("id") != _uuid("verified-fact:" + original)):
            raise ValueError("invalid_manual_identity")
        identity = raw["id"]
        marker = {"kind": "verified_fact", "source_anchor_id": anchor,
                  "evidence_id": identity, "original_fact_view_id": original,
                  "replaces_fact_view_id": original}
    else:
        annotation = {key: canonical_row[key] for key in ("fact_des", "argument", "entities")}
        annotation_hash = hashlib.sha256(canonical(annotation).encode("utf-8")).hexdigest()
        identity = _uuid("manual-fact:" + anchor + ":" + annotation_hash)
        marker = {"kind": "new_manual_annotation", "source_anchor_id": anchor,
                  "evidence_id": identity, "original_annotation_sha256": annotation_hash,
                  "replaces_fact_view_id": original if raw.get("id") == anchor else None}
    return {"id": identity, "source_evidence_id": anchor, "manual_identity": marker}
