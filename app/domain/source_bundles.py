"""Pure source-bound, document-scope paragraph bundles.

The caller supplies complete trusted document/paragraph manifests, normally
projected from the immutable source database. This module has no repository,
model, network, reference-label, retrieval, or publication dependency. It does
not attest that a model read a bundle or that an annotation is a true fact.

Document rows additionally require ``paragraph_manifest`` (ordered original
rows with id/raw_para_id/start_cp/end_cp/text_sha256) and its canonical JSON
``paragraph_manifest_sha256``. Opaque original IDs are never parsed for order.
The manifest defines the original segmentation; text gaps are not guessed.
"""
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from uuid import NAMESPACE_URL, uuid5


SOURCE_BUNDLE_CORE_FIELDS = (
    "span_scope", "paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote",
    "source_paragraph_ids", "document_content_sha256", "paragraph_manifest_sha256", "document_id", "bundle_id",
)
_MANIFEST_FIELDS = {"id", "raw_para_id", "start_cp", "end_cp", "text_sha256"}


def _sha(text):
    return sha256(text.encode("utf-8")).hexdigest()


def paragraph_manifest_sha256(manifest):
    """Hash the exact ordered trusted manifest using UTF-8 canonical JSON."""
    return _sha(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _identity(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source_bundle_invalid_identity")
    return value


def _valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _cap(max_paragraphs):
    if type(max_paragraphs) is not int or not 1 <= max_paragraphs <= 5:
        raise ValueError("source_bundle_max_paragraphs_must_be_1_to_5")


def _prepare(documents, paragraphs):
    if not isinstance(documents, (list, tuple)) or not isinstance(paragraphs, (list, tuple)):
        raise ValueError("source_bundle_source_collections_required")
    docs, expected, raw_docs = {}, {}, set()
    for row in documents:
        if not isinstance(row, Mapping):
            raise ValueError("source_bundle_invalid_document")
        did, raw_did = _identity(row.get("id")), _identity(row.get("raw_doc_id"))
        if did in docs or raw_did in raw_docs:
            raise ValueError("source_bundle_duplicate_document_identity")
        content = row.get("content")
        if not isinstance(content, str) or not _valid_sha(row.get("content_sha256")) or _sha(content) != row["content_sha256"]:
            raise ValueError("source_bundle_document_sha256_mismatch")
        manifest = row.get("paragraph_manifest")
        if not isinstance(manifest, list):
            raise ValueError("source_bundle_complete_paragraph_manifest_required")
        if not _valid_sha(row.get("paragraph_manifest_sha256")) or paragraph_manifest_sha256(manifest) != row["paragraph_manifest_sha256"]:
            raise ValueError("source_bundle_manifest_sha256_mismatch")
        ordered, raw_ids, prior_start = [], set(), 0
        for position, item in enumerate(manifest):
            if not isinstance(item, dict) or set(item) != _MANIFEST_FIELDS:
                raise ValueError("source_bundle_invalid_manifest_row")
            pid, raw_pid = _identity(item.get("id")), _identity(item.get("raw_para_id"))
            if pid in expected or raw_pid in raw_ids:
                raise ValueError("source_bundle_duplicate_paragraph_identity")
            start, end = item.get("start_cp"), item.get("end_cp")
            if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(content):
                raise ValueError("source_bundle_invalid_original_boundary")
            if start < prior_start:
                raise ValueError("source_bundle_manifest_start_order")
            text = content[start:end]
            if not _valid_sha(item.get("text_sha256")) or _sha(text) != item["text_sha256"]:
                raise ValueError("source_bundle_manifest_text_sha256_mismatch")
            original = {"id": pid, "document_id": did, "raw_doc_id": raw_did, "raw_para_id": raw_pid,
                        "document_start_cp": start, "document_end_cp": end, "text": text, "position": position}
            expected[pid] = original
            ordered.append(original)
            raw_ids.add(raw_pid)
            prior_start = start
        if "paragraph_ids" in row and row["paragraph_ids"] != [p["id"] for p in ordered]:
            raise ValueError("source_bundle_paragraph_ids_manifest_mismatch")
        docs[did] = {"id": did, "raw_doc_id": raw_did, "content": content,
                     "content_sha256": row["content_sha256"], "paragraph_manifest_sha256": row["paragraph_manifest_sha256"],
                     "paragraphs": ordered}
        raw_docs.add(raw_did)
    supplied = set()
    for row in paragraphs:
        if not isinstance(row, Mapping):
            raise ValueError("source_bundle_invalid_paragraph")
        pid = _identity(row.get("id"))
        if pid in supplied:
            raise ValueError("source_bundle_duplicate_supplied_paragraph")
        original = expected.get(pid)
        if original is None:
            raise ValueError("source_bundle_unknown_supplied_paragraph")
        for key in ("document_id", "raw_doc_id", "raw_para_id", "text", "document_start_cp", "document_end_cp"):
            if type(row.get(key)) is not type(original[key]) or row.get(key) != original[key]:
                raise ValueError("source_bundle_paragraph_manifest_mismatch")
        document = docs[original["document_id"]]
        if "content_sha256" in row and row["content_sha256"] != document["content_sha256"]:
            raise ValueError("source_bundle_paragraph_document_sha256_mismatch")
        if "text_sha256" in row and row["text_sha256"] != _sha(original["text"]):
            raise ValueError("source_bundle_paragraph_text_sha256_mismatch")
        supplied.add(pid)
    if supplied != set(expected):
        raise ValueError("source_bundle_complete_paragraph_rows_required")
    return docs, expected


def _assemble(docs, paragraphs, start_paragraph_id, end_paragraph_id, max_paragraphs):
    _identity(start_paragraph_id)
    _identity(end_paragraph_id)
    if start_paragraph_id not in paragraphs or end_paragraph_id not in paragraphs:
        raise ValueError("source_bundle_unknown_endpoint")
    first, last = paragraphs[start_paragraph_id], paragraphs[end_paragraph_id]
    if first["document_id"] != last["document_id"]:
        raise ValueError("source_bundle_cross_document_range")
    begin, finish = first["position"], last["position"]
    if begin > finish:
        raise ValueError("source_bundle_reverse_range")
    if finish - begin + 1 > max_paragraphs:
        raise ValueError("source_bundle_paragraph_cap_exceeded")
    document = docs[first["document_id"]]
    parents = document["paragraphs"][begin:finish + 1]
    # Original segmentation can contain overlapping/nested/identical intervals.
    # The citation covers every selected original parent in full. The last
    # manifest parent is the scope endpoint, not necessarily the furthest end.
    start = first["document_start_cp"]
    end = max(p["document_end_cp"] for p in parents)
    canonical = {"span_scope": "document", "paragraph_id": first["id"], "raw_doc_id": first["raw_doc_id"],
        "raw_para_id": first["raw_para_id"], "start_cp": start, "end_cp": end,
        "quote": document["content"][start:end], "source_paragraph_ids": [p["id"] for p in parents],
        "document_content_sha256": document["content_sha256"], "paragraph_manifest_sha256": document["paragraph_manifest_sha256"],
        "document_id": document["id"]}
    canonical["bundle_id"] = str(uuid5(NAMESPACE_URL, "trust-judge:source-bundle:v1:" +
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)))
    return canonical


def assemble_source_bundle(documents, paragraphs, start_paragraph_id, end_paragraph_id, *, max_paragraphs=5):
    """Return a complete contiguous 1–5 original-paragraph source citation.

    The selected scope is contiguous in the trusted original manifest. All
    intervening IDs are mandatory. Original overlaps/containment are legal;
    document start/end form the full selected paragraphs' CP envelope, so end
    can be beyond the last selected parent's end. Scope endpoints are the first
    and last source_paragraph_ids. The quote is a real document slice retaining
    every original separator/gap; no join or text search is used. Coordinates
    are document-relative codepoints, even for one parent.
    Empty original parents are preserved. This supplies source identity only,
    without turning an empty quote, a statement, or an annotation into a fact.
    """
    _cap(max_paragraphs)
    docs, by_id = _prepare(documents, paragraphs)
    return _assemble(docs, by_id, start_paragraph_id, end_paragraph_id, max_paragraphs)


def build_source_bundles(documents, paragraphs, anchors, *, before=1, after=3, max_paragraphs=5):
    """Return neighbor read contexts, distinct from final citation selection.

    ``anchors`` contains internal original paragraph IDs. Contexts use at most
    ``before``/``after`` real neighbors; document edges shorten the range.
    Oversized requested windows are rejected, never silently truncated. Equal
    windows are deduplicated while preserving their anchor IDs and order.
    Later final citations can use ``assemble_source_bundle`` for any complete
    contiguous subset; the caller must independently enforce actual read IDs.
    """
    _cap(max_paragraphs)
    if type(before) is not int or type(after) is not int or before < 0 or after < 0:
        raise ValueError("source_bundle_invalid_neighbor_window")
    if before + 1 + after > max_paragraphs:
        raise ValueError("source_bundle_neighbor_window_exceeds_cap")
    if not isinstance(anchors, (list, tuple)):
        raise ValueError("source_bundle_anchor_ids_required")
    docs, by_id = _prepare(documents, paragraphs)
    result, windows = [], {}
    for pid in anchors:
        _identity(pid)
        if pid not in by_id:
            raise ValueError("source_bundle_unknown_anchor")
        anchor = by_id[pid]
        ordered = docs[anchor["document_id"]]["paragraphs"]
        first, last = max(0, anchor["position"] - before), min(len(ordered) - 1, anchor["position"] + after)
        bundle = _assemble(docs, by_id, ordered[first]["id"], ordered[last]["id"], max_paragraphs)
        if bundle["bundle_id"] not in windows:
            bundle.update(bundle_purpose="read_context", anchor_paragraph_ids=[])
            windows[bundle["bundle_id"]] = bundle
            result.append(bundle)
        target = windows[bundle["bundle_id"]]
        if pid not in target["anchor_paragraph_ids"]:
            target["anchor_paragraph_ids"].append(pid)
    return result


def validate_source_bundle(evidence, documents, paragraphs, *, max_paragraphs=5):
    """Validate exact core source fields; retain detached extra annotations.

    Annotation fields are outside this source-only certificate. No coordinates
    are repaired, no text search resolves repeated quotes, and a cited subset
    must name its real first paragraph rather than a wider read-context anchor.
    """
    _cap(max_paragraphs)
    if not isinstance(evidence, Mapping):
        raise ValueError("source_bundle_evidence_required")
    ids = evidence.get("source_paragraph_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(pid, str) or not pid.strip() for pid in ids):
        raise ValueError("source_bundle_original_paragraph_ids_required")
    docs, by_id = _prepare(documents, paragraphs)
    canonical = _assemble(docs, by_id, ids[0], ids[-1], max_paragraphs)
    for field in SOURCE_BUNDLE_CORE_FIELDS:
        if field not in evidence or type(evidence[field]) is not type(canonical[field]) or evidence[field] != canonical[field]:
            raise ValueError("source_bundle_core_identity_mismatch")
    return deepcopy(dict(evidence))
