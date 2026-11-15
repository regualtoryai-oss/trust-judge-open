"""Complete, bounded source scans with retrieval used only as a priority.

The planner never selects evidence or discards a zero-match paragraph. Batches
retain the caller's original paragraph objects and continuous document order;
source catalogs are built only to measure the exact downstream resource cost.
"""

from hashlib import sha256
import json

from app.domain.retrieval import retrieve
from app.domain.source_catalog import (
    MAX_CATALOG_SPANS,
    MAX_SOURCE_CODEPOINTS,
    SourceCatalogLimitExceeded,
    build_source_catalog,
)


DEFAULT_BATCH_CODEPOINTS = 20_000
DEFAULT_BATCH_CATALOG_SPANS = 128


class CoverageLimitExceeded(SourceCatalogLimitExceeded):
    """A whole immutable paragraph cannot fit the catalog's hard limits."""


def _digest(value) -> str:
    try:
        material = json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
        return sha256(material.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, UnicodeEncodeError):
        raise ValueError("invalid_coverage_fingerprint_material") from None


def _identity(paragraph: dict) -> tuple[str, str]:
    identity, text = paragraph.get("id"), paragraph.get("text")
    if not isinstance(identity, str) or not identity.strip() or not isinstance(text, str) or not text:
        raise ValueError("invalid_coverage_paragraph")
    try:
        identity.encode("utf-8")
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("invalid_source_unicode") from None
    return identity, text


def _document_key(paragraph: dict) -> tuple[str, str]:
    # A paragraph without document metadata is its own source. Never infer a
    # common origin from equal text; sharing a batch does not merge provenance.
    for field in ("document_id", "raw_doc_id"):
        value = paragraph.get(field)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("invalid_coverage_document_identity")
            return field, value
    return "paragraph_id", paragraph["id"]


def plan_source_batches(
    question: str,
    paragraphs: list[dict],
    *,
    max_codepoints: int = DEFAULT_BATCH_CODEPOINTS,
    max_catalog_spans: int = DEFAULT_BATCH_CATALOG_SPANS,
) -> dict:
    """Plan every supplied paragraph once; BM25 changes batch order only.

    Source order within each document is the caller's input order. Consecutive
    whole paragraphs from one document are packed until either batch budget is
    reached. Small whole documents can share a batch; an oversized document is
    split only between paragraphs. Document chunks containing a BM25 match run
    first, followed by all remaining chunks in their original document order.
    No answer labels are consulted.

    Batch budgets are soft packing targets, bounded by ``source_catalog``'s
    hard limits. A paragraph exceeding a soft target gets a dedicated batch;
    it may use up to the catalog's hard limits. A paragraph exceeding either
    hard limit fails explicitly before any partial plan is returned; neither
    its source text nor a source suffix is silently cut away.

    ``paragraphs`` in each returned batch contains the original dict objects,
    with no catalog or duplicated source text added. Source fingerprints are
    independent of input ordering; plan fingerprints also bind question and
    planned batch order. Successful planning is coverage, not proof that a
    model has read or selected all relevant facts.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("coverage_question_must_be_nonempty")
    if not isinstance(paragraphs, list):
        raise ValueError("invalid_coverage_paragraphs")
    if type(max_codepoints) is not int or not 1 <= max_codepoints <= MAX_SOURCE_CODEPOINTS:
        raise ValueError("invalid_coverage_codepoint_budget")
    if type(max_catalog_spans) is not int or not 1 <= max_catalog_spans <= MAX_CATALOG_SPANS:
        raise ValueError("invalid_coverage_span_budget")

    document_groups: dict[tuple[str, str], list[dict]] = {}
    measurements, source_rows, seen = {}, [], set()
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            raise ValueError("invalid_coverage_paragraph")
        identity, text = _identity(paragraph)
        if identity in seen:
            raise ValueError("duplicate_coverage_paragraph")
        seen.add(identity)
        document_key = _document_key(paragraph)
        if len(text) > MAX_SOURCE_CODEPOINTS:
            raise CoverageLimitExceeded("coverage_paragraph_codepoint_limit_exceeded")
        try:
            spans = len(build_source_catalog([paragraph]))
        except SourceCatalogLimitExceeded:
            raise CoverageLimitExceeded("coverage_paragraph_catalog_limit_exceeded") from None
        measurements[identity] = (len(text), spans)
        source_rows.append({"id": identity, "document": list(document_key),
                            "raw_doc_id": paragraph.get("raw_doc_id"),
                            "raw_para_id": paragraph.get("raw_para_id"),
                            "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                            "codepoints": len(text), "catalog_spans": spans})
        document_groups.setdefault(document_key, []).append(paragraph)

    ranked = retrieve(question, paragraphs, limit=max(1, len(paragraphs)))
    priority = {paragraph["id"]: rank for rank, paragraph in enumerate(ranked)}
    source_fingerprint = _digest(sorted(source_rows, key=lambda row: row["id"]))
    batches = []

    def append_batch(rows: list[dict], codepoints: int, spans: int) -> None:
        if not rows:
            return
        paragraph_ids = [row["id"] for row in rows]
        identity_material = [source_by_id[identity] for identity in paragraph_ids]
        batches.append({"id": "B" + _digest(identity_material)[:24],
                        "paragraphs": rows, "paragraph_ids": paragraph_ids,
                        "codepoints": codepoints, "catalog_spans": spans,
                        "soft_budget_exceeded": codepoints > max_codepoints or spans > max_catalog_spans,
                        "bm25_match_count": sum(identity in priority for identity in paragraph_ids)})

    source_by_id = {row["id"]: row for row in source_rows}
    chunks = []
    for rows in document_groups.values():
        pending, codepoints, spans = [], 0, 0
        for paragraph in rows:
            paragraph_codepoints, paragraph_spans = measurements[paragraph["id"]]
            if pending and (codepoints + paragraph_codepoints > max_codepoints
                            or spans + paragraph_spans > max_catalog_spans):
                chunks.append((pending, codepoints, spans))
                pending, codepoints, spans = [], 0, 0
            pending.append(paragraph)
            codepoints += paragraph_codepoints
            spans += paragraph_spans
        if pending:
            chunks.append((pending, codepoints, spans))

    # Keep each bounded document chunk intact when sharing spare batch space.
    # This avoids one paid request per tiny document without scattering its
    # neighboring paragraphs according to independent paragraph scores.
    fallback = len(paragraphs)
    chunks.sort(key=lambda chunk: min((priority.get(row["id"], fallback)
                                      for row in chunk[0]), default=fallback))
    pending, codepoints, spans = [], 0, 0
    for rows, chunk_codepoints, chunk_spans in chunks:
        if pending and (codepoints + chunk_codepoints > max_codepoints
                        or spans + chunk_spans > max_catalog_spans):
            append_batch(pending, codepoints, spans)
            pending, codepoints, spans = [], 0, 0
        pending.extend(rows)
        codepoints += chunk_codepoints
        spans += chunk_spans
    append_batch(pending, codepoints, spans)
    planned = [identity for batch in batches for identity in batch["paragraph_ids"]]
    if len(planned) != len(seen) or set(planned) != seen:
        raise ValueError("incomplete_coverage_plan")
    plan_fingerprint = _digest({"question": question,
                                "source_fingerprint": source_fingerprint,
                                "batch_ids": [batch["id"] for batch in batches],
                                "max_codepoints": max_codepoints,
                                "max_catalog_spans": max_catalog_spans})
    return {"batches": batches, "coverage": {
        "complete": True,
        "paragraph_count": len(paragraphs),
        "planned_paragraph_count": len(planned),
        "batch_count": len(batches),
        "soft_budget_exceeded_batch_count": sum(batch["soft_budget_exceeded"] for batch in batches),
        "max_batch_codepoints": max((batch["codepoints"] for batch in batches), default=0),
        "max_batch_catalog_spans": max((batch["catalog_spans"] for batch in batches), default=0),
        "codepoint_count": sum(row["codepoints"] for row in source_rows),
        "catalog_span_count": sum(row["catalog_spans"] for row in source_rows),
        "bm25_matched_paragraph_count": len(ranked),
        "zero_match_paragraph_count": len(paragraphs) - len(ranked),
        "source_fingerprint": source_fingerprint,
        "plan_fingerprint": plan_fingerprint,
    }}


def merge_exact_candidates(candidate_batches: list[list[dict]]) -> list[dict]:
    """Remove only identical candidate records; retain facts and provenance.

    Literal text, paragraph identity, explicit occurrence and every annotation
    remain part of the identity. Equal text in another document, overlapping
    quotations, and different factual interpretations are all preserved. This
    helper deliberately performs no semantic merge or quote normalization.
    Returned records are the original objects in first-observed order.
    """
    if not isinstance(candidate_batches, list):
        raise ValueError("invalid_candidate_batches")
    result, seen = [], set()
    required = {"paragraph_id", "quote", "start_cp", "end_cp", "fact_des", "argument", "entities"}
    for batch in candidate_batches:
        if not isinstance(batch, list):
            raise ValueError("invalid_candidate_batch")
        for candidate in batch:
            if not isinstance(candidate, dict) or not required.issubset(candidate):
                raise ValueError("invalid_merge_candidate")
            identity = _digest(candidate)
            if identity not in seen:
                seen.add(identity)
                result.append(candidate)
    return result
