"""Experimental, literal query-conditioned views of immutable parent paragraphs.

One fixed 80-codepoint window per parent, not a retrieval/role/evidence filter.
Only original question text and explicitly supplied search terms affect geometry.
No provider/config/model import, concatenated quotes or source normalization.
"""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
import re

from app.domain.question_recall import _features, question_query_plan
from app.domain.semantic_recall import source_manifest
from app.domain.source_catalog import _BOUNDARIES

VERSION = "literal-question-context80-v1"
WINDOW_CHARS = 80
POLICY = {
    "window_codepoints": 80,
    "original_word_weight": 8, "original_bigram_weight": 2,
    "provided_word_weight": 4, "provided_bigram_weight": 1,
    "feature_counting": "unique literal features, maximum weight across queries; no frequency bonus",
    "selection": "maximum fully-contained weighted feature coverage over all continuous 80cp windows",
    "tie_break": "earliest original sentence boundary among maximizers, otherwise earliest maximizer",
    "no_match": "unchanged original prefix; relevance remains unknown",
    "parent_identity": "whole original paragraph; no source/parent removed or reranked",
    "omitted_before_and_after": "unknown, never irrelevant, false, or absent",
}
_FORBIDDEN = {"gold", "reference", "reference_answer", "conclusion", "supporting_facts", "conflicting_facts", "fact_des"}


def _digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _query_weights(plan):
    result = {}
    for facet in plan["facets"]:
        supplied = facet["origin"] == "provided_query"
        for kind, weight in (("word", 4 if supplied else 8), ("bigram", 1 if supplied else 2)):
            for term in _features(facet["text"], kind):
                if len(term) >= 2 or term.isdigit():
                    result[term] = max(result.get(term, 0), weight)
    return tuple(sorted(result.items()))


@lru_cache(maxsize=4096)
def _literal(term):
    # Latin identifiers and numbers must not match inside a larger identifier.
    # Positions come from the ORIGINAL string; lower/casefold never moves spans.
    if re.fullmatch(r"[a-z0-9_.-]+", term, re.I):
        boundary = r"[a-z0-9_.]" if term[0].isdigit() else r"[a-z0-9_]"
        pattern = r"(?<!" + boundary + ")(" + re.escape(term) + r")(?!" + boundary + ")"
    else:
        pattern = "(" + re.escape(term) + ")"
    return re.compile("(?=" + pattern + ")", re.I)


@lru_cache(maxsize=2048)
def _best_window(text, features):
    width = min(WINDOW_CHARS, len(text))
    if len(text) <= WINDOW_CHARS:
        return 0, len(text), "short_parent_complete"
    last = len(text) - width
    events = defaultdict(int)
    events[0] = events[last + 1] = 0
    for term, weight in features:
        if len(term) > width:
            continue
        intervals = []
        for match in _literal(term).finditer(text):
            lo, hi = max(0, match.end(1) - width), min(match.start(1), last)
            if lo > hi:
                continue
            if intervals and lo <= intervals[-1][1] + 1:
                intervals[-1] = intervals[-1][0], max(hi, intervals[-1][1])
            else:
                intervals.append((lo, hi))
        # Union intervals per FEATURE before weighting: repetition gives no votes.
        for lo, hi in intervals:
            events[lo] += weight
            events[hi + 1] -= weight
    positions = sorted(events)
    running, best, maxima = 0, 0, []
    for index, lo in enumerate(positions[:-1]):
        running += events[lo]
        hi = min(last, positions[index + 1] - 1)
        if lo > hi:
            continue
        if running > best:
            best, maxima = running, [(lo, hi)]
        elif running == best:
            maxima.append((lo, hi))
    if not best:
        return 0, width, "no_literal_body_hit_prefix_fallback"
    boundaries = [0] + [match.end() for match in _BOUNDARIES.finditer(text) if match.end() <= last]
    start = next((p for p in boundaries if any(lo <= p <= hi for lo, hi in maxima)), maxima[0][0])
    return start, start + width, "query_conditioned_literal_window"


def verify_context(parent, context):
    """Reject stale geometry or changed text; no semantic verification is implied."""
    manifest = source_manifest([parent])[0]
    text = parent["text"]
    start, end = context.get("start_cp"), context.get("end_cp")
    if (type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text)
            or end - start != min(WINDOW_CHARS, len(text))
            or any(context.get(k) != manifest[k] for k in ("id", "raw_doc_id", "raw_para_id", "source_text_sha256"))
            or context.get("parent_length_cp") != len(text)
            or context.get("excerpt_text") != text[start:end]
            or context.get("excerpt_text_sha256") != sha256(text[start:end].encode()).hexdigest()
            or context.get("title") != parent.get("title")
            or context.get("omitted_before") is not (start > 0)
            or context.get("omitted_after") is not (end < len(text))
            or context.get("omission_means_unknown") is not True
            or context.get("field_or_support_conflict_verified") is not False):
        raise ValueError("query_context_original_span_binding_changed")
    if "document_start_cp" in parent:
        offset = parent["document_start_cp"]
        if (type(offset) is not int or offset < 0 or context.get("document_start_cp") != offset + start
                or context.get("document_end_cp") != offset + end):
            raise ValueError("query_context_original_document_offset_changed")
    elif "document_start_cp" in context or "document_end_cp" in context:
        raise ValueError("query_context_unprovided_document_offset")
    return True


def build_query_context(question, paragraphs, *, query_groups=None):
    """Return one source-addressed view per unchanged parent, in original order.

    Supplied query groups are unverified search hints, never assertions. A zero
    lexical score does not discard a parent or establish irrelevance. Model and
    official evidence output contracts are intentionally outside this module.
    """
    if isinstance(question, dict) and _FORBIDDEN & question.keys():
        raise ValueError("query_context_reference_fields_forbidden")
    manifest = source_manifest(paragraphs)
    if any(_FORBIDDEN & row.keys() for row in paragraphs):
        raise ValueError("query_context_reference_fields_forbidden")
    plan = question_query_plan(question, query_groups)
    features = _query_weights(plan)
    result = []
    for parent, identity in zip(paragraphs, manifest):
        if parent.get("title") is not None and not isinstance(parent["title"], str):
            raise ValueError("query_context_original_title_invalid")
        text = parent["text"]
        start, end, reason = _best_window(text, features)
        excerpt = text[start:end]
        matched = [(term, weight) for term, weight in features if any(
            start <= match.start(1) and match.end(1) <= end for match in _literal(term).finditer(text))]
        row = {**identity, "title": parent.get("title"), "parent_length_cp": len(text),
            "start_cp": start, "end_cp": end, "excerpt_text": excerpt,
            "excerpt_text_sha256": sha256(excerpt.encode()).hexdigest(),
            "omitted_before": start > 0, "omitted_after": end < len(text),
            "omission_means_unknown": True, "selection_reason": reason,
            "lexical_feature_score": sum(weight for _, weight in matched),
            "matched_query_features": [term for term, _ in matched],
            "field_or_support_conflict_verified": False}
        if "document_start_cp" in parent:
            offset = parent["document_start_cp"]
            if type(offset) is not int or offset < 0:
                raise ValueError("query_context_original_document_offset_invalid")
            row.update(document_start_cp=offset + start, document_end_cp=offset + end)
        verify_context(parent, row)
        result.append(row)
    return {"version": VERSION, "policy": dict(POLICY), "query_plan": plan,
        "query_plan_sha256": _digest(plan), "source_manifest_sha256": _digest(manifest),
        "parent_count": len(result), "contexts": result, "model_calls": 0,
        "provided_expansions_semantically_verified": False, "evidence_or_role_filter_applied": False,
        "parent_order_and_selection_unchanged": True, "reference_inputs": False}
