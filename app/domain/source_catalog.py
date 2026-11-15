"""Immutable source slices selected by ID instead of model-generated quotations.

The catalog covers every codepoint of the supplied retrieved paragraphs. It does
not retrieve, normalize, fuzzy-match, pick an occurrence, or change human spans.
"""

from copy import deepcopy
from hashlib import sha256
import json
import re


MAX_CATALOG_SPANS = 512
MAX_SOURCE_CODEPOINTS = 64_000
MAX_SOURCE_SPAN_CODEPOINTS = 384
MAX_SELECTED_SOURCE_SPANS = MAX_CATALOG_SPANS
MAX_SOURCE_WINDOW_UNITS = 3
MAX_SOURCE_WINDOW_CODEPOINTS = MAX_SOURCE_WINDOW_UNITS * MAX_SOURCE_SPAN_CODEPOINTS

# Treat CRLF as a single newline boundary while preserving both characters.
# Standalone CR is retained as ordinary source text, never stripped or rewritten.
_BOUNDARIES = re.compile(r"(?:[。！？!?；;]|\r?\n)+")
_CATALOG_FIELDS = {"id", "paragraph_id", "start_cp", "end_cp", "quote", "paragraph_sha256"}
_SELECTION_FIELDS = {"source_span_ids", "fact_des", "argument", "entities"}
_INTERVAL_FIELDS = {"source_start_id", "source_end_id", "fact_des", "argument", "entities"}
_WINDOW_SELECTION_FIELDS = {"source_window_id", "fact_des", "argument", "entities"}


class SourceCatalogLimitExceeded(ValueError):
    """The complete source catalog cannot fit the explicit runtime budget."""


def _paragraphs(paragraphs: list[dict]) -> list[tuple[str, str, str]]:
    if not isinstance(paragraphs, list):
        raise ValueError("invalid_source_paragraphs")
    seen, rows, total = set(), [], 0
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            raise ValueError("invalid_source_paragraph")
        identity, text = paragraph.get("id"), paragraph.get("text")
        if not isinstance(identity, str) or not identity.strip() or not isinstance(text, str) or not text:
            raise ValueError("invalid_source_paragraph")
        if identity in seen:
            raise ValueError("duplicate_source_paragraph")
        seen.add(identity)
        total += len(text)
        if total > MAX_SOURCE_CODEPOINTS:
            raise SourceCatalogLimitExceeded("source_codepoint_limit_exceeded")
        try:
            digest = sha256(text.encode("utf-8")).hexdigest()
            identity.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("invalid_source_unicode") from None
        rows.append((identity, text, digest))
    return rows


def _span_id(paragraph_id: str, digest: str, start: int, end: int) -> str:
    material = json.dumps([paragraph_id, digest, start, end], ensure_ascii=False, separators=(",", ":"))
    return "S" + sha256(material.encode("utf-8")).hexdigest()[:20]


def build_source_catalog(paragraphs: list[dict]) -> list[dict]:
    """Return complete continuous sentence units from the given source only.

    Boundary punctuation and consecutive delimiters remain attached to the
    preceding unit. A long unit is hard-sliced into at most 384 codepoints.
    Resource limits fail explicitly; no paragraph or suffix is silently omitted.
    """
    result, identities = [], set()
    for paragraph_id, text, digest in _paragraphs(paragraphs):
        start = 0
        ends = [match.end() for match in _BOUNDARIES.finditer(text)]
        if not ends or ends[-1] != len(text):
            ends.append(len(text))
        for unit_end in ends:
            while start < unit_end:
                end = min(unit_end, start + MAX_SOURCE_SPAN_CODEPOINTS)
                if len(result) >= MAX_CATALOG_SPANS:
                    raise SourceCatalogLimitExceeded("source_catalog_span_limit_exceeded")
                identity = _span_id(paragraph_id, digest, start, end)
                if identity in identities:
                    raise ValueError("source_catalog_identity_collision")
                quote = text[start:end]
                if not quote or not 0 <= start < end <= len(text):
                    raise ValueError("invalid_source_catalog_span")
                result.append({"id": identity, "paragraph_id": paragraph_id,
                               "start_cp": start, "end_cp": end, "quote": quote,
                               "paragraph_sha256": digest})
                identities.add(identity)
                start = end
        if start != len(text):
            raise ValueError("incomplete_source_catalog")
    return result


def resolve_source_selection(selection: dict, catalog: list[dict], paragraphs: list[dict]) -> dict:
    """Resolve an explicit ordered sequence of adjacent units into one quote.

    Rebuild the complete catalog from current source text before lookup. This
    rejects stale IDs and any caller mutation of text, coordinates, IDs, hashes,
    catalog ordering or coverage. Model-selected IDs never authorize new text.
    """
    if not isinstance(selection, dict) or set(selection) != _SELECTION_FIELDS:
        raise ValueError("invalid_source_selection")
    selected = selection["source_span_ids"]
    if not isinstance(selected, list) or not 1 <= len(selected) <= MAX_SELECTED_SOURCE_SPANS:
        raise ValueError("invalid_source_selection_count")
    if any(not isinstance(identity, str) or not identity for identity in selected):
        raise ValueError("invalid_source_selection_id")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate_source_selection")
    if not isinstance(selection["fact_des"], str) or not selection["fact_des"].strip() or not isinstance(selection["argument"], dict) or not isinstance(selection["entities"], list):
        raise ValueError("invalid_source_selection_annotation")
    if not isinstance(catalog, list):
        raise ValueError("invalid_source_catalog")
    expected = build_source_catalog(paragraphs)
    for entry in catalog:
        if not isinstance(entry, dict) or set(entry) != _CATALOG_FIELDS:
            raise ValueError("invalid_source_catalog")
        if type(entry["start_cp"]) is not int or type(entry["end_cp"]) is not int:
            raise ValueError("invalid_source_catalog")
        if any(not isinstance(entry[key], str) for key in ("id", "paragraph_id", "quote", "paragraph_sha256")):
            raise ValueError("invalid_source_catalog")
    if catalog != expected:
        raise ValueError("source_catalog_mismatch")
    available = {item["id"]: item for item in expected}
    if any(identity not in available for identity in selected):
        raise ValueError("unknown_source_selection")
    units = [available[identity] for identity in selected]
    paragraph_id = units[0]["paragraph_id"]
    if any(unit["paragraph_id"] != paragraph_id for unit in units):
        raise ValueError("cross_paragraph_source_selection")
    if any(left["end_cp"] != right["start_cp"] for left, right in zip(units, units[1:])):
        raise ValueError("noncontiguous_source_selection")
    text = next(row["text"] for row in paragraphs if row["id"] == paragraph_id)
    start, end = units[0]["start_cp"], units[-1]["end_cp"]
    quote = text[start:end]
    if not quote or quote != "".join(unit["quote"] for unit in units):
        raise ValueError("invalid_source_selection_span")
    return {"paragraph_id": paragraph_id, "quote": quote, "start_cp": start, "end_cp": end,
            "fact_des": deepcopy(selection["fact_des"]),
            "argument": deepcopy(selection["argument"]), "entities": deepcopy(selection["entities"])}


def resolve_source_interval(selection: dict, catalog: list[dict], paragraphs: list[dict]) -> dict:
    """Expand only an explicit inclusive, same-paragraph source interval.

    S1..Sn are local aliases for the supplied immutable catalog in its current
    order. This is a distinct wire contract: old sparse ID lists are rejected,
    never completed by guessing omitted units. Rebuilding and validating the
    catalog prevents mutated source text/geometry from changing what an alias
    means. The caller additionally binds this catalog to its request identity.
    """
    if not isinstance(selection, dict) or set(selection) != _INTERVAL_FIELDS:
        raise ValueError("invalid_source_interval")
    if not isinstance(catalog, list):
        raise ValueError("source_catalog_mismatch")
    expected = build_source_catalog(paragraphs)
    for row in catalog:
        if not isinstance(row, dict) or set(row) != _CATALOG_FIELDS:
            raise ValueError("source_catalog_mismatch")
        if type(row["start_cp"]) is not int or type(row["end_cp"]) is not int:
            raise ValueError("source_catalog_mismatch")
        if any(not isinstance(row[key], str) for key in ("id", "paragraph_id", "quote", "paragraph_sha256")):
            raise ValueError("source_catalog_mismatch")
    if catalog != expected:
        raise ValueError("source_catalog_mismatch")
    aliases = {f"S{index + 1}": index for index in range(len(expected))}
    first, last = selection["source_start_id"], selection["source_end_id"]
    if not isinstance(first, str) or first not in aliases:
        raise ValueError("unknown_source_start")
    if not isinstance(last, str) or last not in aliases:
        raise ValueError("unknown_source_end")
    start_index, end_index = aliases[first], aliases[last]
    if start_index > end_index:
        raise ValueError("reversed_source_interval")
    units = expected[start_index:end_index + 1]
    if any(row["paragraph_id"] != units[0]["paragraph_id"] for row in units):
        raise ValueError("cross_paragraph_interval")
    # Resolve through the original strict primitive. Every intervening unit is
    # part of this explicitly selected interval, not a repaired discontinuity.
    return resolve_source_selection({
        "source_span_ids": [row["id"] for row in units],
        "fact_des": selection["fact_des"], "argument": deepcopy(selection["argument"]),
        "entities": deepcopy(selection["entities"]),
    }, catalog, paragraphs)


def build_source_windows(paragraphs: list[dict], *, include_paragraphs: bool = False) -> list[dict]:
    """Predefine a same-paragraph window at every original unit boundary.

    Each window contains its starting unit plus up to two following units, with
    no model-controlled geometry. Every original unit starts a window, so the
    union covers every source codepoint, including paragraph/document tails.
    Raw source limits are measured on the base catalog, not duplicated windows.
    ``include_paragraphs`` appends a whole-paragraph choice when no compact
    window already has that geometry. Whole choices may exceed the compact
    1152-codepoint bound but remain within the validated base source limits.
    """
    units = build_source_catalog(paragraphs)
    texts = {row["id"]: row["text"] for row in paragraphs}
    windows, identities = [], set()
    for index, first in enumerate(units):
        following = [first]
        for unit in units[index + 1:index + MAX_SOURCE_WINDOW_UNITS]:
            if unit["paragraph_id"] != first["paragraph_id"]:
                break
            following.append(unit)
        start, end = first["start_cp"], following[-1]["end_cp"]
        quote = texts[first["paragraph_id"]][start:end]
        if (not quote or len(quote) > MAX_SOURCE_WINDOW_CODEPOINTS
                or quote != "".join(row["quote"] for row in following)):
            raise ValueError("invalid_source_window_catalog")
        material = json.dumps([first["paragraph_id"], first["paragraph_sha256"], start, end],
                              ensure_ascii=False, separators=(",", ":"))
        identity = "W" + sha256(material.encode("utf-8")).hexdigest()[:20]
        if identity in identities:
            raise ValueError("invalid_source_window_catalog")
        identities.add(identity)
        windows.append({"id": identity, "paragraph_id": first["paragraph_id"], "start_cp": start,
                        "end_cp": end, "quote": quote, "paragraph_sha256": first["paragraph_sha256"]})
    # One window per base unit is an exact source-coverage invariant. A window
    # begins with that unit and may extend only into its existing neighbors.
    if len(windows) != len(units) or any(
        window["paragraph_id"] != unit["paragraph_id"] or window["start_cp"] != unit["start_cp"]
        or window["end_cp"] < unit["end_cp"] for window, unit in zip(windows, units)
    ):
        raise ValueError("invalid_source_window_catalog")
    if include_paragraphs:
        # Additional fixed choices retain a statement's qualifications even
        # when they extend past three units. Existing compact aliases keep
        # their positions; no text is copied or assembled by the model.
        geometry = {(row["paragraph_id"], row["start_cp"], row["end_cp"]) for row in windows}
        for paragraph_id, text, paragraph_digest in _paragraphs(paragraphs):
            if (paragraph_id, 0, len(text)) in geometry:
                continue
            material = json.dumps([paragraph_id, paragraph_digest, 0, len(text)],
                                  ensure_ascii=False, separators=(",", ":"))
            identity = "W" + sha256(material.encode("utf-8")).hexdigest()[:20]
            if identity in identities:
                raise ValueError("invalid_source_window_catalog")
            identities.add(identity)
            windows.append({"id": identity,
                "paragraph_id": paragraph_id, "start_cp": 0, "end_cp": len(text),
                "quote": text, "paragraph_sha256": paragraph_digest})
    return windows


def resolve_source_window(selection: dict, windows: list[dict], paragraphs: list[dict], *, include_paragraphs: bool = False) -> dict:
    """Resolve one predefined window; reject all old model geometry fields.

    Alias W1..Wn comes only from the validated catalog's program-owned order.
    External alias maps, lists and interval endpoints are not accepted. The
    provider separately binds this complete catalog and question per request.
    """
    if not isinstance(selection, dict) or set(selection) != _WINDOW_SELECTION_FIELDS:
        raise ValueError("invalid_source_window")
    if not isinstance(windows, list):
        raise ValueError("invalid_source_window_catalog")
    expected = build_source_windows(paragraphs, include_paragraphs=include_paragraphs)
    for row in windows:
        if not isinstance(row, dict) or set(row) != _CATALOG_FIELDS:
            raise ValueError("invalid_source_window_catalog")
        if type(row["start_cp"]) is not int or type(row["end_cp"]) is not int:
            raise ValueError("invalid_source_window_catalog")
        if any(not isinstance(row[key], str) for key in ("id", "paragraph_id", "quote", "paragraph_sha256")):
            raise ValueError("invalid_source_window_catalog")
    if windows != expected:
        raise ValueError("invalid_source_window_catalog")
    aliases = {f"W{index + 1}": row for index, row in enumerate(expected)}
    chosen = selection["source_window_id"]
    if not isinstance(chosen, str) or chosen not in aliases:
        raise ValueError("unknown_source_window")
    window = aliases[chosen]
    units = build_source_catalog(paragraphs)
    selected_units = [row["id"] for row in units if row["paragraph_id"] == window["paragraph_id"]
                      and window["start_cp"] <= row["start_cp"] and row["end_cp"] <= window["end_cp"]]
    resolved = resolve_source_selection({
        "source_span_ids": selected_units, "fact_des": selection["fact_des"],
        "argument": deepcopy(selection["argument"]), "entities": deepcopy(selection["entities"]),
    }, units, paragraphs)
    if (resolved["quote"], resolved["start_cp"], resolved["end_cp"]) != (
        window["quote"], window["start_cp"], window["end_cp"]
    ):
        raise ValueError("invalid_source_window")
    return resolved
