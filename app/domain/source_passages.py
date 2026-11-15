"""Pure, source-bound complete sentence/line-group geometry within one parent.

This module never selects a fact, shortens a long sentence, reads a model
response or attests that a model received source text. All original sources
are required and validated. A passage is an exact document slice with the
original paragraph identity and explicit paragraph-relative coordinates.

Sentence indexes are zero-based and half-open. Conservative abbreviation
handling may keep several sentences in one group. Newlines are original line
boundaries, including blank lines, not inferred semantic boundaries. Geometry
does not prove that selected context contains every relevant qualification.
Existing whole-paragraph bundles are a different protocol; callers must
explicitly opt in to this geometry in their own frozen run/proof contracts.
"""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
import re
from uuid import NAMESPACE_URL, uuid5

from app.domain.source_bundles import SOURCE_BUNDLE_CORE_FIELDS, _prepare


SOURCE_GEOMETRY = "complete-sentence-context-v1"
SENTENCE_MANIFEST_FORMAT = "source-sentence-manifest-v1"
SOURCE_PASSAGE_CORE_FIELDS = SOURCE_BUNDLE_CORE_FIELDS + (
    "source_geometry", "passage_id", "parent_start_cp", "parent_end_cp",
    "local_start_cp", "local_end_cp", "parent_text_sha256",
    "start_sentence_index", "end_sentence_index", "sentence_manifest_sha256",
)
_CLOSERS = frozenset('"\'”’」』】》〉）)]}')
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon",
    "vs", "etc", "no", "fig", "eq", "inc", "ltd", "co", "corp", "dept",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "a.m", "p.m", "e.g", "i.e", "u.s", "u.k",
})
_DOTTED_INITIALS = re.compile(r"(?:[A-Za-z]\.){2,}\Z")
_TOKEN_BEFORE_PERIOD = re.compile(r"[A-Za-z.]+\Z")


def _require(condition, reason):
    if not condition:
        raise ValueError("source_passage_" + reason)


def _sha(text):
    return sha256(text.encode("utf-8", errors="strict")).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _period_boundary(text, index):
    if index + 1 < len(text) and (text[index + 1].isdigit() or text[index + 1] == "."):
        return False
    match = _TOKEN_BEFORE_PERIOD.search(text[:index + 1])
    token = match.group() if match else ""
    stem = token[:-1].casefold()
    if stem in _ABBREVIATIONS or _DOTTED_INITIALS.fullmatch(token):
        return False
    if len(stem) == 1 and stem.isalpha():
        return False
    after = index + 1
    while after < len(text) and text[after] in _CLOSERS:
        after += 1
    return after == len(text) or text[after].isspace() or text[after] == "\x00"


def _sentence_slices(text):
    """Return exhaustive ordered CP slices; never hard-cut at a size limit."""
    start, index, size = 0, 0, len(text)
    result = []
    while index < size:
        char = text[index]
        if char in "\r\n":
            end = index + (2 if text[index:index + 2] == "\r\n" else 1)
        elif char in "。！？!?" or (char == "." and _period_boundary(text, index)):
            end = index + 1
            while end < size and text[end] in "。！？!?":
                end += 1
            while end < size and text[end] in _CLOSERS:
                end += 1
            # Exact trailing separators remain part of the previous group.
            # NUL is source text too, never stripped or treated as an EOF.
            while end < size and (text[end].isspace() or text[end] == "\x00") and text[end] not in "\r\n":
                end += 1
            if end < size and text[end] in "\r\n":
                end += 2 if text[end:end + 2] == "\r\n" else 1
        else:
            index += 1
            continue
        result.append((start, end))
        start = index = end
    if start < size:
        result.append((start, size))
    return result


def _parent_manifest(document, parent):
    text, absolute = parent["text"], parent["document_start_cp"]
    sentences = []
    for index, (start, end) in enumerate(_sentence_slices(text)):
        literal = text[start:end]
        sentences.append({"index": index, "local_start_cp": start, "local_end_cp": end,
                          "start_cp": absolute + start, "end_cp": absolute + end,
                          "text": literal, "text_sha256": _sha(literal)})
    manifest = {"source_geometry": SOURCE_GEOMETRY, "paragraph_id": parent["id"],
        "document_id": parent["document_id"], "raw_doc_id": parent["raw_doc_id"],
        "raw_para_id": parent["raw_para_id"], "parent_start_cp": absolute,
        "parent_end_cp": parent["document_end_cp"], "parent_text_sha256": _sha(text),
        "document_content_sha256": document["content_sha256"],
        "paragraph_manifest_sha256": document["paragraph_manifest_sha256"], "sentences": sentences}
    manifest["sentence_manifest_sha256"] = _sha(_canonical(manifest))
    return manifest


def build_sentence_manifest(documents, paragraphs):
    """Bind complete source parents to full-coverage sentence/line manifests.

    Every sentence row is a literal original slice. Joining the ordered rows
    exactly reconstructs its full parent, including all newlines and controls.
    Empty parents have zero sentence rows, with their original empty identity
    retained. Document/manifest order is never inferred from opaque raw IDs.
    """
    docs, _ = _prepare(documents, paragraphs)
    bindings, parents = [], []
    for document in docs.values():
        bindings.append({"document_id": document["id"], "raw_doc_id": document["raw_doc_id"],
            "document_content_sha256": document["content_sha256"],
            "paragraph_manifest_sha256": document["paragraph_manifest_sha256"],
            "source_paragraph_ids": [p["id"] for p in document["paragraphs"]]})
        parents.extend(_parent_manifest(document, parent) for parent in document["paragraphs"])
    manifest = {"format": SENTENCE_MANIFEST_FORMAT, "source_geometry": SOURCE_GEOMETRY,
                "documents": bindings, "paragraphs": parents}
    manifest["manifest_sha256"] = _sha(_canonical(manifest))
    return manifest


def _assemble(document, parent, start_sentence_index, end_sentence_index):
    manifest = _parent_manifest(document, parent)
    sentences = manifest["sentences"]
    if start_sentence_index is None and end_sentence_index is None:
        start_sentence_index, end_sentence_index = 0, len(sentences)
    _require(type(start_sentence_index) is int and type(end_sentence_index) is int,
             "sentence_indexes_required_together")
    if not sentences:
        _require(start_sentence_index == end_sentence_index == 0, "empty_parent_sentence_range")
        local_start = local_end = 0
    else:
        _require(0 <= start_sentence_index < end_sentence_index <= len(sentences), "sentence_range")
        local_start = sentences[start_sentence_index]["local_start_cp"]
        local_end = sentences[end_sentence_index - 1]["local_end_cp"]
    start, end = parent["document_start_cp"] + local_start, parent["document_start_cp"] + local_end
    canonical = {"span_scope": "document", "source_geometry": SOURCE_GEOMETRY,
        "paragraph_id": parent["id"], "raw_doc_id": parent["raw_doc_id"], "raw_para_id": parent["raw_para_id"],
        "start_cp": start, "end_cp": end, "quote": document["content"][start:end],
        "source_paragraph_ids": [parent["id"]], "document_id": document["id"],
        "document_content_sha256": document["content_sha256"],
        "paragraph_manifest_sha256": document["paragraph_manifest_sha256"],
        "parent_start_cp": parent["document_start_cp"], "parent_end_cp": parent["document_end_cp"],
        "local_start_cp": local_start, "local_end_cp": local_end, "parent_text_sha256": _sha(parent["text"]),
        "start_sentence_index": start_sentence_index, "end_sentence_index": end_sentence_index,
        "sentence_manifest_sha256": manifest["sentence_manifest_sha256"]}
    canonical["passage_id"] = str(uuid5(NAMESPACE_URL, "trust-judge:source-passage:" + SOURCE_GEOMETRY + ":" +
                                       _canonical(canonical)))
    canonical["bundle_id"] = canonical["passage_id"]
    return canonical


def assemble_source_passage(documents, paragraphs, paragraph_id,
                            start_sentence_index=None, end_sentence_index=None):
    """Return one exact contiguous group inside one original parent.

    Both omitted indexes mean the whole original parent; both supplied indexes
    must describe a legal zero-based half-open sentence range. There is no
    maximum sentence/character count and no implicit truncation, repair or
    expansion into neighboring paragraphs. An empty parent has the valid
    source-only range [0, 0); it is not thereby useful evidence.
    """
    _require(isinstance(paragraph_id, str) and bool(paragraph_id.strip()), "invalid_parent_identity")
    docs, by_id = _prepare(documents, paragraphs)
    _require(paragraph_id in by_id, "unknown_parent")
    parent = by_id[paragraph_id]
    return _assemble(docs[parent["document_id"]], parent, start_sentence_index, end_sentence_index)


def validate_source_passage(evidence, documents, paragraphs):
    """Rebuild all geometry from immutable source; retain detached annotations.

    Annotations remain unverified. In particular a ``read`` marker, role,
    summary or confidence added by the caller is not certified here. No
    substring search, model coordinates, semantic truth or actual delivery is
    used to repair or accept a citation. Whole-bundle protocol rows cannot
    silently become this explicitly versioned geometry.
    """
    _require(isinstance(evidence, Mapping), "evidence_required")
    _require(all(key in evidence for key in SOURCE_PASSAGE_CORE_FIELDS), "geometry_core_required")
    canonical = assemble_source_passage(documents, paragraphs, evidence["paragraph_id"],
        evidence["start_sentence_index"], evidence["end_sentence_index"])
    for key in SOURCE_PASSAGE_CORE_FIELDS:
        _require(type(evidence[key]) is type(canonical[key]) and evidence[key] == canonical[key],
                 "core_identity_mismatch")
    return deepcopy(dict(evidence))
