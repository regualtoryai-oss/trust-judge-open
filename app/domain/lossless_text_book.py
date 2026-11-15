"""An exact, reversible text dictionary for a future model-input projection.

Only the explicitly named text field is packed. All other metadata is copied
unchanged, and every record and every occurrence of text remains addressable.
This is not a source-selection, passage, summarization or publication codec.

The local audit is deliberately separate from ``native_text_book``. A caller
must validate against its trusted original records before sending the native
view; self-consistent hashes alone cannot authenticate an untrusted book.
Sentence/line boundaries are conservative navigation boundaries, not an NLP
claim. No character, including whitespace, is normalized or removed.
"""

from copy import deepcopy
from hashlib import sha256
import json
import math
import re


FORMAT = "lossless-text-book-v1"
AUDIT_FORMAT = "lossless-text-book-audit-v1"
SPLIT_POLICY = "conservative-sentence-or-line-exact-characters-v1"
_VIEW_KEYS = frozenset({"format", "text_key", "id_key", "records", "text_blocks"})
_BOOK_KEYS = _VIEW_KEYS | {"audit"}
_CLOSERS = frozenset('"\'”’」』】》〉）)]}')
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "rev", "hon",
    "vs", "etc", "no", "fig", "eq", "inc", "ltd", "co", "corp", "dept",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "a.m", "p.m", "e.g", "i.e", "u.s", "u.k",
})
_DOTTED_INITIALS = re.compile(r"(?:[A-Za-z]\.){2,}\Z")
_TOKEN_BEFORE_PERIOD = re.compile(r"[A-Za-z.]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, reason):
    if not condition:
        raise ValueError("lossless_text_book_" + reason)


def _utf8(text):
    try:
        return text.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError("lossless_text_book_invalid_unicode") from exc


def _json_value(value):
    # JSON's encoder silently converts integer object keys and accepts tuples.
    # Refuse both rather than mutate metadata during a supposedly lossless pass.
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        _require(math.isfinite(value), "nonfinite_number")
        return
    if type(value) is str:
        _utf8(value)
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "metadata_object_keys")
        for key, item in value.items():
            _utf8(key)
            _json_value(item)
        return
    raise ValueError("lossless_text_book_non_json_metadata")


def _canonical(value):
    _json_value(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fingerprint(value):
    return sha256(_canonical(value)).hexdigest()


def _keys(text_key, id_key):
    _require(type(text_key) is str and bool(text_key)
             and type(id_key) is str and bool(id_key) and text_key != id_key,
             "field_names")
    _utf8(text_key)
    _utf8(id_key)


def _records(records, text_key, id_key, *, packed=False):
    _keys(text_key, id_key)
    _require(type(records) is list, "record_list")
    _json_value(records)
    identifiers = []
    for row in records:
        _require(type(row) is dict and text_key in row and id_key in row,
                 "record_fields")
        identifier = row[id_key]
        _require(type(identifier) is str and bool(identifier), "record_identity")
        identifiers.append(identifier)
        if packed:
            descriptor = row[text_key]
            _require(type(descriptor) is dict and set(descriptor) == {"text_ref"}
                     and type(descriptor["text_ref"]) is list
                     and all(type(ref) is str for ref in descriptor["text_ref"]),
                     "text_reference_shape")
        else:
            _require(type(row[text_key]) is str, "text_field")
    _require(len(identifiers) == len(set(identifiers)), "duplicate_record_identity")
    return identifiers


def _period_boundary(text, index):
    # Decimals, dotted initials, titles and common abbreviations conservatively
    # stay with their following sentence. This never changes source bytes.
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
    return after == len(text) or text[after].isspace()


def _sentence_blocks(text):
    """Yield nonempty, exact slices; no maximum length or hidden truncation."""
    start, index, size = 0, 0, len(text)
    while index < size:
        char = text[index]
        if char in "\r\n":
            end = index + (2 if char == "\r" and text[index:index + 2] == "\r\n" else 1)
        elif char in "。！？!?" or (char == "." and _period_boundary(text, index)):
            end = index + 1
            while end < size and text[end] in "。！？!?":
                end += 1
            while end < size and text[end] in _CLOSERS:
                end += 1
            # Trailing source spaces belong to this block, including at most
            # one complete newline. CRLF is never separated into two blocks.
            while end < size and text[end].isspace() and text[end] not in "\r\n":
                end += 1
            if end < size and text[end] in "\r\n":
                end += 2 if text[end:end + 2] == "\r\n" else 1
        else:
            index += 1
            continue
        yield text[start:end]
        start = index = end
    if start < size:
        yield text[start:]


def pack_texts(records, *, text_key="quote", id_key="id"):
    """Pack exact repeated sentence/line blocks, retaining every input row.

    The original text digest hashes strict UTF-8 bytes. The records digest
    hashes canonical JSON, not an original file's serialization or byte order.
    A text with no safe boundary is one complete block, however long it is.
    """
    order = _records(records, text_key, id_key)
    packed, blocks, block_audit, record_audit, by_text = [], [], [], [], {}
    for original in records:
        text = original[text_key]
        row, refs, spans = deepcopy(original), [], []
        start_cp = start_utf8 = 0
        for literal in _sentence_blocks(text):
            identifier = by_text.get(literal)
            if identifier is None:
                identifier = "B" + str(len(blocks) + 1)
                by_text[literal] = identifier
                encoded = _utf8(literal)
                blocks.append({"id": identifier, "text": literal})
                block_audit.append({"id": identifier, "sha256": sha256(encoded).hexdigest(),
                                    "codepoints": len(literal), "utf8_bytes": len(encoded)})
            end_cp, end_utf8 = start_cp + len(literal), start_utf8 + len(_utf8(literal))
            refs.append(identifier)
            spans.append({"block_id": identifier, "start_cp": start_cp, "end_cp": end_cp,
                          "start_utf8": start_utf8, "end_utf8": end_utf8})
            start_cp, start_utf8 = end_cp, end_utf8
        _require(start_cp == len(text) and start_utf8 == len(_utf8(text)), "internal_coverage")
        row[text_key] = {"text_ref": refs}
        packed.append(row)
        record_audit.append({"id": original[id_key], "sha256": sha256(_utf8(text)).hexdigest(),
                             "codepoints": len(text), "utf8_bytes": len(_utf8(text)), "refs": spans})
    view = {"format": FORMAT, "text_key": text_key, "id_key": id_key,
            "records": packed, "text_blocks": blocks}
    return {**view, "audit": {
        "format": AUDIT_FORMAT, "split_policy": SPLIT_POLICY,
        "original_records_sha256": _fingerprint(records),
        "native_view_sha256": _fingerprint(view), "record_order": order,
        "records": record_audit, "blocks": block_audit,
        "exact_dedup_only": True, "lossless": True, "new_model_calls": 0,
        "semantic_verified": False,
    }}


def restore_texts(book, *, expected_original_sha256=None):
    """Restore after checking exact refs, order, offsets and all frozen hashes.

    For externally supplied books, pass a trusted expected digest or use
    ``validate_text_book``. An attacker can recompute a self-consistent book.
    No unknown, unused or duplicate dictionary block is silently ignored.
    """
    _require(type(book) is dict and set(book) == _BOOK_KEYS and book.get("format") == FORMAT,
             "book_shape")
    _json_value(book)
    text_key, id_key = book["text_key"], book["id_key"]
    _records(book["records"], text_key, id_key, packed=True)
    _require(type(book["text_blocks"]) is list and type(book["audit"]) is dict, "dictionary_shape")
    by_id = {}
    for position, block in enumerate(book["text_blocks"], 1):
        _require(type(block) is dict and set(block) == {"id", "text"}
                 and block["id"] == "B" + str(position)
                 and type(block["text"]) is str and bool(block["text"]), "block_identity_or_literal")
        by_id[block["id"]] = block["text"]
    restored, used = [], set()
    for packed in book["records"]:
        row = deepcopy(packed)
        refs = row[text_key]["text_ref"]
        _require(all(ref in by_id for ref in refs), "unknown_text_reference")
        used.update(refs)
        row[text_key] = "".join(by_id[ref] for ref in refs)
        restored.append(row)
    _require(used == set(by_id), "unused_text_block")
    # Repacking is a deterministic independent derivation of every offset,
    # dictionary allocation, occurrence and audit claim, not a hash-only check.
    expected = pack_texts(restored, text_key=text_key, id_key=id_key)
    _require(_canonical(book) == _canonical(expected), "integrity_or_order")
    if expected_original_sha256 is not None:
        _require(type(expected_original_sha256) is str
                 and bool(_SHA256.fullmatch(expected_original_sha256)), "expected_digest_shape")
        _require(expected["audit"]["original_records_sha256"] == expected_original_sha256,
                 "trusted_original_digest")
    return restored


def validate_text_book(trusted_original, book):
    """Authenticate the projection against trusted rows; return a local receipt."""
    _require(type(book) is dict and "text_key" in book and "id_key" in book, "book_shape")
    expected = pack_texts(trusted_original, text_key=book["text_key"], id_key=book["id_key"])
    restored = restore_texts(book, expected_original_sha256=expected["audit"]["original_records_sha256"])
    _require(_canonical(restored) == _canonical(trusted_original)
             and _canonical(book) == _canonical(expected), "trusted_original_mismatch")
    return {"format": "lossless-text-book-validation-v1", "source_texts_equal": True,
            "original_records_sha256": expected["audit"]["original_records_sha256"],
            "native_view_sha256": expected["audit"]["native_view_sha256"],
            "record_count": len(trusted_original), "new_model_calls": 0,
            "semantic_verified": False}


def native_text_book(book, *, trusted_original=None, expected_original_sha256=None):
    """Return only the reversible model view, after local integrity validation.

    A caller supplying neither trust anchor gets internal consistency only.
    Integration must bind a trusted original and a frozen projection mode.
    """
    if trusted_original is not None:
        validate_text_book(trusted_original, book)
    restore_texts(book, expected_original_sha256=expected_original_sha256)
    return {key: deepcopy(book[key]) for key in book if key in _VIEW_KEYS}
