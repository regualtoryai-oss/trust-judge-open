"""Conservative visible-JSON decoding using json-repair plus an insertion audit.

Only missing backslashes immediately before existing string-internal quotes may
be recovered. No model call, truncation completion, field repair or value repair.
The caller still validates the response envelope, tool identity and full schema.
"""
from hashlib import sha256
from importlib.metadata import version
import json
import math

from json_repair import repair_json


class StrictJSONSyntaxError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str):
    raise StrictJSONSyntaxError(code)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def _constant(_value):
    _fail("non_finite_json_number")


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        _fail("non_finite_json_number")
    if isinstance(value, dict):
        for child in value.values(): _finite(child)
    elif isinstance(value, list):
        for child in value: _finite(child)


def _strict_loads(text):
    result = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    _finite(result)
    return result


def _quote_insertions(original: str, proposed: str) -> list[dict]:
    """Compare against valid JSON; never generate or infer a replacement value."""
    trimmed = original.strip(" \t\r\n")
    leading = len(original) - len(original.lstrip(" \t\r\n"))
    left = right = 0
    in_string = escaped = False
    insertions = []
    while left < len(trimmed) and right < len(proposed):
        char = proposed[right]
        if trimmed[left] == char:
            if in_string:
                if escaped: escaped = False
                elif char == "\\": escaped = True
                elif char == '"': in_string = False
            elif char == '"':
                in_string = True
            left += 1; right += 1
        elif char == "\\" and proposed[right:right + 2] == '\\"' and trimmed[left] == '"' and in_string and not escaped:
            insertions.append({"offset": leading + left, "insert": "\\", "before": '"'})
            escaped = True; right += 1
        else:
            _fail("json_syntax_not_quote_only")
    if left != len(trimmed) or right != len(proposed) or not insertions:
        _fail("json_syntax_not_quote_only")
    return insertions


def decode_visible_arguments(arguments: str) -> tuple[dict, dict | None]:
    """Strict valid-JSON fast path, otherwise audited quote escaping only.

    Audit offsets are Unicode/codepoint positions in the original *untrimmed*
    argument string. The repaired hash covers that original string with only
    the recorded backslashes inserted; original outer whitespace is preserved.
    """
    if not isinstance(arguments, str):
        _fail("visible_arguments_string_required")
    try:
        decoded = _strict_loads(arguments)
    except json.JSONDecodeError:
        pass
    else:
        if not isinstance(decoded, dict): _fail("visible_arguments_object_required")
        return decoded, None
    # One extra closing delimiter is recoverable only when a complete scan
    # proves it is outside strings and mismatches the active container stack.
    # No library guesses, missing closure/value completion or other edits.
    stack, unmatched = [], []
    in_string = escaped = False
    for offset, char in enumerate(arguments):
        if in_string:
            if escaped: escaped = False
            elif char == "\\": escaped = True
            elif char == '"': in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            expected = "[" if char == "]" else "{"
            if stack and stack[-1] == expected:
                stack.pop()
            else:
                unmatched.append((offset, char))
    if not in_string and not stack and len(unmatched) == 1:
        offset, char = unmatched[0]
        repaired = arguments[:offset] + arguments[offset + 1:]
        try:
            decoded = _strict_loads(repaired)
        except json.JSONDecodeError:
            pass  # Another syntax defect must not be repaired together.
        else:
            if not isinstance(decoded, dict): _fail("visible_arguments_object_required")
            return decoded, {
                "format": "strict-json-single-unmatched-closer-v1",
                "original_arguments_sha256": sha256(arguments.encode("utf-8")).hexdigest(),
                "repaired_arguments_sha256": sha256(repaired.encode("utf-8")).hexdigest(),
                "deletions": [{"offset": offset, "delete": char}], "insertions": [],
                "library": "deterministic-json-container-stack", "library_version": "1",
            }
    try:
        candidate = repair_json(arguments, return_objects=True, ensure_ascii=False, strict=True)
        if not isinstance(candidate, dict): _fail("visible_arguments_object_required")
        proposed = json.dumps(candidate, ensure_ascii=False, allow_nan=False)
    except StrictJSONSyntaxError:
        raise
    except (ValueError, TypeError, RecursionError):
        _fail("json_syntax_repair_rejected")
    insertions = _quote_insertions(arguments, proposed)
    repaired = arguments
    for insertion in reversed(insertions):
        offset = insertion["offset"]
        repaired = repaired[:offset] + insertion["insert"] + repaired[offset:]
    try:
        decoded = _strict_loads(repaired)
    except json.JSONDecodeError:
        _fail("json_syntax_repair_invalid")
    if decoded != candidate:
        _fail("json_syntax_candidate_mismatch")
    return decoded, {
        "format": "strict-json-quote-escape-v1",
        "original_arguments_sha256": sha256(arguments.encode("utf-8")).hexdigest(),
        "repaired_arguments_sha256": sha256(repaired.encode("utf-8")).hexdigest(),
        "insertions": insertions, "library": "json-repair", "library_version": version("json-repair"),
    }
