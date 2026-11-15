"""No model/network: only synthetic quote escaping and rejected broader repair."""
from hashlib import sha256
import json

import pytest

from app.agents import strict_json_syntax as syntax


def test_valid_json_fast_path_keeps_all_values_without_calling_repair(monkeypatch):
    def forbidden(*_args, **_kwargs): pytest.fail("valid JSON must not enter repair")
    monkeypatch.setattr(syntax, "repair_json", forbidden)
    original = ' \n{"statement":"保留空白 与\\\"引号\\\"。","n":1.50,"a":[true,null]}\t'
    decoded, audit = syntax.decode_visible_arguments(original)
    assert decoded == json.loads(original) and audit is None


def test_one_outside_string_unmatched_closer_has_exact_single_deletion_audit():
    original = ' {"a":{"text":"] stays quoted"}],"b":[1,2]} '
    decoded, audit = syntax.decode_visible_arguments(original)
    assert decoded == {"a": {"text": "] stays quoted"}, "b": [1, 2]}
    assert audit["format"] == "strict-json-single-unmatched-closer-v1"
    assert audit["insertions"] == [] and len(audit["deletions"]) == 1
    deletion = audit["deletions"][0]
    assert deletion["delete"] == "]" and original[deletion["offset"]] == "]"
    repaired = original[:deletion["offset"]] + original[deletion["offset"] + 1:]
    assert json.loads(repaired) == decoded
    assert sha256(repaired.encode()).hexdigest() == audit["repaired_arguments_sha256"]


@pytest.mark.parametrize("raw", ['{"a":1]],"b":2}', '{"a":1],"b":}', '{"a":1],"a":2}', '{"a":1],"b":NaN}'])
def test_unmatched_closer_never_repairs_other_defects_duplicate_keys_or_values(raw):
    with pytest.raises(syntax.StrictJSONSyntaxError):
        syntax.decode_visible_arguments(raw)


@pytest.mark.parametrize("raw", ['{"key":1,"key":2}', '{"a":{"key":1,"key":2}}'])
def test_duplicate_keys_never_enter_heuristic_repair(raw, monkeypatch):
    def forbidden(*_args, **_kwargs): pytest.fail("duplicate keys must not enter repair")
    monkeypatch.setattr(syntax, "repair_json", forbidden)
    with pytest.raises(syntax.StrictJSONSyntaxError, match="duplicate_json_key"):
        syntax.decode_visible_arguments(raw)


@pytest.mark.parametrize("raw", ['{"n":NaN}', '{"n":Infinity}', '{"n":-Infinity}', '{"n":1e400}'])
def test_nonfinite_values_are_rejected(raw):
    with pytest.raises(syntax.StrictJSONSyntaxError, match="non_finite_json_number"):
        syntax.decode_visible_arguments(raw)


def test_unescaped_quotes_recover_only_by_backslash_insertions_with_exact_original_offsets():
    original = ' \n{"statement": "😀他说"原话"。", "evidence_ids": ["E1"], "n": 2}\t'
    decoded, audit = syntax.decode_visible_arguments(original)
    assert decoded == {"statement": '😀他说"原话"。', "evidence_ids": ["E1"], "n": 2}
    assert audit["library"] == "json-repair" and audit["library_version"]
    assert len(audit["insertions"]) == 2
    repaired = original
    for insertion in reversed(audit["insertions"]):
        assert insertion["insert"] == "\\" and insertion["before"] == '"'
        assert original[insertion["offset"]] == '"'
        repaired = repaired[:insertion["offset"]] + "\\" + repaired[insertion["offset"]:]
    assert json.loads(repaired) == decoded
    assert audit["original_arguments_sha256"] == sha256(original.encode()).hexdigest()
    assert audit["repaired_arguments_sha256"] == sha256(repaired.encode()).hexdigest()
    assert original.startswith(" \n") and original.endswith("\t")


@pytest.mark.parametrize("raw", [
    '{"statement": "原话", "n": 2',  # Missing container closure.
    '{"statement": "原话',  # Truncated string and container.
    '{"statement": "原话", "n": }',  # Missing value.
    "{'statement': '原话'}",  # Replacing quote characters is forbidden.
    '{"statement": "😀他说"原话"。", "n": 2}\ntrailing text',
])
def test_completion_field_value_or_nonquote_repairs_are_rejected(raw):
    with pytest.raises(syntax.StrictJSONSyntaxError): syntax.decode_visible_arguments(raw)


@pytest.mark.parametrize("candidate", [
    {"statement": '他说"原话"。', "n": 3},  # Changed scalar.
    {"statement": '他说"原话"。', "n": 2, "new": "field"},  # Added field.
    {"statement": "他说原话。", "n": 2},  # Deleted literal quotes.
    {"statement": '他说"别的话"。', "n": 2},  # Changed text.
    {"statement": '他说"原话"。'},  # Dropped field.
])
def test_candidate_library_cannot_change_scalars_or_keys(candidate, monkeypatch):
    original = '{"statement": "他说"原话"。", "n": 2}'
    monkeypatch.setattr(syntax, "repair_json", lambda *_args, **_kwargs: candidate)
    with pytest.raises(syntax.StrictJSONSyntaxError, match="json_syntax_not_quote_only"):
        syntax.decode_visible_arguments(original)


@pytest.mark.parametrize("value", [None, {}, 1])
def test_argument_type_is_explicit(value):
    with pytest.raises(syntax.StrictJSONSyntaxError, match="visible_arguments_string_required"):
        syntax.decode_visible_arguments(value)


@pytest.mark.parametrize("value", ["[]", '"text"', "42", "null"])
def test_tool_arguments_must_be_an_object(value):
    with pytest.raises(syntax.StrictJSONSyntaxError, match="visible_arguments_object_required"):
        syntax.decode_visible_arguments(value)
