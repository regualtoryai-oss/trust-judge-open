"""Generic source-local numeric occurrence tests; no sample/reference material."""
from copy import deepcopy

import pytest

from app.domain.numeric_literals import numeric_claims, unsupported_numeric_claims


@pytest.mark.parametrize("source,model,value", [
    ("一名人员", "1人", "1"), ("两名人员", "2人", "2"),
    ("十名人员", "10人", "10"), ("十二名人员", "12人", "12"),
    ("一千零二名人员", "1002人", "1002"),
    ("1万名人员", "10000人", "10000"), ("十万人", "100000人", "100000"),
    ("一万零二名人员", "10002人", "10002"),
    ("２名人员", "2人", "2"),
])
def test_standard_chinese_counts_and_explicit_multipliers(source, model, value):
    assert unsupported_numeric_claims(model, source) == []
    assert numeric_claims(source)[0]["value"] == value
    assert numeric_claims(source)[0]["dimension"] == "count:person"


@pytest.mark.parametrize("source,model", [
    ("二〇二六年十月十九日", "2026年10月19日"),
    ("2026年10月19日", "2026-10-19"),
    ("2026/10/19", "二〇二六年十月十九日"),
    ("二〇二六", "2026"), ("十二", "12"),
])
def test_year_digit_sequences_and_date_components(source, model):
    assert unsupported_numeric_claims(model, source) == []


@pytest.mark.parametrize("source,model", [
    ("百分之二", "2%"), ("2％", "百分之二"),
    ("千分之二十", "2%"), ("万分之五", "0.05%"),
    ("负百分之二", "-2%"), ("百分之九点五", "9.5%"),
    ("1,234.50元", "1234.5元"), ("一千二百三十四点五元", "1,234.50元"),
    ("两千米", "2公里"), ("2公斤", "两千克"),
])
def test_percent_decimal_thousands_and_explicit_unit_spelling(source, model):
    assert unsupported_numeric_claims(model, source) == []


@pytest.mark.parametrize("model,source", [
    ("3人", "两名人员"), ("2人", "两国"), ("2人", "百分之二"),
    ("2%", "两名人员"), ("95%", "9.5%"), ("9.5%", "95%"),
    ("1人", "1万名人员"), ("10人", "十万人"),
    ("2美元", "两元"), ("2人", "型号为2"),
    ("171", "AI171航班"), ("AI172", "AI171航班"),
    ("3200人", "三千二人"), ("12000人", "1万2人"),
])
def test_novel_values_wrong_dimensions_and_abbreviation_guesses_remain_unsupported(model, source):
    assert unsupported_numeric_claims(model, source)


def test_does_not_create_quantities_from_lexical_single_chinese_digits_or_substrings():
    source = "一方面，应一起调查二氧化碳问题；两国专家出席。"
    assert numeric_claims("一方面一起调查二氧化碳") == []
    start = source.index("两国")
    assert numeric_claims(source) == [{"raw": "两国", "start": start, "end": start + 2, "value": "2", "dimension": "count:国"}]
    for value in ("1人", "2人"):
        assert unsupported_numeric_claims(value, source)
    assert [row["value"] for row in numeric_claims("十万人")] == ["100000"]
    assert [row["value"] for row in numeric_claims("1,234.50元")] == ["1234.5"]
    assert [row["value"] for row in numeric_claims("9.5%")] == ["0.095"]


def test_bare_model_number_can_match_known_source_value_but_not_ratio_or_identifier():
    assert unsupported_numeric_claims("2", "两名人员") == []
    assert unsupported_numeric_claims("10000", "1万人") == []
    assert unsupported_numeric_claims("2026", "二〇二六年") == []
    assert unsupported_numeric_claims("1", "100%")
    assert unsupported_numeric_claims("1", "1万名人员")
    assert unsupported_numeric_claims("171", "AI171")


def test_unknown_numeric_tokens_require_the_same_complete_raw_value():
    rows = numeric_claims("1万2人")
    assert rows == [{"raw": "1万2人", "start": 0, "end": 4, "value": "1万2", "dimension": "unknown:count:person"}]
    assert unsupported_numeric_claims("1万2人", "记录为1万2人。") == []
    assert unsupported_numeric_claims("1", "1万2人")
    assert unsupported_numeric_claims("2人", "1万2人")


def test_output_keeps_literal_codepoint_spans_and_never_rewrites_source_or_fields():
    source = "\n😀两名人员幸存；其余9.5%待核。AI171。\r\n"
    fields = '{"argument":{"数值口径":"2人"},"fact_des":"原文称两名人员幸存。"}'
    original = deepcopy((source, fields))
    claims = numeric_claims(source)
    for row in claims:
        assert source[row["start"]:row["end"]] == row["raw"]
    assert unsupported_numeric_claims(fields, source) == []
    assert (source, fields) == original
    assert [row["dimension"] for row in claims] == ["count:person", "ratio", "identifier"]


def test_bare_punctuation_is_not_part_of_a_new_number_and_large_arabic_values_stay_exact():
    assert numeric_claims("2, then 9.5.") == [
        {"raw": "2", "start": 0, "end": 1, "value": "2", "dimension": "number"},
        {"raw": "9.5", "start": 8, "end": 11, "value": "9.5", "dimension": "number"},
    ]
    value = "1234567890123456789012345678901234567890"
    assert numeric_claims(value)[0]["value"] == value
    assert numeric_claims("12,34")[0]["dimension"].startswith("unknown:")
    assert unsupported_numeric_claims("1234", "12,34")
