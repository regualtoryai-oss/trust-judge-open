import pytest

from app.domain.source_metadata import clean_publish_time, source_credibility_value


@pytest.mark.parametrize("raw,normalized,status", [
    ("2026-09-28", "2026-09-28", "explicit_date"),
    ("2026年9月28日", "2026-09-28", "explicit_date"),
    ("2026/9/28", "2026-09-28", "explicit_date"),
    ("2026-09-28T08:30:00+08:00", "2026-09-28T08:30:00+08:00", "explicit_datetime"),
    ("2026-09-28 08:30", "2026-09-28T08:30:00", "explicit_datetime_timezone_unknown"),
    ("2026-09-28 08:30北京时间", "2026-09-28T08:30:00+08:00", "explicit_datetime"),
    ("9月28日", None, "year_unknown"), ("3天前", None, "relative_time_unknown"),
    ("昨天", None, "relative_time_unknown"), ("2026-09", None, "incomplete_date"),
    ("2026", None, "incomplete_date"), ("2026年9月", None, "incomplete_date"),
    ("人民日报", None, "unknown"), ("", None, "missing"), (None, None, "missing"),
    ("2026-02-30", None, "invalid_calendar"), ("2026-02-30T08:30Z", None, "invalid_calendar"),
])
def test_publication_cleaning_never_infers_missing_year_day_or_zone(raw, normalized, status):
    result = clean_publish_time(raw)
    assert result["raw"] == raw
    assert result["normalized"] == normalized and result["status"] == status


@pytest.mark.parametrize("value,expected", [("0.95", .95), (.9, .9), (0, 0), (1, 1),
                                             ("null", None), (None, None), (True, None),
                                             ("官方", None), (float('nan'), None), (1.01, None),
                                             ({"grade": .95}, None)])
def test_credibility_uses_original_numeric_metadata_only(value, expected):
    grade, basis = source_credibility_value({"source_credibility_json": value})
    assert grade == expected
    assert "not_" in basis
