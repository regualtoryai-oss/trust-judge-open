"""Conservative source metadata cleaning, without clocks or inferred years."""
from datetime import date, datetime
import json
import math
import re


def clean_publish_time(raw):
    result = {"raw": raw, "status": "unknown", "normalized": None, "time_precision": "unknown"}
    if raw is None or raw == "":
        return {**result, "status": "missing"}
    if not isinstance(raw, str):
        return result
    text = raw.strip()
    if not text:
        return {**result, "status": "missing"}
    if re.search(r"(?:\d+|[一二三四五六七八九十]+)\s*(?:秒|分钟|小时|天|日|周|个月|月|年)前|昨天|今天|刚刚|前天", text):
        return {**result, "status": "relative_time_unknown"}
    full_date = re.fullmatch(r"(\d{4})(?:-|/|年)(\d{1,2})(?:-|/|月)(\d{1,2})日?", text)
    if full_date:
        try:
            value = date(*(int(p) for p in full_date.groups())).isoformat()
        except ValueError:
            return {**result, "status": "invalid_calendar"}
        return {**result, "status": "explicit_date", "normalized": value, "time_precision": "date"}
    iso = text
    if text.endswith("北京时间"):
        iso = text[:-4].strip() + "+08:00"
    # Only complete explicit dates can enter datetime parsing. A month or a
    # source name must not be interpreted using the host date or timezone.
    if re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", iso):
        try:
            value = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return {**result, "status": "invalid_calendar"}
        return {**result, "status": "explicit_datetime" if value.tzinfo else "explicit_datetime_timezone_unknown",
                "normalized": value.isoformat(), "time_precision": "datetime"}
    if re.match(r"^\d{1,2}月\d{1,2}日", text):
        return {**result, "status": "year_unknown"}
    if re.fullmatch(r"\d{4}(?:年|[-/]\d{1,2}|年\d{1,2}月)?", text):
        return {**result, "status": "incomplete_date"}
    return result


def source_credibility_value(paragraph):
    """Read the supplied numeric grade; unknown is not a fabricated zero."""
    value = paragraph.get("source_credibility_json", paragraph.get("source_credibility"))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            value = None
    if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1:
        return float(value), "original_source_credibility_metadata_not_model_rank_or_truth_probability"
    return None, "source_credibility_missing_or_invalid_not_inferred"
