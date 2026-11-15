"""Source-preserving numeric occurrence checks, using cn2an for Chinese values.

This checks whether a number of the stated dimension occurs in supplied text.
It does not establish the associated subject, truth, time or statistical scope.
No rewritten source text, inferred range midpoint or numeric subtoken is used.
"""
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import re
import unicodedata

import cn2an
from cn2an.conf import NUMBER_CN2AN, NUMBER_LOW_AN2CN


_CN = "零〇○一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟点"
_NUMBER = rf"[+\-负正]?[\d{_CN}.,]+"
_LEFT = r"(?<![A-Za-z0-9_.])"
_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]{1,12}\d+(?:[-._][A-Za-z0-9]+)*(?![A-Za-z0-9_])")
_ISO_DATE = re.compile(r"(?<![A-Za-z0-9])(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})(?!\d)")
_PERCENT = re.compile(
    rf"{_LEFT}(?:(?P<sign>[+\-负正]?)(?P<prefix>百分之|千分之|万分之)(?P<cn>{_NUMBER})|(?P<suffix_number>{_NUMBER})\s*(?P<suffix>[%％‰‱]))")

# Explicit spelling aliases and physical scale factors, not inferred entities.
# Ambiguous lexical 起 is deliberately absent: 一起调查 is not a count.
_UNITS = {
    "人": ("count:person", "1"), "名": ("count:person", "1"),
    **{unit: (f"count:{unit}", "1") for unit in ("个", "件", "项", "条", "次", "辆", "架", "艘", "户", "家", "所", "座", "例", "台", "笔", "国")},
    "年": ("time:year", "1"), "月": ("time:month", "1"), "日": ("time:day", "1"), "天": ("time:day", "1"),
    "小时": ("time:hour", "1"), "分钟": ("time:minute", "1"), "秒": ("time:second", "1"),
    "元": ("currency:元", "1"), "美元": ("currency:美元", "1"), "欧元": ("currency:欧元", "1"), "英镑": ("currency:英镑", "1"),
    "米": ("length:m", "1"), "公里": ("length:m", "1000"), "毫米": ("length:m", "0.001"), "厘米": ("length:m", "0.01"),
    "克": ("mass:g", "1"), "公斤": ("mass:g", "1000"), "吨": ("mass:g", "1000000"),
    "升": ("volume:l", "1"), "毫升": ("volume:l", "0.001"),
}
_QUANTITY = re.compile(rf"{_LEFT}(?P<number>{_NUMBER})\s*(?P<unit>{'|'.join(map(re.escape, sorted(_UNITS, key=len, reverse=True)))})")
_BARE = re.compile(rf"{_LEFT}(?P<number>{_NUMBER})")
_ARABIC = re.compile(r"[+\-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)")
_ARABIC_SCALE = re.compile(r"(?P<number>[+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?P<scale>十|百|千|万|亿|万亿)")


def _value(token: str) -> Decimal | None:
    text = unicodedata.normalize("NFKC", token).replace("○", "〇")
    # Sign spelling is normalized without implementing Chinese arithmetic.
    text = ("-" + text[1:]) if text.startswith("负") else text
    text = ("+" + text[1:]) if text.startswith("正") else text
    try:
        if _ARABIC.fullmatch(text):
            value = Decimal(text.replace(",", ""))
        elif _ARABIC_SCALE.fullmatch(text):
            match = _ARABIC_SCALE.fullmatch(text)
            value = _scale(Decimal(match["number"].replace(",", "")), Decimal(str(cn2an.cn2an("一" + match["scale"], "strict"))))
        elif not re.search(r"[0-9,]", text):
            sign = Decimal(-1) if text.startswith("-") else Decimal(1)
            unsigned = text.lstrip("+-")
            # Use the package's own digit variants and strict grammar. Its
            # normal mode alone also guesses colloquial 三千二 -> 3200.
            normalized = "".join(NUMBER_LOW_AN2CN[NUMBER_CN2AN[ch]] if ch in NUMBER_CN2AN else ch for ch in unsigned)
            integer, point, fraction = normalized.partition("点")
            digits = set(NUMBER_LOW_AN2CN.values())
            if not integer or (point and (not fraction or any(ch not in digits for ch in fraction))):
                return None
            if any(ch not in digits for ch in integer):
                cn2an.cn2an(integer, "strict")  # Validate, do not infer omitted place values.
            value = sign * Decimal(str(cn2an.cn2an(normalized, "normal")))
        else:
            return None  # No smart-mode guessing of mixed/abbreviated values.
        return value if value.is_finite() else None
    except (ValueError, TypeError, KeyError, OverflowError, InvalidOperation):
        return None


def _canonical(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _scale(value: Decimal, factor: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = max(28, len(value.as_tuple().digits) + len(factor.as_tuple().digits) + 8)
        return value * factor


def numeric_claims(text: str) -> list[dict]:
    """Return exact Python/codepoint spans and JSON-safe normalized value strings.

    Percent/per-mille values are dimension ``ratio`` at their fractional value.
    Failed numeric parses remain unknown claims; they never disappear or split
    into smaller values. Single bare Chinese digits are lexical-ambiguous and
    require an explicit supported unit; multi-character numeral tokens are kept.
    """
    if not isinstance(text, str):
        raise TypeError("numeric_text_required")
    claims, occupied = [], []
    def free(start, end):
        return all(end <= left or start >= right for left, right in occupied)
    def add(start, end, value, dimension):
        claims.append({"raw": text[start:end], "start": start, "end": end,
                       "value": _canonical(value) if isinstance(value, Decimal) else value, "dimension": dimension})
    def quantity(start, end, token, dimension, factor=Decimal(1)):
        value = _value(token)
        if value is None:
            add(start, end, unicodedata.normalize("NFKC", token), "unknown:" + dimension)
        else:
            add(start, end, _scale(value, factor), dimension)
        occupied.append((start, end))

    for match in _ISO_DATE.finditer(text):
        try:
            date(int(match["year"]), int(match["month"]), int(match["day"]))
        except ValueError:
            continue
        for key in ("year", "month", "day"):
            add(*match.span(key), Decimal(match[key]), "time:" + key)
        occupied.append(match.span())
    for match in _IDENTIFIER.finditer(text):
        if free(*match.span()):
            add(*match.span(), match.group(), "identifier"); occupied.append(match.span())
    for match in _PERCENT.finditer(text):
        if not free(*match.span()):
            continue
        token = match["cn"] if match["prefix"] else match["suffix_number"]
        if match["prefix"] and match["sign"]:
            token = match["sign"] + token
        divisor = {"百分之": 100, "千分之": 1000, "万分之": 10000, "%": 100, "％": 100, "‰": 1000, "‱": 10000}[match["prefix"] or match["suffix"]]
        quantity(*match.span(), token, "ratio", Decimal(1) / Decimal(divisor))
    for match in _QUANTITY.finditer(text):
        if free(*match.span()):
            dimension, factor = _UNITS[match["unit"]]
            quantity(*match.span(), match["number"], dimension, Decimal(factor))
    for match in _BARE.finditer(text):
        if not free(*match.span()):
            continue
        token = match["number"].rstrip(",.")
        if not token:
            continue
        unsigned = token.lstrip("+-负正")
        if not re.search(r"\d", token) and len(unsigned) < 2:
            continue
        quantity(match.start(), match.start() + len(token), token, "number")
    return sorted(claims, key=lambda row: (row["start"], row["end"]))


def unsupported_numeric_claims(fields_text: str, source_text: str) -> list[dict]:
    """Return unsupported model occurrences without changing either input.

    Typed quantities require equal dimension/value. Only an untyped model
    number may match an equally valued known source number/quantity. Ratios,
    identifiers and failed parses never supply a bare-number value.
    """
    source = numeric_claims(source_text)
    exact = {(row["dimension"], row["value"]) for row in source}
    untyped_values = {row["value"] for row in source if row["dimension"] not in {"identifier", "ratio"}
                      and not row["dimension"].startswith("unknown:")}
    return [row for row in numeric_claims(fields_text)
            if (row["dimension"], row["value"]) not in exact
            and not (row["dimension"] == "number" and row["value"] in untyped_values)]
