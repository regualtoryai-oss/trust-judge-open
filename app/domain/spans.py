"""Codepoint half-open source spans. Never normalize or search-repair a quote."""


def validate_span(text: str, quote: str, start_cp: int, end_cp: int) -> None:
    if not isinstance(text, str) or not isinstance(quote, str) or not quote:
        raise ValueError("text and a nonempty quote must be strings")
    if type(start_cp) is not int or type(end_cp) is not int:
        raise ValueError("span coordinates must be integer codepoints")
    if not 0 <= start_cp < end_cp <= len(text):
        raise ValueError("span coordinates are out of bounds")
    if text[start_cp:end_cp] != quote:
        raise ValueError("quote does not match the specified source span")


def cp_to_utf16(text: str, index: int) -> int:
    if not isinstance(text, str) or type(index) is not int or not 0 <= index <= len(text):
        raise ValueError("invalid codepoint position")
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text[:index])
