"""Literal mentioned-source routes; metadata matches are not evidence or votes.

No provider, configuration, network, authenticity, credibility or semantic claim.
The original title/URL and body are never rewritten; title prefix is query-only.
"""
import re

FAMILY_ORDER = ("original_question", "mentioned_source", "F", "R", "P", "L", "C")


def latin_mentions(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("complete_original_question_required")
    result = []
    for token in re.findall(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*(?![A-Za-z0-9_])", question):
        if len(token) >= 3 and any("A" <= c <= "Z" for c in token) and token.casefold() not in {x.casefold() for x in result}:
            result.append(token)
    return result


def mentioned_source_routes(question, documents):
    tokens = latin_mentions(question)
    if not isinstance(documents, list):
        raise ValueError("raw_document_list_required")
    routes, seen = [], set()
    for document in documents:
        if not isinstance(document, dict) or not isinstance(document.get("doc_id"), str) or not document["doc_id"] or document["doc_id"] in seen:
            raise ValueError("unique_original_document_ids_required")
        seen.add(document["doc_id"])
        title, url = document.get("title"), document.get("content_url")
        title_text = title if isinstance(title, str) else ""
        url_text = url if isinstance(url, str) else ""
        matches = []
        for token in tokens:
            pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", re.I | re.ASCII)
            fields = [key for key, value in (("title", title_text), ("content_url", url_text)) if pattern.search(value)]
            if fields:
                matches.append({"question_token": token, "metadata_fields": fields})
        if matches:
            routes.append({"raw_doc_id": document["doc_id"], "original_title": title, "original_content_url": url,
                "matches": matches, "query_title_view": title_text.split("|", 1)[0],
                "query": question + "\n检索目标（仅查询线索，非事实）：" + title_text.split("|", 1)[0],
                "metadata_match_is_not_source_authentication": True, "semantic_verified": False,
                "independent_vote_or_confidence_boost": False})
    return routes


def roundrobin_with_source(paragraphs, old_families, source_family):
    """One source family, not a quota per matched document. Empty is identity."""
    known = {row["id"] for row in paragraphs}
    if len(known) != len(paragraphs) or "original_question" not in old_families or not set(old_families) <= set(FAMILY_ORDER) - {"mentioned_source"}:
        raise ValueError("original_family_source_identity_invalid")
    families = dict(old_families)
    if source_family:
        families["mentioned_source"] = source_family
    ordered = [key for key in FAMILY_ORDER if key in families]
    for key in ordered:
        ids = [row["id"] for row in families[key]]
        if len(set(ids)) != len(ids) or not set(ids) <= known:
            raise ValueError("unknown_or_duplicate_family_parent")
    positions = dict.fromkeys(ordered, 0)
    selected, seen = [], set()
    while len(selected) < 40:
        added = False
        for key in ordered:
            rows = families[key]
            while positions[key] < len(rows) and rows[positions[key]]["id"] in seen:
                positions[key] += 1
            if positions[key] == len(rows):
                continue
            rank = positions[key]
            positions[key] += 1
            seen.add(rows[rank]["id"])
            selected.append(dict(rows[rank], selected_from_family=key, family_rank=rank + 1))
            added = True
            if len(selected) == 40:
                break
        if not added:
            break
    return selected
