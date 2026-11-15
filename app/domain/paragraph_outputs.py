"""Program-owned paragraph output consolidation; never a semantic selector.

One source key remains one output record. Mirrors retain their own keys. Existing
annotations are joined without adding claims; this offline merge is not a new
model summary or renewed semantic verification. Cross-paragraph quotations need
the original immutable document and its paragraph coordinates.
"""
from copy import deepcopy
import hashlib
import json

ARGUMENTS = ("时间", "主体", "地点", "内容", "影响", "数值口径")
VERSION = "paragraph-output-one-source-key-v1"


def _key(row):
    key = (row.get("doc_id", row.get("raw_doc_id")), row.get("para_id", row.get("raw_para_id")))
    if not all(isinstance(v, str) and v for v in key):
        raise ValueError("paragraph_output_source_key")
    return key


def _unique(values):
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value.strip()))


def whole_source_quote(source, original_quote):
    """Return the full first paragraph or a verified continuous document span.

    The document span starts at the declared first paragraph and ends at the
    last covered paragraph. No normalization, fuzzy repair or invented joining
    delimiter is permitted. Unknown document geometry is an explicit error.
    """
    text = source.get("text")
    if not isinstance(text, str) or not text or not isinstance(original_quote, str) or not original_quote:
        raise ValueError("paragraph_output_empty_source_or_quote")
    if original_quote in text:
        return text, {"start_cp": 0, "end_cp": len(text), "scope": "one_original_paragraph"}
    document, begin = source.get("document_text"), source.get("document_start_cp")
    segments = source.get("document_paragraphs")
    if (not isinstance(document, str) or type(begin) is not int or begin < 0 or not isinstance(segments, list)
            or document[begin:begin + len(text)] != text):
        raise ValueError("paragraph_output_cross_span_requires_original_document")
    offset = document.find(original_quote, begin)
    if offset < begin or offset >= begin + len(text):
        raise ValueError("paragraph_output_quote_not_in_declared_start_paragraph")
    another = document.find(original_quote, offset + 1)
    if begin <= another < begin + len(text):
        raise ValueError("paragraph_output_ambiguous_cross_span")
    finish = offset + len(original_quote)
    covered = [p for p in segments if type(p.get("start_cp")) is int and type(p.get("end_cp")) is int
               and p["end_cp"] > begin and p["start_cp"] < finish]
    if not covered or min(p["start_cp"] for p in covered) != begin:
        raise ValueError("paragraph_output_cross_span_geometry")
    covered.sort(key=lambda p: p["start_cp"])
    previous_end = begin
    for p in covered:
        if (not 0 <= p["start_cp"] < p["end_cp"] <= len(document) or p["start_cp"] < previous_end
                or not isinstance(p.get("text"), str) or document[p["start_cp"]:p["end_cp"]] != p["text"]):
            raise ValueError("paragraph_output_cross_span_source_changed")
        previous_end = p["end_cp"]
    if covered[0].get("para_id") != _key(source)[1]:
        raise ValueError("paragraph_output_cross_span_start_id")
    end = covered[-1]["end_cp"]
    if end < finish:
        raise ValueError("paragraph_output_cross_span_geometry")
    return document[begin:end], {"start_cp": begin, "end_cp": end, "scope": "original_document_contiguous_span",
                                "covered_para_ids": [p["para_id"] for p in covered]}


def consolidate_evidence(rows, source_paragraphs):
    """Return (standard evidence, old-index map, separate provenance audit)."""
    source = {}
    for paragraph in source_paragraphs:
        key = _key(paragraph)
        if key in source:
            raise ValueError("paragraph_output_duplicate_source_key")
        source[key] = paragraph
    grouped = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("paragraph_output_invalid_record")
        key = _key(row)
        if key not in source:
            raise ValueError("paragraph_output_unknown_source")
        quote, geometry = whole_source_quote(source[key], row.get("quote"))
        grouped.setdefault(key, []).append((index, row, quote, geometry))
    output, mapping, audit = [], {}, []
    for key, peers in grouped.items():
        # Different verified lengths with the same starting paragraph retain
        # the longest continuous whole-paragraph extent, never a quote union.
        longest = max(peers, key=lambda peer: len(peer[2]))
        if any(not longest[2].startswith(peer[2]) for peer in peers):
            raise ValueError("paragraph_output_incompatible_contiguous_spans")
        descriptions = _unique(peer[1].get("fact_des") for peer in peers)
        if not descriptions:
            raise ValueError("paragraph_output_missing_fact_description")
        argument = {field: "；".join(_unique(peer[1].get("argument", {}).get(field) for peer in peers))
                    for field in ARGUMENTS}
        entities = _unique(entity for peer in peers for entity in peer[1].get("entities", []))
        row = {"doc_id": key[0], "para_id": key[1], "quote": longest[2],
               "fact_des": "；".join(descriptions), "argument": argument, "entities": entities}
        target = len(output)
        output.append(row)
        for index, *_ in peers:
            mapping[index] = target
        audit.append({"doc_id": key[0], "para_id": key[1], "old_evidence_indices": [p[0] for p in peers],
                      "new_evidence_index": target, "geometry": longest[3],
                      "annotation_method": "join_existing_descriptions_without_new_claims",
                      "new_summary_generated": False,
                      "source_quote_sha256": hashlib.sha256(longest[2].encode()).hexdigest()})
    return output, mapping, {"version": VERSION, "input_records": len(rows), "output_records": len(output),
                             "new_model_calls": 0, "semantic_revalidation_performed": False,
                             "mirrors_not_removed": True, "records": audit}


def consolidate_submission(submission, source_paragraphs):
    """Offline bundle adapter. Relation membership is preserved, not inferred."""
    result = deepcopy(submission)
    old = submission["evidence"]
    evidence, mapping, audit = consolidate_evidence(old, source_paragraphs)
    by_key = {_key(row): index for index, row in enumerate(evidence)}
    result["evidence"] = evidence
    relation_audits = []
    for conclusion in result.get("conclusions", []):
        for side in ("supporting_facts", "conflicting_facts"):
            refs, seen = [], set()
            original = conclusion.get(side, [])
            for ref in original:
                if isinstance(ref, dict) and "evidence_index" in ref:
                    identity = ref["evidence_index"]
                    if type(identity) is not int or identity not in mapping:
                        raise ValueError("paragraph_output_unknown_relation_index")
                    target = mapping[identity]
                else:
                    key = _key(ref)
                    if key not in by_key:
                        raise ValueError("paragraph_output_unknown_relation_key")
                    target = by_key[key]
                if target not in seen:
                    seen.add(target)
                    refs.append({"evidence_index": target})
            conclusion[side] = refs
            relation_audits.append({"question_id": conclusion.get("question_id"), "side": side,
                                    "before": len(original), "after": len(refs)})
    audit["relation_counts"] = relation_audits
    audit["output_sha256"] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                                       separators=(",", ":")).encode()).hexdigest()
    return result, audit
