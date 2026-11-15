"""Pure raw-byte adapter for CLOSED v22 source candidates, never a v19 proof.

No files, retrieval, queries, models, publication or reference answers are read.
Full parent completeness and coordinates come from bound raw document slices.
"""
from copy import deepcopy
from hashlib import sha256
import json
import re

POOL_FORMAT = "v22-four-independent-channel-quota-candidates-v1"
REFERENCE_FIELDS = frozenset({"evidence_samples", "evidence_samples.json", "reference_answers", "gold", "gold_answers",
    "expected", "expected_answers", "conclusion", "conclusions", "supporting_facts", "conflicting_facts", "answer"})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _reject_references(value):
    if isinstance(value, dict):
        if set(value) & REFERENCE_FIELDS:
            raise ValueError("reference_or_answer_fields_forbidden")
        for part in value.values():
            _reject_references(part)
    elif isinstance(value, list):
        for part in value:
            _reject_references(part)


def _decode_bound(data, expected):
    if (not isinstance(data, bytes) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or sha256(data).hexdigest() != expected):
        raise ValueError("frozen_input_bytes_sha256_mismatch")
    def pairs(items):
        value = {}
        for key, part in items:
            if key in value:
                raise ValueError("duplicate_json_field_forbidden")
            value[key] = part
        return value
    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs)
    _reject_references(value)
    return value


def _one(rows, predicate, check):
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise ValueError(check)
    return matches[0]


def build_frozen_selection_input(*, raw_input_bytes, pool_input_bytes, topic_id, question_id,
                                 expected_raw_sha256, expected_pool_sha256, variant="four_channel_quota400"):
    """Preserve one existing ranked source pool; do not select or create facts."""
    raw = _decode_bound(raw_input_bytes, expected_raw_sha256)
    pool = _decode_bound(pool_input_bytes, expected_pool_sha256)
    if (not isinstance(raw, dict) or set(raw) != {"documents", "questions", "title"}
            or pool.get("format") != POOL_FORMAT or pool.get("reference_reads") != []
            or not isinstance(topic_id, str) or not topic_id or not isinstance(question_id, str) or not question_id):
        raise ValueError("supported_closed_raw_pool_schema_required")
    source = _one(pool["inputs"], lambda row: row["topic_id"] == topic_id, "unique_pool_topic_required")
    if source["sha256"] != expected_raw_sha256 or source["title"] != raw["title"]:
        raise ValueError("raw_source_not_bound_to_pool_topic")
    case = _one(pool["cases"], lambda row: row["topic_id"] == topic_id and row["question_id"] == question_id, "unique_frozen_case_required")
    question = _one(raw["questions"], lambda row: row["question_id"] == question_id, "unique_raw_question_required")
    if set(question) != {"question_id", "question"} or case["question"] != question["question"] or case["title"] != raw["title"]:
        raise ValueError("complete_original_question_or_title_changed")
    output = _one(case["outputs"], lambda row: row["variant"] == variant, "unique_existing_candidate_variant_required")
    candidates = output["candidates"]
    if not isinstance(candidates, list) or sha256(canonical(candidates)).hexdigest() != output["candidate_sha256"]:
        raise ValueError("frozen_candidate_rows_hash_mismatch")
    known, documents, locations = {}, set(), set()
    for doc in raw["documents"]:
        did, content, title = doc["doc_id"], doc["content"], doc["title"]
        if not isinstance(did, str) or not did or did in documents or not isinstance(content, str) or not isinstance(title, str):
            raise ValueError("unique_literal_raw_document_required")
        documents.add(did)
        for para in doc["paragraphs"]:
            pid, text, start, end = para["para_id"], para["text"], para["char_start"], para["char_end"]
            identity = f"{topic_id}/{did}/{pid}"
            if (not isinstance(pid, str) or not pid or not isinstance(text, str) or identity in known or (did, pid) in locations
                    or type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(content) or content[start:end] != text):
                raise ValueError("raw_parent_identity_or_codepoint_span_invalid")
            locations.add((did, pid)); known[identity] = (doc, para)
    catalog, metadata, seen = [], {}, set()
    for rank, row in enumerate(candidates, 1):
        pid = row["id"]
        if pid in seen or pid not in known:
            raise ValueError("duplicate_or_unknown_frozen_source")
        seen.add(pid); doc, para = known[pid]; text = para["text"]; ref = f"P{rank}"
        text_sha = sha256(text.encode("utf-8")).hexdigest()
        if row["raw_doc_id"] != doc["doc_id"] or row["raw_para_id"] != para["para_id"]:
            raise ValueError("frozen_raw_source_identity_changed")
        if "source_text_sha256" in row and row["source_text_sha256"] != text_sha:
            raise ValueError("frozen_literal_utf8_text_sha256_mismatch")
        catalog.append({"paragraph_ref": ref, "text": text, "title": doc["title"], "doc_id": doc["doc_id"],
            "para_id": para["para_id"], "paragraph_complete": True, "parent_complete": True})
        metadata[ref] = {"paragraph_id": pid, "doc_id": doc["doc_id"], "para_id": para["para_id"], "candidate_rank": rank,
            "document_start_cp": para["char_start"], "document_end_cp": para["char_end"], "parent_start_cp": 0, "parent_end_cp": len(text),
            "source_text_sha256_utf8": text_sha, "v19_text_json_digest": sha256(canonical(text)).hexdigest(),
            "document_content_sha256_utf8": sha256(doc["content"].encode("utf-8")).hexdigest(),
            "source_credibility": deepcopy(doc.get("source_credibility")), "content_url": deepcopy(doc.get("content_url")),
            "raw_publish_time": deepcopy(doc.get("publish_time")), "cache_text_sha_supplied": "source_text_sha256" in row,
            "parent_complete": True, "raw_source_coordinate_binding_verified": True, "semantic_verified": False,
            "field_or_support_conflict_verified": False}
    return {"format": "frozen-unified-selection-input-v1", "topic_id": topic_id, "question_id": question_id,
        "question": deepcopy(question), "topic_title": deepcopy(raw["title"]), "catalog": catalog, "source_metadata": metadata,
        "binding": {"raw_input_file_sha256": expected_raw_sha256, "pool_input_file_sha256": expected_pool_sha256, "variant": variant,
            "raw_question_sha256": sha256(canonical(question)).hexdigest(), "candidate_rows_sha256": output["candidate_sha256"],
            "catalog_sha256": sha256(canonical(catalog)).hexdigest(), "source_metadata_sha256": sha256(canonical(metadata)).hexdigest()},
        "stats": {"candidate_count": len(catalog), "source_parent_count": len(known), "source_document_count": len(documents),
            "catalog_parent_codepoints": sum(len(row["text"]) for row in catalog), "max_parent_codepoints": max((len(row["text"]) for row in catalog), default=0),
            "unprovided_source_parent_count": len(known)-len(catalog), "unprovided_context_state": "unknown_not_irrelevant",
            "model_reading_executed": False, "model_input_tokens": None, "token_status": "unknown_exact_messages_not_built",
            "source_credibility_inferred_or_modified": False}, "publishable": False, "v19_formal_proof_created": False}
