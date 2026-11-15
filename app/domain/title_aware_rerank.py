"""Pure local rerank inputs/parent aggregation; no model, IO or truth claims."""
from hashlib import sha256
import json
import math

from app.domain.semantic_recall import window_catalog

VERSION = "original-title-complete-window-bge-v1"


def fingerprint(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def prepare_title_pairs(question, catalog, *, model_fingerprint):
    """Query is unchanged; title and every literal 320/80 window are data.

    Caller binds original source bytes externally. Pair hashes include title,
    original query, model and formatter, so no-title logits cannot be reused.
    Tokenizer checks happen before inference, never by cutting this payload.
    """
    text = question.get("question") if isinstance(question, dict) else question
    if (not isinstance(text, str) or not text.strip() or not isinstance(catalog, list)
            or not isinstance(model_fingerprint, str) or not model_fingerprint):
        raise ValueError("original_question_catalog_and_model_binding_required")
    rows, metadata, seen, locations = [], {}, set(), set()
    for row in catalog:
        ref = row.get("paragraph_ref") if isinstance(row, dict) else None
        if (not isinstance(ref, str) or not ref or ref in seen
                or any(not isinstance(row.get(k), str) or not row[k] for k in ("doc_id", "para_id"))
                or not isinstance(row.get("title"), str) or not isinstance(row.get("text"), str)
                or row.get("paragraph_complete") is not True or row.get("parent_complete") is not True):
            raise ValueError("unique_complete_original_parent_required")
        location = row["doc_id"], row["para_id"]
        if location in locations:
            raise ValueError("duplicate_original_source_required_to_fail")
        seen.add(ref); locations.add(location)
        rows.append({"id": ref, "raw_doc_id": row["doc_id"], "raw_para_id": row["para_id"], "text": row["text"]})
        metadata[ref] = {"paragraph_ref": ref, "doc_id": row["doc_id"], "para_id": row["para_id"],
                         "title": row["title"], "source_text_sha256_utf8": sha256(row["text"].encode()).hexdigest()}
    records = []
    for window in window_catalog(rows, window_chars=320, overlap_chars=80):
        ref = window["paragraph_id"]
        document = metadata[ref]["title"] + "\n" + window["text"]
        payload = {"formatter": VERSION, "model_fingerprint": model_fingerprint, "query": text, "document": document}
        records.append({"paragraph_ref": ref, "window_id": window["window_id"], "start_cp": window["start_cp"],
                        "end_cp": window["end_cp"], "window_text": window["text"], "model_document": document,
                        "window_text_sha256": window["window_text_sha256"], "pair_sha256": fingerprint(payload)})
    scored_refs = {row["paragraph_ref"] for row in records}
    return {"version": VERSION, "question": text, "model_fingerprint": model_fingerprint,
            "catalog_sha256": fingerprint(catalog), "parents": list(metadata.values()), "records": records,
            "parents_without_windows": [r["paragraph_ref"] for r in catalog if r["paragraph_ref"] not in scored_refs],
            "window_chars": 320, "overlap_chars": 80, "source_binding_verified_here": False,
            "original_title_is_unverified_context": True, "scores_are_truth_probabilities": False}


def rank_title_parents(plan, scores):
    """Require every frozen window before parent-max; never infer missing=0."""
    if plan.get("version") != VERSION or plan.get("parents_without_windows"):
        raise ValueError("all_original_parent_windows_required")
    expected = {row["pair_sha256"] for row in plan["records"]}
    if (not isinstance(scores, dict) or set(scores) != expected
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in scores.values())):
        raise ValueError("complete_exact_pair_finite_scores_required")
    best = {}
    for record in plan["records"]:
        ref, score = record["paragraph_ref"], scores[record["pair_sha256"]]
        if ref not in best or score > best[ref]["local_logit"]:
            best[ref] = {"local_logit": score, "best_window_id": record["window_id"],
                         "best_window_start_cp": record["start_cp"], "best_window_end_cp": record["end_cp"]}
    order = {row["paragraph_ref"]: i for i, row in enumerate(plan["parents"])}
    if set(best) != set(order):
        raise ValueError("unscored_parent_cannot_be_assigned_zero")
    by_ref = {row["paragraph_ref"]: row for row in plan["parents"]}
    return [{**by_ref[ref], **best[ref]} for ref in sorted(best, key=lambda r: (-best[r]["local_logit"], order[r]))]


def fuse_title_hybrid(plan, title_ranking, old_hybrid_refs, *, limit=40):
    """Equal RRF60 of complete title-BGE ranking and existing hybrid40.

    Preserve each original source; no quota, neighbor or reference-dependent
    fallback. This function does not change which parents were eligible.
    """
    known = {row["paragraph_ref"]: row for row in plan["parents"]}
    positions = {ref: i for i, ref in enumerate(known)}
    if type(limit) is not int or not 1 <= limit <= 40 or len(old_hybrid_refs) > 40:
        raise ValueError("fixed_forty_candidate_budget_required")
    lists = [[row["paragraph_ref"] for row in title_ranking], old_hybrid_refs]
    if set(lists[0]) != set(known):
        raise ValueError("complete_title_parent_ranking_required")
    scores, provenance = {}, {}
    for channel, refs in enumerate(lists):
        if not isinstance(refs, list) or len(refs) != len(set(refs)) or any(r not in known for r in refs):
            raise ValueError("unknown_or_duplicate_source_in_ranking")
        for rank, ref in enumerate(refs, 1):
            scores[ref] = scores.get(ref, 0.0) + 1.0 / (60 + rank)
            provenance.setdefault(ref, []).append({"channel": channel, "rank": rank})
    refs = sorted(scores, key=lambda r: (-scores[r], positions[r]))[:limit]
    return [{**known[ref], "rrf60": scores[ref], "ranking_signals": provenance[ref]} for ref in refs]
