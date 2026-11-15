"""Read complete, verified local query/window scores without inference.

The parent rank's selected top32 is deliberately irrelevant here. Each row is
one original body window under one literal query facet, including final tails,
empty parents and payloads shared by multiple original sources. Scores remain
navigation logits; they establish neither model delivery nor factual support.
"""
from copy import deepcopy
from hashlib import sha256
import math

from app.domain import local_parent_rank as rank


FORMAT = "verified-local-passage-window-scores-v1"


def load_bound_window_scores(raw, rank_plan, proof, expected_model_manifest_sha256, raw_case_sha256):
    """Return every CLOSED query/window pair, preserving original order/bytes.

    No scores or paths are accepted apart from the SHA-bound CLOSED proof.
    Verification keeps the current producer/source checks and the old cache
    origin's own archive checks in ``verify_local_rank`` unchanged. Re-reading
    each descriptor after verification detects changed/missing bound files.
    ``maxlogit`` is a single exact query/window score, with no parent, facet or
    top-k aggregation despite the compatibility field name.
    """
    verified = rank.verify_local_rank(raw, rank_plan, proof,
        expected_model_manifest_sha256=expected_model_manifest_sha256,
        raw_case_sha256=raw_case_sha256)
    closed = rank._read(proof)
    plan = rank._read(closed["window_plan"])
    payloads = rank._read(closed["pair_payloads"])
    scores = rank._read(closed["pair_scores"])
    parents = {row["id"]: row for row in raw["paragraphs"]}
    queries = {row["query_id"]: row["text"] for row in plan["queries"]}
    rows = []
    for window in plan["records"]:
        key = window["pair_sha256"]
        parent = parents[window["paragraph_id"]]
        query = queries[window["query_id"]]
        payload = payloads[key]
        score = scores[key]
        rank._require(type(score) in (int, float) and math.isfinite(score)
            and rank.fingerprint(payload) == key
            and payload["query"] == query and payload["document"] == window["text"]
            and payload["model_manifest_sha256"] == expected_model_manifest_sha256
            and parent["text"][window["start_cp"]:window["end_cp"]] == window["text"],
            "passage_window_score_binding_changed")
        rows.append({
            "window_id": window["window_id"],
            "paragraph_id": window["paragraph_id"],
            "raw_doc_id": window["raw_doc_id"],
            "raw_para_id": window["raw_para_id"],
            "start_cp": window["start_cp"],
            "end_cp": window["end_cp"],
            "text": window["text"],
            "source_text_sha256": window["source_text_sha256"],
            "window_text_sha256": window["window_text_sha256"],
            "query_facet": window["query_id"],
            "query": query,
            "query_sha256": sha256(query.encode("utf8")).hexdigest(),
            "pair_sha256": key,
            "maxlogit": score,
        })
    return {
        "format": FORMAT,
        "binding": {
            "raw_case_sha256": raw_case_sha256,
            "raw_case_canonical_sha256": rank.fingerprint(raw),
            "rank_plan_sha256": rank.fingerprint(rank_plan),
            "proof": deepcopy(proof),
            "window_plan": deepcopy(closed["window_plan"]),
            "pair_payloads": deepcopy(closed["pair_payloads"]),
            "pair_scores": deepcopy(closed["pair_scores"]),
            "model_manifest_sha256": expected_model_manifest_sha256,
            "formatter_config_sha256": closed["formatter_config_sha256"],
        },
        "queries": deepcopy(plan["queries"]),
        "parent_count": len(parents),
        "query_window_count": len(rows),
        "unique_pair_count": len(scores),
        "records": rows,
        "score_semantics": "single_exact_query_window_logit_no_parent_or_facet_aggregation",
        "all_original_windows_preserved": True,
        "selected_top32_filter_applied": False,
        "rank_verification": verified,
        "new_local_inference_calls": 0,
        "provider_calls": 0,
        "semantic_verified": False,
        "model_delivery_verified": False,
    }
