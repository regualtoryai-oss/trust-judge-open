"""Deterministic BM25 navigation over original complete body windows.

This module derives only literal question facets. It never plans an answer,
changes source text, calls a model, or labels a source as support/conflict.
The context caller must first validate the separately CLOSED BGE window input.
"""
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version as package_version
import math
import re

from rank_bm25 import BM25Okapi

from app.domain.retrieval import _tokens


VERSION = "literal-question-title-body-bm25-family-v1"
POLICY = {"version": VERSION, "library": "rank_bm25.BM25Okapi", "library_version": "0.2.2", "k1": 1.5, "b": 0.75,
    "epsilon": 0.25, "rrf_k": 60, "family_aggregation": "mean-rrf60-over-literal-query-ranks",
    "index": "original-title-newline-complete-original-body-window", "title_is_verified_fact": False,
    "query_derivation": "whole-question-plus-original-clause-quoted-or-latin-span",
    "whole_original_question_in_every_query": True, "added_synonyms_or_claims": False,
    "topic_name_publisher_year_or_source_id_branches": False,
    "zero_token_match": "no-rank-zero-contribution",
    "tie_breaker": "raw-parent-order-then-local-start-then-local-end"}
_DELIMITERS = re.compile(r"还是|或者|抑或|[？?；;。]")
_QUOTED = re.compile(r'“([^”]+)”|「([^」]+)」|『([^』]+)』|《([^》]+)》|"([^"\n]+)"')
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_'\-]*")


def _require(condition, reason):
    if not condition:
        raise ValueError("context_lexical_rank_" + reason)


def literal_queries(question):
    """Keep all conditions in every query; focus text is an exact raw slice."""
    _require(isinstance(question, str) and bool(question.strip()), "original_question_required")
    spans = []
    def add(start, end, kind):
        while start < end and question[start].isspace(): start += 1
        while end > start and question[end - 1].isspace(): end -= 1
        if start < end and question[start:end] != question:
            spans.append((start, end, kind))
    start = 0
    for match in _DELIMITERS.finditer(question):
        add(start, match.start(), "literal_clause")
        start = match.end()
    add(start, len(question), "literal_clause")
    for match in _QUOTED.finditer(question):
        group = next(i for i in range(1, match.re.groups + 1) if match.group(i) is not None)
        add(*match.span(group), "quoted_span")
    for match in _LATIN.finditer(question):
        add(*match.span(), "latin_span")
    # Repeated words do not create extra facets; keep the first literal offset
    # and all derivation kinds for audit. No names or answer lexicon are used.
    unique = {}
    for begin, end, kind in sorted(spans):
        focus = question[begin:end]
        entry = unique.setdefault(focus, {"start_cp": begin, "end_cp": end, "text": focus, "kinds": []})
        if kind not in entry["kinds"]: entry["kinds"].append(kind)
    queries = [{"query_id": "L0", "text": question, "whole_question": question, "focus": None}]
    for focus in unique.values():
        queries.append({"query_id": f"L{len(queries)}", "text": question + "\n重点核验：" + focus["text"],
                        "whole_question": question, "focus": focus})
    return queries


def rank_lexical_windows(raw_case, windows):
    """Return a locally derived mean RRF contribution for every source window."""
    _require(isinstance(raw_case, dict) and isinstance(windows, list), "raw_and_windows_required")
    _require(package_version("rank-bm25") == POLICY["library_version"], "bm25_library_version_changed")
    queries = literal_queries(raw_case["question"]["question"])
    parents = {p["id"]: p for p in raw_case["paragraphs"]}
    documents = {d["id"]: d for d in raw_case["documents"]}
    _require(len(parents) == len(raw_case["paragraphs"]) and len(documents) == len(raw_case["documents"]),
             "unique_original_source_ids_required")
    order = {pid: i for i, pid in enumerate(parents)}
    records, texts, seen = [], [], set()
    for window in windows:
        pid, start, end = window.get("paragraph_id"), window.get("start_cp"), window.get("end_cp")
        parent = parents.get(pid)
        _require(parent is not None and type(start) is int and type(end) is int
            and 0 <= start <= end <= len(parent["text"]), "original_window_geometry_required")
        key = (pid, start, end)
        _require(key not in seen, "duplicate_original_window")
        seen.add(key)
        doc = documents.get(parent["document_id"])
        _require(doc is not None and isinstance(doc.get("title", ""), str), "original_document_title_required")
        body, title = parent["text"][start:end], doc.get("title", "")
        _require(window.get("window_text_sha256") == sha256(body.encode("utf8")).hexdigest(),
                 "original_window_body_changed")
        texts.append(_tokens(title + "\n" + body))
        records.append({"window_id": window["window_id"], "paragraph_id": pid,
            "start_cp": start, "end_cp": end, "body_sha256": sha256(body.encode("utf8")).hexdigest(),
            "title_sha256": sha256(title.encode("utf8")).hexdigest(), "facet_scores": [], "mean_rrf_score": 0.0})
    model = BM25Okapi(texts, k1=POLICY["k1"], b=POLICY["b"], epsilon=POLICY["epsilon"]) if any(texts) else None
    rankings = []
    for query in queries:
        tokens = _tokens(query["text"])
        scores = model.get_scores(tokens).tolist() if model is not None else [0.0] * len(records)
        _require(len(scores) == len(records) and all(math.isfinite(s) for s in scores), "finite_bm25_scores_required")
        query_terms = set(tokens)
        matched = [i for i, words in enumerate(texts) if query_terms.intersection(words)]
        ranked = sorted(matched, key=lambda i: (-scores[i], order[records[i]["paragraph_id"]],
            records[i]["start_cp"], records[i]["end_cp"]))
        positions = {index: position for position, index in enumerate(ranked, 1)}
        rankings.append({"query_id": query["query_id"], "ranked_window_ids": [records[i]["window_id"] for i in ranked]})
        for i, record in enumerate(records):
            position = positions.get(i)
            contribution = 1.0 / (POLICY["rrf_k"] + position) if position is not None else 0.0
            record["mean_rrf_score"] += contribution / len(queries)
            record["facet_scores"].append({"query_id": query["query_id"], "bm25_score": float(scores[i]),
                "rank": position, "rrf_contribution": contribution,
                "family_mean_contribution": contribution / len(queries)})
    return {"format": "derived-local-context-lexical-ranks-v1", "policy": deepcopy(POLICY),
        "question_sha256": sha256(raw_case["question"]["question"].encode("utf8")).hexdigest(),
        "queries": queries, "window_count": len(records), "windows": records, "rankings": rankings,
        "provider_calls": 0, "new_local_inference_calls": 0,
        "semantic_verified": False, "model_delivery_verified": False}
