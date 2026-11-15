"""Local exploration views inside a fixed source pool; never evidence selection.

Literal question facets are search hints. Executing them or exhausting this
queue does not establish factual, semantic or counterevidence completeness.
"""
from collections import defaultdict
from hashlib import sha256

from app.domain.frozen_selection_inputs import canonical, _decode_bound
from app.domain.question_recall import _bm25, question_query_plan, recall_question_candidates
from app.domain.selection_read_tools import SelectionReadTools

POLICY = {
    "version": "literal-facet-local-exploration-v1",
    "query_source": "full_original_question_literal_clauses_plus_explicit_unverified_hints_if_supplied",
    "frozen_comparison_extra_queries": 0,
    "comparison_search_domains": ["frozen_widepool", "raw_all_paragraphs"],
    "pool_change": False, "ranking_depth": 60, "rrf_k": 30,
    "granularity_weights": {"word": 1.0, "bigram": .65},
    "facet_anchor": "equal_RRF_of_complete_question_and_literal_facet_window_rankings",
    "fusion": "original_window_multi_then_nonempty_unique_facets_round_robin_deduplicated",
    "comparison_limits": [5, 20, 40], "full_original_question_always_bound": True,
    "same_budget_controls": "unchanged_window_multi_limit20_and_limit40",
    "title_or_source_role_branches": False, "automatic_neighbours": False,
    "empty_query_or_zero_body_hits_counts_as_explored": False,
    "read_or_already_exposed_refs_repeat": False,
    "no_candidate_or_query_means_irrelevant": False,
}


def _identity(query): return " ".join(query.split()).casefold()


def _scope(values, sources):
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError("exploration_scope_collection_required")
    values = list(values)
    if any(not isinstance(v, str) or v not in sources for v in values) or len(values) != len(set(values)):
        raise ValueError("exploration_known_unique_scope_required")
    return set(values)


def _rrf(rankings):
    scores = defaultdict(float)
    for weight, ranking in rankings:
        for rank, (_, index) in enumerate(ranking[:60], 1):
            scores[index] += weight / (30 + rank)
    return sorted(scores, key=lambda i: (-scores[i], i))[:60]


def build_exploration_schedule(frozen_case, *, read_refs=(), already_exposed_refs=(), additional_queries=()):
    """Return bootstrap5 and one common exposure queue's prefixes20/40.

    Extra queries, if explicitly supplied, are unverified hints and undergo the
    same policy. The frozen eighteen-question comparison supplies none. Original
    raw-byte authenticity belongs to the caller's adapter rebuild, not a dict.
    """
    SelectionReadTools(frozen_case, max_search_calls=0, max_read_calls=0)
    return _schedule(frozen_case, "frozen_widepool", read_refs, already_exposed_refs, additional_queries)


def build_full_raw_exploration_input(raw_input_bytes, expected_raw_sha256, topic_id, question_id):
    """Independent all-parent source projection, not a v41 fixed-pool case."""
    raw = _decode_bound(raw_input_bytes, expected_raw_sha256)
    if set(raw) != {"title", "questions", "documents"} or not isinstance(topic_id, str) or not topic_id:
        raise ValueError("exploration_original_raw_projection_required")
    questions = [q for q in raw["questions"] if q["question_id"] == question_id]
    if len(questions) != 1 or set(questions[0]) != {"question_id", "question"}:
        raise ValueError("exploration_unique_original_question_required")
    catalog, metadata, documents = [], {}, set()
    for doc in raw["documents"]:
        did, content, title = doc["doc_id"], doc["content"], doc["title"]
        if not isinstance(did, str) or not did or did in documents or not isinstance(content, str) or not isinstance(title, str):
            raise ValueError("exploration_unique_original_document_required")
        documents.add(did); content_sha = sha256(content.encode()).hexdigest()
        for para in doc["paragraphs"]:
            pid, text, start, end = (para[k] for k in ("para_id", "text", "char_start", "char_end"))
            ref = f"{topic_id}/{did}/{pid}"
            if (not isinstance(pid, str) or not pid or ref in metadata or not isinstance(text, str)
                    or type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(content) or content[start:end] != text):
                raise ValueError("exploration_original_paragraph_identity_or_span_changed")
            catalog.append({"paragraph_ref": ref, "doc_id": did, "para_id": pid, "title": title,
                            "text": text, "paragraph_complete": True})
            metadata[ref] = {"doc_id": did, "para_id": pid, "document_start_cp": start, "document_end_cp": end,
                "source_text_sha256_utf8": sha256(text.encode()).hexdigest(), "document_content_sha256_utf8": content_sha,
                "source_credibility": doc.get("source_credibility"), "semantic_verified": False}
    return {"format": "full-raw-local-exploration-input-v1", "topic_id": topic_id, "question_id": question_id,
        "question": questions[0], "topic_title": raw["title"], "catalog": catalog, "source_metadata": metadata,
        "binding": {"raw_input_file_sha256": expected_raw_sha256, "catalog_sha256": sha256(canonical(catalog)).hexdigest()},
        "not_a_frozen_widepool_case": True, "model_calls": 0, "publishable": False}


def build_full_raw_exploration_schedule(raw_input_bytes, expected_raw_sha256, topic_id, question_id):
    source = build_full_raw_exploration_input(raw_input_bytes, expected_raw_sha256, topic_id, question_id)
    return _schedule(source, "raw_all_paragraphs", (), (), ())


def _schedule(source, search_domain, read_refs, already_exposed_refs, additional_queries):
    catalog = source["catalog"]; question = source["question"]["question"]
    sources = {row["paragraph_ref"]: row for row in catalog}
    skipped = _scope(read_refs, sources) | _scope(already_exposed_refs, sources)
    if not isinstance(additional_queries, (list, tuple)) or any(not isinstance(q, str) or len(q) > 160 for q in additional_queries):
        raise ValueError("exploration_explicit_short_query_hints_required")
    parents = [{"id": r["paragraph_ref"], "raw_doc_id": r["doc_id"], "raw_para_id": r["para_id"],
                "title": r["title"], "text": r["text"]} for r in catalog]
    base = recall_question_candidates(question, parents, limit=60, variant="window_multi")
    # The old bootstrap's internal ranking depth depends on its limit. Reusing
    # top5 from a depth60 request is not guaranteed to reproduce limit5.
    bootstrap = recall_question_candidates(question, parents, limit=5, variant="window_multi")
    controls = {n: recall_question_candidates(question, parents, limit=n, variant="window_multi") for n in (20, 40)}
    original_ranking = [r["paragraph_id"] for r in base["candidates"]]
    facets = question_query_plan(question)["facets"]
    planned = [(row["origin"], row["text"]) for row in facets]
    planned += [("explicit_unverified_hint", query) for query in additional_queries]
    anchor = [(weight, _bm25(parents, question, kind, True)) for kind, weight in POLICY["granularity_weights"].items()]
    channels, attempts, seen = [], [], set()
    for origin, query in planned:
        identity = _identity(query); ref = "Q" + str(len(attempts) + 1)
        attempt = {"query_ref": ref, "origin": origin, "query": query,
                   "full_original_question": question, "semantic_verified": False}
        if not identity:
            attempt.update(status="empty_query_not_explored", ranked_refs=[])
        elif identity in seen:
            attempt.update(status="duplicate_query_not_explored", ranked_refs=[])
        else:
            seen.add(identity)
            if query == question:
                ranking = original_ranking
            else:
                local = [(weight, _bm25(parents, query, kind, True)) for kind, weight in POLICY["granularity_weights"].items()]
                # A zero-hit facet cannot inherit the question list and claim
                # that its own condition was explored successfully.
                ranking = [parents[i]["id"] for i in _rrf([*anchor, *local])] if any(rows for _, rows in local) else []
            attempt.update(status="ranked_local_hint" if ranking else "no_body_match_not_explored", ranked_refs=ranking)
            if ranking: channels.append((ref, ranking))
        attempts.append(attempt)
    cursors = [0] * len(channels); chosen, origins, seen_refs = [], {}, set(skipped)
    while len(chosen) < 40:
        advanced = False
        for index, (query_ref, ranking) in enumerate(channels):
            while cursors[index] < len(ranking) and ranking[cursors[index]] in seen_refs:
                cursors[index] += 1
            if cursors[index] >= len(ranking): continue
            ref = ranking[cursors[index]]; cursors[index] += 1
            seen_refs.add(ref); chosen.append(ref); origins[ref] = query_ref; advanced = True
            if len(chosen) == 40: break
        if not advanced: break
    return {"format": "fixed-pool-local-exploration-schedule-v1", "policy_sha256": sha256(canonical(POLICY)).hexdigest(),
        "search_domain": search_domain, "input_format": source["format"],
        "topic_id": source["topic_id"], "question_id": source["question_id"], "question": source["question"],
        "pool_catalog_sha256": source["binding"]["catalog_sha256"], "pool_count": len(catalog),
        "query_attempts": attempts, "successful_nonempty_unique_query_lists": len(channels),
        "bootstrap5": [r["paragraph_id"] for r in bootstrap["candidates"] if r["paragraph_id"] not in skipped],
        "original20": [r["paragraph_id"] for r in controls[20]["candidates"] if r["paragraph_id"] not in skipped],
        "original40": [r["paragraph_id"] for r in controls[40]["candidates"] if r["paragraph_id"] not in skipped],
        "multi_facet20": chosen[:20], "multi_facet40": chosen,
        "first_exposure_query": origins, "read_refs_skipped": [r for r in sources if r in set(read_refs)],
        "already_exposed_refs_skipped": [r for r in sources if r in set(already_exposed_refs)],
        "unexposed_scope_unknown": True, "facet_execution_proves_conditions_understood": False,
        "model_calls": 0, "full_text_sent_to_model": False, "parent_raw_bytes_reverified": False,
        "semantic_verified": False, "formal_evidence_generated": False, "publishable": False}
