"""Pure full-window RRF navigation to exact, complete sentence contexts.

The caller supplies a separately verified and SHA-pinned window-score value.
This module checks its complete original query/window geometry, rebuilds every
candidate, and never treats local scores as truth, model reading or evidence.
It does not load paths, call a model, cut a sentence, or modify source text.
"""
from copy import deepcopy
from hashlib import sha256
import math

from app.domain.local_parent_rank import fingerprint, prepare_pairs
from app.domain.local_passage_rank import FORMAT as WINDOW_FORMAT
from app.domain.context_lexical_rank import POLICY as LEXICAL_POLICY, VERSION as LEXICAL_VERSION
from app.domain.source_passages import (
    SOURCE_GEOMETRY, assemble_source_passage, build_sentence_manifest,
)


FORMAT = "trusted-sentence-context-candidate-plan-v1"
POLICY = {
    "version": "all-window-equal-rrf60-complete-sentence-context-v1",
    "rrf_k": 60,
    "candidate_limit": 32,
    "per_document_soft_cap": 3,
    "per_parent_soft_cap": 2,
    "window_facet_aggregation": "equal-weight-rrf-over-complete-window-ranks",
    "exact_context_merge": "same-parent-and-sentence-index-range",
    "merged_context_score": "max-origin-window-rrf-no-duplicate-votes",
    "context_neighbors_each_side": 1,
    "neighbor_counting": "nearest-nonblank-sentence-group-keep-intervening-source",
    "long_sentence_policy": "keep-complete-no-hard-character-cut",
    "tie_breaker": "raw-parent-order-then-local-window-or-sentence-range",
    "fill_passes": [
        {"name": "document-and-parent-diversity", "document_cap": 3, "parent_cap": 2},
        {"name": "relax-document-retain-parent", "document_cap": None, "parent_cap": 2},
        {"name": "fill-remaining-ranked-contexts", "document_cap": None, "parent_cap": None},
    ],
}
HYBRID_POLICY = {**deepcopy(POLICY),
    "version": "all-window-bge-bm25-family-rrf60-context-v2",
    "window_facet_aggregation": "equal-bge-and-lexical-family-mean-rrf60",
    "hybrid_retrieval": {"enabled": True, "module": "app.domain.context_lexical_rank",
        "module_version": LEXICAL_VERSION, "family_weights": {"bge": 0.5, "lexical": 0.5},
        "within_family": "mean-of-query-rrf60-contributions", "lexical_policy": deepcopy(LEXICAL_POLICY)}}


def _require(condition, reason):
    if not condition:
        raise ValueError("context_candidates_" + reason)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _window_input(raw, windows, expected_digest):
    _require(_sha(expected_digest) and fingerprint(windows) == expected_digest, "window_scores_digest_changed")
    _require(isinstance(windows, dict) and windows.get("format") == WINDOW_FORMAT
        and windows.get("all_original_windows_preserved") is True
        and windows.get("selected_top32_filter_applied") is False
        and windows.get("provider_calls") == 0 and windows.get("new_local_inference_calls") == 0
        and windows.get("semantic_verified") is False and windows.get("model_delivery_verified") is False,
        "verified_full_window_input_required")
    binding, verification = windows.get("binding"), windows.get("rank_verification")
    _require(isinstance(binding, dict) and isinstance(verification, dict)
        and binding.get("raw_case_canonical_sha256") == fingerprint(raw)
        and _sha(binding.get("model_manifest_sha256")) and _sha(binding.get("rank_plan_sha256"))
        and verification.get("rank_plan_sha256") == binding["rank_plan_sha256"]
        and isinstance(binding.get("proof"), dict)
        and verification.get("proof_sha256") == binding["proof"].get("sha256"), "window_source_binding_changed")
    expected, payloads = prepare_pairs(raw, binding["model_manifest_sha256"])
    rows = windows.get("records")
    _require(windows.get("queries") == expected["queries"] and isinstance(rows, list)
        and len(rows) == len(expected["records"])
        and windows.get("query_window_count") == len(rows)
        and windows.get("parent_count") == len(raw["paragraphs"])
        and windows.get("unique_pair_count") == len(payloads), "complete_window_facet_coverage_changed")
    values = {}
    for row, original in zip(rows, expected["records"]):
        _require(isinstance(row, dict)
            and all(row.get(key) == original[key] for key in (
                "window_id", "paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "text",
                "source_text_sha256", "window_text_sha256", "pair_sha256"))
            and row.get("query_facet") == original["query_id"], "original_window_geometry_changed")
        payload = payloads[original["pair_sha256"]]
        _require(row.get("query") == payload["query"]
            and row.get("query_sha256") == sha256(payload["query"].encode("utf8")).hexdigest(), "original_query_changed")
        score = row.get("maxlogit")
        _require(type(score) in (float, int) and math.isfinite(score), "finite_window_score_required")
        key = original["pair_sha256"]
        _require(key not in values or values[key] == score, "same_exact_payload_score_changed")
        values[key] = score
    return binding, expected, rows


def _window_fusion(raw, queries, rows):
    order = {p["id"]: i for i, p in enumerate(raw["paragraphs"])}
    by_window = {}
    for row in rows:
        key = (row["paragraph_id"], row["start_cp"], row["end_cp"])
        fused = by_window.setdefault(key, {"window_id": row["window_id"], "paragraph_id": key[0],
            "start_cp": key[1], "end_cp": key[2], "window_text_sha256": row["window_text_sha256"],
            "facet_scores": {}, "rrf_score": 0.0})
        qid = row["query_facet"]
        _require(qid not in fused["facet_scores"], "duplicate_window_facet")
        fused["facet_scores"][qid] = {"query_facet": qid, "maxlogit": row["maxlogit"],
            "pair_sha256": row["pair_sha256"]}
    for query in queries:
        qid = query["query_id"]
        _require(all(qid in row["facet_scores"] for row in by_window.values()), "missing_window_facet")
        ranked = sorted(by_window, key=lambda key: (-by_window[key]["facet_scores"][qid]["maxlogit"],
            order[key[0]], key[1], key[2]))
        for position, key in enumerate(ranked, 1):
            contribution = 1.0 / (POLICY["rrf_k"] + position)
            by_window[key]["facet_scores"][qid].update(rank=position, rrf_contribution=contribution)
            by_window[key]["rrf_score"] += contribution
    result = []
    for key in sorted(by_window, key=lambda key: (order[key[0]], key[1], key[2])):
        row = by_window[key]
        row["facet_scores"] = [row["facet_scores"][q["query_id"]] for q in queries]
        result.append(row)
    return result


def _context_range(sentences, start, end):
    hits = [i for i, row in enumerate(sentences)
        if row["local_start_cp"] < end and start < row["local_end_cp"]]
    if not hits:
        return None
    first, last = hits[0], hits[-1]
    left, right = first - 1, last + 1
    while left >= 0 and not sentences[left]["text"].strip():
        left -= 1
    while right < len(sentences) and not sentences[right]["text"].strip():
        right += 1
    return (left if left >= 0 else first, right + 1 if right < len(sentences) else last + 1)


def build_context_plan(raw_case, bound_window_scores, *, full_window_scores_digest, hybrid_retrieval=False):
    """Build at most 32 navigation contexts; supplied scores must be pinned.

    The digest is an external trust boundary: callers must obtain it from a
    verified loader/closed artifact, never from a model's self-reported scores.
    This pure module rechecks complete geometry/query coverage, not inference.
    """
    _require(type(hybrid_retrieval) is bool, "explicit_hybrid_mode_required")
    binding, expected, rows = _window_input(raw_case, bound_window_scores, full_window_scores_digest)
    manifest = build_sentence_manifest(raw_case["documents"], raw_case["paragraphs"])
    parent_manifests = {row["paragraph_id"]: row for row in manifest["paragraphs"]}
    parents = {row["id"]: row for row in raw_case["paragraphs"]}
    order = {pid: i for i, pid in enumerate(parents)}
    windows = _window_fusion(raw_case, expected["queries"], rows)
    policy, lexical = deepcopy(POLICY), None
    if hybrid_retrieval:
        from app.domain.context_lexical_rank import rank_lexical_windows
        lexical = rank_lexical_windows(raw_case, windows)
        policy = deepcopy(HYBRID_POLICY)
        by_window = {(r["paragraph_id"], r["start_cp"], r["end_cp"]): r for r in lexical["windows"]}
        for window in windows:
            key = (window["paragraph_id"], window["start_cp"], window["end_cp"])
            bge = window["rrf_score"] / len(expected["queries"])
            lex = by_window[key]["mean_rrf_score"]
            window["family_contributions"] = {"bge_mean_rrf": bge, "lexical_mean_rrf": lex,
                "bge_weighted": 0.5 * bge, "lexical_weighted": 0.5 * lex}
            window["rrf_score"] = 0.5 * bge + 0.5 * lex
    contexts, unquotable = {}, []
    for window in windows:
        parent = parents[window["paragraph_id"]]
        sentences = parent_manifests[parent["id"]]["sentences"]
        span = _context_range(sentences, window["start_cp"], window["end_cp"])
        if span is None or not parent["text"][window["start_cp"]:window["end_cp"]].strip():
            unquotable.append({"window_id": window["window_id"], "paragraph_id": parent["id"],
                "start_cp": window["start_cp"], "end_cp": window["end_cp"], "reason": "empty-or-blank-original-window"})
            continue
        start, end = span
        key = (parent["id"], start, end)
        context = contexts.setdefault(key, {"context_id": "CX" + fingerprint({
            "source_geometry": SOURCE_GEOMETRY, "paragraph_id": key[0],
            "sentence_manifest_sha256": parent_manifests[key[0]]["sentence_manifest_sha256"],
            "start_sentence_index": start, "end_sentence_index": end})[:32],
            "paragraph_id": key[0], "document_id": parent["document_id"],
            "start_sentence_index": start, "end_sentence_index": end,
            "window_origins": [], "rrf_score": window["rrf_score"]})
        context["window_origins"].append(deepcopy(window))
        context["rrf_score"] = max(context["rrf_score"], window["rrf_score"])
    ranked = sorted(contexts.values(), key=lambda row: (-row["rrf_score"], order[row["paragraph_id"]],
        row["start_sentence_index"], row["end_sentence_index"]))
    selected, selections, doc_counts, parent_counts, pass_audits = [], {}, {}, {}, []
    for step in POLICY["fill_passes"]:
        pass_audit = {"pass": step["name"], "selected_context_ids": [], "deferred_contexts": [],
            "not_visited_after_candidate_limit": []}
        for position, row in enumerate(ranked):
            if len(selected) == POLICY["candidate_limit"]:
                pass_audit["not_visited_after_candidate_limit"] = [r["context_id"] for r in ranked[position:]
                    if r["context_id"] not in selections]
                break
            cid, doc, pid = row["context_id"], row["document_id"], row["paragraph_id"]
            if cid in selections:
                continue
            reasons = []
            if step["document_cap"] is not None and doc_counts.get(doc, 0) >= step["document_cap"]:
                reasons.append("document-soft-cap")
            if step["parent_cap"] is not None and parent_counts.get(pid, 0) >= step["parent_cap"]:
                reasons.append("parent-soft-cap")
            if reasons:
                pass_audit["deferred_contexts"].append({"context_id": cid, "reasons": reasons})
                continue
            selections[cid] = step["name"]
            selected.append(row)
            pass_audit["selected_context_ids"].append(cid)
            doc_counts[doc], parent_counts[pid] = doc_counts.get(doc, 0) + 1, parent_counts.get(pid, 0) + 1
        pass_audits.append(pass_audit)
    candidates = []
    for row in selected:
        passage = assemble_source_passage(raw_case["documents"], raw_case["paragraphs"], row["paragraph_id"],
            row["start_sentence_index"], row["end_sentence_index"])
        candidates.append({"context_id": row["context_id"], "source_passage": passage,
            "rrf_score": row["rrf_score"], "window_origins": deepcopy(row["window_origins"]),
            "selection_pass": selections[row["context_id"]]})
    plan = {
        "format": FORMAT,
        "source_binding": {
            "raw_case_canonical_sha256": fingerprint(raw_case),
            "question_sha256": fingerprint(raw_case["question"]),
            "documents_sha256": fingerprint(raw_case["documents"]),
            "paragraphs_sha256": fingerprint(raw_case["paragraphs"]),
            "sentence_manifest_sha256": manifest["manifest_sha256"],
            "rank_plan_sha256": binding["rank_plan_sha256"],
            "rank_proof_sha256": binding["proof"]["sha256"],
            "model_manifest_sha256": binding["model_manifest_sha256"],
            "formatter_config_sha256": binding["formatter_config_sha256"],
        },
        "policy": policy,
        "full_window_scores_digest": full_window_scores_digest,
        "candidates": candidates,
        "audit": {
            "corpus_parent_count": len(parents), "query_facet_count": len(expected["queries"]),
            "full_query_window_count": len(rows), "fused_original_window_count": len(windows),
            "unique_context_count": len(ranked), "candidate_count": len(candidates),
            "ranked_contexts": [{"context_id": row["context_id"], "paragraph_id": row["paragraph_id"],
                "start_sentence_index": row["start_sentence_index"], "end_sentence_index": row["end_sentence_index"],
                "rrf_score": row["rrf_score"], "selected": row["context_id"] in selections,
                "selection_pass": selections.get(row["context_id"])} for row in ranked],
            "selection_pass_audit": pass_audits,
            "unquotable_original_windows": unquotable,
            "not_selected_contexts_are_unknown_not_irrelevant": True,
            "all_full_windows_used_before_context_selection": True,
        },
        "provider_calls": 0, "new_local_inference_calls": 0,
        "semantic_verified": False, "model_delivery_verified": False,
        "all_original_paragraphs_read_by_model": False,
    }
    if hybrid_retrieval:
        plan["audit"]["hybrid_retrieval"] = lexical
    return plan


def validate_context_plan(raw_case, bound_window_scores, context_plan, *, full_window_scores_digest, hybrid_retrieval=False):
    """Recompute source geometry, full RRF/cap/fill policy and every audit row."""
    expected = build_context_plan(raw_case, bound_window_scores,
        full_window_scores_digest=full_window_scores_digest, hybrid_retrieval=hybrid_retrieval)
    _require(type(context_plan) is dict and context_plan == expected, "plan_changed")
    return deepcopy(expected)
