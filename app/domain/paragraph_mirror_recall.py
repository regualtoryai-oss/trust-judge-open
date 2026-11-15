"""Bounded direct mirror additions to an immutable BM25 base; never formal evidence."""
from collections import Counter, defaultdict
import hashlib
import json

from app.domain.paragraph_mirrors import character_similarity, SIMILARITY


def expand_base_mirrors(paragraphs, base_ids, *, candidate_limit=60):
    """Compare at most the base anchors with all qualified source paragraphs.

    Exact buckets cover the entire input. Near comparisons use proven length
    and character-overlap upper bounds before bitset LCS. No all-pairs index,
    transitive expansion, annotations, selected labels or reference data enter.
    A strongest direct anchor is sufficient audit provenance for each addition.
    """
    if type(candidate_limit) is not int or not 0 <= candidate_limit <= 120 or not isinstance(paragraphs, list):
        raise ValueError("mirror_recall_budget_or_sources_invalid")
    rows, keys, ids = [], set(), set()
    for row in paragraphs:
        if (not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k]
                for k in ("id", "raw_doc_id", "raw_para_id", "text"))):
            raise ValueError("mirror_recall_source_invalid")
        key = row["raw_doc_id"], row["raw_para_id"]
        if row["id"] in ids or key in keys:
            raise ValueError("mirror_recall_duplicate_source")
        ids.add(row["id"]); keys.add(key); rows.append(row)
    if (not isinstance(base_ids, list) or any(not isinstance(pid, str) for pid in base_ids)
            or len(base_ids) != len(set(base_ids)) or not set(base_ids) <= ids or len(base_ids) > 60):
        raise ValueError("mirror_recall_base_invalid")
    byid = {row["id"]: row for row in rows}; anchors = [byid[pid] for pid in base_ids]
    exact = defaultdict(list)
    for source_index, row in enumerate(rows):
        exact[row["text"]].append((source_index, row))
    exact_matches = {}
    for anchor_index, anchor in enumerate(anchors):
        for source_index, row in exact[anchor["text"]]:
            if row["id"] not in base_ids and row["raw_doc_id"] != anchor["raw_doc_id"]:
                exact_matches.setdefault(row["id"], (1.0, anchor_index, anchor, "exact"))
    # Histograms are source text descriptors, never model/evidence decisions.
    histograms = {}
    def histogram(text):
        if text not in histograms: histograms[text] = Counter(text)
        return histograms[text]
    additions, lcs_calls, comparisons = [], 0, 0
    for source_index, row in enumerate(rows):
        if row["id"] in base_ids:
            continue
        best = exact_matches.get(row["id"])
        if best is None:
            for anchor_index, anchor in enumerate(anchors):
                if anchor["raw_doc_id"] == row["raw_doc_id"]:
                    continue
                comparisons += 1
                left, right = anchor["text"], row["text"]
                size = len(left) + len(right)
                if 20 * min(len(left), len(right)) < 9 * size:
                    continue
                overlap = sum((histogram(left) & histogram(right)).values())
                if 20 * overlap < 9 * size:
                    continue
                lcs_calls += 1
                similarity = character_similarity(left, right)
                if similarity >= .9 and (best is None or similarity > best[0]):
                    best = (similarity, anchor_index, anchor, "similar")
        if best is not None:
            similarity, anchor_index, anchor, match = best
            additions.append(((-similarity, anchor_index, source_index), {
                "paragraph_id": row["id"], "raw_doc_id": row["raw_doc_id"], "raw_para_id": row["raw_para_id"],
                "candidate_only": True, "requires_independent_ranking_and_field_verification": True,
                "origins": [{"paragraph_id": anchor["id"], "raw_doc_id": anchor["raw_doc_id"],
                    "raw_para_id": anchor["raw_para_id"], "match": match, "similarity": similarity}]}))
    additions.sort(key=lambda pair: pair[0]); selected = [row for _, row in additions[:candidate_limit]]
    manifest = [{"id": row["id"], "doc_id": row["raw_doc_id"], "para_id": row["raw_para_id"],
                 "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()} for row in rows]
    return {"enabled": True, "method": "direct_base_whole_paragraph_exact_or_0.9", "similarity_metric": SIMILARITY,
        "threshold": .9, "candidate_budget": candidate_limit, "base_anchor_count": len(anchors), "exact_source_text_buckets": len(exact),
        "eligible_source_count": len(rows), "source_manifest_sha256": hashlib.sha256(json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "direct_nonexact_comparisons": comparisons, "bitset_lcs_calls": lcs_calls,
        "matched_before_budget": len(additions), "budget_truncated": len(additions) > candidate_limit,
        "added_count": len(selected), "additions": selected,
        "origin_policy": "strongest_direct_base_anchor_stable_ties_no_transitive_expansion",
        "evidence_created_or_deleted": False, "independent_source_votes_added": 0}
