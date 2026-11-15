"""Raw paragraph mirror candidates, never evidence or independent source votes.

Similarity is normalized indel similarity: 2*LCS/(len(left)+len(right)) on
unmodified Unicode code points. Whitespace, punctuation and case are retained.
Groups are only navigation metadata; expansions use direct edges, not transitive
similarity. Every returned location needs its own source and fact verification.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
import math

VERSION = "raw-paragraph-mirror-candidates-v1"
SIMILARITY = "unicode-normalized-indel-2lcs-over-lengths-v1"


def _digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _threshold(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("invalid_mirror_threshold")
    return Fraction(str(value))


def character_similarity(left: str, right: str) -> float:
    """Exact codepoint LCS using integer bit sets; no external dependency."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError("invalid_mirror_text")
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    # Use the shorter string as the bitset width.
    if len(left) < len(right):
        left, right = right, left
    masks = {}
    for index, char in enumerate(right):
        masks[char] = masks.get(char, 0) | (1 << index)
    state = 0
    for char in left:
        combined = state | masks.get(char, 0)
        state = combined & ~(combined - ((state << 1) | 1))
    return 2 * state.bit_count() / (len(left) + len(right))


def _rows(paragraphs):
    if not isinstance(paragraphs, list):
        raise ValueError("mirror_paragraphs_not_list")
    rows, seen = [], set()
    for source in paragraphs:
        if not isinstance(source, dict):
            raise ValueError("invalid_mirror_paragraph")
        doc = source.get("raw_doc_id", source.get("doc_id"))
        para = source.get("raw_para_id", source.get("para_id"))
        text = source.get("text")
        if (not isinstance(doc, str) or not doc or not isinstance(para, str)
                or not para or not isinstance(text, str) or not text):
            raise ValueError("invalid_mirror_paragraph")
        key = (doc, para)
        if key in seen:
            raise ValueError("duplicate_mirror_location")
        seen.add(key)
        row = {"doc_id": doc, "para_id": para, "text": text}
        internal = source.get("paragraph_id", source.get("id"))
        if internal is not None:
            if not isinstance(internal, str) or not internal:
                raise ValueError("invalid_mirror_internal_id")
            row["paragraph_id"] = internal
        rows.append(row)
    return sorted(rows, key=lambda row: (row["doc_id"], row["para_id"]))


def _identity(row):
    return {key: row[key] for key in ("doc_id", "para_id", "paragraph_id") if key in row}


def _cross(left, right, only):
    return any(a["doc_id"] != b["doc_id"] for a in left for b in right) if only else True


def build_paragraph_mirror_index(paragraphs, *, threshold=0.9, cross_document_only=True):
    """Return a deterministic source-bound index without reading annotations.

    ``exact_groups`` keep compact raw-text buckets; ``similar_edges`` link
    distinct buckets. Components are transitive navigation groups, never a
    claim that all member pairs meet the threshold or share one provenance.
    """
    bound = _threshold(threshold)
    if type(cross_document_only) is not bool:
        raise ValueError("invalid_mirror_cross_document_policy")
    rows = _rows(paragraphs)
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["text"]].append(_identity(row))
    texts = sorted(buckets, key=lambda text: (len(text), text))
    text_ids = {text: sha256(text.encode("utf-8")).hexdigest() for text in texts}
    # Cryptographic hashes are IDs, never an equality substitute.
    if len(set(text_ids.values())) != len(text_ids):
        raise ValueError("mirror_text_hash_collision")
    exact = [{"text_sha256": text_ids[text], "codepoints": len(text), "members": buckets[text]}
        for text in texts if len(buckets[text]) > 1 and _cross(buckets[text], buckets[text], cross_document_only)]
    histograms = {text: Counter(text) for text in texts}
    edges = []
    for index, left in enumerate(texts):
        for right in texts[index + 1:]:
            size = len(left) + len(right)
            # Proven upper bounds avoid expensive LCS without false negatives.
            if 2 * len(left) * bound.denominator < size * bound.numerator:
                break
            if not _cross(buckets[left], buckets[right], cross_document_only):
                continue
            overlap = sum((histograms[left] & histograms[right]).values())
            if 2 * overlap * bound.denominator < size * bound.numerator:
                continue
            score = character_similarity(left, right)
            if score >= float(bound):
                edges.append({"left_text_sha256": text_ids[left], "right_text_sha256": text_ids[right], "similarity": score})
    members = {text_ids[text]: {"codepoints": len(text), "members": buckets[text]} for text in texts}
    parents = {identity: identity for identity in members}
    def root(identity):
        while parents[identity] != identity:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity
    touched = {row["text_sha256"] for row in exact}
    for edge in edges:
        left, right = edge["left_text_sha256"], edge["right_text_sha256"]
        touched.update((left, right))
        a, b = sorted((root(left), root(right)))
        parents[b] = a
    components = defaultdict(list)
    for identity in sorted(touched):
        components[root(identity)].append(identity)
    body = {"version": VERSION, "similarity_metric": SIMILARITY, "threshold": float(bound),
        "cross_document_only": cross_document_only,
        "source_manifest": [{**_identity(row), "text_sha256": text_ids[row["text"]], "codepoints": len(row["text"])} for row in rows],
        "text_buckets": members, "exact_groups": exact, "similar_edges": edges,
        "components": [{"group_id": "PM" + _digest(ids), "text_sha256s": ids,
            "meaning": "transitive_candidate_component_not_pairwise_equivalence_or_independence"}
            for _, ids in sorted(components.items())],
        "evidence_created_or_deleted": False, "independent_source_votes_added": 0,
        "meaning": "candidate_source_locations_only_each_requires_own_literal_fact_and_material_verification"}
    return {**body, "index_sha256": _digest(body)}


def expand_paragraph_mirror_candidates(index, anchors):
    """Expand raw anchors through exact buckets and direct threshold edges only.

    No gold, fact text, source-group count, quote or prior selected label is read.
    Original anchors stay available to callers; this result contains additions
    only and never replaces or deduplicates the existing evidence list.
    """
    if (not isinstance(index, dict) or index.get("version") != VERSION
            or index.get("index_sha256") != _digest({k: v for k, v in index.items() if k != "index_sha256"})):
        raise ValueError("mirror_index_binding")
    if not isinstance(anchors, list):
        raise ValueError("mirror_anchors_not_list")
    manifest = {(row["doc_id"], row["para_id"]): row for row in index["source_manifest"]}
    selected = set()
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise ValueError("invalid_mirror_anchor")
        key = (anchor.get("raw_doc_id", anchor.get("doc_id")), anchor.get("raw_para_id", anchor.get("para_id")))
        if key not in manifest or key in selected:
            raise ValueError("unknown_or_duplicate_mirror_anchor")
        selected.add(key)
    linked = defaultdict(list)
    for edge in index["similar_edges"]:
        a, b = edge["left_text_sha256"], edge["right_text_sha256"]
        linked[a].append((b, edge["similarity"]))
        linked[b].append((a, edge["similarity"]))
    output = {}
    for key in sorted(selected):
        source = manifest[key]
        text_id = source["text_sha256"]
        for neighbor, score in [(text_id, 1.0), *linked[text_id]]:
            for candidate in index["text_buckets"][neighbor]["members"]:
                target = (candidate["doc_id"], candidate["para_id"])
                if target in selected or (index["cross_document_only"] and target[0] == key[0]):
                    continue
                entry = output.setdefault(target, {**candidate, "text_sha256": neighbor,
                    "requires_independent_fact_verification": True, "candidate_only": True, "origins": []})
                entry["origins"].append({"doc_id": key[0], "para_id": key[1],
                    "match": "exact" if neighbor == text_id else "similar", "similarity": score})
    return [output[key] for key in sorted(output)]


def find_literal_fragment_mirror_candidates(paragraphs, fragments, *, min_codepoints=12):
    """Find exact shared source fragments despite different enclosing paragraphs.

    The caller supplies literal fragments from its source context, not a model
    paraphrase. This is a different rule from whole-paragraph similarity >=.9.
    All repeated occurrences keep their explicit half-open codepoint spans.
    Results are candidate locations, never a claim about equivalent facts,
    time/attribution, source independence or official reference collection.
    """
    if type(min_codepoints) is not int or min_codepoints < 1:
        raise ValueError("invalid_mirror_fragment_minimum")
    if (not isinstance(fragments, list) or any(not isinstance(s, str) or len(s) < min_codepoints for s in fragments)
            or len(fragments) != len(set(fragments))):
        raise ValueError("invalid_mirror_fragments")
    rows = _rows(paragraphs)
    found = []
    for fragment in fragments:
        occurrences = []
        for row in rows:
            offset = 0
            while (position := row["text"].find(fragment, offset)) >= 0:
                occurrences.append({**_identity(row), "start_cp": position,
                    "end_cp": position + len(fragment), "quote": fragment})
                offset = position + 1
        found.append({"fragment": fragment, "fragment_sha256": sha256(fragment.encode("utf-8")).hexdigest(),
            "occurrences": occurrences,
            "cross_document_mirror": len({row["doc_id"] for row in occurrences}) > 1})
    body = {"version": VERSION, "match_rule": "exact_literal_fragment_not_whole_paragraph_similarity",
        "min_codepoints": min_codepoints,
        "source_sha256": _digest([{**_identity(row), "text_sha256": sha256(row["text"].encode("utf-8")).hexdigest()} for row in rows]),
        "fragments": found, "candidate_only": True, "requires_independent_fact_verification": True,
        "evidence_created_or_deleted": False, "independent_source_votes_added": 0}
    return {**body, "index_sha256": _digest(body)}
