"""Bounded, raw-only paragraph recall; scores are relevance hints, never evidence.

Document titles cannot buy a candidate slot without a body match. Query facets
come only from the original question or explicitly supplied search groups. No
answer, reference ID, provider, model or network is used here.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import math
import re

from app.domain.retrieval import _tokens

VERSION = "question-body-multigranular-v20"
VARIANTS = ("body_multi", "balanced", "feedback", "window_multi", "window_balanced", "window_feedback")
_CHUNK = re.compile(r"[\u3400-\u9fff]+|[a-z]+[a-z0-9_-]*|\d+(?:\.\d+)?", re.I)
_SPLIT = re.compile(r"[，,。；;？！?!：:]|还是|抑或|而非|到底|是否|能否|以及")
_GENERIC = {"应当", "应该", "认定", "称", "所称", "到底", "是否", "能否", "为准", "哪一", "哪方", "说法", "成立", "支持", "属实", "问题", "证据"}


@lru_cache(maxsize=20000)
def _features(text, kind):
    text = text.lower()
    if kind == "word":
        return tuple(t for t in _tokens(text) if len(t) > 1 and t not in _GENERIC)
    tokens = []
    for chunk in _CHUNK.findall(text):
        if re.fullmatch(r"[\u3400-\u9fff]+", chunk):
            tokens.extend(chunk[i:i+2] for i in range(len(chunk)-1))
        else:
            tokens.append(chunk)
    return tuple(tokens)


def question_query_plan(question, query_groups=None):
    """Preserve full conditions while exposing literal, complementary facets."""
    if isinstance(question, dict):
        question = question.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question_recall_question_invalid")
    if query_groups is not None and (not isinstance(query_groups, (list, tuple)) or len(query_groups) > 3):
        raise ValueError("question_recall_groups_invalid")
    facets = [("question", question)]
    for part in _SPLIT.split(question):
        part = part.strip(" \n\t‘’“”「」？?")
        # A contrast such as "不愿走" may have just one jieba noun token;
        # character features still retain the literal predicate. Generic
        # interrogative tails alone do not provide a separate search signal.
        if (not re.match(r"^(?:哪|以哪|这一说法)", part)
                and (len(set(_features(part, "word"))) >= 2 or len(set(_features(part, "bigram"))) >= 3)):
            if len(facets) < 9:
                facets.append(("literal_clause", part))
    for group in query_groups or ():
        if not isinstance(group, dict) or not isinstance(group.get("terms"), list):
            raise ValueError("question_recall_groups_invalid")
        terms = group["terms"]
        if len(terms) > 8 or any(not isinstance(t, str) or not t.strip() or len(t) > 160 for t in terms):
            raise ValueError("question_recall_groups_invalid")
        facets.extend(("provided_query", term) for term in terms)
    result, seen = [], set()
    for origin, text in facets:
        # Negation/word order may disappear in word tokenization. Exact query
        # reuse is redundant, but opposite literal predicates are distinct.
        identity = (tuple(sorted(set(_features(text, "word")))), tuple(sorted(set(_features(text, "bigram")))))
        if not any(identity) or identity in seen:
            continue
        seen.add(identity)
        result.append({"facet_id": "F" + str(len(result)+1), "origin": origin,
                       "role": "complete_question" if origin == "question" else "literal_subcondition" if origin == "literal_clause" else "provided_expansion",
                       "text": text, "full_original_question": question})
    # At most one full question + eight raw clauses + 3*8 supplied terms.
    # Do not drop a final official/language query merely because earlier groups
    # used their allowances first.
    return {"version": VERSION, "original_question": question, "facets": result}


@lru_cache(maxsize=20000)
def _windows(text):
    # Source identity remains the entire original paragraph. Overlap prevents
    # a local chunk boundary from hiding a literal condition or time phrase.
    return tuple(text[start:start+240] for start in range(0, max(1, len(text)-80), 160))


def _bm25(rows, query, kind, windowed=False):
    counts = [[Counter(_features(t, kind)) for t in (_windows(row["text"]) if windowed else (row["text"],))] for row in rows]
    # IDF counts source paragraphs, not chunks: one long source cannot inflate
    # document frequency or vote repeatedly through several overlapping windows.
    df = Counter(term for chunks in counts for term in {t for terms in chunks for t in terms})
    lengths = [[sum(c.values()) for c in chunks] for chunks in counts]
    average = sum(sum(x) for x in lengths)/sum(map(len,lengths)) if rows else 0
    terms = set(_features(query, kind))
    idfs = {t: math.log(1 + (len(rows)-df[t]+.5)/(df[t]+.5)) for t in terms if df[t]}
    normalizer = math.sqrt(sum(idfs.values())) or 1
    scores = []
    for index, chunks in enumerate(counts):
        if not average:
            continue
        score = max(sum(idfs[t] * words[t] * 2.5 / (words[t] + 1.5*(.25+.75*length/average))
                    for t in idfs.keys() & words.keys()) / normalizer for words,length in zip(chunks,lengths[index]))
        if score > 0:
            # Site-navigation headings cannot dominate full source sentences.
            score *= min(1.0, max(.15, len(rows[index]["text"].strip())/35))
            scores.append((score, index))
    return sorted(scores, key=lambda row: (-row[0], row[1]))


def _source_order(row, index):
    number = re.search(r"#p([0-9]+)$", row.get("raw_para_id", ""))
    return (0, int(number.group(1))) if number else (1, index)


def recall_question_candidates(question, paragraphs, *, limit=40, query_groups=None,
                               excluded_ids=(), variant="window_multi"):
    """Return <= limit unique raw source IDs and inspectable deterministic hints.

    ``balanced`` reserves up to 15% of slots for immediate source context and
    caps one document at four slots during its first pass. Its second pass may
    fill remaining slots. Near/exact mirrors retain their own raw source IDs.
    All variants are development retrieval strategies, not a semantic filter.
    """
    if type(limit) is not int or not 1 <= limit <= 60 or variant not in VARIANTS or not isinstance(paragraphs, list):
        raise ValueError("question_recall_parameters_invalid")
    if not isinstance(excluded_ids, (list, tuple, set, frozenset)) or any(not isinstance(x, str) for x in excluded_ids):
        raise ValueError("question_recall_exclusions_invalid")
    plan = question_query_plan(question, query_groups)
    windowed = variant.startswith("window_")
    diverse = variant in {"balanced", "feedback", "window_balanced", "window_feedback"}
    seen, source_keys, rows, original_order = set(), set(), [], {}
    for index, row in enumerate(paragraphs):
        if (not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k]
                for k in ("id", "raw_doc_id", "raw_para_id")) or not isinstance(row.get("text"), str)):
            raise ValueError("question_recall_source_invalid")
        key = row["raw_doc_id"], row["raw_para_id"]
        if row["id"] in seen or key in source_keys:
            raise ValueError("question_recall_duplicate_source")
        seen.add(row["id"]); source_keys.add(key); original_order[row["id"]] = index
        if row["id"] not in excluded_ids and row["text"].strip():
            rows.append(row)
    if not set(excluded_ids) <= seen:
        raise ValueError("question_recall_unknown_exclusion")
    fusion, signals = defaultdict(float), defaultdict(list)
    queries = plan["facets"]
    def add_queries(searches, weight=1.0):
        for qi, query in enumerate(searches):
            for kind, kind_weight in (("word", 1.0), ("bigram", .65)):
                ranked = _bm25(rows, query["text"], kind, windowed)
                for rank, (score, index) in enumerate(ranked[:max(limit*3, 60)], 1):
                    w = weight * kind_weight * (1.35 if query["origin"] == "question" else 1.0)
                    fusion[index] += w/(30+rank)
                    signals[index].append({"origin": query["origin"], "facet": qi,
                        "granularity": kind, "rank": rank, "bm25_score": score})
    add_queries(queries)
    if variant in {"feedback", "window_feedback"} and fusion:
        top = sorted(fusion, key=lambda i: (-fusion[i], i))[:5]
        query_terms = set(_features(plan["original_question"], "word"))
        all_df = Counter(t for row in rows for t in set(_features(row["text"], "word")))
        top_df = Counter(t for i in top for t in set(_features(rows[i]["text"], "word")))
        expansion = sorted((t for t in top_df if t not in query_terms and top_df[t] >= 2 and len(t) >= 2),
            key=lambda t: (-top_df[t] * math.log(1+(len(rows)-all_df[t]+.5)/(all_df[t]+.5)), t))[:8]
        if expansion:
            feedback = {"origin": "raw_top5_feedback", "text": " ".join(expansion),
                        "full_original_question": plan["original_question"]}
            plan["feedback"] = feedback
            add_queries([feedback], weight=.45)
    # Titles are a bounded tie preference and never produce a body-zero hit.
    qwords = set(_features(plan["original_question"], "word"))
    for i in fusion:
        overlap = qwords & set(_features(rows[i].get("title") or "", "word"))
        fusion[i] *= 1 + .08 * (len(overlap)/max(1, len(qwords)))
    order = sorted(fusion, key=lambda i: (-fusion[i], i))
    reserve = min(6, limit//6) if diverse else 0
    main_limit = limit-reserve
    selected, document_counts = [], Counter()
    for i in order:
        if diverse and document_counts[rows[i]["raw_doc_id"]] >= 4:
            continue
        selected.append(i); document_counts[rows[i]["raw_doc_id"]] += 1
        if len(selected) >= main_limit:
            break
    if reserve:
        bydoc = defaultdict(list)
        for i, row in enumerate(rows):
            bydoc[row["raw_doc_id"]].append(i)
        positions = {}
        for doc, indices in bydoc.items():
            indices.sort(key=lambda i: _source_order(rows[i], original_order[rows[i]["id"]]))
            for pos, i in enumerate(indices):
                positions[i] = (indices, pos)
        context = []
        for anchor_rank, i in enumerate(selected[:12]):
            indices, pos = positions[i]
            for distance in (-1, 1):
                if 0 <= pos+distance < len(indices):
                    other = indices[pos+distance]
                    if other not in selected and other in fusion:
                        context.append((-(fusion[other]+fusion[i]*.15), anchor_rank, other, i))
        context.sort()
        for _, _, i, anchor in context:
            if i in selected or document_counts[rows[i]["raw_doc_id"]] >= 5:
                continue
            selected.append(i); document_counts[rows[i]["raw_doc_id"]] += 1
            signals[i].append({"origin": "immediate_source_context", "anchor_id": rows[anchor]["id"]})
            if len(selected) >= limit:
                break
    for restricted in ((True, False) if diverse else (False,)):
        for i in order:
            if len(selected) >= limit:
                break
            if i not in selected and (not restricted or document_counts[rows[i]["raw_doc_id"]] < 5):
                selected.append(i); document_counts[rows[i]["raw_doc_id"]] += 1
    return {"version": VERSION, "variant": variant, "query_plan": plan,
        "source_count": len(paragraphs), "eligible_count": len(rows), "limit": limit,
        "candidates": [{"paragraph_id": rows[i]["id"], "raw_doc_id": rows[i]["raw_doc_id"],
            "raw_para_id": rows[i]["raw_para_id"], "score": fusion[i], "signals": signals[i]}
            for i in selected], "model_calls": 0, "reference_inputs": False}
