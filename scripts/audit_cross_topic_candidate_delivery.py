"""Zero-provider candidate and literal delivery audit, never final evidence P/R.

`generate` reads raw-only source projections and CLOSED cached title rankings.
`score` is a separate offline process and may read references only after closure.
No generated candidates, scores, or reference labels enter the runtime database.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICY = {
    "version": "raw-all-window-bm25-closed-title-family-mirror-audit-v1",
    "primary_variant": "three_family",
    "families": ["bm25_body", "bm25_title_body", "closed_title_bge"],
    "family_weights": [1 / 3, 1 / 3, 1 / 3],
    "bm25": {"library": "rank_bm25.BM25Okapi", "k1": 1.5, "b": 0.75, "epsilon": 0.25},
    "window_cp": 320, "overlap_cp": 80,
    "query_policy": "complete-question-plus-literal-focus-context_lexical_rank.literal_queries",
    "parent_aggregation": "max-window-within-same-query-then-query-rank",
    "within_family": "mean-RRF60-over-query-parent-ranks",
    "between_families": "equal-weight-family-contribution-no-logit-comparison",
    "mirror_group": "exact-full-paragraph-UTF8-SHA256-only-with-all-original-identities",
    "mirror_group_score": "best-member-score;members-are-candidates-not-evidence",
    "decision_threshold": None,
    "rank_curve": [20, 40, 60, 100, 200],
    "input_proxy_budgets": [6000, 10000, 16000],
    "budget_policy": "ordered-whole-original-paragraph-groups;stop-before-first-over-budget",
    "model_token_proxy": "cl100k_base;not-provider-usage-or-whole-episode-budget",
    "reference_allowed_during_generation": False,
    "final_selection_or_relation_predictions": False,
}
FORBIDDEN = {".env", "evidence_samples.json", "conclusions.json"}
ATTEMPTS: list[str] = []


def digest(data):
    if not isinstance(data, bytes):
        data = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(data).hexdigest()


def read(path, expected=None):
    path = Path(path).resolve()
    data = path.read_bytes()
    if expected and digest(data) != expected:
        raise ValueError("source_hash_changed:" + path.name)
    return json.loads(data.decode("utf-8-sig")), {"file": str(path), "sha256": digest(data)}


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def parents(raw):
    if set(raw) != {"documents", "questions", "title"}:
        raise ValueError("raw_source_only_required")
    result, seen = [], set()
    for document in raw["documents"]:
        for paragraph in document["paragraphs"]:
            start, end = paragraph["char_start"], paragraph["char_end"]
            key = document["doc_id"], paragraph["para_id"]
            if key in seen or not 0 <= start < end <= len(document["content"]):
                raise ValueError("original_paragraph_identity_or_range_changed")
            seen.add(key)
            text = document["content"][start:end]
            if text != paragraph["text"]:
                raise ValueError("original_paragraph_bytes_changed")
            result.append({"doc_id": key[0], "para_id": key[1], "title": document["title"],
                "start_cp": start, "end_cp": end, "text": text,
                "text_sha256": sha256(text.encode()).hexdigest()})
    return result


def windows(text):
    if len(text) <= 320:
        return [text]
    starts = list(range(0, len(text) - 320 + 1, 240))
    if starts[-1] != len(text) - 320:
        starts.append(len(text) - 320)
    return [text[start:start + 320] for start in starts]


def lexical_indices(rows):
    from app.domain.retrieval import _tokens
    from rank_bm25 import BM25Okapi
    indices, bodies, titles = [], [], []
    for index, row in enumerate(rows):
        for window in windows(row["text"]):
            indices.append(index)
            bodies.append(_tokens(window))
            titles.append(_tokens(row["title"] + "\n" + window))
    options = POLICY["bm25"]
    models = [BM25Okapi(values, k1=options["k1"], b=options["b"], epsilon=options["epsilon"])
              for values in (bodies, titles)]
    return indices, (bodies, titles), models


def lexical_rankings(question, rows, index):
    from app.domain.context_lexical_rank import literal_queries
    from app.domain.retrieval import _tokens
    windows_to_parent, corpora, models = index
    queries = literal_queries(question)
    families = []
    for corpus, model in zip(corpora, models):
        total = [0.0] * len(rows)
        vocab = [set(tokens) for tokens in corpus]
        for query in queries:
            tokens = _tokens(query["text"])
            token_set = set(tokens)
            scores = model.get_scores(tokens).tolist()
            best = {}
            for ix, score in enumerate(scores):
                if not math.isfinite(score):
                    raise ValueError("finite_local_score_required")
                if token_set.intersection(vocab[ix]):
                    parent = windows_to_parent[ix]
                    best[parent] = max(best.get(parent, -math.inf), score)
            ranked = sorted(best, key=lambda ix: (-best[ix], ix))
            for position, ix in enumerate(ranked, 1):
                total[ix] += 1 / (60 + position) / len(queries)
        families.append(total)
    return queries, families


def groups_for_order(rows, scores):
    by_hash = defaultdict(list)
    for index, row in enumerate(rows):
        by_hash[row["text_sha256"]].append(index)
    grouped = []
    for key, members in by_hash.items():
        score = max(scores[index] for index in members)
        if score <= 0:
            continue
        ranked = sorted(members, key=lambda index: (-scores[index], index))
        # All exact copies are candidate aliases, including unranked mirrors.
        grouped.append({"text_sha256": key, "score": score,
            "representative_index": ranked[0], "member_indices": members,
            "source_member_count": len(members), "quote_cp": len(rows[members[0]]["text"])})
    return sorted(grouped, key=lambda group: (-group["score"], group["representative_index"]))


def generate(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "policy-before-data.json", POLICY)
    guards = [{"file": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).read_bytes())}]
    lexical_source = ROOT / "app/domain/context_lexical_rank.py"
    guards.append({"file": str(lexical_source), "sha256": digest(lexical_source.read_bytes())})
    pool, guard = read(ROOT / ".runtime/v22-channel-pool-r2/fixed-candidates.json")
    guards.append(guard)
    freeze, guard = read(ROOT / ".runtime/title-aware-rerank-r1/candidate-freeze.json",
        "3d5da6cdcf3f456bf4715016a1738e12bca103940f6e812c434727ce1946de4f")
    guards.append(guard)
    if freeze.get("generation_complete") is not True or freeze.get("reference_reads") != [] or freeze.get("paid_api_calls") != 0:
        raise ValueError("closed_raw_only_cached_rankings_required")
    ranking, guard = read(ROOT / ".runtime/title-aware-rerank-r1/all-parent-ranking.json", freeze["all_parent_ranking_sha256"])
    guards.append(guard)
    rankings = {(case["topic_id"], case["question_id"]): case for case in ranking["cases"]}
    baseline, guard = read(ROOT / "docs/reports/v20-semantic/fixed-candidates.json")
    guards.append(guard)
    baselines = {(case["topic_id"], case["question_id"]): case for case in baseline["cases"]}
    sources, cases = [], []
    for source in pool["inputs"]:
        raw, guard = read(source["path"], source["sha256"])
        guards.append(guard)
        rows = parents(raw)
        indexed = lexical_indices(rows)
        topic_id = source["topic_id"]
        lookup = {(row["doc_id"], row["para_id"]): index for index, row in enumerate(rows)}
        sources.append({"topic_id": topic_id, "raw": guard,
            "paragraphs": [{key: value for key, value in row.items() if key != "text"} for row in rows]})
        for question in raw["questions"]:
            qid = question["question_id"]
            cached = rankings[(topic_id, qid)]
            if cached["question"] != question["question"] or cached["title"] != raw["title"]:
                raise ValueError("complete_original_question_changed")
            queries, families = lexical_rankings(question["question"], rows, indexed)
            bge = [0.0] * len(rows)
            for position, candidate in enumerate(cached["candidates"], 1):
                key = candidate["doc_id"], candidate["para_id"]
                ix = lookup[key]
                if candidate["source_text_sha256_utf8"] != rows[ix]["text_sha256"]:
                    raise ValueError("cached_original_text_changed")
                bge[ix] = 1 / (60 + position)
            totals = {
                "three_family": [(a + b + c) / 3 for a, b, c in zip(*families, bge)],
                "bm25_two_family": [(a + b) / 2 for a, b in zip(*families)],
                "closed_title_bge": bge,
            }
            variants = {}
            for variant, scores in totals.items():
                order = sorted((ix for ix in range(len(rows)) if scores[ix] > 0), key=lambda ix: (-scores[ix], ix))
                variants[variant] = {"source_order": order, "mirror_groups": groups_for_order(rows, scores)}
            old = next(item for item in baselines[(topic_id, qid)]["outputs"] if item["variant"] == "hybrid40")
            old_indices = [lookup[(row["raw_doc_id"], row["raw_para_id"])] for row in old["candidates"]]
            cases.append({"topic_id": topic_id, "question_id": qid, "question": question["question"],
                "title": raw["title"], "queries": queries, "all_original_paragraphs": len(rows),
                "closed_title_bge_parent_scope": len(cached["candidates"]),
                "baseline_hybrid40_indices": old_indices, "variants": variants})
        print(json.dumps({"topic": topic_id, "closed_cases": len(cases), "provider_calls": 0}), flush=True)
    fixed = {"format": "raw-candidate-delivery-catalog-v1", "policy": POLICY,
        "sources": sources, "cases": cases, "provider_calls": 0, "new_local_inference_calls": 0,
        "reference_access": [], "final_evidence_or_relation_predictions": False}
    save(output / "fixed-candidates.json", fixed)
    for guard in guards:
        if digest(Path(guard["file"]).read_bytes()) != guard["sha256"]:
            raise ValueError("source_changed_during_generation")
    closure = {"status": "CLOSED", "candidate_sha256": digest((output / "fixed-candidates.json").read_bytes()),
        "policy_sha256": digest((output / "policy-before-data.json").read_bytes()),
        "source_guards": guards, "cases": len(cases), "topics": len(sources),
        "provider_calls": 0, "new_local_inference_calls": 0, "reference_access": []}
    save(output / "closed.json", closure)
    print(json.dumps(closure), flush=True)


def selected_rows(indices, rows):
    return [rows[index] for index in indices]


def coverage(indices, rows, references, documents):
    submitted = {(rows[index]["doc_id"], rows[index]["para_id"]) for index in indices}
    gold = {(item.get("doc_id"), item.get("para_id")) for item in references}
    hits = gold.intersection(submitted)
    by_key = {(row["doc_id"], row["para_id"]): row for row in rows}
    literal, anchored_literal = set(), set()
    issues = []
    for ref in references:
        key = ref.get("doc_id"), ref.get("para_id")
        row = by_key.get(key)
        if row is None or not isinstance(ref.get("quote"), str):
            issues.append({"doc_id": key[0], "para_id": key[1], "reason": "unknown-source-or-missing-quote"})
            continue
        content = documents[row["doc_id"]]["content"]
        quote = ref["quote"]
        parent = content[row["start_cp"]:row["end_cp"]]
        if quote and quote in parent:
            literal.add(key)
            if key in submitted:
                anchored_literal.add(key)
        else:
            issues.append({"doc_id": key[0], "para_id": key[1], "reason": "reference-quote-not-contained-in-declared-single-parent",
                "declared_parent_cp": len(parent), "reference_quote_cp": len(quote),
                "found_in_declared_document": bool(quote and quote in content)})
    return {"candidate_sources": len(submitted), "reference_sources": len(gold), "source_id_hits": len(hits),
        "candidate_source_id_coverage": len(hits) / len(gold) if gold else None,
        "full_parent_reference_quote_deliverable": len(anchored_literal),
        "reference_quote_single_parent_containable": len(literal), "reference_issues": issues,
        "missed_sources": [{"doc_id": key[0], "para_id": key[1]} for key in sorted(gold - submitted)]}


def score(folder, sample_root, score_output):
    from scripts.offline_reference_tools import load_reference_tools
    import tiktoken
    folder = Path(folder).resolve()
    closed, guard = read(folder / "closed.json")
    if closed.get("status") != "CLOSED" or closed.get("cases") != 18 or closed.get("provider_calls") != 0 or closed.get("reference_access") != []:
        raise ValueError("closed_full_eighteen_reference_free_generation_required")
    fixed, _ = read(folder / "fixed-candidates.json", closed["candidate_sha256"])
    for source in closed["source_guards"]:
        if digest(Path(source["file"]).read_bytes()) != source["sha256"]:
            raise ValueError("source_closure_changed")
    mapper, engine, tool_guards = load_reference_tools()
    encoder = tiktoken.get_encoding("cl100k_base")
    catalogs = {source["topic_id"]: source for source in fixed["sources"]}
    raws, references = {}, {}
    for topic_id, source in catalogs.items():
        raw, _ = read(source["raw"]["file"], source["raw"]["sha256"])
        raws[topic_id] = raw
        topic_dir = Path(sample_root) / raw["title"]
        evidence, _ = read(topic_dir / "evidence_samples.json")
        conclusions, _ = read(topic_dir / "conclusions.json")
        references[topic_id] = evidence, {row["question_id"]: row for row in conclusions}
    results = []
    for case in fixed["cases"]:
        topic, qid = case["topic_id"], case["question_id"]
        rows = catalogs[topic]["paragraphs"]
        documents = {row["doc_id"]: row for row in raws[topic]["documents"]}
        evidence, conclusion_by_id = references[topic]
        conclusion = conclusion_by_id[qid]
        union, issues = mapper.question_union_reference(evidence, conclusion)
        categories = {}
        for side in ("supporting_facts", "conflicting_facts"):
            categories[side] = engine._relation_rows(conclusion[side], evidence, submitted=False,
                issues=issues, context=[topic, qid, side])
        # The historical union mapper intentionally returns identities only.
        # Keep its exact denominator but restore original reference quotes for
        # this offline literal-delivery diagnostic, never for generation.
        by_identity = defaultdict(list)
        for relation in (*categories["supporting_facts"], *categories["conflicting_facts"]):
            by_identity[(relation.get("doc_id"), relation.get("para_id"))].append(relation)
        union_with_quotes = []
        for identity in union:
            related = by_identity.get((identity.get("doc_id"), identity.get("para_id")))
            union_with_quotes.extend(related or [identity])
        categories["union"] = union_with_quotes
        def metrics(indices):
            return {label: coverage(indices, rows, refs, documents) for label, refs in categories.items()}
        before = metrics(case["baseline_hybrid40_indices"])
        variants = {}
        for label, variant in case["variants"].items():
            source_curve = {str(k): metrics(variant["source_order"][:k]) for k in POLICY["rank_curve"]}
            group_curve, budget_curve = {}, {}
            payloads, group_tokens = [], []
            for group in variant["mirror_groups"]:
                representative = rows[group["representative_index"]]
                content = documents[representative["doc_id"]]["content"]
                payload = {"quote": content[representative["start_cp"]:representative["end_cp"]],
                    "aliases": [[rows[ix]["doc_id"], rows[ix]["para_id"]] for ix in group["member_indices"]]}
                payloads.append(payload)
                group_tokens.append(len(encoder.encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))))
            for k in POLICY["rank_curve"]:
                indices = [ix for group in variant["mirror_groups"][:k] for ix in group["member_indices"]]
                group_curve[str(k)] = {"metrics": metrics(indices), "unique_full_quotes": min(k, len(payloads)),
                    "literal_payload_CP": sum(len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) for payload in payloads[:k]),
                    "cl100k_payload_token_proxy": sum(group_tokens[:k])}
            overhead = len(encoder.encode(json.dumps({"question": case["question"], "sources": []}, ensure_ascii=False)))
            for budget in POLICY["input_proxy_budgets"]:
                used, count = overhead, 0
                for cost in group_tokens:
                    if used + cost > budget:
                        break
                    used += cost
                    count += 1
                indices = [ix for group in variant["mirror_groups"][:count] for ix in group["member_indices"]]
                budget_curve[str(budget)] = {"metrics": metrics(indices), "unique_full_quotes": count,
                    "cl100k_payload_token_proxy": used, "not_actual_model_input_or_usage": True}
            variants[label] = {"same_source_count_curve": source_curve,
                "same_full_quote_group_curve": group_curve, "whole_group_budget_prefix_curve": budget_curve}
        results.append({"topic_id": topic, "question_id": qid, "all_original_paragraphs": case["all_original_paragraphs"],
            "closed_title_bge_parent_scope": case["closed_title_bge_parent_scope"],
            "baseline_hybrid40": before, "variants": variants, "reference_mapping_issues": issues})
        print(json.dumps({"scored": topic + "/" + qid, "provider_calls": 0}), flush=True)
    report = {"format": "cross-topic-candidate-delivery-score-v1", "cases": results,
        "generation_closure": guard, "reference_reads_only_after_generation_closure": True,
        "provider_calls": 0, "provider_tokens": 0, "new_local_inference_calls": 0,
        "final_evidence_P_R_or_official_score": False,
        "literal_delivery_is_capability_not_actual_model_reading": True, "score_tool_guards": tool_guards}
    save(score_output, report)


def audit(event, args):
    if event.startswith("socket."):
        ATTEMPTS.append(event)
        raise PermissionError("offline_network_forbidden")
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        name = Path(os.fsdecode(args[0])).name.lower()
        if name == ".env" or (sys.argv[1:2] == ["generate"] and name in FORBIDDEN):
            ATTEMPTS.append(name)
            raise PermissionError("offline_generation_reference_or_credential_read_forbidden")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "score"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-root")
    parser.add_argument("--generation")
    args = parser.parse_args()
    sys.addaudithook(audit)
    os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if args.mode == "generate":
        generate(args.output)
    else:
        score(args.generation, args.sample_root, args.output)
