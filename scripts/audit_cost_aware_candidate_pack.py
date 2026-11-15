"""One fixed length-aware navigation experiment. No model or final evidence.

generate: raw-only, CLOSED title-BGE ranking, no reference IO/network.
score: separate process; exact source identities/literal quote coverage only.
"""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_cross_topic_candidate_delivery import read, save

POLICY = {"version": "closed-title-bge-rrf60-over-sqrt-whole-parent-content-cost-v1",
    "input_content_token_proxy_budget": 15000, "maximum_original_paragraphs": 60,
    "score": "(1/(60+one_based_original_title_BGE_rank))/sqrt(single_parent_payload_tokens)",
    "payload_keys": ["doc_id", "para_id", "title", "quote"],
    "payload_quote": "complete-original-parent", "per_record_separator_proxy_tokens": 2,
    "cost_tokenizer": "tiktoken.cl100k_base;not-provider-usage-or-whole-episode-budget",
    "tie_breaker": "original-title-BGE-rank", "cannot_fit": "unprovided_budget;continue-next-ranked-parent",
    "threshold": None, "new_model_calls": 0, "reference_access_during_generation": [],
    "unprovided_means_irrelevant": False, "final_evidence_or_relation_created": False}


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def pack(rows, *, budget, limit=60, stop_at_first_overflow=False):
    """Rows carry immutable source indices and nonnegative full-payload costs."""
    selected, unprovided, used = [], [], 0
    stopped = False
    for row in rows:
        if stopped:
            reason = "unprovided_after_prefix_budget_stop"
        elif len(selected) >= limit:
            reason = "unprovided_candidate_limit"
        elif used + row["payload_tokens"] > budget:
            reason = "unprovided_budget"
            if stop_at_first_overflow:
                stopped = True
        else:
            selected.append(row["paragraph_index"])
            used += row["payload_tokens"]
            continue
        unprovided.append({"paragraph_index": row["paragraph_index"], "reason": reason})
    return {"selected_indices": selected, "unprovided": unprovided, "content_payload_proxy_tokens": used}


def generate(output):
    import tiktoken
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "policy-before-data.json", POLICY)
    folder = ROOT / ".runtime/candidate-delivery-audit-20261021-r4"
    closed, guard = read(folder / "closed.json")
    fixed, fixed_guard = read(folder / "fixed-candidates.json", closed["candidate_sha256"])
    guards = [guard, fixed_guard, {"file": str(Path(__file__).resolve()), "sha256": file_sha(__file__)}]
    if closed["status"] != "CLOSED" or closed["reference_access"] != [] or len(fixed["cases"]) != 18:
        raise ValueError("closed_complete_raw_only_input_required")
    for guard in closed["source_guards"]:
        if file_sha(guard["file"]) != guard["sha256"]:
            raise ValueError("original_source_code_or_raw_changed")
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = lambda value: len(encoder.encode(json.dumps(value, ensure_ascii=False, separators=(",", ":"))))
    raw_by_topic = {}
    catalogs = {source["topic_id"]: source["paragraphs"] for source in fixed["sources"]}
    for source in fixed["sources"]:
        raw, guard = read(source["raw"]["file"], source["raw"]["sha256"])
        raw_by_topic[source["topic_id"]] = raw
        guards.append(guard)
    cases = []
    for case in fixed["cases"]:
        topic = case["topic_id"]
        original = catalogs[topic]
        documents = {doc["doc_id"]: doc for doc in raw_by_topic[topic]["documents"]}
        scored_rows = []
        for position, index in enumerate(case["variants"]["closed_title_bge"]["source_order"], 1):
            row = original[index]
            text = documents[row["doc_id"]]["content"][row["start_cp"]:row["end_cp"]]
            if sha256(text.encode()).hexdigest() != row["text_sha256"]:
                raise ValueError("complete_original_parent_text_changed")
            payload = {key: row[key] for key in ("doc_id", "para_id", "title")}
            payload["quote"] = text
            cost = token_count(payload) + POLICY["per_record_separator_proxy_tokens"]
            scored_rows.append({"paragraph_index": index, "original_rank": position, "payload_tokens": cost,
                "cost_aware_navigation_score": (1 / (60 + position)) / math.sqrt(cost)})
        overhead = token_count({"question": case["question"], "sources": []})
        available = POLICY["input_content_token_proxy_budget"] - overhead
        density_order = sorted(scored_rows, key=lambda row: (-row["cost_aware_navigation_score"], row["original_rank"]))
        before60 = {"selected_indices": [row["paragraph_index"] for row in scored_rows[:60]],
            "content_payload_proxy_tokens": sum(row["payload_tokens"] for row in scored_rows[:60]),
            "unprovided": [{"paragraph_index": row["paragraph_index"], "reason": "unprovided_candidate_limit"} for row in scored_rows[60:]]}
        variants = {"title_BGE_first60": before60,
            "title_BGE_15k_prefix_max60": pack(scored_rows, budget=available, stop_at_first_overflow=True),
            "cost_aware_15k_pack_max60": pack(density_order, budget=available)}
        for variant in variants.values():
            variant["content_input_proxy_tokens"] = variant["content_payload_proxy_tokens"] + overhead
            variant["within_15k_content_input_proxy"] = variant["content_input_proxy_tokens"] <= 15000
        cases.append({"topic_id": topic, "question_id": case["question_id"], "question": case["question"],
            "parent_costs_in_original_title_BGE_order": scored_rows, "density_order_indices": [row["paragraph_index"] for row in density_order],
            "question_overhead_tokens": overhead, "variants": variants})
    artifact = {"format": "closed-title-cost-aware-candidates-v1", "policy": POLICY,
        "raw_catalog_source": fixed_guard, "cases": cases, "source_guards": guards,
        "new_model_calls": 0, "reference_reads": [], "actual_model_reading_verified": False}
    save(output / "fixed-candidates.json", artifact)
    for guard in guards:
        if file_sha(guard["file"]) != guard["sha256"]:
            raise ValueError("source_changed_during_generation")
    save(output / "closed.json", {"status": "CLOSED", "fixed_sha256": file_sha(output / "fixed-candidates.json"),
        "policy_sha256": file_sha(output / "policy-before-data.json"), "reference_reads": [], "new_model_calls": 0,
        "cases": len(cases), "primary_variant": "cost_aware_15k_pack_max60"})
    print(json.dumps({"status": "CLOSED", "cases": len(cases), "new_model_calls": 0,
        "fixed_sha256": file_sha(output / "fixed-candidates.json")}), flush=True)


def score(folder, output):
    from scripts.offline_reference_tools import load_reference_tools
    from scripts.audit_cross_topic_candidate_delivery import coverage
    folder = Path(folder).resolve()
    closed, _ = read(folder / "closed.json")
    if closed["status"] != "CLOSED" or closed["cases"] != 18 or closed["reference_reads"] != []:
        raise ValueError("closed_eighteen_raw_only_candidates_required")
    fixed, _ = read(folder / "fixed-candidates.json", closed["fixed_sha256"])
    for guard in fixed["source_guards"]:
        if file_sha(guard["file"]) != guard["sha256"]:
            raise ValueError("closed_source_changed")
    catalogs, _ = read(fixed["raw_catalog_source"]["file"], fixed["raw_catalog_source"]["sha256"])
    sources = {source["topic_id"]: source for source in catalogs["sources"]}
    raw_by_topic, references = {}, {}
    sample = ROOT.parent / "多智能体可信研判与证据链溯源/样例集-拓尔思-20261012"
    for topic, source in sources.items():
        raw, _ = read(source["raw"]["file"], source["raw"]["sha256"])
        raw_by_topic[topic] = raw
        evidence, _ = read(sample / raw["title"] / "evidence_samples.json")
        conclusions, _ = read(sample / raw["title"] / "conclusions.json")
        references[topic] = evidence, {row["question_id"]: row for row in conclusions}
    mapper, engine, _ = load_reference_tools()
    cases = []
    for case in fixed["cases"]:
        topic, qid = case["topic_id"], case["question_id"]
        evidence, conclusions = references[topic]
        conclusion = conclusions[qid]
        union, issues = mapper.question_union_reference(evidence, conclusion)
        categories = {}
        for side in ("supporting_facts", "conflicting_facts"):
            categories[side] = engine._relation_rows(conclusion[side], evidence, submitted=False, issues=issues, context=[])
        by_key = {}
        for row in (*categories["supporting_facts"], *categories["conflicting_facts"]):
            by_key.setdefault((row.get("doc_id"), row.get("para_id")), []).append(row)
        categories["union"] = [row for identity in union for row in
            by_key.get((identity.get("doc_id"), identity.get("para_id")), [identity])]
        documents = {doc["doc_id"]: doc for doc in raw_by_topic[topic]["documents"]}
        variants = {}
        for name, variant in case["variants"].items():
            variants[name] = {"metrics": {label: coverage(variant["selected_indices"], sources[topic]["paragraphs"], rows, documents)
                for label, rows in categories.items()}, "selected_paragraphs": len(variant["selected_indices"]),
                "content_input_proxy_tokens": variant["content_input_proxy_tokens"],
                "within_15k_content_input_proxy": variant["within_15k_content_input_proxy"],
                "all_unprovided_retained_unknown": True}
        cases.append({"topic_id": topic, "question_id": qid, "variants": variants})
    report = {"format": "closed-cost-aware-source-delivery-score-v1", "cases": cases,
        "candidate_fixed_sha256": closed["fixed_sha256"], "new_model_calls": 0, "new_model_tokens": 0,
        "candidate_source_ID_coverage_not_final_precision_recall": True, "whole_episode_budget_measured": False,
        "actual_model_reading_verified": False, "official_score": False}
    save(output, report)


if __name__ == "__main__":
    def audit(event, args):
        if event.startswith("socket."):
            raise PermissionError("offline_network_forbidden")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            name = Path(os.fsdecode(args[0])).name.lower()
            if name == ".env" or (sys.argv[1] == "generate" and name in {"evidence_samples.json", "conclusions.json"}):
                raise PermissionError("raw_only_generation_reference_or_credentials_forbidden")
    sys.addaudithook(audit)
    if sys.argv[1] == "generate":
        generate(sys.argv[2])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        raise ValueError("generate_or_score_required")
