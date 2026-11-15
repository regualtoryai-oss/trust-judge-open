"""CLOSED original-block mirror expansion; zero inference, separate scoring."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected=None):
    path = Path(path).resolve()
    if expected and sha(path) != expected:
        raise ValueError("original_binding_changed")
    return json.loads(path.read_text("utf-8-sig")), {"file": str(path), "sha256": sha(path)}


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def generate(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    policy = {"version": "closed-literal-source-block-mirror-navigation-v1",
        "method": "exact-case-sensitive-original-block-in-complete-raw-document",
        "parent_aliases": "all-original-parents-which-fully-contain-one-exact-occurrence",
        "shorter_or_partial_or_similar_matches": False, "cross_parent_hits_retained_separately": True,
        "formal_evidence_created": False, "source_block_is_candidate_not_fact": True,
        "reference_reads": [], "new_model_calls": 0}
    save(folder / "policy-before-data.json", policy)
    guards = [{"file": str(Path(__file__).resolve()), "sha256": sha(__file__)}]
    v39 = ROOT / ".runtime/v39-source-alias-feedback-r2"
    receipt, guard = read(v39 / "feedback-final-receipt.json",
        "388079f65d415f781879943b03166db9eb3e7e6fc1fd654432f4aab56489c23d")
    guards.append(guard)
    hints, guard = read(receipt["output_artifact"]["file"], receipt["output_artifact"]["sha256"])
    guards.append(guard)
    previous, guard = read(ROOT / ".runtime/v39-feedback-retrieval-r1/fixed-candidates.json",
        "a23ecfda40d340ece4a0bee1ea0ca45985bc748211b372e68702f0e6e7f186d8")
    guards.append(guard)
    if receipt["status"] != "completed" or previous.get("reference_reads") != [] or hints.get("semantic_verified") is not False:
        raise ValueError("closed_unverified_original_hints_required")
    source = previous["inputs"][0]
    raw, guard = read(source["path"], source["sha256"])
    guards.append(guard)
    case = previous["cases"][0]
    if hints["topic_id"] != case["topic_id"] or hints["question_id"] != case["question_id"]:
        raise ValueError("original_question_identity_changed")
    baseline = case["outputs"][0]["candidates"]
    rows, documents = {}, {doc["doc_id"]: doc for doc in raw["documents"]}
    for doc in raw["documents"]:
        for parent in doc["paragraphs"]:
            key = doc["doc_id"], parent["para_id"]
            start, end = parent["char_start"], parent["char_end"]
            if key in rows or doc["content"][start:end] != parent["text"]:
                raise ValueError("source_original_parent_changed")
            rows[key] = {"doc_id": key[0], "para_id": key[1], "start_cp": start, "end_cp": end,
                "text_sha256": sha256(parent["text"].encode()).hexdigest()}
    groups = []
    for hint in hints["terms"]:
        query, bound = hint["query"], hint["source_span"]
        source_doc = documents[bound["raw_doc_id"]]
        if (hint["original_question"] != case["question"] or hint["semantic_verified"] is not False
            or source_doc["content"][bound["document_start_cp"]:bound["document_end_cp"]] != query
            or sha256(query.encode()).hexdigest() != bound["block_text_sha256"]):
            raise ValueError("source_block_hint_is_not_bound_original_text")
        occurrences, aliases, cross_parent = [], {}, []
        for doc in raw["documents"]:
            offset = 0
            while (offset := doc["content"].find(query, offset)) >= 0:
                end = offset + len(query)
                occurrence = {"doc_id": doc["doc_id"], "start_cp": offset, "end_cp": end}
                containing = [parent for parent in doc["paragraphs"]
                    if parent["char_start"] <= offset < end <= parent["char_end"]]
                occurrences.append(occurrence)
                if not containing:
                    cross_parent.append(occurrence)
                for parent in containing:
                    key = doc["doc_id"], parent["para_id"]
                    aliases[key] = dict(rows[key], original_block_document_span=[offset, end])
                offset += 1
        groups.append({"original_source_ref": hint["source_ref"], "original_unverified_role": hint["role"],
            "quote": query, "quote_sha256": bound["block_text_sha256"], "original_source_span": bound,
            "exact_document_occurrences": occurrences, "parent_aliases": list(aliases.values()),
            "cross_parent_occurrences": cross_parent})
    baseline_rows = [rows[(row["raw_doc_id"], row["raw_para_id"])] for row in baseline]
    expanded = {(row["doc_id"], row["para_id"]): row for row in baseline_rows}
    for group in groups:
        for row in group["parent_aliases"]:
            expanded[(row["doc_id"], row["para_id"])] = rows[(row["doc_id"], row["para_id"])]
    fixed = {"format": "closed-grounded-block-source-navigation-v1", "topic_id": case["topic_id"],
        "question_id": case["question_id"], "question": case["question"], "source": source,
        "baseline_candidates": baseline_rows, "expanded_candidates": list(expanded.values()),
        "grounded_block_groups": groups, "model_calls": 0, "new_local_inference_calls": 0,
        "reference_reads": [], "semantic_verdict": None, "workbench_e2e": False}
    save(folder / "fixed-candidates.json", fixed)
    for guard in guards:
        if sha(guard["file"]) != guard["sha256"]:
            raise ValueError("input_changed_during_generation")
    save(folder / "closed.json", {"status": "CLOSED", "fixed_sha256": sha(folder / "fixed-candidates.json"),
        "policy_sha256": sha(folder / "policy-before-data.json"), "source_guards": guards,
        "reference_reads": [], "new_model_calls": 0})
    print(json.dumps({"baseline_sources": len(baseline_rows), "expanded_sources": len(expanded),
        "literal_original_blocks": len(groups), "new_model_calls": 0}), flush=True)


def score(folder, output):
    from scripts.offline_reference_tools import load_reference_tools
    folder = Path(folder).resolve()
    closed, _ = read(folder / "closed.json")
    if closed["status"] != "CLOSED" or closed["reference_reads"] != []:
        raise ValueError("closed_reference_free_generation_required")
    fixed, _ = read(folder / "fixed-candidates.json", closed["fixed_sha256"])
    for guard in closed["source_guards"]:
        if sha(guard["file"]) != guard["sha256"]:
            raise ValueError("original_source_guard_changed")
    raw, _ = read(fixed["source"]["path"], fixed["source"]["sha256"])
    root = ROOT.parent / "多智能体可信研判与证据链溯源/样例集-拓尔思-20261012" / raw["title"]
    evidence, _ = read(root / "evidence_samples.json")
    conclusions, _ = read(root / "conclusions.json")
    conclusion = next(row for row in conclusions if row["question_id"] == fixed["question_id"])
    mapper, engine, _ = load_reference_tools()
    union, issues = mapper.question_union_reference(evidence, conclusion)
    sides = {"union": union}
    for label in ("supporting_facts", "conflicting_facts"):
        sides[label] = engine._relation_rows(conclusion[label], evidence, submitted=False, issues=issues, context=[])
    result = {"reference_mapping_issues": issues, "variants": {}, "new_model_calls": 0,
        "new_model_tokens": 0, "candidate_presence_not_final_recall": True}
    for variant in ("baseline_candidates", "expanded_candidates"):
        result["variants"][variant] = {side: engine.compare_rows(fixed["topic_id"], fixed[variant], refs)["unique_paragraphs"]["A_id_only"]["metrics"]
            for side, refs in sides.items()}
    # The literal delivered block can be shorter than the reference quotation.
    # Report this separately; a containing parent alias is not proof of reading.
    selected_keys = {(row["doc_id"], row["para_id"]) for row in fixed["expanded_candidates"]}
    qrefs = [row for row in evidence if (row["doc_id"], row["para_id"]) in
        {(row.get("doc_id"), row.get("para_id")) for row in union}]
    literal_quote_keys = set()
    for ref in qrefs:
        for group in fixed["grounded_block_groups"]:
            if ref["quote"] in group["quote"] and any((alias["doc_id"], alias["para_id"]) == (ref["doc_id"], ref["para_id"])
                for alias in group["parent_aliases"]):
                literal_quote_keys.add((ref["doc_id"], ref["para_id"]))
    result["only_grounded_blocks_full_reference_quote_delivered"] = len(literal_quote_keys)
    result["navigation_groups"] = [{"original_source_ref": group["original_source_ref"],
        "candidate_parent_aliases": len(group["parent_aliases"]), "quote_cp": len(group["quote"])} for group in fixed["grounded_block_groups"]]
    result["raw_only_fixed_sha256"] = closed["fixed_sha256"]
    save(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    def audit(event, args):
        if event.startswith("socket."):
            raise PermissionError("offline_network_forbidden")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            name = Path(os.fsdecode(args[0])).name.lower()
            if name == ".env" or (sys.argv[1] == "generate" and name in {"evidence_samples.json", "conclusions.json"}):
                raise PermissionError("raw_only_generation_credential_or_reference_forbidden")
    sys.addaudithook(audit)
    if sys.argv[1] == "generate":
        generate(sys.argv[2])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        raise ValueError("generate_or_score_required")
