"""Local-only title-BGE preparation and one authorized 8-pair CPU probe."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.domain.title_aware_rerank import prepare_title_pairs

MODEL = ROOT / ".runtime/models/bge-reranker-v2-m3-6f5ff6529851-int8"
MANIFEST = "d8a13027b4c4bef94c870988c1749e6dff02fcfc4cfcef0e73a002227945ff58"
TOKENIZER = "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3"
INDEX = ROOT / ".runtime/v41-frozen-selection-inputs-r1/input-index.json"
INDEX_SHA = "9ee10a5595c256855eb66e7f46998e4ed8f405727c66d22be1852ed232e08832"
POOL = ROOT / ".runtime/v22-channel-pool-r2/fixed-candidates.json"
POOL_SHA = "6544144737d01c0bed15206f3925ec7a548e3540dcbbeebf7ab92774e229945a"
OLD = ROOT / "docs/reports/v20-semantic/fixed-candidates.json"
OLD_SHA = "709052e31f73272948b40c3afcb1c93b70e429208c869d984a4ad42e051bc436"
OUT = ROOT / ".runtime/title-aware-rerank-r1"


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":")); stream.write("\n")


def prepare():
    guards = []
    def read(path, expected=None):
        actual = digest(path)
        if expected and actual != expected: raise ValueError("frozen_input_sha_changed")
        guards.append({"file": str(Path(path).resolve()), "sha256": actual})
        return json.loads(Path(path).read_text("utf-8"))
    index, pool, old = read(INDEX, INDEX_SHA), read(POOL, POOL_SHA), read(OLD, OLD_SHA)
    read(MODEL / "local-model-manifest.json", MANIFEST)
    if digest(MODEL / "tokenizer.json") != TOKENIZER: raise ValueError("tokenizer_changed")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json")); tokenizer.no_truncation(); tokenizer.no_padding()
    for binding in pool["inputs"]:
        if digest(binding["path"]) != binding["sha256"]: raise ValueError("original_raw_source_changed")
        guards.append({"file": binding["path"], "sha256": binding["sha256"]})
    cases, unique, overflow = [], {}, []
    for item in index["cases"]:
        case = read(item["file"], item["sha256"])
        planned = prepare_title_pairs(case["question"], case["catalog"], model_fingerprint=MANIFEST)
        wide = next(c for c in pool["cases"] if (c["topic_id"], c["question_id"]) == (item["topic_id"], item["question_id"]))
        parents = next(o["candidates"] for o in wide["outputs"] if o["variant"] == "four_channel_quota400")
        if [(r["raw_doc_id"], r["raw_para_id"]) for r in parents] != [(r["doc_id"], r["para_id"]) for r in case["catalog"]]: raise ValueError("wide_pool_order_changed")
        if wide["question"] != planned["question"]: raise ValueError("complete_question_changed")
        old_case = next(c for c in old["cases"] if (c["topic_id"], c["question_id"]) == (item["topic_id"], item["question_id"]))
        by_source = {(r["doc_id"], r["para_id"]): r["paragraph_ref"] for r in case["catalog"]}
        old_refs = [by_source[(r["raw_doc_id"], r["raw_para_id"])] for r in next(o["candidates"] for o in old_case["outputs"] if o["variant"] == "hybrid40")]
        records = []
        for row in planned["records"]:
            count = len(tokenizer.encode(planned["question"], row["model_document"]).ids)
            value = {"query": planned["question"], "document": row["model_document"], "tokens": count}
            key = row["pair_sha256"]
            if key in unique and unique[key] != value: raise ValueError("pair_collision")
            unique[key] = value
            if count > 512: overflow.append({"topic_id": item["topic_id"], "question_id": item["question_id"], "p": row["paragraph_ref"], "tokens": count})
            records.append({k: v for k, v in row.items() if k not in {"window_text", "model_document"}})
        planned["records"] = records
        cases.append({"topic_id": item["topic_id"], "question_id": item["question_id"], "title": case["topic_title"],
                      "case_input": item["file"], "case_sha256": item["sha256"], "plan": planned, "old_hybrid_refs": old_refs,
                      "original_ids": {r["paragraph_ref"]: p["id"] for r, p in zip(case["catalog"], parents)}})
    probe = []
    for topic in ("01", "02"):
        case = next(c for c in cases if c["topic_id"] == topic and c["question_id"] == "q_001")
        keys = list(dict.fromkeys(r["pair_sha256"] for r in case["plan"]["records"]))[:4]
        if len(keys) != 4: raise ValueError("four_raw_order_pairs_per_topic_required")
        probe.extend(keys)
    for g in guards:
        if digest(g["file"]) != g["sha256"]: raise ValueError("prepare_source_changed")
    OUT.mkdir(parents=True, exist_ok=False)
    plan = {"format": "title-aware-bge-widepool-preparation-v1", "cases": cases, "pairs": unique,
            "schedule": sorted(unique, key=lambda key: (unique[key]["tokens"], key)), "probe_keys": probe,
            "overflow": overflow, "guards": guards, "model_manifest_sha256": MANIFEST, "tokenizer_sha256": TOKENIZER,
            "threads": 4, "batch_size": 4, "probe_scores_reused_for_full_run": False,
            "script_sha256": digest(__file__), "helper_sha256": digest(ROOT / "app/domain/title_aware_rerank.py"),
            "parents": sum(len(c["plan"]["parents"]) for c in cases), "windows": sum(len(c["plan"]["records"]) for c in cases),
            "maximum_tokens": max(v["tokens"] for v in unique.values()), "queries_truncated": 0,
            "reference_reads": [], "paid_api_calls": 0, "full_run_authorized_by_prepare": False}
    save(OUT / "preparation.json", plan)
    print(json.dumps({"prepared": True, "cases": len(cases), "parents": plan["parents"], "windows": plan["windows"], "unique_pairs": len(unique), "maximum_tokens": plan["maximum_tokens"], "overflows": len(overflow), "preparation_sha256": digest(OUT / "preparation.json")}), flush=True)


def probe(expected_sha):
    path = OUT / "preparation.json"
    if digest(path) != expected_sha: raise ValueError("exact_preparation_sha_required")
    plan = json.loads(path.read_text("utf-8"))
    if plan["overflow"] or plan["script_sha256"] != digest(__file__) or plan["helper_sha256"] != digest(ROOT / "app/domain/title_aware_rerank.py"): raise ValueError("title_pair_overflow_or_code_changed")
    for guard in plan["guards"]:
        if digest(guard["file"]) != guard["sha256"]: raise ValueError("source_changed")
    manifest = json.loads((MODEL / "local-model-manifest.json").read_text("utf-8"))
    for item in manifest["files"]:
        if digest(MODEL / item["file"]) != item["sha256"]: raise ValueError("local_model_file_changed")
    save(OUT / "probe-start.json", {"preparation_sha256": expected_sha, "threads": 4, "batch_size": 4, "pairs": 8, "paid_api_calls": 0})
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    started = time.monotonic()
    model = TextCrossEncoder(model_name="BAAI/bge-reranker-v2-m3-int8", specific_model_path=str(MODEL),
        cache_dir=str(MODEL / "runtime-cache"), local_files_only=True, threads=4, providers=["CPUExecutionProvider"])
    model.model.tokenizer.no_truncation()
    load_seconds = time.monotonic() - started
    measured = []
    for offset in (0, 4):
        keys = plan["probe_keys"][offset:offset+4]
        values = [plan["pairs"][key] for key in keys]
        tick = time.monotonic(); scores = list(model.rerank_pairs([(p["query"], p["document"]) for p in values], batch_size=4)); seconds = time.monotonic() - tick
        if len(scores) != 4 or any(not math.isfinite(v) for v in scores): raise ValueError("finite_complete_probe_scores_required")
        measured.append({"keys": keys, "seconds": seconds, "scores": scores, "tokens": [p["tokens"] for p in values]})
    result = {"status": "CLOSED", "preparation_sha256": expected_sha, "model_load_seconds": load_seconds,
              "batches": measured, "elapsed_seconds": time.monotonic()-started, "pair_count": 8,
              "estimated_full_seconds_not_measured": sum(b["seconds"] for b in measured)/8*len(plan["pairs"])+load_seconds,
              "scores_reused_for_full_run": False, "reference_reads": [], "paid_api_calls": 0, "network_attempts": []}
    save(OUT / "probe-final-receipt.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    def audit(event, args):
        if event == "socket.bind" and len(args)>1 and isinstance(args[1], tuple) and args[1][0] in {"::1", "127.0.0.1"} and args[1][1] == 0: return
        if event in {"socket.connect", "socket.connect_ex", "socket.bind", "socket.getaddrinfo"}: raise PermissionError("local_probe_network_forbidden")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            name = Path(os.fsdecode(args[0])).name.lower()
            if name in {".env", "evidence_samples.json", "conclusions.json", "raw_documents.json"}: raise PermissionError("local_probe_reference_or_credential_forbidden")
    sys.addaudithook(audit)
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    if sys.argv[1] == "prepare": prepare()
    elif sys.argv[1] == "probe": probe(sys.argv[2])
    else: raise ValueError("prepare_or_probe_required")
