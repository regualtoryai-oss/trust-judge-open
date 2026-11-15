"""Two disjoint CPU4 local-inference shards; merge only after both CLOSED."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_title_aware_rerank import OUT, MODEL, digest, save
from app.domain.title_aware_rerank import rank_title_parents, fuse_title_hybrid

PREP_SHA = "c8a335b672b6c47429500f276dee1f11a1d6ebe8b596e570156361f9945b7c97"
NETWORK, REFERENCE_ATTEMPTS, LOCAL_PROBES = [], [], []


def load_plan():
    if digest(OUT / "preparation.json") != PREP_SHA: raise ValueError("preparation_changed")
    plan = json.loads((OUT / "preparation.json").read_text("utf-8"))
    if plan["overflow"] or plan["script_sha256"] != digest(ROOT / "scripts/probe_title_aware_rerank.py") or plan["helper_sha256"] != digest(ROOT / "app/domain/title_aware_rerank.py"): raise ValueError("prepared_code_or_token_gate_changed")
    for g in plan["guards"]:
        if digest(g["file"]) != g["sha256"]: raise ValueError("prepared_source_changed")
    return plan


def batch_binding(number, keys, plan):
    value = {"preparation_sha256": PREP_SHA, "global_batch": number, "ordered_pair_sha256": keys,
             "threads": 4, "batch_size": 4, "model_manifest_sha256": plan["model_manifest_sha256"]}
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(shard):
    if shard not in (0, 1): raise ValueError("fixed_two_shards_required")
    plan = load_plan(); schedule = plan["schedule"]; count = (len(schedule)+3)//4
    start, end = (0, count//2) if shard == 0 else (count//2, count)
    folder = OUT / f"shard-{shard}"; folder.mkdir(exist_ok=False)
    manifest = json.loads((MODEL / "local-model-manifest.json").read_text("utf-8"))
    for item in manifest["files"]:
        if digest(MODEL / item["file"]) != item["sha256"]: raise ValueError("local_model_changed")
    save(folder / "start.json", {"preparation_sha256": PREP_SHA, "shard": shard, "global_batch_start": start,
        "global_batch_end_exclusive": end, "threads": 4, "batch_size": 4, "script_sha256": digest(__file__), "paid_api_calls": 0})
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    tick = time.monotonic()
    model = TextCrossEncoder(model_name="BAAI/bge-reranker-v2-m3-int8", specific_model_path=str(MODEL),
        cache_dir=str(MODEL / "runtime-cache"), local_files_only=True, threads=4, providers=["CPUExecutionProvider"])
    model.model.tokenizer.no_truncation()
    load_seconds = time.monotonic()-tick
    inference, completed, last_notice = 0.0, 0, tick
    for number in range(start, end):
        keys = schedule[number*4:number*4+4]; values = [plan["pairs"][key] for key in keys]
        if model.model.tokenizer.truncation is not None: raise ValueError("silent_truncation_forbidden")
        if any(len(model.model.tokenizer.encode(p["query"], p["document"]).ids)>512 for p in values): raise ValueError("runtime_pair_too_long")
        before = time.monotonic(); scores = list(model.rerank_pairs([(p["query"], p["document"]) for p in values], batch_size=4)); seconds = time.monotonic()-before
        if len(scores) != len(keys) or any(not math.isfinite(v) for v in scores): raise ValueError("incomplete_or_nonfinite_batch")
        if NETWORK or REFERENCE_ATTEMPTS: raise ValueError("guard_attempt_prevents_publication")
        save(folder / f"batch-{number:05d}.json", {"binding_sha256": batch_binding(number, keys, plan), "keys": keys, "scores": scores, "seconds": seconds})
        inference += seconds; completed += len(keys)
        now = time.monotonic()
        if now-last_notice>=30 or number==end-1:
            elapsed = now-tick
            total = min(len(schedule), end*4)-start*4
            print(json.dumps({"shard": shard, "completed": completed, "total": total, "elapsed_seconds": elapsed,
                              "remaining_seconds_estimate": elapsed/max(1,completed)*(total-completed), "phase_closed": False}), flush=True)
            last_notice = now
    load_plan()
    if NETWORK or REFERENCE_ATTEMPTS: raise ValueError("guard_failure")
    receipt = {"closed": True, "shard": shard, "preparation_sha256": PREP_SHA, "global_batch_start": start,
        "global_batch_end_exclusive": end, "pairs": completed, "model_load_seconds": load_seconds,
        "inference_seconds": inference, "elapsed_seconds": time.monotonic()-tick, "paid_api_calls": 0,
        "network_attempts": NETWORK, "reference_attempts": REFERENCE_ATTEMPTS, "local_loopback_import_probes": LOCAL_PROBES,
        "script_sha256": digest(__file__), "probe_scores_reused": False}
    save(folder / "final-receipt.json", receipt); print(json.dumps(receipt), flush=True)


def merge():
    plan = load_plan(); scores, receipts = {}, []
    for shard in (0,1):
        folder = OUT / f"shard-{shard}"; receipt = json.loads((folder / "final-receipt.json").read_text("utf-8"))
        if not receipt["closed"] or receipt["preparation_sha256"]!=PREP_SHA or receipt["script_sha256"]!=digest(__file__) or receipt["network_attempts"] or receipt["reference_attempts"]: raise ValueError("closed_clean_shard_required")
        receipts.append({"file": str(folder / "final-receipt.json"), "sha256": digest(folder / "final-receipt.json")})
        for number in range(receipt["global_batch_start"], receipt["global_batch_end_exclusive"]):
            keys = plan["schedule"][number*4:number*4+4]
            batch = json.loads((folder / f"batch-{number:05d}.json").read_text("utf-8"))
            if batch["binding_sha256"]!=batch_binding(number,keys,plan) or batch["keys"]!=keys or len(batch["scores"])!=len(keys) or any(k in scores for k in keys): raise ValueError("exact_disjoint_full_batches_required")
            scores.update(zip(keys,batch["scores"]))
    if set(scores)!=set(plan["pairs"]): raise ValueError("not_all_pairs_scored")
    final_cases, all_ranked, fits = [], [], []
    fit_dir = OUT / "native-fit-inputs"; fit_dir.mkdir(exist_ok=False)
    for ordinal, case in enumerate(plan["cases"],1):
        parent_plan = case["plan"]; local = {r["pair_sha256"]: scores[r["pair_sha256"]] for r in parent_plan["records"]}
        ranked = rank_title_parents(parent_plan,local); fused = fuse_title_hybrid(parent_plan,ranked,case["old_hybrid_refs"])
        def source_rows(rows):
            return [{**r, "id": case["original_ids"][r["paragraph_ref"]], "raw_doc_id": r["doc_id"], "raw_para_id": r["para_id"],
                     "source_text_sha256": r["source_text_sha256_utf8"]} for r in rows]
        common = {k: case[k] for k in ("topic_id","question_id","title")}; common["question"] = parent_plan["question"]
        final_cases.append({**common, "outputs": [{"variant":"title_bge_only40","candidates":source_rows(ranked[:40])},
                                                   {"variant":"title_bge_hybrid_rrf40","candidates":source_rows(fused)}]})
        all_ranked.append({**common,"candidates":source_rows(ranked)})
        source = json.loads(Path(case["case_input"]).read_text("utf-8")); variants=[]
        for label, refs in (("baseline_hybrid40",case["old_hybrid_refs"]),("title_bge_hybrid_rrf40",[r["paragraph_ref"] for r in fused])):
            selected=set(refs)
            raw_case={"format":"dsh-local-rerank-raw-subset-v1","question":source["question"],"topic_title":source["topic_title"],
                "catalog":[r for r in source["catalog"] if r["paragraph_ref"] in selected],
                "source_metadata":{r:m for r,m in source["source_metadata"].items() if r in selected},
                "binding":{"parent_case_file":case["case_input"],"parent_case_sha256":case["case_sha256"],"variant":label},
                "unprovided_context_state":"unknown_not_irrelevant","publishable":False}
            path=fit_dir/f"case-{ordinal:03d}-{label}.json";save(path,raw_case)
            variants.append({"label":label,"file":str(path),"sha256":digest(path)})
        fits.append({"topic_id":case["topic_id"],"question_id":case["question_id"],"variants":variants})
    fixed={"format":"title-aware-full-window-local-rerank-candidates-v1","cases":final_cases,"preparation_sha256":PREP_SHA,
        "paid_api_calls":0,"reference_reads":[],"all_parent_windows_scored":True,"source_scope":"frozen V22 four_channel_quota400 only"}
    save(OUT/"fixed-candidates.json",fixed);save(OUT/"all-parent-ranking.json",{"cases":all_ranked})
    save(OUT/"native-fit-index.json",{"format":"frozen-local-rerank-native-fit-inputs-v1","cases":fits})
    freeze={"generation_complete":True,"cases":18,"parents":plan["parents"],"windows":plan["windows"],"unique_pairs":len(scores),
        "fixed_candidates_sha256":digest(OUT/"fixed-candidates.json"),"all_parent_ranking_sha256":digest(OUT/"all-parent-ranking.json"),
        "native_fit_index_sha256":digest(OUT/"native-fit-index.json"),"preparation_sha256":PREP_SHA,"shard_receipts":receipts,
        "reference_reads":[],"paid_api_calls":0,"script_sha256":digest(__file__)}
    load_plan();save(OUT/"candidate-freeze.json",freeze);print(json.dumps(freeze),flush=True)


if __name__=="__main__":
    def audit(event,args):
        if event=="socket.bind" and len(args)>1 and isinstance(args[1],tuple) and args[1][0] in {"::1","127.0.0.1"} and args[1][1]==0:
            LOCAL_PROBES.append({"event":event,"host":args[1][0],"port":0});return
        if event in {"socket.connect","socket.connect_ex","socket.bind","socket.getaddrinfo"}:
            NETWORK.append(event);raise PermissionError("local_full_rerank_network_forbidden")
        if event=="open" and args and isinstance(args[0],(str,bytes)):
            if Path(os.fsdecode(args[0])).name.lower() in {".env","evidence_samples.json","conclusions.json","raw_documents.json"}:
                REFERENCE_ATTEMPTS.append(Path(os.fsdecode(args[0])).name);raise PermissionError("references_and_credentials_forbidden")
    sys.addaudithook(audit);os.environ.update(HF_HUB_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",TOKENIZERS_PARALLELISM="false")
    if sys.argv[1]=="run":run(int(sys.argv[2]))
    elif sys.argv[1]=="merge":merge()
    else:raise ValueError("run_shard_or_merge_required")
