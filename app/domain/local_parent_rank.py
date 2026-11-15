"""Closed, body-only local BGE ranking. No provider, download or agent loop.

Every original parent has a score for every literal question facet. Windows
include their final tail; empty parents are scored too. Scores are navigation
logits, not truth probabilities. The separate rank-plan codec owns selection.
"""
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

from app.domain.semantic_recall import window_catalog
from app.domain.source_bundle_tools import SourceBundleTools


ROOT = Path(__file__).resolve().parents[2]
FORMATTER_ID = "original-full-question-literal-alternatives-body-bge-v1"
FORMAT = "closed-local-parent-rank-v1"
MODEL_NAME = "BAAI/bge-reranker-v2-m3-int8"
MODEL_FILES = {"onnx/model_int8.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
CONFIG = {"formatter_id": FORMATTER_ID, "window_chars": 320, "overlap_chars": 80,
    "body_only": True, "full_question_in_every_query": True,
    "split_policy": "first_literal_还是_no_strip_no_rewriting",
    "empty_parent_policy": "one_complete_literal_window", "max_pair_tokens": 512,
    "truncation": False, "batch_size": 4, "threads": 4, "providers": ["CPUExecutionProvider"],
    "model_name": MODEL_NAME}
RUNTIME_PACKAGES = {"fastembed": "0.9.0", "onnxruntime": "1.31.0", "tokenizers": "0.23.3", "numpy": "2.5.4"}
INFERENCE_TIMEOUT_SECONDS = 900


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def fingerprint(value):
    return sha256(canonical(value)).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError("local_parent_rank_" + reason)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file_sha(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    return {"file": str(path), "sha256": _file_sha(path), "bytes": path.stat().st_size}


def decode(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("local_parent_rank_nonfinite_json")))


def _read(bound):
    _require(isinstance(bound, dict) and isinstance(bound.get("file"), str)
        and Path(bound["file"]).is_absolute() and _sha(bound.get("sha256")), "descriptor_required")
    path = Path(bound["file"])
    _require(path.is_file() and ("bytes" not in bound or path.stat().st_size == bound["bytes"])
        and _file_sha(path) == bound["sha256"], "artifact_changed")
    return decode(path.read_bytes())


def _save(directory, name, value):
    path = Path(directory) / name
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
    return descriptor(path)


def _plan_api():
    # Pure codec; lazy import keeps this module free of provider/ONNX imports.
    from app.domain.lc_bundle_case import rank_queries, rank_source_binding, build_rank_plan, validate_rank_plan
    return rank_queries, rank_source_binding, build_rank_plan, validate_rank_plan


def _raw(raw_case):
    _require(isinstance(raw_case, dict) and raw_case.get("format") == "frozen-source-bundle-case-v1"
        and not set(raw_case) - {"format", "question", "documents", "paragraphs", "limits"}, "raw_case_shape")
    SourceBundleTools(raw_case["documents"], raw_case["paragraphs"], raw_case["question"], **raw_case.get("limits", {}))


def prepare_pairs(raw_case, model_manifest_sha256):
    """Deterministic complete payloads; no model or file IO."""
    _raw(raw_case)
    _require(_sha(model_manifest_sha256), "model_manifest_sha_required")
    queries_fn, binding_fn, _, _ = _plan_api()
    queries = queries_fn(raw_case)
    windows = window_catalog(raw_case["paragraphs"], window_chars=320, overlap_chars=80)
    by_id = {row["id"]: row for row in raw_case["paragraphs"]}
    # window_catalog intentionally skips whitespace; this adapter scores every
    # parent rather than silently substituting zero for an unscored parent.
    grouped = {pid: [] for pid in by_id}
    for window in windows:
        grouped[window["paragraph_id"]].append(window)
    windows = []
    for pid, parent in by_id.items():
        entries = grouped[pid] or [{"paragraph_id": pid, "raw_doc_id": parent["raw_doc_id"],
            "raw_para_id": parent["raw_para_id"], "start_cp": 0, "end_cp": len(parent["text"]),
            "text": parent["text"], "source_text_sha256": sha256(parent["text"].encode()).hexdigest(),
            "window_text_sha256": sha256(parent["text"].encode()).hexdigest()}]
        for entry in entries:
            windows.append({**entry, "window_id": f"W{len(windows) + 1}"})
    records, payloads = [], {}
    for query in queries:
        query_id = query["query_id"]
        for window in windows:
            payload = {"formatter_id": FORMATTER_ID, "formatter_config_sha256": fingerprint(CONFIG),
                "model_manifest_sha256": model_manifest_sha256, "query": query["text"], "document": window["text"]}
            key = fingerprint(payload)
            payloads[key] = payload
            records.append({**window, "query_id": query_id, "pair_sha256": key})
    return {"format": "local-parent-window-plan-v1", "source_binding": binding_fn(raw_case), "queries": queries,
        "records": records, "parent_ids": list(by_id), "formatter_config_sha256": fingerprint(CONFIG)}, payloads


def parent_scores(raw_case, plan, scores):
    """Max over every body window separately for each query; never compare facets."""
    expected = {row["pair_sha256"] for row in plan["records"]}
    _require(isinstance(scores, dict) and set(scores) == expected
        and all(type(v) in (float, int) and math.isfinite(v) for v in scores.values()), "complete_finite_pair_scores_required")
    best = {}
    for row in plan["records"]:
        key = row["paragraph_id"], row["query_id"]
        best[key] = max(best.get(key, -math.inf), scores[row["pair_sha256"]])
    query_ids = [q["query_id"] for q in plan["queries"]]
    return [{"paragraph_id": row["id"], "source_text_sha256": sha256(row["text"].encode()).hexdigest(),
        "facet_scores": [{"query_id": qid, "max_window_logit": best[row["id"], qid]} for qid in query_ids]}
        for row in raw_case["paragraphs"]]


def validate_model(manifest_path, expected_sha256):
    """Hash the explicit local manifest and every required weight/tokenizer file."""
    _require(_sha(expected_sha256), "expected_model_sha_required")
    bound = descriptor(manifest_path)
    _require(bound["sha256"] == expected_sha256, "model_manifest_changed")
    manifest, base = _read(bound), Path(bound["file"]).parent
    files = manifest.get("files")
    _require(manifest.get("fastembed_supported_name") == MODEL_NAME
        and manifest.get("original_model") == "BAAI/bge-reranker-v2-m3"
        and manifest.get("model_file") == "onnx/model_int8.onnx"
        and isinstance(manifest.get("revision"), str) and bool(manifest["revision"])
        and manifest.get("production_auto_download") is False and isinstance(files, list)
        and len(files) == len(MODEL_FILES) and {f.get("file") for f in files} == MODEL_FILES, "model_not_allowlisted")
    result = {}
    for item in files:
        path = (base / item["file"]).resolve()
        _require(path.is_relative_to(base) and path.is_file() and type(item.get("bytes")) is int
            and item["bytes"] >= 0 and _sha(item.get("sha256")), "model_file_invalid")
        actual = descriptor(path)
        _require(actual["bytes"] == item["bytes"] and actual["sha256"] == item["sha256"], "model_file_changed")
        result[item["file"]] = actual
    return {"manifest": bound, "files": result, "model_name": MODEL_NAME, "revision": manifest["revision"]}


def _sources():
    return {name: descriptor(ROOT / name) for name in (
        "app/domain/local_parent_rank.py", "scripts/local_parent_rank.py",
        "app/domain/semantic_recall.py", "app/domain/lc_bundle_case.py")}


def _archive_sources(output_dir):
    result = {}
    for name, source in _sources().items():
        path = Path(output_dir) / "source-archive" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(Path(source["file"]).read_bytes())
        result[name] = descriptor(path)
        _require(result[name]["sha256"] == source["sha256"], "runner_source_changed")
    return result


def _source_check(sources, *, require_current=True):
    """Verify bound archive bytes; only a current producer must match live code.

    Exact-pair cache origins retain their own CLOSED source archive. A later
    codec revision does not change an identical model/query/body payload, but
    it must not make a missing or modified origin archive acceptable.
    """
    _require(isinstance(sources, dict) and set(sources) == set(_sources()), "runner_sources_required")
    for name, current in _sources().items():
        _read_bytes = Path(sources[name].get("file", ""))
        _require(_read_bytes.is_absolute() and _read_bytes.is_file()
            and _sha(sources[name].get("sha256"))
            and ("bytes" not in sources[name] or _read_bytes.stat().st_size == sources[name]["bytes"])
            and _file_sha(_read_bytes) == sources[name].get("sha256")
            and (not require_current or current["sha256"] == sources[name]["sha256"]), "runner_source_changed")


def _closed_values(proof, *, expected_model_manifest_sha256, check_model=True, cache_origin=False):
    closed = _read(proof)
    _require(closed.get("format") == FORMAT and closed.get("status") == "CLOSED"
        and closed.get("provider_calls") == 0 and closed.get("external_network_calls") == 0
        and closed.get("semantic_verified") is False and closed.get("config") == CONFIG
        and closed.get("formatter_config_sha256") == fingerprint(CONFIG), "closed_receipt_invalid")
    _source_check(closed["source_archives"], require_current=not cache_origin)
    model = closed["model"]
    _require(model["manifest"]["sha256"] == expected_model_manifest_sha256, "expected_model_changed")
    _read(model["manifest"])
    if check_model:
        _require(validate_model(model["manifest"]["file"], expected_model_manifest_sha256) == model, "model_binding_changed")
    payloads, plan, scores, parents = (_read(closed[key]) for key in ("pair_payloads", "window_plan", "pair_scores", "parent_scores"))
    _require(isinstance(payloads, dict) and set(payloads) == set(scores)
        and all(fingerprint(payload) == key for key, payload in payloads.items()), "pair_payload_binding_changed")
    return closed, payloads, plan, scores, parents


def _legacy_values(proof, expected_sha):
    """Import only a CLOSED body/facet audit, rebuilding every actual pair.

    Legacy rank/selection results are ignored. Only exact local score payloads
    are reusable, including the legacy formatter-to-payload-key translation.
    """
    closed = _read(proof)
    formats = {FORMATTER_ID, "original-full-question-raw-body-window-bge-v1"}
    _require(closed.get("status") == "CLOSED" and closed.get("format") in formats
        and closed.get("provider_calls") == 0 and closed.get("external_network_calls") == 0
        and closed.get("reference_reads") == [] and closed.get("query_and_windows_untruncated") is True
        and closed.get("titles_never_in_body_payload") is True
        and closed.get("formatter_sha256") == sha256(closed["format"].encode()).hexdigest(), "legacy_cache_not_closed_body_only")
    script = Path(proof["file"]).parent / "local_rank.py"
    _require(script.is_file() and _file_sha(script) == closed.get("script_sha256"), "legacy_runner_changed")
    _require(closed["model_manifest"]["sha256"] == expected_sha, "legacy_model_changed")
    model_manifest = _read(closed["model_manifest"])
    _require(closed.get("weight_sha256") == next(f["sha256"] for f in model_manifest["files"]
        if f["file"] == model_manifest["model_file"]), "legacy_weights_changed")
    raw, old_plan, old_scores = (_read(closed[k]) for k in ("raw_case", "plan", "scores"))
    plan, payloads = prepare_pairs(raw, expected_sha)
    is_facets = closed["format"] == FORMATTER_ID
    records = plan["records"] if is_facets else [r for r in plan["records"] if r["query_id"] == "F0"]
    queries = {q["query_id"]: q["text"] for q in plan["queries"]}
    _require(old_plan.get("title_in_model_document") is False and old_plan.get("window_chars") == 320
        and old_plan.get("overlap_chars") == 80 and len(old_plan.get("records", [])) == len(records), "legacy_window_policy_changed")
    if is_facets:
        _require({q.get("id"): q.get("query") for q in old_plan.get("queries", [])} == queries
            and all(q.get("query_sha256_utf8") == sha256(q["query"].encode()).hexdigest()
                for q in old_plan["queries"]), "legacy_queries_changed")
    else:
        _require(old_plan.get("question") == raw["question"]["question"], "legacy_query_changed")
    result, legacy_keys = {}, set()
    for old, row in zip(old_plan["records"], records):
        _require(all(old.get(k) == row[k] for k in ("paragraph_id", "start_cp", "end_cp", "text", "source_text_sha256", "window_text_sha256")), "legacy_original_window_changed")
        query = queries[row["query_id"]]
        if is_facets:
            _require(old.get("facet_id") == row["query_id"], "legacy_facet_changed")
            old_payload = {"model_manifest_sha256": expected_sha, "query": query, "document": row["text"]}
        else:
            old_payload = {"formatter": closed["format"], "model_manifest_sha256": expected_sha,
                "question": query, "window_text": row["text"]}
        old_key = fingerprint(old_payload)
        _require(old.get("pair_sha256") == old_key and old_key in old_scores
            and type(old_scores[old_key]) in (int, float) and math.isfinite(old_scores[old_key]), "legacy_pair_score_changed")
        legacy_keys.add(old_key)
        key = row["pair_sha256"]
        _require(key not in result or result[key] == old_scores[old_key], "legacy_conflicting_duplicate_payload")
        result[key] = old_scores[old_key]
    _require(set(old_scores) == legacy_keys, "legacy_score_coverage_changed")
    return {key: payloads[key] for key in result}, result


def seed_local_cache(cache_dir, closed_receipt, *, expected_model_manifest_sha256):
    """Explicit exact-payload import; RUNNING/changed/foreign-model audits fail."""
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    closed = _read(closed_receipt)
    if closed.get("format") == FORMAT:
        _, payloads, _, scores, _ = _closed_values(closed_receipt,
            expected_model_manifest_sha256=expected_model_manifest_sha256, cache_origin=True)
    else:
        payloads, scores = _legacy_values(closed_receipt, expected_model_manifest_sha256)
    for key in payloads:
        entry = {"format": "local-parent-pair-cache-v1", "pair_sha256": key, "closed_receipt": closed_receipt}
        path = cache_dir / (key + ".json")
        if path.exists():
            _require(decode(path.read_bytes()) == entry, "cache_entry_conflict")
        else:
            _save(cache_dir, path.name, entry)
    return {"imported_exact_payloads": len(scores), "closed_receipt": closed_receipt, "provider_calls": 0}


def _cache_scores(cache_dir, payloads, expected_sha):
    result, origins, memo = {}, {}, {}
    for key, payload in payloads.items():
        path = Path(cache_dir) / (key + ".json")
        if not path.exists():
            continue
        entry = decode(path.read_bytes())
        _require(entry.get("format") == "local-parent-pair-cache-v1" and entry.get("pair_sha256") == key, "cache_entry_changed")
        proof = entry["closed_receipt"]
        ident = fingerprint(proof)
        if ident not in memo:
            closed = _read(proof)
            if closed.get("format") == FORMAT:
                _, prior_payloads, _, prior_scores, _ = _closed_values(proof,
                    expected_model_manifest_sha256=expected_sha, check_model=False, cache_origin=True)
            else:
                prior_payloads, prior_scores = _legacy_values(proof, expected_sha)
            memo[ident] = prior_payloads, prior_scores
        prior_payloads, prior_scores = memo[ident]
        _require(prior_payloads.get(key) == payload and key in prior_scores, "cache_exact_payload_mismatch")
        result[key], origins[key] = prior_scores[key], proof
    return result, origins


def _run_inference(job, output_dir):
    interpreter = ROOT / ".runtime/v20-embedding-env/Scripts/python.exe"
    script = ROOT / "scripts/local_parent_rank.py"
    _require(interpreter.is_file() and script.is_file(), "local_runtime_missing")
    job_bound = _save(output_dir, "inference-job.json", job)
    result_path = Path(output_dir) / "inference-result.json"
    try:
        process = subprocess.run([str(interpreter), str(script), "--infer-job", job_bound["file"],
            "--job-sha256", job_bound["sha256"], "--infer-output", str(result_path)], cwd=ROOT,
            capture_output=True, timeout=INFERENCE_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        raise ValueError("local_parent_rank_local_inference_timeout") from None
    _require(process.returncode == 0 and result_path.is_file(), "local_inference_failed")
    response = decode(result_path.read_bytes())
    _require(response.get("format") == "local-parent-inference-result-v1" and response.get("status") == "CLOSED"
        and response.get("job_sha256") == job_bound["sha256"] and response.get("provider_calls") == 0
        and response.get("external_network_calls") == 0 and response.get("runtime_packages") == RUNTIME_PACKAGES
        and response.get("truncation") is False, "local_inference_receipt_invalid")
    return response, descriptor(result_path)


def build_local_rank(raw_case, *, raw_case_sha256, model_manifest, expected_manifest_sha256, cache_dir, output_dir):
    """Build once from a caller-verified raw-file SHA; no remote fallback.

    Missing exact pairs are scored in the fixed local embedding interpreter.
    The main worker can import and verify this module without ONNX installed.
    """
    _require(_sha(raw_case_sha256), "raw_file_sha_required")
    output_dir, cache_dir = Path(output_dir).resolve(), Path(cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _require(not (output_dir / "closed.json").exists(), "output_already_closed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan, payloads = prepare_pairs(raw_case, expected_manifest_sha256)
        model = validate_model(model_manifest, expected_manifest_sha256)
        sources = _archive_sources(output_dir)
        raw_bound = _save(output_dir, "raw-case.json", raw_case)
        plan_bound = _save(output_dir, "window-plan.json", plan)
        payload_bound = _save(output_dir, "pair-payloads.json", payloads)
        scores, origins = _cache_scores(cache_dir, payloads, expected_manifest_sha256)
        missing = [key for key in payloads if key not in scores]
        inference, inference_bound = None, None
        if missing:
            job = {"format": "local-parent-inference-job-v1", "model": model, "config": CONFIG,
                "runner_sources": sources, "pairs": {key: payloads[key] for key in missing}}
            inference, inference_bound = _run_inference(job, output_dir)
            new_scores = inference.get("scores")
            _require(isinstance(new_scores, dict) and set(new_scores) == set(missing)
                and all(type(v) in (int, float) and math.isfinite(v) for v in new_scores.values()), "inference_pair_coverage_changed")
            scores.update(new_scores)
        _source_check(sources)
        parents = parent_scores(raw_case, plan, scores)
        scores_bound = _save(output_dir, "pair-scores.json", scores)
        parents_bound = _save(output_dir, "parent-scores.json", parents)
        closed = {"format": FORMAT, "status": "CLOSED", "raw_case": raw_bound,
            "raw_case_file_sha256": raw_case_sha256, "raw_case_canonical_sha256": fingerprint(raw_case),
            "model": model, "config": CONFIG, "formatter_config_sha256": fingerprint(CONFIG),
            "source_archives": sources, "window_plan": plan_bound, "pair_payloads": payload_bound,
            "pair_scores": scores_bound, "parent_scores": parents_bound,
            "cache_origins": origins, "new_pair_keys": missing, "inference_result": inference_bound,
            "all_parent_count": len(parents), "all_query_window_records": len(plan["records"]),
            "unique_payload_pairs": len(payloads), "cache_hit_pairs": len(origins), "new_local_pairs": len(missing),
            "provider_calls": 0, "external_network_calls": 0, "semantic_verified": False,
            "scores_are_truth_probabilities": False, "model_delivery_verified": False}
        proof = _save(output_dir, "closed.json", closed)
        producer = {"runner_sha256": sources["scripts/local_parent_rank.py"]["sha256"],
            "model_manifest_sha256": expected_manifest_sha256,
            "weights_sha256": model["files"]["onnx/model_int8.onnx"]["sha256"],
            "tokenizer_sha256": model["files"]["tokenizer.json"]["sha256"],
            "formatter_id": FORMATTER_ID, "formatter_sha256": sha256(FORMATTER_ID.encode()).hexdigest(),
            "formatter_config_sha256": fingerprint(CONFIG), "pair_payloads_sha256": payload_bound["sha256"],
            "window_plan_sha256": plan_bound["sha256"], "pair_scores_sha256": scores_bound["sha256"],
            "parent_scores_sha256": parents_bound["sha256"], "closed_receipt_sha256": proof["sha256"]}
        _, _, build_plan, _ = _plan_api()
        rank_plan = build_plan(raw_case, parents, producer)
        _save(output_dir, "rank-plan.json", rank_plan)
        # Cache publication follows a complete audit only; no partial score files
        # are discoverable through this exact-payload cache.
        for key in missing:
            _save(cache_dir, key + ".json", {"format": "local-parent-pair-cache-v1",
                "pair_sha256": key, "closed_receipt": proof})
        return {"rank_plan": rank_plan, "rank_plan_sha256": fingerprint(rank_plan), "proof": proof}
    except Exception as error:
        failure = output_dir / "failed.json"
        if not failure.exists():
            reason = str(error)
            _save(output_dir, "failed.json", {"format": FORMAT, "status": "FAILED", "provider_calls": 0,
                "error_code": reason if reason.startswith("local_parent_rank_") and " " not in reason else "local_parent_rank_failed"})
        raise


def verify_local_rank(raw_case, rank_plan, proof, *, expected_model_manifest_sha256, raw_case_sha256=None):
    """Pure hash/source/math verification. Never loads ONNX or a mutable cache."""
    closed, payloads, plan, scores, parents = _closed_values(proof,
        expected_model_manifest_sha256=expected_model_manifest_sha256)
    _require(_read(closed["raw_case"]) == raw_case and closed["raw_case_canonical_sha256"] == fingerprint(raw_case)
        and _sha(closed.get("raw_case_file_sha256"))
        and (raw_case_sha256 is None or raw_case_sha256 == closed["raw_case_file_sha256"]), "raw_source_changed")
    expected_plan, expected_payloads = prepare_pairs(raw_case, expected_model_manifest_sha256)
    _require(plan == expected_plan and payloads == expected_payloads, "complete_original_payloads_changed")
    _require(parents == parent_scores(raw_case, plan, scores), "parent_max_changed")
    origins, missing = closed["cache_origins"], closed["new_pair_keys"]
    _require(isinstance(origins, dict) and isinstance(missing, list) and len(missing) == len(set(missing))
        and not set(origins) & set(missing) and set(origins) | set(missing) == set(payloads), "score_origin_coverage_changed")
    memo = {}
    for key, origin in origins.items():
        ident = fingerprint(origin)
        if ident not in memo:
            prior = _read(origin)
            if prior.get("format") == FORMAT:
                _, ps, _, ss, _ = _closed_values(origin, expected_model_manifest_sha256=expected_model_manifest_sha256,
                    check_model=False, cache_origin=True)
            else:
                ps, ss = _legacy_values(origin, expected_model_manifest_sha256)
            memo[ident] = ps, ss
        ps, ss = memo[ident]
        _require(ps.get(key) == payloads[key] and ss.get(key) == scores[key], "cache_origin_changed")
    if missing:
        inference = _read(closed["inference_result"])
        _require(inference.get("format") == "local-parent-inference-result-v1" and inference.get("status") == "CLOSED"
            and inference.get("provider_calls") == 0 and inference.get("external_network_calls") == 0
            and inference.get("runtime_packages") == RUNTIME_PACKAGES and inference.get("truncation") is False
            and inference.get("scores") == {key: scores[key] for key in missing}, "inference_scores_changed")
        job_path = Path(closed["inference_result"]["file"]).parent / "inference-job.json"
        job_bound = descriptor(job_path)
        job = _read(job_bound)
        _require(inference.get("job_sha256") == job_bound["sha256"] and job.get("pairs") == {key: payloads[key] for key in missing}
            and job.get("model") == closed["model"] and job.get("config") == CONFIG
            and job.get("runner_sources") == closed["source_archives"], "inference_job_changed")
    else:
        _require(closed["inference_result"] is None, "unexpected_inference_receipt")
    _require(closed["all_parent_count"] == len(parents) and closed["all_query_window_records"] == len(plan["records"])
        and closed["unique_payload_pairs"] == len(payloads) and closed["cache_hit_pairs"] == len(origins)
        and closed["new_local_pairs"] == len(missing), "closed_counts_changed")
    sources, model = closed["source_archives"], closed["model"]
    producer = {"runner_sha256": sources["scripts/local_parent_rank.py"]["sha256"],
        "model_manifest_sha256": expected_model_manifest_sha256,
        "weights_sha256": model["files"]["onnx/model_int8.onnx"]["sha256"],
        "tokenizer_sha256": model["files"]["tokenizer.json"]["sha256"], "formatter_id": FORMATTER_ID,
        "formatter_sha256": sha256(FORMATTER_ID.encode()).hexdigest(), "formatter_config_sha256": fingerprint(CONFIG),
        **{key + "_sha256": closed[key]["sha256"] for key in ("pair_payloads", "window_plan", "pair_scores", "parent_scores")},
        "closed_receipt_sha256": proof["sha256"]}
    _, _, build_plan, validate_plan = _plan_api()
    _require(rank_plan == build_plan(raw_case, parents, producer), "rank_plan_changed")
    receipt = validate_plan(raw_case, rank_plan, fingerprint(rank_plan))
    return {"format": "verified-local-parent-rank-v1", "rank_plan_sha256": fingerprint(rank_plan),
        "proof_sha256": proof["sha256"], "parent_count": len(parents), "query_count": len(plan["queries"]),
        "provider_calls": 0, "new_inference_calls": 0, "semantic_verified": False, "selection": receipt}
