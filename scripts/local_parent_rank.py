"""Fixed, offline BGE entry point; worker imports need only the standard library."""
import argparse
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain import local_parent_rank as rank  # noqa: E402


def _infer(job_path, expected_sha, output):
    job_bound = rank.descriptor(job_path)
    rank._require(job_bound["sha256"] == expected_sha, "inference_job_changed")
    job = rank._read(job_bound)
    rank._require(job.get("format") == "local-parent-inference-job-v1" and job.get("config") == rank.CONFIG,
        "inference_job_invalid")
    rank._source_check(job["runner_sources"])
    model = rank.validate_model(job["model"]["manifest"]["file"], job["model"]["manifest"]["sha256"])
    rank._require(model == job["model"], "inference_model_changed")
    pairs = job.get("pairs")
    rank._require(isinstance(pairs, dict) and bool(pairs)
        and all(rank.fingerprint(payload) == key and payload.get("formatter_id") == rank.FORMATTER_ID
            and payload.get("formatter_config_sha256") == rank.fingerprint(rank.CONFIG)
            and payload.get("model_manifest_sha256") == model["manifest"]["sha256"]
            and isinstance(payload.get("query"), str) and isinstance(payload.get("document"), str)
            for key, payload in pairs.items()), "inference_payload_changed")
    packages = {name: importlib.metadata.version(name) for name in rank.RUNTIME_PACKAGES}
    rank._require(packages == rank.RUNTIME_PACKAGES, "local_runtime_version_changed")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(model["files"]["tokenizer.json"]["file"])
    tokenizer.no_truncation()
    tokenizer.no_padding()
    counts = {key: len(tokenizer.encode(payload["query"], payload["document"]).ids) for key, payload in pairs.items()}
    # All payloads are admitted before the first local inference. Overlong
    # original questions fail; neither query nor source is silently truncated.
    rank._require(all(0 < n <= 512 for n in counts.values()), "pair_token_limit_exceeded")
    import onnxruntime
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    onnxruntime.disable_telemetry_events()
    folder = Path(model["manifest"]["file"]).parent
    engine = TextCrossEncoder(model_name=rank.MODEL_NAME, specific_model_path=str(folder),
        cache_dir=str(folder / "runtime-cache"), local_files_only=True, threads=4, providers=["CPUExecutionProvider"])
    engine.model.tokenizer.no_truncation()
    keys, scores = sorted(pairs, key=lambda key: (counts[key], key)), {}
    import math
    for offset in range(0, len(keys), 4):
        batch = keys[offset:offset + 4]
        rank._require(engine.model.tokenizer.truncation is None, "tokenizer_truncation_enabled")
        values = [float(value) for value in engine.rerank_pairs(
            [(pairs[key]["query"], pairs[key]["document"]) for key in batch], batch_size=4)]
        rank._require(len(values) == len(batch) and all(math.isfinite(v) for v in values), "local_scores_invalid")
        scores.update(zip(batch, values))
    rank._source_check(job["runner_sources"])
    result = {"format": "local-parent-inference-result-v1", "status": "CLOSED", "job_sha256": expected_sha,
        "scores": scores, "pair_token_counts": counts, "runtime_packages": packages, "truncation": False,
        "provider_calls": 0, "external_network_calls": 0}
    rank._save(Path(output).parent, Path(output).name, result)
    return {"status": "CLOSED", "new_local_pairs": len(scores), "provider_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-job", type=Path)
    parser.add_argument("--job-sha256")
    parser.add_argument("--infer-output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-sha256")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--import-cache", nargs=2, action="append", default=[], metavar=("CLOSED_FILE", "SHA256"))
    args = parser.parse_args(argv)
    try:
        if args.infer_job:
            rank._require(args.job_sha256 and args.infer_output, "inference_arguments_required")
            result = _infer(args.infer_job, args.job_sha256, args.infer_output)
        else:
            rank._require(all((args.input, args.input_sha256, args.model_manifest, args.expected_manifest_sha256,
                args.cache_dir, args.output_dir)), "cli_arguments_required")
            bound = rank.descriptor(args.input)
            rank._require(bound["sha256"] == args.input_sha256, "raw_file_changed")
            raw = rank._read(bound)
            for path, expected in args.import_cache:
                proof = rank.descriptor(path)
                rank._require(proof["sha256"] == expected, "import_receipt_changed")
                rank.seed_local_cache(args.cache_dir, proof, expected_model_manifest_sha256=args.expected_manifest_sha256)
            result = rank.build_local_rank(raw, raw_case_sha256=args.input_sha256, model_manifest=args.model_manifest,
                expected_manifest_sha256=args.expected_manifest_sha256, cache_dir=args.cache_dir, output_dir=args.output_dir)
        code = 0
    except Exception as error:
        reason = str(error)
        result = {"status": "FAILED", "provider_calls": 0,
            "error_code": reason if reason.startswith("local_parent_rank_") and " " not in reason else "local_parent_rank_failed"}
        if args.infer_output and not args.infer_output.exists():
            rank._save(args.infer_output.parent, args.infer_output.name, result)
        code = 1
    sys.stdout.buffer.write(rank.canonical(result) + b"\n")
    return code


if __name__ == "__main__":
    def offline_audit(event, args):
        if event == "socket.bind" and len(args) > 1 and isinstance(args[1], tuple) and args[1][0] in {"127.0.0.1", "::1"} and args[1][1] == 0:
            return
        if event in {"socket.connect", "socket.connect_ex", "socket.bind", "socket.getaddrinfo"}:
            raise PermissionError("local_parent_rank_network_forbidden")
    sys.addaudithook(offline_audit)
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    raise SystemExit(main())
