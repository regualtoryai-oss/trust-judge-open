"""Rebuild ALL source-only selector inputs from a SHA-bound CLOSED v22 pool.

Usage: python scripts/build_frozen_selection_inputs.py --pool PATH
       --pool-sha256 SHA256 --output NEW_DIRECTORY
No answers, model clients, credentials, retrieval or embedding operations.
"""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN = []

def guard(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.bind"}:
        FORBIDDEN.append(event); raise RuntimeError("frozen_input_build_network_forbidden")
    if event == "import" and str(args[0]).startswith(("app.agents", "app.config", "openai", "fastembed")):
        raise RuntimeError("frozen_input_build_model_or_settings_forbidden")
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        name = str(args[0]).replace("\\", "/").split("/")[-1]
        if name in {".env", "evidence_samples.json", "conclusions.json", "raw_documents.json", "expected.json"}:
            FORBIDDEN.append(name); raise RuntimeError("frozen_input_build_reference_or_credentials_forbidden")

sys.addaudithook(guard)
from app.domain.frozen_selection_inputs import build_frozen_selection_input, canonical


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def build(pool_path, expected_pool_sha, output):
    begin = time.monotonic(); pool_path = Path(pool_path).resolve(); output = Path(output).resolve()
    if output.exists():
        raise ValueError("new_selector_input_directory_required")
    pool_bytes = pool_path.read_bytes()
    if sha256(pool_bytes).hexdigest() != expected_pool_sha:
        raise ValueError("closed_pool_bytes_sha256_mismatch")
    pool = json.loads(pool_bytes); source_bytes = {}; sources = []
    for item in pool["inputs"]:
        tid = item["topic_id"]
        if tid in source_bytes:
            raise ValueError("duplicate_pool_topic")
        data = Path(item["path"]).read_bytes()
        if sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("raw_source_bytes_sha256_mismatch")
        source_bytes[tid] = data; sources.append(item)
    code = {str(p): file_sha(p) for p in (Path(__file__), ROOT / "app/domain/frozen_selection_inputs.py")}
    prepared = []; seen = set()
    for ordinal, case in enumerate(pool["cases"], 1):
        key = case["topic_id"], case["question_id"]
        if key in seen:
            raise ValueError("duplicate_frozen_question")
        seen.add(key)
        source = next(s for s in sources if s["topic_id"] == key[0])
        value = build_frozen_selection_input(raw_input_bytes=source_bytes[key[0]], pool_input_bytes=pool_bytes,
            topic_id=key[0], question_id=key[1], expected_raw_sha256=source["sha256"], expected_pool_sha256=expected_pool_sha,
            variant="four_channel_quota400")
        prepared.append((ordinal, value))
    if {key[0] for key in seen} != set(source_bytes):
        raise ValueError("all_pool_topics_required")
    if code != {p: file_sha(p) for p in code} or file_sha(pool_path) != expected_pool_sha or FORBIDDEN:
        raise ValueError("frozen_input_build_source_changed")
    for source in sources:
        if file_sha(source["path"]) != source["sha256"]:
            raise ValueError("raw_source_changed_during_build")
    output.mkdir(parents=True); cases = []
    for ordinal, value in prepared:
        path = output / f"case-{ordinal:03d}.json"; save(path, value)
        cases.append({"ordinal": ordinal, "topic_id": value["topic_id"], "question_id": value["question_id"],
            "file": str(path), "sha256": file_sha(path), "binding": value["binding"], "stats": value["stats"]})
    index = {"format": "frozen-unified-selection-input-index-v1", "source_pool": {"file": str(pool_path), "sha256": expected_pool_sha},
        "variant": "four_channel_quota400", "source_inputs": sources, "case_count": len(cases), "topic_count": len(sources), "cases": cases,
        "candidate_occurrences": sum(c["stats"]["candidate_count"] for c in cases),
        "catalog_parent_codepoints": sum(c["stats"]["catalog_parent_codepoints"] for c in cases),
        "new_candidates_created": False, "new_queries": 0, "new_embeddings": 0, "model_calls": 0,
        "tokens_status": "unknown_exact_messages_not_built", "reference_reads": [], "network_attempts": [], "source_code": code}
    save(output / "input-index.json", index)
    save(output / "input-freeze.json", {"status": "CLOSED_RAW_INPUT_MAPPING_ONLY", "all_existing_pool_cases_included": True,
        "index": {"file": str(output / "input-index.json"), "sha256": file_sha(output / "input-index.json")},
        "source_pool": index["source_pool"], "source_inputs": sources, "source_code": code,
        "model_calls": 0, "new_candidates_created": False, "reference_reads": [], "network_attempts": [],
        "elapsed_seconds": time.monotonic() - begin, "publishable": False, "v19_proof_created": False})
    print(json.dumps({"status": "CLOSED", "index": {"file": str(output / "input-index.json"), "sha256": file_sha(output / "input-index.json")},
        "freeze": {"file": str(output / "input-freeze.json"), "sha256": file_sha(output / "input-freeze.json")},
        "cases": len(cases), "candidate_occurrences": index["candidate_occurrences"], "codepoints": index["catalog_parent_codepoints"],
        "elapsed_seconds": time.monotonic() - begin}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True); parser.add_argument("--pool-sha256", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); build(args.pool, args.pool_sha256, args.output)
