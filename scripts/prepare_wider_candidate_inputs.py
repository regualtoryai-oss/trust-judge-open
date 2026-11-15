"""Freeze raw-only, complete-parent top-k inputs from an existing local ranking.

No retrieval, model, scoring, reference-answer, or network operation occurs here.
The ranking, source cases, titles, complete questions, source metadata and source
text hashes are checked before creating an exclusive output directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = ROOT / ".runtime/title-aware-rerank-r1/candidate-freeze.json"
DEFAULT_FREEZE_SHA = "3d5da6cdcf3f456bf4715016a1738e12bca103940f6e812c434727ce1946de4f"
FORBIDDEN_FILES = {".env", "evidence_samples.json", "conclusions.json"}
FORBIDDEN_KEYS = {"evidence_samples", "conclusions", "supporting_facts", "conflicting_facts",
                  "reference_answer", "reference_answers", "gold", "answers", "answer"}


def digest(path: Path | str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def read_bound(path: Path | str, expected: str) -> dict:
    path = Path(path)
    if path.name.lower() in FORBIDDEN_FILES:
        raise ValueError("reference_or_credential_input_forbidden")
    if digest(path) != expected:
        raise ValueError("bound_input_sha_changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("object_input_required")
    return value


def reject_answer_fields(value) -> None:
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise ValueError("reference_field_in_raw_input")
        for child in value.values():
            reject_answer_fields(child)
    elif isinstance(value, list):
        for child in value:
            reject_answer_fields(child)


def question_text(value) -> str:
    text = value.get("question") if isinstance(value, dict) else value
    if not isinstance(text, str) or not text.strip():
        raise ValueError("complete_original_question_required")
    return text


def select_case(source: dict, ranking: dict, *, limit: int) -> tuple[dict, list[dict]]:
    """Apply one rank cutoff to every question without changing original text."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("positive_integer_limit_required")
    reject_answer_fields(source)
    reject_answer_fields(ranking)
    if question_text(source.get("question")) != question_text(ranking.get("question")):
        raise ValueError("complete_original_question_changed")
    if source.get("topic_title") != ranking.get("title"):
        raise ValueError("original_topic_title_changed")
    for key in ("topic_id", "question_id"):
        if key in source and source[key] != ranking.get(key):
            raise ValueError("case_identity_changed")
    catalog = source.get("catalog")
    metadata = source.get("source_metadata")
    candidates = ranking.get("candidates")
    if not isinstance(catalog, list) or not isinstance(metadata, dict) or not isinstance(candidates, list):
        raise ValueError("raw_catalog_metadata_and_ranking_required")
    by_ref, identities = {}, set()
    for row in catalog:
        ref = row["paragraph_ref"]
        identity = row["doc_id"], row["para_id"]
        if ref in by_ref or identity in identities:
            raise ValueError("duplicate_parent_alias_or_source_identity")
        if row.get("paragraph_complete") is not True or row.get("parent_complete") is not True:
            raise ValueError("complete_parent_required")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("nonempty_parent_text_required")
        by_ref[ref] = row
        identities.add(identity)
    if set(metadata) != set(by_ref):
        raise ValueError("exact_catalog_metadata_aliases_required")
    ranked_refs, previous_score = set(), math.inf
    for row in candidates:
        ref = row["paragraph_ref"]
        if ref in ranked_refs or ref not in by_ref:
            raise ValueError("ranking_has_duplicate_or_unknown_parent")
        ranked_refs.add(ref)
        original = by_ref[ref]
        score = row.get("local_logit")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("finite_local_score_required")
        if score > previous_score:
            raise ValueError("descending_frozen_rank_required")
        previous_score = score
        text_sha = sha256(original["text"].encode("utf-8")).hexdigest()
        if (row.get("doc_id"), row.get("para_id")) != (original["doc_id"], original["para_id"]):
            raise ValueError("ranked_source_identity_changed")
        if row.get("title") != original.get("title"):
            raise ValueError("ranked_original_title_changed")
        if row.get("source_text_sha256_utf8") != text_sha:
            raise ValueError("ranked_full_parent_text_changed")
        meta = metadata[ref]
        if (meta.get("doc_id"), meta.get("para_id")) != (original["doc_id"], original["para_id"]):
            raise ValueError("source_metadata_identity_changed")
        if meta.get("source_text_sha256_utf8") != text_sha:
            raise ValueError("source_metadata_text_hash_changed")
    if ranked_refs != set(by_ref):
        raise ValueError("complete_parent_ranking_required")
    selected = candidates[:limit]
    refs = [row["paragraph_ref"] for row in selected]
    result = {
        "format": "dsh-local-rerank-raw-subset-v1",
        "question": deepcopy(source["question"]), "topic_title": source["topic_title"],
        "topic_id": ranking["topic_id"], "question_id": ranking["question_id"],
        "catalog": [deepcopy(by_ref[ref]) for ref in refs],
        "source_metadata": {ref: deepcopy(metadata[ref]) for ref in refs},
        "unprovided_context_state": "unknown_not_irrelevant", "publishable": False,
    }
    provenance = [{"rank": number, "paragraph_ref": row["paragraph_ref"],
                   "doc_id": row["doc_id"], "para_id": row["para_id"],
                   "local_logit": row["local_logit"],
                   "source_text_sha256_utf8": row["source_text_sha256_utf8"],
                   "full_parent_codepoints": len(by_ref[row["paragraph_ref"]]["text"])}
                  for number, row in enumerate(selected, 1)]
    return result, provenance


def save_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def prepare(*, freeze_path: Path, freeze_sha: str, output: Path, limit: int = 60) -> dict:
    freeze_path, output = freeze_path.resolve(), output.resolve()
    freeze = read_bound(freeze_path, freeze_sha)
    if freeze.get("generation_complete") is not True or freeze.get("reference_reads") != [] or freeze.get("paid_api_calls") != 0:
        raise ValueError("closed_raw_only_local_ranking_required")
    ranking_path = freeze_path.parent / "all-parent-ranking.json"
    preparation_path = freeze_path.parent / "preparation.json"
    ranking = read_bound(ranking_path, freeze["all_parent_ranking_sha256"])
    original_plan = read_bound(preparation_path, freeze["preparation_sha256"])
    bindings = [{"path": str(freeze_path), "sha256": freeze_sha},
                {"path": str(ranking_path), "sha256": freeze["all_parent_ranking_sha256"]},
                {"path": str(preparation_path), "sha256": freeze["preparation_sha256"]}]
    plans = {}
    for row in original_plan["cases"]:
        key = row["topic_id"], row["question_id"]
        if key in plans:
            raise ValueError("duplicate_source_case")
        plans[key] = row
    ranked_cases, prepared, seen = ranking["cases"], [], set()
    for ordinal, row in enumerate(ranked_cases, 1):
        key = row["topic_id"], row["question_id"]
        if key in seen or key not in plans:
            raise ValueError("ranking_case_identity_invalid")
        seen.add(key)
        plan = plans[key]
        source_path = Path(plan["case_input"]).resolve()
        source = read_bound(source_path, plan["case_sha256"])
        selected, provenance = select_case(source, row, limit=limit)
        selected["binding"] = {
            "parent_case_file": str(source_path), "parent_case_sha256": plan["case_sha256"],
            "variant": f"title_bge_only{limit}", "frozen_ranking_sha256": freeze["all_parent_ranking_sha256"],
            "rank_cutoff": limit, "source_aliases_preserved": True,
        }
        bindings.append({"path": str(source_path), "sha256": plan["case_sha256"]})
        prepared.append((ordinal, row, selected, provenance))
    if seen != set(plans) or len(prepared) != freeze["cases"]:
        raise ValueError("all_frozen_cases_required")
    if output.exists():
        raise FileExistsError("exclusive_output_directory_required")
    # Validate every binding before writing anything and again before closing.
    for binding in bindings:
        if digest(binding["path"]) != binding["sha256"]:
            raise ValueError("input_changed_during_preparation")
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    index, ranks = [], []
    for ordinal, row, selected, provenance in prepared:
        file = inputs / f"case-{ordinal:03d}.json"
        save_exclusive(file, selected)
        lengths = [len(parent["text"]) for parent in selected["catalog"]]
        index.append({"ordinal": ordinal, "topic_id": row["topic_id"], "question_id": row["question_id"],
                      "file": str(file), "sha256": digest(file), "paragraphs": len(lengths),
                      "body_codepoints": sum(lengths), "largest_parent_codepoints": max(lengths),
                      "title_codepoints": sum(len(parent.get("title", "")) for parent in selected["catalog"]),
                      "question_codepoints": len(question_text(selected["question"])), "reference_reads": [], "paid_api_calls": 0})
        ranks.append({"topic_id": row["topic_id"], "question_id": row["question_id"], "rows": provenance})
    index_path, provenance_path = output / "input-index.json", output / "ranking-provenance.json"
    save_exclusive(index_path, {"format": "raw-only-wider-bge-input-index-v1", "cases": index})
    save_exclusive(provenance_path, {"format": "raw-only-wider-bge-ranking-provenance-v1", "cases": ranks})
    for binding in bindings:
        if digest(binding["path"]) != binding["sha256"]:
            raise ValueError("input_changed_during_preparation")
    receipt = {
        "status": "CLOSED", "format": "raw-only-wider-bge-input-freeze-v1", "variant": f"title_bge_only{limit}",
        "cases": len(index), "paragraphs": sum(case["paragraphs"] for case in index),
        "input_index_sha256": digest(index_path), "ranking_provenance_sha256": digest(provenance_path),
        "source_bindings": bindings, "script_sha256": digest(Path(__file__)),
        "reference_reads": [], "paid_api_calls": 0, "network_attempts": [],
        "new_local_model_calls": 0, "final_selected_evidence_not_generated": True,
        "budget_status": "not_measured_requires_official_native_preflight", "publishable": False,
    }
    save_exclusive(output / "candidate-freeze.json", receipt)
    return receipt


def main() -> None:
    def audit(event, args):
        if event in {"socket.connect", "socket.connect_ex", "socket.bind", "socket.getaddrinfo"}:
            raise PermissionError("raw_only_preparation_network_forbidden")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            if Path(os.fsdecode(args[0])).name.lower() in FORBIDDEN_FILES:
                raise PermissionError("raw_only_preparation_reference_or_credential_forbidden")
    sys.addaudithook(audit)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--freeze-sha256", default=DEFAULT_FREEZE_SHA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(prepare(freeze_path=args.freeze, freeze_sha=args.freeze_sha256,
                             output=args.output, limit=args.limit), ensure_ascii=False))


if __name__ == "__main__":
    main()
