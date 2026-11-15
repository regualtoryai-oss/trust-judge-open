"""Raw-only future-question preparation. No gold, Repository, API or runtime write.

The serialized selection request is an explicitly labelled v1-schema cost proxy;
the new runtime must freeze its final v2 wire before any real provider call.
"""
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess

from langchain_openai import ChatOpenAI

from app.agents.strict_evidence import prepare_selection_request, validate_raw_case
from app.agents.strict_transport import strict_tool, TOKENIZER, TOKENIZER_SHA
from app.domain.paragraph_ranked import source_manifest, source_material_policy
from app.domain.question_recall import recall_question_candidates
from app.domain.source_metadata import source_credibility_value
from scripts.audit_dsh_closed_episode import bound
from scripts.score_workbench_real_episode import verify_source_rows, verify_legacy_sources, known_usage

ROOT = Path(__file__).resolve().parents[1]


def load_json(path, expected=None):
    path = Path(path)
    if path.name.lower() in {".env", "conclusions.json", "evidence_samples.json"} or path.name.lower().startswith("gold_"):
        raise ValueError("raw_preparation_reference_or_credential_forbidden")
    raw, binding = bound(path, expected)
    return json.loads(raw), binding


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return bound(path)[1]


def read_workbench(database, question_text, raw_question_id):
    """One SQLite read transaction; no model configuration or credential IO."""
    with sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        questions = [dict(row) for row in db.execute(
            "SELECT * FROM questions WHERE question=? AND raw_question_id=?", (question_text, raw_question_id))]
        if len(questions) != 1:
            raise ValueError("one_existing_workbench_raw_question_required")
        question = questions[0]
        topic = dict(db.execute("SELECT * FROM topics WHERE id=?", (question["topic_id"],)).fetchone())
        paragraphs = [dict(row) for row in db.execute("""SELECT p.id,p.document_id,p.raw_para_id,p.text,
            p.start_cp document_start_cp,p.end_cp document_end_cp,d.raw_doc_id,d.title,d.content_sha256,d.content_url,
            d.raw_publish_time,d.parsed_publish_time,d.time_precision,d.source_credibility_json
            FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE d.topic_id=? ORDER BY d.rowid,p.rowid""", (topic["id"],))]
        runs = []
        for record in db.execute("SELECT id,status,question_ids_json,usage_json,error_json,created_at,completed_at FROM runs WHERE topic_id=? ORDER BY created_at", (topic["id"],)):
            row = dict(record)
            if question["id"] not in json.loads(row.pop("question_ids_json")):
                continue
            usage = json.loads(row.pop("usage_json")).get(question["id"], [])
            row["question_usage"] = usage
            row["question_known_usage"] = known_usage(usage)
            row["error"] = json.loads(row.pop("error_json")) if row.get("error_json") else None
            runs.append(row)
        return question, topic, paragraphs, runs


def prepare_raw(question, paragraphs, *, variant="window_balanced", candidate_limit=60):
    manifest = source_manifest(paragraphs)
    material = source_material_policy(question, manifest)
    eligible = [row for row in manifest if row["id"] in set(material["eligible_paragraph_ids"]) and row["text"].strip()]
    recall = recall_question_candidates(question, eligible, limit=candidate_limit, variant=variant)
    sources = {row["id"]: row for row in paragraphs}
    catalog, metadata = [], {}
    for i, selected in enumerate(recall["candidates"], 1):
        row = sources[selected["paragraph_id"]]; p = "P" + str(i)
        catalog.append({"paragraph_ref": p, "doc_id": row["raw_doc_id"], "para_id": row["raw_para_id"],
            "title": row["title"], "text": row["text"], "paragraph_complete": True, "parent_complete": True})
        grade, _basis = source_credibility_value(row)
        metadata[p] = {"paragraph_id": row["id"], "doc_id": row["raw_doc_id"], "para_id": row["raw_para_id"],
            "source_credibility": grade, "raw_publish_time": row.get("raw_publish_time"),
            "source_text_sha256_utf8": sha256(row["text"].encode()).hexdigest(),
            "parent_start_cp": 0, "parent_end_cp": len(row["text"]),
            "document_start_cp": row["document_start_cp"], "document_end_cp": row["document_end_cp"]}
    raw = {"format": "strict-workbench-raw-case-v2-preparation-only", "publishable": False,
        "question": {"question_id": question["raw_question_id"], "question": question["question"]},
        "catalog": catalog, "source_metadata": metadata,
        "retrieval": {"strategy": f"{variant}{candidate_limit}_full_original_parents", "candidate_limit": candidate_limit,
            "source_manifest_sha256": sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "material_policy": material, "all_corpus_read_by_model": False, "model_calls": 0}}
    return validate_raw_case(raw), recall


def selection_wire_proxy(raw_case, *, compact=False):
    if compact:
        from app.agents.compact_selection import prepare_compact_selection_request
        schema, messages = prepare_compact_selection_request(raw_case)
    else:
        schema, messages = prepare_selection_request(raw_case)
    model = ChatOpenAI(model="deepseek-v4-pro", api_key="explicit-no-network-preparation-placeholder",
        base_url="https://api.deepseek.com/beta", max_retries=0, temperature=0, use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}, "max_tokens": 4096})
    structured = model.with_structured_output(strict_tool(schema), method="function_calling", strict=True, include_raw=True)
    kwargs = {key: value for key, value in structured.steps[0].steps__["raw"].kwargs.items()
        if key not in {"ls_structured_output_format", "structured_output_format"}}
    wire = model._get_request_payload(messages, stop=None, **kwargs)
    wire.update(wire.pop("extra_body", {}))
    if any(key in wire for key in ("api_key", "authorization", "headers")):
        raise ValueError("no_credentials_in_proxy_wire")
    return wire


def prepare(inventory_path, expected_inventory_sha, database, raw_input, topic_id, question_id, output,
            *, variant="window_balanced", candidate_limit=60):
    output = Path(output).resolve()
    output.relative_to(ROOT / "docs/reports")
    inventory, inventory_binding = load_json(inventory_path, expected_inventory_sha)
    candidates = [row for row in inventory["candidates"] if row["topic_id"] == topic_id and row["question"]["question_id"] == question_id]
    if len(candidates) != 1:
        raise ValueError("one_existing_frozen_baseline_required")
    baseline = candidates[0]
    old_result, result_binding = load_json(baseline["legacy_result"]["file"], baseline["legacy_result"]["sha256"])
    old_terminal, terminal_binding = load_json(baseline["legacy_terminal"]["file"], baseline["legacy_terminal"]["sha256"])
    raw, raw_binding = load_json(raw_input, baseline["legacy_raw_input"]["sha256"])
    old_raw, old_raw_binding = load_json(baseline["legacy_raw_input"]["file"], baseline["legacy_raw_input"]["sha256"])
    if old_raw != raw or old_terminal["status"] != "succeeded" or old_result["origin_run_id"] != old_terminal["run_id"]:
        raise ValueError("unchanged_original_and_effective_closed_baseline_required")
    question, topic, paragraphs, runs = read_workbench(database, baseline["question"]["question"], question_id)
    contexts, _sources = verify_source_rows(raw, paragraphs)
    verify_legacy_sources(old_result, contexts)
    raw_case, recall = prepare_raw(question, paragraphs, variant=variant, candidate_limit=candidate_limit)
    wire = selection_wire_proxy(raw_case)
    if output.exists():
        raise ValueError("preparation_output_must_be_new_directory")
    output.mkdir(parents=True)
    case_binding = save(output / "candidate-case.json", raw_case)
    recall_binding = save(output / "raw-recall-proof.json", recall)
    paragraph_binding = save(output / "workbench-source-paragraphs.json", paragraphs)
    wire_path = output / "selection-v1-schema-cost-proxy-wire.json"
    with wire_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(wire, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    if bound(TOKENIZER)[1]["sha256"] != TOKENIZER_SHA:
        raise ValueError("bound_tokenizer_changed")
    command = [str(ROOT / ".runtime/v20-embedding-env/Scripts/python.exe"), "-c",
        "from tokenizers import Tokenizer;from pathlib import Path;import sys;print(len(Tokenizer.from_file(sys.argv[1]).encode(Path(sys.argv[2]).read_text('utf8')).ids))", str(TOKENIZER), str(wire_path)]
    count = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    compact_wire = selection_wire_proxy(raw_case, compact=True)
    compact_path = output / "selection-v2-integer-cost-proxy-wire.json"
    with compact_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(compact_wire, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    compact_command = [*command[:-1], str(compact_path)]
    compact_count = subprocess.run(compact_command, capture_output=True, text=True, timeout=60, check=True)
    selected = raw_case["catalog"]
    body_lengths = sorted((len(row["text"]) for row in selected), reverse=True)
    result = {"format": "raw-only-window-balanced-scoped-preparation-v1", "status": "PREPARATION_NOT_LAUNCH_AUTHORITY",
        "topic_id": topic_id, "question": baseline["question"], "workbench_question": question, "workbench_topic": topic,
        "references_read": False, "model_calls": 0, "new_tokens": 0, "runtime_files_modified": False,
        "inventory": inventory_binding, "raw_input": raw_binding, "baseline_original_raw": old_raw_binding,
        "baseline": {"result": result_binding, "terminal": terminal_binding, "historical_usage": baseline["legacy_usage"],
            "usage_is_new_cost": False, "evidence_records": len(old_result["evidence"]),
            "unique_original_parents": len({(row["raw_doc_id"], row["raw_para_id"]) for row in old_result["evidence"]}),
            "same_original_raw_verified": True, "quote_spans_verified": True,
            "historical_terminal_has_embedded_result_sha": False},
        "main_DB_prior_same_question_runs": runs,
        "candidate_case": case_binding, "recall_proof": recall_binding, "source_paragraphs": paragraph_binding,
        "scope": {"recall_strategy": variant, "source_paragraphs": len(paragraphs), "candidate_parents": len(selected),
            "full_original_parent_codepoints": sum(body_lengths), "distinct_docs": len({row["doc_id"] for row in selected}),
            "per_doc_counts": dict(Counter(row["doc_id"] for row in selected)),
            "longest_parent_CP": body_lengths[0] if body_lengths else 0,
            "longest_ten_full_parents_CP": sum(body_lengths[:10]),
            "all_parents_complete": True, "candidate_coverage_of_gold": "not_measured"},
        "proxy": {"wire": bound(wire_path)[1], "input_token_estimate": int(count.stdout.strip()),
            "compact_wire": bound(compact_path)[1], "compact_integer_input_token_estimate": int(compact_count.stdout.strip()),
            "wire_codepoints": len(wire_path.read_text("utf8")), "is_actual_provider_usage": False,
            "is_frozen_final_v2_wire": False, "schema": "existing_v1_required_P_selection_proxy_only",
            "tokenizer": bound(TOKENIZER)[1], "reservation_example": {"selection_output": 4096, "safety": 512}},
        "source_guards": [bound(Path(__file__))[1], *[bound(ROOT / file)[1] for file in (
            "app/domain/question_recall.py", "app/domain/paragraph_ranked.py", "app/agents/strict_evidence.py", "app/agents/strict_transport.py")]]}
    return result, save(output / "raw-only-preparation.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "expected-inventory-sha", "database", "raw-input", "topic-id", "question-id", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--recall-strategy", choices=("window_balanced", "balanced"), default="window_balanced")
    parser.add_argument("--candidate-limit", type=int, choices=(40, 60), default=60)
    args = parser.parse_args()
    result, binding = prepare(args.inventory, args.expected_inventory_sha, args.database, args.raw_input,
        args.topic_id, args.question_id, args.output, variant=args.recall_strategy, candidate_limit=args.candidate_limit)
    print(json.dumps({"binding": binding, "scope": result["scope"], "proxy_tokens": result["proxy"]["input_token_estimate"],
        "compact_proxy_tokens": result["proxy"]["compact_integer_input_token_estimate"],
        "prior_same_question_main_DB_runs": len(result["main_DB_prior_same_question_runs"])}, ensure_ascii=False))
