"""Close a terminal UI bundle task without models, references, or DB writes."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import httpx

from app.domain.source_bundles import paragraph_manifest_sha256
from scripts.capture_workbench_real import bound, load, save


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "workbench-source-bundle-real-closed-v1"


def _get(api_base, path, *, raw=False):
    if not api_base.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("capture_local_workbench_only")
    response = httpx.get(api_base.rstrip("/") + path, timeout=30)
    response.raise_for_status()
    return response.content if raw else response.json()


def usage_observation(run):
    totals = {key: 0 for key in ("input_tokens", "output_tokens", "total_tokens")}
    unknown = 0
    for rows in run.get("usage", {}).values():
        for record in rows:
            if (record.get("usage_source") != "provider" or any(type(record.get(k)) is not int or record[k] < 0 for k in totals)
                    or record["input_tokens"] + record["output_tokens"] != record["total_tokens"]):
                unknown += 1
                continue
            for key in totals: totals[key] += record[key]
    return {"known_totals": totals, "unknown_usage_records": unknown, "all_observed_usage_known": unknown == 0,
            "usage_missing_does_not_mean_no_cost": True}


def capture_episode(directory):
    """Bind explicit audit files only; never inspect session/reasoning logs."""
    directory = Path(directory).resolve()
    tail = {"requests": []}
    for name in ("episode.json", "source-bundle-episode.json", "resume-prior-usage.json"):
        if (directory / name).is_file(): tail[name] = bound(directory / name)
    for pre in sorted(directory.glob("request-*-preflight.json")):
        prefix = pre.name.removesuffix("preflight.json")
        item = {"preflight": bound(pre)}
        for suffix in ("wire", "actual-wire", "finished", "settlement", "response", "syntax-recovery", "relation-projection", "analysis-input-projection"):
            path = directory / (prefix + suffix + ".json")
            if path.is_file(): item[suffix] = bound(path)
        tail["requests"].append(item)
    tail["budget_forecasts"] = [bound(path) for path in sorted(directory.glob("budget*.json"))]
    cached = []
    for hit in sorted(directory.glob("cache-*-hit.json")):
        prefix = hit.name.removesuffix("hit.json")
        row = {"operation": load(hit)["operation"], "cache_hit": bound(hit)}
        for suffix in ("wire", "response", "finished", "syntax-recovery", "relation-projection", "analysis-input-projection"):
            path = directory / (prefix + suffix + ".json")
            if path.is_file(): row[suffix] = bound(path)
        cached.append(row)
    if cached:
        tail["cached_requests"] = cached
    front = directory / "source-bundle"
    native = front / "native"
    front_files = {}
    for path in [front / name for name in ("raw-case.json", "documents.json", "paragraphs.json", "previous-usage.json", "launch-manifest.json", "provider-response.json",
            "prepared-case.json", "model-output.json", "artifact.json", "scope.json", "closed-receipt.json", "usage-report.json", "sealed-front-recovery.json")]:
        if path.is_file(): front_files[path.name] = bound(path)
    for name in ("closed-receipt.json", "budget.jsonl", "native-presence.jsonl", "native-request-index.jsonl", "submit-request.json", "tool-submission.json", "artifact.json", "usage-report.json"):
        if (native / name).is_file(): front_files["native/" + name] = bound(native / name)
    # Requests are exact native JSON only; response SSE may contain private
    # reasoning fields and is deliberately not opened or copied by capture.
    front_files["native_requests"] = [bound(path) for path in sorted(native.glob("native-request-*.json"))]
    front_files["budget_forecasts"] = [bound(path) for path in sorted(front.glob("budget*.json"))]
    provider_response = front / "provider-response.json"
    if provider_response.is_file():
        proof = load(provider_response).get("proof", {})
        if proof.get("format") == "langchain-source-bundle-workbench-proof-v1":
            front_files["langchain_requests"] = proof["front_requests"]
            if (proof.get("graded_selection") is True or proof.get("compact_analyses") is True
                    or proof.get("paragraph_quality") is True):
                front_files["selection_mode"] = {"selection_only": proof.get("selection_only", False),
                    "graded_selection": proof.get("graded_selection", False),
                    "graded_selection_policy": proof.get("graded_selection_policy"),
                    "selection_output_tokens": proof.get("selection_output_tokens"),
                    "compact_analyses": proof.get("compact_analyses", False), "use_type_is_final_relation": False}
                if proof.get("paragraph_quality") is True:
                    front_files["selection_mode"].update(paragraph_quality=True,
                        paragraph_quality_policy=proof.get("paragraph_quality_policy"))
                if proof.get("paragraph_quality_compact") is True:
                    front_files["selection_mode"].update(paragraph_quality_compact=True,
                        paragraph_quality_compact_policy=proof.get("paragraph_quality_compact_policy"))
            if proof.get("local_parent_rank") is not None:
                front_files["local_parent_rank"] = proof["local_parent_rank"]
            if proof.get("parent_token_pack") is True:
                front_files.setdefault("selection_mode", {}).update(parent_token_pack=True,
                    parent_token_pack_policy=proof.get("parent_token_pack_policy"))
    # Failed LC fronts also have immutable wire/settlement descriptors. They
    # remain cost ancestors even when no business provider-response was made.
    receipt = front / "closed-receipt.json"
    if receipt.is_file():
        closed = load(receipt)
        if closed.get("format") == "langchain-source-bundle-closed-receipt-v1":
            front_files.setdefault("langchain_requests", closed.get("front_requests", []))
    return {"format": "bundle-front-and-strict-tail-files-v1", "episode_directory": str(directory),
            "front": front_files, "tail": tail, "model_calls_by_capture": 0}


def read_run_lineage(database, run_id):
    """Read explicit cost/checkpoint ancestry; no Repository or checkpoint body IO."""
    with sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        rows, pending = {}, [run_id]
        while pending:
            rid = pending.pop()
            if rid in rows:
                continue
            row = db.execute("SELECT id,topic_id,question_ids_json,target_revision_id,model_config_json,status,usage_json FROM runs WHERE id=?", (rid,)).fetchone()
            option = db.execute("SELECT options_json FROM run_options WHERE run_id=?", (rid,)).fetchone()
            if row is None or option is None:
                raise ValueError("resume_ancestor_run_missing")
            config, options = json.loads(row["model_config_json"]), json.loads(option[0])
            revision = db.execute("SELECT input_manifest_sha256 FROM graph_revisions WHERE id=?", (row["target_revision_id"],)).fetchone()
            if revision is None:
                raise ValueError("resume_ancestor_manifest_missing")
            rows[rid] = {"id": rid, "topic_id": row["topic_id"], "question_ids": json.loads(row["question_ids_json"]),
                "status": row["status"], "usage": json.loads(row["usage_json"]), "input_manifest_sha256": revision[0],
                "model_config_sha256": sha256(row["model_config_json"].encode()).hexdigest(),
                "sealed_front_recovery": config.get("source_bundle_config", {}).get("sealed_front_recovery"),
                "cost_parent_run_ids": config.get("strict_evidence_cost_parent_run_ids", []),
                "resume_from_run_id": options.get("resume_from_run_id"),
                "resume_checkpoint_sources": options.get("resume_checkpoint_sources", {})}
            pending.extend(rows[rid]["cost_parent_run_ids"])
            if rows[rid]["resume_from_run_id"]:
                pending.append(rows[rid]["resume_from_run_id"])
    return {"format": "readonly-run-cost-checkpoint-lineage-v1", "current_run_id": run_id, "runs": rows}


def capture_resume_lineage(database, run, result, output, ancestor_episodes, getter):
    """Keep ancestor manifests separate: never transplant old files into a new run."""
    lineage = read_run_lineage(database, run["id"])
    expected = set(lineage["runs"]) - {run["id"]}
    if set(ancestor_episodes) != expected:
        raise ValueError("resume_requires_each_explicit_ancestor_episode_once")
    provenance, checkpoint = result.get("execution_provenance", {}), None
    if run.get("resumed_from_run_id"):
        checkpoint_database = Path(database).resolve().parent / "checkpoints.sqlite3"
        with sqlite3.connect(f"file:{checkpoint_database.as_posix()}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            # Read identity columns only, never serialized graph or model state.
            checkpoint = db.execute("SELECT thread_id,checkpoint_ns,checkpoint_id,parent_checkpoint_id FROM checkpoints WHERE thread_id=? AND checkpoint_ns='' AND checkpoint_id=?",
                (provenance.get("source_thread_id"), provenance.get("checkpoint_id"))).fetchone()
            if checkpoint is None:
                raise ValueError("resume_original_checkpoint_identity_missing")
    ancestors = []
    for rid in sorted(expected):
        ancestor_run = getter("/runs/" + rid)
        if ancestor_run.get("id") != rid:
            raise ValueError("resume_ancestor_API_identity_changed")
        ancestors.append({"run_id": rid,
            "run": save(output / f"ancestor-{rid}-run.json", ancestor_run),
            "activity": save(output / f"ancestor-{rid}-activity.json", getter("/runs/" + rid + "/activity")),
            "native_manifest": save(output / f"ancestor-{rid}-native.json", capture_episode(ancestor_episodes[rid]))})
    return {"format": "explicit-checkpoint-source-ancestor-files-v1", "database_projection": save(output / "run-lineage.json", lineage),
        "ancestors": ancestors, "execution_provenance": result.get("execution_provenance"),
        **({"checkpoint_identity": save(output / "checkpoint-identity.json", {"format": "readonly-original-checkpoint-identity-v1", **dict(checkpoint),
            "serialized_checkpoint_state_read": False})} if checkpoint is not None else {}),
        "ancestor_files_are_current_requests": False, "new_model_calls": 0}


def read_database(database, topic_id, revision_id):
    """Use a read-only SQLite transaction, never Repository construction."""
    with sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        revision_row = db.execute("SELECT * FROM graph_revisions WHERE id=? AND topic_id=?", (revision_id, topic_id)).fetchone()
        snapshot_row = db.execute("SELECT payload_json FROM revision_artifacts WHERE revision_id=?", (revision_id,)).fetchone()
        if revision_row is None or snapshot_row is None:
            raise ValueError("terminal_success_requires_actual_published_snapshot")
        documents = [dict(row) for row in db.execute("SELECT id,raw_doc_id,title,content,content_sha256 FROM documents WHERE topic_id=? ORDER BY rowid", (topic_id,))]
        parents = [dict(row) for row in db.execute("""SELECT p.id,p.document_id,p.raw_para_id,p.start_cp,p.end_cp,p.text_sha256
            FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE d.topic_id=? ORDER BY d.rowid,p.start_cp,p.rowid""", (topic_id,))]
        by_doc = {row["id"]: [] for row in documents}
        for p in parents: by_doc[p["document_id"]].append({key: p[key] for key in ("id", "raw_para_id", "start_cp", "end_cp", "text_sha256")})
        for doc in documents:
            doc["paragraph_manifest"] = by_doc[doc["id"]]
            doc["paragraph_manifest_sha256"] = paragraph_manifest_sha256(doc["paragraph_manifest"])
        paragraphs = [dict(row) for row in db.execute("""SELECT p.id,p.document_id,p.id paragraph_id,p.raw_para_id,p.text,
            p.start_cp document_start_cp,p.end_cp document_end_cp,d.raw_doc_id,d.title,d.content_sha256,d.content_url,
            d.raw_publish_time,d.parsed_publish_time,d.time_precision,d.source_credibility_json
            FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE d.topic_id=? ORDER BY d.rowid,p.rowid""", (topic_id,))]
    return {"revision": dict(revision_row), "snapshot": json.loads(snapshot_row[0]), "source_documents": documents, "source_paragraphs": paragraphs}


def verify_ui_start(proof, run, freeze_sha256):
    if (proof.get("format") != "actual-workbench-ui-start-v1" or proof.get("actual_browser_UI_start") is not True
            or proof.get("run_id") != run["id"] or proof.get("topic_id") != run["topic_id"]
            or run.get("question_ids") != [proof.get("question_internal_id")]
            or proof.get("input_freeze_sha256") != freeze_sha256):
        raise ValueError("actual_UI_start_must_bind_same_run_question_and_freeze")
    observation = proof.get("browser_observation")
    if not isinstance(observation, dict) or bound(observation["file"])["sha256"] != observation.get("sha256"):
        raise ValueError("actual_UI_start_browser_observation_hash_required")


def capture(run_id, episode_dir, input_freeze, ui_start_proof, output, raw_topic_id, raw_question_id, *, database=None, api_base="http://127.0.0.1:8790/api", getter=None, ancestor_episodes=None):
    get = getter or (lambda path, raw=False: _get(api_base, path, raw=raw))
    run = get("/runs/" + run_id)
    if run.get("id") != run_id or run.get("status") not in {"succeeded", "failed", "cancelled", "superseded"} or not run.get("completed_at"):
        raise ValueError("only_terminal_actual_UI_tasks_can_be_captured")
    freeze_d, ui_d = bound(input_freeze), bound(ui_start_proof)
    verify_ui_start(load(ui_start_proof), run, freeze_d["sha256"])
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    artifacts = {"run": save(output / "run.json", run), "activity": save(output / "activity.json", get("/runs/" + run_id + "/activity")),
                 "native_episode_manifest": save(output / "native-files.json", capture_episode(episode_dir))}
    closure = {"format": FORMAT, "closed": True, "fixture": False, "real_provider": True, "status": run["status"],
        "run_id": run_id, "topic_id": run["topic_id"], "question_internal_id": load(ui_start_proof)["question_internal_id"],
        "raw_topic_id": raw_topic_id, "raw_question_id": raw_question_id, "input_freeze": freeze_d, "ui_start_proof": ui_d,
        "usage_observation": usage_observation(run), "artifacts": artifacts, "references_read": False,
        "new_model_calls": 0, "final_precision_recall": None}
    if run["status"] != "succeeded":
        return save(output / "failed-closure.json", closure)
    captured = read_database(database or ROOT / ".runtime/business.sqlite3", run["topic_id"], run["target_revision_id"])
    snapshot = captured["snapshot"]
    qid = closure["question_internal_id"]
    if snapshot.get("run_id") != run_id or captured["revision"].get("state") != "ready" or snapshot.get("question_results", {}).get(qid, {}).get("origin_run_id") != run_id:
        raise ValueError("success_must_use_this_run_published_result_not_old_snapshot")
    prefix = f"/topics/{run['topic_id']}/revisions/{run['target_revision_id']}"
    api_publication = {"graph": get(prefix + "/graph"), "artifacts": get(prefix + "/artifacts")}
    for section in api_publication.values():
        for key, value in section.items():
            if key in snapshot and snapshot[key] != value:
                raise ValueError("actual_API_publication_differs_from_read_only_snapshot")
    captured.update(question_result=snapshot["question_results"][qid], questions=get(f"/topics/{run['topic_id']}/questions"),
        export_readiness=get(prefix + "/export-readiness"), api_publication=api_publication)
    if run.get("resumed_from_run_id") or ancestor_episodes:
        artifacts["resume_lineage"] = save(output / "resume-lineage.json", capture_resume_lineage(
            database or ROOT / ".runtime/business.sqlite3", run, captured["question_result"], output, ancestor_episodes or {}, get))
    for key, value in captured.items(): artifacts[key] = save(output / (key + ".json"), value)
    preview_bytes = get(prefix + "/submission-preview", raw=True)
    preview = output / "api-submission-preview.json"
    with preview.open("xb") as stream: stream.write(preview_bytes)
    artifacts["api_submission_preview"] = bound(preview)
    return save(output / "closed-inputs.json", closure)


def bind_browser_download(closure_path, proof_path, output):
    closure, proof = load(closure_path), load(proof_path)
    run = load(closure["artifacts"]["run"]["file"])
    if closure.get("status") != "succeeded" or proof.get("actual_browser_UI_download") is not True or proof.get("run_id") != run["id"] or proof.get("revision_id") != run["target_revision_id"]:
        raise ValueError("actual_browser_download_must_bind_successful_revision")
    downloaded = Path(proof["download_file"]["file"]).read_bytes()
    api = Path(closure["artifacts"]["api_submission_preview"]["file"]).read_bytes()
    if (sha256(downloaded).hexdigest() != proof["download_file"]["sha256"] or downloaded != api
            or sha256(api).hexdigest() != closure["artifacts"]["api_submission_preview"]["sha256"]
            or proof.get("exact_API_preview_sha256") != sha256(api).hexdigest()):
        raise ValueError("browser_download_must_equal_actual_API_bytes")
    return save(output, {"format": "bundle-actual-browser-download-binding-v1", "closed_inputs": bound(closure_path),
        "ui_export_proof": bound(proof_path), "download_file": proof["download_file"], "exact_API_preview_bytes_equal": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-id", "episode-dir", "input-freeze", "ui-start-proof", "raw-topic-id", "raw-question-id"):
        parser.add_argument("--" + name)
    parser.add_argument("--output", required=True)
    parser.add_argument("--database", default=str(ROOT / ".runtime/business.sqlite3"))
    parser.add_argument("--api-base", default="http://127.0.0.1:8790/api")
    parser.add_argument("--bind-closed")
    parser.add_argument("--browser-download-proof")
    parser.add_argument("--ancestor-episode", action="append", default=[], help="RUN_ID=ORIGINAL_EPISODE_DIRECTORY (all cost/checkpoint ancestors)")
    args = parser.parse_args()
    if args.bind_closed:
        result = bind_browser_download(args.bind_closed, args.browser_download_proof, args.output)
    else:
        required = (args.run_id, args.episode_dir, args.input_freeze, args.ui_start_proof, args.raw_topic_id, args.raw_question_id)
        if not all(required): parser.error("all real capture identity/freeze/UI arguments are required")
        ancestors = {}
        for value in args.ancestor_episode:
            rid, directory = value.split("=", 1)
            if rid in ancestors:
                parser.error("duplicate ancestor run ID")
            ancestors[rid] = directory
        result = capture(*required[:4], args.output, *required[4:], database=args.database, api_base=args.api_base, ancestor_episodes=ancestors)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
