"""Read-only closure of a real UI task and its actual provider audit files.

No model, credential, reference or runtime mutation. Call after terminal state.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import httpx

ROOT = Path(__file__).resolve().parents[1]


def bound(path):
    path = Path(path).resolve()
    return {"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return bound(path)


def load(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def get(path):
    response = httpx.get("http://127.0.0.1:8790/api" + path, timeout=30)
    response.raise_for_status()
    return response.json()


def capture_native(directory, run, output, *, parents=()):
    if run["status"] not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("only_terminal_real_tasks_can_close")
    directory = Path(directory).resolve()
    requests = []
    for pre in sorted(directory.glob("request-*-preflight.json")):
        prefix = pre.name.removesuffix("preflight.json")
        row = {"preflight": bound(pre)}
        for key in ("wire", "finished", "settlement", "response", "field-projection", "syntax-recovery", "relation-projection"):
            path = directory / (prefix + key + ".json")
            if path.exists():
                row[key] = bound(path)
        if "finished" not in row:
            raise ValueError("request_not_durably_finished")
        requests.append(row)
    usage = {k: 0 for k in ("input_tokens", "output_tokens", "total_tokens")}
    for rows in run["usage"].values():
        for record in rows:
            for key in usage:
                if type(record.get(key)) is not int:
                    raise ValueError("unknown_usage_cannot_be_closed_as_known")
                usage[key] += record[key]
    result = {"closed": True, "fixture": False, "real_provider": True, "run_id": run["id"],
        "episode": bound(directory / "episode.json"), "requests": requests, "known_usage": usage,
        "costparents": list(parents)}
    cached = []
    ordered = []
    for hit in sorted(directory.glob("cache-*-hit.json")):
        prefix = hit.name.removesuffix("hit.json")
        details = load(hit)
        row = {"operation":details["operation"], "cache_hit":bound(hit)}
        for key in ("wire", "response", "finished", "field-projection", "syntax-recovery"):
            path = directory / (prefix + key + ".json")
            if path.exists():
                row[key] = bound(path)
        cached.append(row)
        ordered.append((int(prefix.split('-')[1]),details["operation"]))
    if cached:
        result["cached_requests"] = cached
    for row in requests:
        p = load(row["preflight"]["file"])
        finished = load(row["finished"]["file"])
        if p.get("admitted") is True and finished.get("request_started") is True:
            ordered.append((int(Path(row["preflight"]["file"]).name.split('-')[1]),p["operation"]))
    result["logical_operations"] = [operation for _,operation in sorted(ordered)]
    prior = directory / "resume-prior-usage.json"
    if prior.exists():
        result["resume_prior_usage"] = bound(prior)
    return save(output, result)


def main():
    parser = argparse.ArgumentParser()
    for name in ("run-id", "episode-dir", "output", "raw-topic-id", "raw-question-id", "input-freeze", "candidate-case"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--cost-parent", action="append", default=[], help="RUN_ID=EPISODE_DIRECTORY")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = get("/runs/" + args.run_id)
    parents = []
    captured_parents = {}
    for index, item in enumerate(args.cost_parent):
        rid, directory = item.split("=", 1)
        parent = get("/runs/" + rid)
        run_binding = save(output / f"parent-{index}-run.json", parent)
        activity = get("/runs/" + rid + "/activity")
        ancestors = set()
        for event in activity.get("events", []):
            if event.get("type") == "cost_lineage_seeded":
                ancestors.update(event["data"]["prior_run_ids"])
        if ancestors - set(captured_parents):
            raise ValueError("capture_cost_parents_in_ancestor_first_order")
        native = capture_native(directory, parent, output / f"parent-{index}-native-manifest.json",
            parents=[captured_parents[ancestor] for ancestor in sorted(ancestors)])
        captured_parents[rid] = {"run": run_binding, "native_manifest": native}
        parents.append(captured_parents[rid])
    native = capture_native(args.episode_dir, run, output / "native-episode-manifest.json", parents=parents)
    artifacts = {"run": save(output / "run.json", run), "native_episode_manifest": native,
        "candidate_catalog": bound(args.candidate_case)}
    tid, rev = run["topic_id"], run["target_revision_id"]
    artifacts["activity"] = save(output / "activity.json", get("/runs/" + args.run_id + "/activity"))
    if run["status"] != "succeeded":
        result = save(output / "failed-closure.json", {"closed": True, "status": run["status"],
            "artifacts": artifacts, "final_precision_recall": None, "references_read": False})
        print(json.dumps(result))
        return
    with sqlite3.connect(f"file:{(ROOT / '.runtime/business.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        revision = dict(db.execute("SELECT * FROM graph_revisions WHERE id=?", (rev,)).fetchone())
        snapshot = json.loads(db.execute("SELECT payload_json FROM revision_artifacts WHERE revision_id=?", (rev,)).fetchone()[0])
        paragraphs = [dict(row) for row in db.execute("""SELECT p.id,p.document_id,p.id paragraph_id,p.raw_para_id,p.text,
          p.start_cp document_start_cp,p.end_cp document_end_cp,d.raw_doc_id,d.title,d.content_sha256,d.content_url,
          d.raw_publish_time,d.parsed_publish_time,d.time_precision,d.source_credibility_json
          FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE d.topic_id=? ORDER BY d.rowid,p.rowid""", (tid,))]
    questions = get(f"/topics/{tid}/questions")
    qid = run["question_ids"][0]
    for key, value in {"revision": revision, "snapshot": snapshot, "question_result": snapshot["question_results"][qid],
        "source_paragraphs": paragraphs, "questions": questions,
        "export_readiness": get(f"/topics/{tid}/revisions/{rev}/export-readiness")}.items():
        artifacts[key] = save(output / (key + ".json"), value)
    closure = {"format": "workbench-real-quality-closed-v1", "closed": True, "fixture": False, "real_provider": True,
        "topic_id": tid, "run_id": args.run_id, "question_internal_id": qid,
        "raw_topic_id": args.raw_topic_id, "raw_question_id": args.raw_question_id,
        "artifacts": artifacts, "input_freeze": bound(args.input_freeze)}
    print(json.dumps(save(output / "closed-inputs.json", closure)))


if __name__ == "__main__":
    main()
