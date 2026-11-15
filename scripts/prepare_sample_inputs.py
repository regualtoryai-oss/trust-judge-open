"""Project public development samples into answer-free runtime input files.

Preparation boundaries (this script and the explicit browser import preview)
may project conclusions.json to question_id/question only. The strict runtime
importer rejects answer-bearing input. No evidence_samples file is opened.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data.ingest import canonical_json, validate_inputs


def prepare(source_dir: Path, output_dir: Path) -> dict:
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if not source_dir.is_dir():
        raise ValueError("source directory does not exist")
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("runtime output must be outside the source sample directory")
    topics = []
    pending = []
    for document_path in sorted(source_dir.glob("*/raw_documents.json")):
        folder = document_path.parent
        question_path = folder / "conclusions.json"
        documents = json.loads(document_path.read_text(encoding="utf-8-sig"))
        source_questions = json.loads(question_path.read_text(encoding="utf-8-sig"))
        if not isinstance(source_questions, list) or not all(isinstance(q, dict) and "question_id" in q and "question" in q for q in source_questions):
            raise ValueError(f"invalid question source in {folder.name}")
        questions = [{"question_id": q["question_id"], "question": q["question"]} for q in source_questions]
        docs, qs = validate_inputs(documents, questions)
        payload = {"title": folder.name, "documents": docs, "questions": qs}
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        relative = Path(folder.name) / "input.json"
        pending.append((output_dir / relative, encoded))
        topics.append({"title": folder.name, "input_file": relative.as_posix(), "sha256": sha256(encoded).hexdigest(), "counts": {"documents": len(docs), "paragraphs": sum(len(d["paragraphs"]) for d in docs), "questions": len(qs)}})
    if not topics:
        raise ValueError("no raw_documents.json files found one directory below source")
    # Validate the entire collection before writing any runtime artifact.
    for path, encoded in pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)
    manifest = {"input_format": "trust-judge-runtime-input-v1", "purpose": "development regression; these public examples are not blind evaluation", "topics": topics, "counts": {key: sum(t["counts"][key] for t in topics) for key in ("documents", "paragraphs", "questions")}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.source_dir, args.output_dir)
    print(json.dumps({"topics": len(result["topics"]), **result["counts"], "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
