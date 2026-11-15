"""No-model source-integrity demo using original synthetic fixtures only.

Runs in a disposable database, never reads .env, and never constructs a model.
This demonstrates import/idempotency/Unicode spans, not LLM quality.
"""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile

from app.data.ingest import import_topic


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = sorted((root / "fixtures" / "import_samples").glob("*/input.json"))
    results = []
    with tempfile.TemporaryDirectory(prefix="trust-judge-synthetic-") as temporary:
        database = Path(temporary) / "demo.sqlite"
        for source in samples:
            payload = json.loads(source.read_text(encoding="utf-8"))
            first = import_topic(database, payload["documents"], payload["questions"], payload["title"])
            repeated = import_topic(database, payload["documents"], payload["questions"], payload["title"])
            if first["topic_id"] != repeated["topic_id"] or not repeated["idempotent"]:
                raise RuntimeError("synthetic import idempotency failed")
            results.append({"scenario": source.parent.name, "counts": first["counts"], "repeat_is_idempotent": True})
        with closing(sqlite3.connect(database)) as connection:
            rows = connection.execute(
                "SELECT d.content,p.text,p.start_cp,p.end_cp "
                "FROM paragraphs p JOIN documents d ON d.id=p.document_id"
            ).fetchall()
            if not rows or any(content[start:end] != text for content, text, start, end in rows):
                raise RuntimeError("synthetic Unicode codepoint span check failed")
        print(json.dumps({"mode": "synthetic-no-model", "model_calls": 0,
                          "paragraph_spans_verified": len(rows), "scenarios": results},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
