"""Data-boundary and immutable-provenance tests, not blind model evaluation."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from app.data.db import connect, initialize, transaction
from app.data.ingest import import_topic
from scripts.prepare_sample_inputs import prepare

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "多智能体可信研判与证据链溯源" / "样例集-拓尔思-20261012"


def source_input():
    # Explicit synthetic source for boundary tests; never used by the application.
    return [{"doc_id": "doc_001", "title": "测试资料", "content": "🌊首句。末句。", "paragraphs": [{"para_id": "doc_001#p1", "char_start": 0, "char_end": 7}], "publish_time": "", "content_url": None, "source_credibility": 0.8}], [{"question_id": "q_001", "question": "测试问题是什么？"}]


def test_migration_repeatable_foreign_keys_and_transactions(tmp_path):
    db = tmp_path / "test.sqlite"
    initialize(db)
    initialize(db)
    conn = connect(db)
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 10000
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0] == 22
        with pytest.raises(sqlite3.IntegrityError), transaction(conn):
            conn.execute("INSERT INTO questions VALUES ('q','missing-topic','q_1','test')")
        assert not conn.in_transaction
        with transaction(conn):
            conn.execute("CREATE TABLE test_counter(value INTEGER)")
            conn.execute("INSERT INTO test_counter VALUES (1)")
            with pytest.raises(RuntimeError), transaction(conn):
                conn.execute("INSERT INTO test_counter VALUES (2)")
                raise RuntimeError("nested rollback")
        assert [row[0] for row in conn.execute("SELECT value FROM test_counter")] == [1]
    finally:
        conn.close()


def test_import_idempotency_and_changed_snapshot(tmp_path):
    docs, questions = source_input()
    path = tmp_path / "test.sqlite"
    first = import_topic(path, docs, questions, "topic")
    again = import_topic(path, deepcopy(docs), deepcopy(questions), "topic")
    assert first["topic_id"] == again["topic_id"]
    assert not first["idempotent"] and again["idempotent"]
    assert first["counts"] == {"documents": 1, "paragraphs": 1, "questions": 1}
    questions[0]["question"] = "已改的问题"
    changed = import_topic(path, docs, questions, "topic")
    assert changed["topic_id"] != first["topic_id"]


@pytest.mark.parametrize("field", ["conclusion", "supporting_facts", "conflicting_facts", "conclusion_id", "argument"])
def test_reject_gold_fields_before_database_creation(tmp_path, field):
    docs, questions = source_input()
    questions[0][field] = "forbidden"
    with pytest.raises(ValueError, match="runtime accepts only"):
        import_topic(tmp_path / "invalid.sqlite", docs, questions, "topic")
    assert not (tmp_path / "invalid.sqlite").exists()


def test_reject_annotation_hidden_in_document_or_bad_span(tmp_path):
    docs, questions = source_input()
    docs[0]["evidence"] = []
    with pytest.raises(ValueError):
        import_topic(tmp_path / "test.sqlite", docs, questions, "topic")
    docs[0].pop("evidence")
    docs[0]["paragraphs"][0]["text"] = "🌊首句。改句。"
    with pytest.raises(ValueError, match="does not match"):
        import_topic(tmp_path / "test.sqlite", docs, questions, "topic")


def test_source_identity_immutable_by_sql(tmp_path):
    docs, questions = source_input()
    path = tmp_path / "test.sqlite"
    result = import_topic(path, docs, questions, "topic")
    conn = connect(path)
    try:
        row = conn.execute("SELECT d.*,p.text,p.start_cp,p.end_cp FROM documents d JOIN paragraphs p ON p.document_id=d.id").fetchone()
        assert row["raw_doc_id"] == "doc_001" and row["text"] == docs[0]["content"]
        assert json.loads(row["source_credibility_json"]) == 0.8
        for statement in ["UPDATE documents SET content='altered'", "DELETE FROM documents", "UPDATE paragraphs SET text='altered'", "DELETE FROM paragraphs", "UPDATE questions SET question='altered'", "DELETE FROM questions"]:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute(statement)
        assert conn.execute("SELECT topic_id FROM questions").fetchone()[0] == result["topic_id"]
    finally:
        conn.close()


def test_two_real_topics_keep_colliding_raw_ids_isolated(tmp_path):
    if not SAMPLE_DIR.is_dir():
        pytest.skip("official development samples are not distributed with the code")
    path = tmp_path / "real.sqlite"
    inputs = []
    for raw in sorted(SAMPLE_DIR.glob("*/raw_documents.json"))[:2]:
        docs = json.loads(raw.read_text(encoding="utf-8-sig"))
        # Gold annotations never enter the importer: explicit preprocessing only.
        questions = [{"question_id": q["question_id"], "question": q["question"]} for q in json.loads(raw.with_name("conclusions.json").read_text(encoding="utf-8-sig"))]
        inputs.append(import_topic(path, docs, questions, raw.parent.name))
    conn = connect(path)
    try:
        rows = conn.execute("SELECT id,topic_id FROM documents WHERE raw_doc_id='doc_001'").fetchall()
        assert len(rows) == 2 and len({row["id"] for row in rows}) == 2
        assert {row["topic_id"] for row in rows} == {item["topic_id"] for item in inputs}
        assert conn.execute("SELECT COUNT(*) FROM questions WHERE raw_question_id='q_001'").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0] == sum(item["counts"]["paragraphs"] for item in inputs)
    finally:
        conn.close()


def test_preparation_projects_questions_and_does_not_write_to_sources(tmp_path):
    source = tmp_path / "source"
    topic = source / "case"
    topic.mkdir(parents=True)
    docs, questions = source_input()
    questions[0]["conclusion"] = "SYNTHETIC GOLD MUST NOT CROSS BOUNDARY"
    (topic / "raw_documents.json").write_text(json.dumps(docs), encoding="utf-8")
    (topic / "conclusions.json").write_text(json.dumps(questions), encoding="utf-8")
    (topic / "evidence_samples.json").write_text("this file must not even be parsed", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in topic.iterdir()}
    output = tmp_path / "runtime"
    manifest = prepare(source, output)
    payload = json.loads((output / "case/input.json").read_text(encoding="utf-8"))
    assert set(payload) == {"title", "documents", "questions"}
    assert set(payload["questions"][0]) == {"question_id", "question"}
    assert "SYNTHETIC GOLD" not in (output / "case/input.json").read_text(encoding="utf-8")
    assert before == {p.name: p.read_bytes() for p in topic.iterdir()}
    assert manifest["counts"]["questions"] == 1
    with pytest.raises(ValueError, match="outside"):
        prepare(source, source / "runtime")
