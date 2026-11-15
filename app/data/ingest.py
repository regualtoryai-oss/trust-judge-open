"""Strict source/question imports; answers and evidence annotations are forbidden.

``import_topic`` returns ``{topic_id, idempotent, counts}``, where counts contains
documents/paragraphs/questions. Source IDs stay unchanged within a topic and all
database IDs are separate UUIDs. A content change produces a new source snapshot.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from uuid import uuid4

from app.data.db import connect, initialize, transaction
from app.domain.spans import validate_span

DOC_KEYS = {"doc_id", "title", "content", "paragraphs", "publish_time", "source_credibility", "content_url"}
PARA_KEYS = {"para_id", "char_start", "char_end", "text"}
QUESTION_KEYS = {"question_id", "question"}


def _string(value, label, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{label} must be a {'possibly empty ' if empty else 'nonempty '}string")
    return value


def _allow_keys(value, allowed, required, label):
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValueError(f"{label} contains unsupported fields or lacks required fields; runtime accepts only raw sources and projected questions")


def _hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_inputs(documents: list[dict], questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return canonical validated copies. No gold-answer projection occurs here."""
    if not isinstance(documents, list) or not documents or not isinstance(questions, list) or not questions:
        raise ValueError("documents and projected questions must be nonempty lists")
    docs, seen_docs = [], set()
    for doc in documents:
        _allow_keys(doc, DOC_KEYS, {"doc_id", "title", "content", "paragraphs"}, "document")
        raw_id = _string(doc["doc_id"], "doc_id")
        if raw_id in seen_docs:
            raise ValueError("duplicate raw document ID within topic")
        seen_docs.add(raw_id)
        content = _string(doc["content"], "content")
        title = _string(doc["title"], "document title", empty=True)
        if not isinstance(doc["paragraphs"], list) or not doc["paragraphs"]:
            raise ValueError("document paragraphs must be a nonempty list")
        paras, seen_paras = [], set()
        for para in doc["paragraphs"]:
            _allow_keys(para, PARA_KEYS, {"para_id", "char_start", "char_end"}, "paragraph")
            para_id = _string(para["para_id"], "para_id")
            if para_id in seen_paras:
                raise ValueError("duplicate paragraph ID within document")
            seen_paras.add(para_id)
            start, end = para["char_start"], para["char_end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(content):
                raise ValueError("invalid document-relative codepoint paragraph span")
            text = para.get("text", content[start:end])
            validate_span(content, text, start, end)
            paras.append({"para_id": para_id, "char_start": start, "char_end": end, "text": text})
        publish_time = doc.get("publish_time")
        content_url = doc.get("content_url")
        for value, label in [(publish_time, "publish_time"), (content_url, "content_url")]:
            if value is not None:
                _string(value, label, empty=True)
        credibility = doc.get("source_credibility")
        if credibility is not None and (type(credibility) not in (int, float) or not math.isfinite(credibility) or not 0 <= credibility <= 1):
            raise ValueError("source_credibility must be a finite source grade from 0 to 1, or null")
        docs.append({"doc_id": raw_id, "title": title, "content": content, "paragraphs": sorted(paras, key=lambda p: p["para_id"]), "publish_time": publish_time, "content_url": content_url, "source_credibility": credibility})
    qs, seen_qs = [], set()
    for question in questions:
        _allow_keys(question, QUESTION_KEYS, QUESTION_KEYS, "question")
        qid = _string(question["question_id"], "question_id")
        if qid in seen_qs:
            raise ValueError("duplicate question ID within topic")
        seen_qs.add(qid)
        qs.append({"question_id": qid, "question": _string(question["question"], "question")})
    return sorted(docs, key=lambda d: d["doc_id"]), sorted(qs, key=lambda q: q["question_id"])


def import_topic(db_path: Path, documents: list[dict], questions: list[dict], title: str) -> dict:
    title = _string(title, "topic title")
    docs, qs = validate_inputs(documents, questions)
    manifest_hash = _hash(canonical_json({"title": title, "documents": docs, "questions": qs}))
    counts = {"documents": len(docs), "paragraphs": sum(len(d["paragraphs"]) for d in docs), "questions": len(qs)}
    initialize(db_path)
    conn = connect(db_path)
    try:
        with transaction(conn):
            old = conn.execute("SELECT id FROM topics WHERE source_manifest_sha256=?", (manifest_hash,)).fetchone()
            if old:
                return {"topic_id": old["id"], "idempotent": True, "counts": counts}
            topic_id = str(uuid4())
            conn.execute("INSERT INTO topics(id,title,source_manifest_sha256,created_at) VALUES (?,?,?,?)", (topic_id, title, manifest_hash, datetime.now(timezone.utc).isoformat()))
            for doc in docs:
                doc_id = str(uuid4())
                from app.domain.source_metadata import clean_publish_time
                publish = clean_publish_time(doc["publish_time"])
                conn.execute("INSERT INTO documents(id,topic_id,raw_doc_id,title,content,content_sha256,content_url,raw_publish_time,source_credibility_json,parsed_publish_time,time_precision) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (doc_id, topic_id, doc["doc_id"], doc["title"], doc["content"], _hash(doc["content"]), doc["content_url"], doc["publish_time"], canonical_json(doc["source_credibility"]), publish["normalized"], publish["time_precision"]))
                for para in doc["paragraphs"]:
                    conn.execute("INSERT INTO paragraphs(id,document_id,raw_para_id,start_cp,end_cp,text,text_sha256) VALUES (?,?,?,?,?,?,?)", (str(uuid4()), doc_id, para["para_id"], para["char_start"], para["char_end"], para["text"], _hash(para["text"])))
            for q in qs:
                conn.execute("INSERT INTO questions(id,topic_id,raw_question_id,question) VALUES (?,?,?,?)", (str(uuid4()), topic_id, q["question_id"], q["question"]))
            return {"topic_id": topic_id, "idempotent": False, "counts": counts}
    finally:
        conn.close()
