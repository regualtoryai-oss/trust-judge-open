"""Read-only real API source inspection; synthetic originals, no model traffic."""
from hashlib import sha256

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.data.ingest import import_topic
from app.main import create_app


@pytest.fixture
def source_client(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, model_api_key="", model_name="")
    chunks = ["🦊开场。", "限定：尚未确认。", "原始数字4人。", "另方称5人。", "需核验时间。", "结束。"]
    gaps = ["\r\n", "  \t", "\n🧭\n", "\r\n\r\n", " / "]
    content, paragraphs = "", []
    # Deliberately non-numeric IDs whose lexical order disagrees with CP order.
    for index, (raw_id, chunk) in enumerate(zip(["z", "a", "m", "b", "x", "c"], chunks)):
        start = len(content)
        content += chunk
        paragraphs.append({"para_id": raw_id, "char_start": start, "char_end": len(content)})
        if index < len(gaps):
            content += gaps[index]
    first = {"doc_id": "原始-doc", "title": "合成连续原文", "content": content, "paragraphs": paragraphs}
    # Legal overlap/nesting and same-interval distinct original IDs.
    second = {"doc_id": "overlap", "title": "原始重叠切分", "content": "🧭甲乙丙丁戊。",
              "paragraphs": [{"para_id": "outer", "char_start": 0, "char_end": 7},
                             {"para_id": "inner", "char_start": 1, "char_end": 3},
                             {"para_id": "same", "char_start": 1, "char_end": 3}]}
    topic = import_topic(settings.database_path, [first, second], [{"question_id": "q", "question": "合成核验？"}], "连续原文测试")
    with TestClient(create_app(settings=settings)) as client:
        repo = client.app.state.repo
        rows = repo.paragraphs(topic["topic_id"])
        yield client, repo, topic["topic_id"], first, {p["raw_para_id"]: p for p in rows}


def database_counts(repo):
    with repo.connection() as db:
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {name: db.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0] for name in names}


def test_context_preserves_real_source_slice_gaps_unicode_and_opaque_ids(source_client):
    client, repo, _, first, parents = source_client
    before = database_counts(repo)
    response = client.get(f"/api/paragraphs/{parents['a']['id']}/context")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["span_scope"] == "document" and body["bundle_purpose"] == "read_context"
    assert body["anchor_paragraph_id"] == parents["a"]["id"]
    assert body["paragraph_id"] == parents["z"]["id"]  # Range start, not clicked anchor.
    assert body["paragraph_count"] == 5
    assert [m["raw_para_id"] for m in body["members"]] == ["z", "a", "m", "b", "x"]
    assert body["source_paragraph_ids"] == [m["paragraph_id"] for m in body["members"]]
    assert body["quote"] == first["content"][body["start_cp"]:body["end_cp"]]
    assert "\r\n" in body["quote"] and "  \t" in body["quote"] and "\n🧭\n" in body["quote"]
    assert body["anchor_start_cp"] == parents["a"]["document_start_cp"]
    assert body["anchor_end_cp"] == parents["a"]["document_end_cp"]
    assert body["document_content_sha256"] == sha256(first["content"].encode()).hexdigest()
    assert database_counts(repo) == before
    assert client.get("/api/health").json()["model_configured"] is False
    # Same source request yields the same immutable proof, not a new revision.
    assert client.get(f"/api/paragraphs/{parents['a']['id']}/context").json() == body


@pytest.mark.parametrize("key,before,after,expected", [("z", 1, 3, ["z", "a", "m", "b"]), ("c", 1, 3, ["x", "c"]), ("b", 0, 0, ["b"])])
def test_context_edges_and_single_paragraph(source_client, key, before, after, expected):
    client, _, _, _, parents = source_client
    body = client.get(f"/api/paragraphs/{parents[key]['id']}/context?before={before}&after={after}").json()
    assert [member["raw_para_id"] for member in body["members"]] == expected


def test_context_preserves_overlaps_without_truncating_outer_interval(source_client):
    client, _, _, _, parents = source_client
    body = client.get(f"/api/paragraphs/{parents['inner']['id']}/context").json()
    assert [m["raw_para_id"] for m in body["members"]] == ["outer", "inner", "same"]
    assert body["quote"] == "🧭甲乙丙丁戊。" and body["end_cp"] == 7
    assert len(set(body["source_paragraph_ids"])) == 3


@pytest.mark.parametrize("query", ["before=-1", "after=-1", "before=5", "after=5", "before=4&after=1", "before=1.5", "after=nan"])
def test_invalid_windows_rejected_without_model_or_run(source_client, query):
    client, repo, _, _, parents = source_client
    before = database_counts(repo)
    assert client.get(f"/api/paragraphs/{parents['a']['id']}/context?{query}").status_code == 422
    assert database_counts(repo) == before


def test_unknown_anchor_and_integrity_failure_are_safe(source_client, monkeypatch):
    client, repo, _, _, parents = source_client
    assert client.get("/api/paragraphs/unknown-original/context").status_code == 404
    original = repo.source_documents
    def tampered(*args, **kwargs):
        docs = original(*args, **kwargs)
        docs[0]["content"] += "SYNTHETIC_DO_NOT_ECHO"
        return docs
    monkeypatch.setattr(repo, "source_documents", tampered)
    failed = client.get(f"/api/paragraphs/{parents['a']['id']}/context")
    assert failed.status_code == 409
    assert failed.json()["detail"]["code"] == "source_context_integrity_failed"
    assert "SYNTHETIC_DO_NOT_ECHO" not in failed.text


def test_source_manifest_uses_original_starts_not_ids_and_remains_topic_scoped(source_client):
    _, repo, topic_id, _, parents = source_client
    docs = repo.source_documents(topic_id, document_id=parents["z"]["document_id"])
    assert len(docs) == 1
    assert docs[0]["title"] == "合成连续原文"
    assert [p["raw_para_id"] for p in docs[0]["paragraph_manifest"]] == ["z", "a", "m", "b", "x", "c"]
    assert repo.source_documents("another-topic", document_id=parents["z"]["document_id"]) == []


def test_repository_source_projection_can_initialize_the_actual_complete_reader(source_client):
    from app.domain.source_bundle_tools import SourceBundleTools
    _, repo, topic_id, _, parents = source_client
    tools = SourceBundleTools(repo.source_documents(topic_id), repo.paragraphs(topic_id),
        {"question_id": "q", "question": "原始数字与限定是什么？"})
    tools.search("限定", limit=10)
    result = tools.read([parents["a"]["id"]], before=1, after=3)
    assert result["bundles"][0]["quote"] == repo.source_context(parents["a"]["id"])["quote"]
