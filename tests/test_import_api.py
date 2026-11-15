"""Browser import security and provenance tests using synthetic inputs only."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.data import import_service
from app.data.db import connect
from app.main import create_app


def source_input():
    content = "🦊初报为3人。更正为4人。"
    return {"title": "合成导入校验", "documents": [{"doc_id": "原文_001", "title": "合成原始资料", "content": content,
            "paragraphs": [{"para_id": "段落_001", "char_start": 0, "char_end": len(content)}],
            "publish_time": None, "content_url": None, "source_credibility": 0.8}],
            "questions": [{"question_id": "题目_001", "question": "更正通报后的数字是多少？"}]}


def upload(payload=None):
    return {"entries": [{"name": "input.json", "content": json.dumps(payload or source_input(), ensure_ascii=True)}]}


@pytest.fixture
def service_client(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="", model_api_key="")
    with TestClient(create_app(settings=settings)) as client:
        yield client, settings


def table_count(settings, table):
    db = connect(settings.database_path)
    try:
        return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        db.close()


def preview(client, body=None):
    response = client.post("/api/imports/preview", json=body or upload())
    assert response.status_code == 200, response.text
    return response.json()


def commit(client, planned, **extra):
    return client.post("/api/imports/commit", json={"preview_id": planned["preview_id"], "fingerprint": planned["fingerprint"], **extra})


def test_preview_is_read_only_commit_preserves_raw_unicode_ids_and_coordinates(service_client):
    client, settings = service_client
    planned = preview(client)
    assert planned["counts"] == {"documents": 1, "paragraphs": 1, "questions": 1}
    assert table_count(settings, "topics") == table_count(settings, "questions") == table_count(settings, "runs") == 0
    response = commit(client, planned)
    assert response.status_code == 200, response.text
    assert response.json()["idempotent"] is False
    topic_id = response.json()["topic_id"]
    db = connect(settings.database_path)
    try:
        source = db.execute("SELECT d.*,p.* FROM documents d JOIN paragraphs p ON p.document_id=d.id").fetchone()
        assert source["raw_doc_id"] == "原文_001" and source["raw_para_id"] == "段落_001"
        assert source["content"] == source["text"] == source_input()["documents"][0]["content"]
        assert source["end_cp"] == len(source["content"])
        assert db.execute("SELECT raw_question_id FROM questions").fetchone()[0] == "题目_001"
    finally:
        db.close()
    assert client.get(f"/api/topics/{topic_id}/questions").json()[0]["raw_question_id"] == "题目_001"
    assert table_count(settings, "runs") == table_count(settings, "evidence") == 0


def test_same_content_reuses_topic_changed_content_creates_snapshot_and_original_immutable(service_client):
    client, settings = service_client
    first = commit(client, preview(client)).json()
    repeated_preview = preview(client)
    assert repeated_preview["existing_topic_id"] == first["topic_id"]
    repeated = commit(client, repeated_preview).json()
    assert repeated["topic_id"] == first["topic_id"] and repeated["idempotent"] is True
    changed = source_input()
    changed["questions"][0]["question"] = "另一个问题？"
    next_topic = commit(client, preview(client, upload(changed))).json()
    assert next_topic["topic_id"] != first["topic_id"]
    assert table_count(settings, "topics") == 2
    assert client.get(f"/api/topics/{first['topic_id']}/questions").json()[0]["question"] == source_input()["questions"][0]["question"]


def test_question_answer_projection_is_explicit_requires_ack_and_never_retains_values(service_client):
    client, settings = service_client
    payload = source_input()
    payload["questions"][0].update({"conclusion": "SYNTHETIC_GOLD_DO_NOT_KEEP", "supporting_facts": [{"secret": "SYNTHETIC_GOLD_DO_NOT_KEEP"}]})
    planned = preview(client, upload(payload))
    isolated_previews = import_service.ImportService(client.app.state.repo)
    isolated_previews.preview(upload(payload))
    assert "SYNTHETIC_GOLD" not in repr(isolated_previews.previews)
    assert planned["excluded_fields"] == [{"path": "questions[]", "fields": ["conclusion", "supporting_facts"], "count": 2}]
    assert "SYNTHETIC_GOLD" not in json.dumps(planned)
    assert commit(client, planned).json()["detail"]["code"] == "import_projection_ack_required"
    assert table_count(settings, "topics") == 0
    imported = commit(client, planned, acknowledge_excluded_fields=True)
    assert imported.status_code == 200
    assert b"SYNTHETIC_GOLD" not in settings.database_path.read_bytes()
    db = connect(settings.database_path)
    try:
        assert "SYNTHETIC_GOLD" not in "".join(db.iterdump())
    finally:
        db.close()


def test_raw_documents_and_legacy_conclusions_pair_projects_question_only(service_client):
    client, _ = service_client
    payload = source_input()
    payload["questions"][0]["conclusion"] = "synthetic excluded answer"
    body = {"title": "官方文件形式合成测试", "entries": [
        {"name": "raw_documents.json", "content": json.dumps(payload["documents"])},
        {"name": "conclusions.json", "content": json.dumps(payload["questions"])},
    ]}
    planned = preview(client, body)
    assert planned["title"] == body["title"] and planned["excluded_fields"][0]["count"] == 1
    assert commit(client, planned, acknowledge_excluded_fields=True).status_code == 200


def test_documents_and_questions_pair_and_bom_and_title_override(service_client):
    client, _ = service_client
    payload = source_input()
    body = {"title": "自定义标题", "entries": [
        {"name": "documents.json", "content": "\ufeff" + json.dumps(payload["documents"])},
        {"name": "questions.json", "content": json.dumps(payload["questions"])},
    ]}
    assert preview(client, body)["title"] == "自定义标题"
    body = upload()
    body["title"] = "另一个标题"
    assert preview(client, body)["title"] == "另一个标题"


@pytest.mark.parametrize("location", ["top", "document", "paragraph"])
def test_answers_outside_question_projection_are_rejected_before_writes(service_client, location):
    client, settings = service_client
    payload = source_input()
    target = payload if location == "top" else payload["documents"][0] if location == "document" else payload["documents"][0]["paragraphs"][0]
    target["evidence_samples"] = [{"quote": "synthetic answer"}]
    response = client.post("/api/imports/preview", json=upload(payload))
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "import_unsupported_fields"
    assert table_count(settings, "topics") == 0


@pytest.mark.parametrize("filename", ["evidence_samples.json", "answers.json", "data.zip", "../input.json", "a\\input.json", "C:input.json"])
def test_unsupported_evidence_archive_and_path_names_are_rejected_without_parsing(service_client, filename):
    client, settings = service_client
    response = client.post("/api/imports/preview", json={"entries": [{"name": filename, "content": "UNPARSEABLE_SECRET"}]})
    assert response.status_code == 422
    assert "UNPARSEABLE_SECRET" not in response.text
    assert table_count(settings, "topics") == 0


@pytest.mark.parametrize("change,code", [
    (lambda p: p["documents"].append(deepcopy(p["documents"][0])), "import_duplicate_document_id"),
    (lambda p: p["documents"][0]["paragraphs"].append(deepcopy(p["documents"][0]["paragraphs"][0])), "import_duplicate_paragraph_id"),
    (lambda p: p["questions"].append(deepcopy(p["questions"][0])), "import_duplicate_question_id"),
    (lambda p: p["documents"][0]["paragraphs"][0].update(char_start=True), "import_invalid_span"),
    (lambda p: p["documents"][0]["paragraphs"][0].update(char_end=10000), "import_invalid_span"),
    (lambda p: p["documents"][0]["paragraphs"][0].update(text="🦊改写原文"), "import_span_text_mismatch"),
    (lambda p: p["documents"][0].update(source_credibility=True), "import_invalid_credibility"),
    (lambda p: p["documents"][0].update(source_credibility=10**1000), "import_invalid_credibility"),
    (lambda p: p["documents"][0].update(content="\ud800"), "import_invalid_unicode"),
    (lambda p: p["questions"][0].update(question_id=""), "import_invalid_string"),
    (lambda p: p["questions"][0].pop("question"), "import_missing_field"),
    (lambda p: p.update(documents=[]), "import_invalid_shape"),
])
def test_precise_validation_errors_are_controlled_and_import_nothing(service_client, change, code):
    client, settings = service_client
    payload = source_input()
    change(payload)
    response = client.post("/api/imports/preview", json=upload(payload))
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == code
    assert table_count(settings, "topics") == 0


def test_utf16_emoji_coordinate_rejected_not_silently_normalized(service_client):
    client, _ = service_client
    payload = source_input()
    content = payload["documents"][0]["content"]
    payload["documents"][0]["paragraphs"][0]["char_end"] = len(content.encode("utf-16-le")) // 2
    response = client.post("/api/imports/preview", json=upload(payload))
    assert response.json()["detail"]["code"] == "import_invalid_span"
    assert "UTF-16" in response.json()["detail"]["message"]


def test_overlapping_paragraphs_are_legal_as_in_official_raw_sources(service_client):
    client, _ = service_client
    payload = source_input()
    payload["documents"][0]["paragraphs"].append({"para_id": "段落_002", "char_start": 1, "char_end": 3})
    assert preview(client, upload(payload))["counts"]["paragraphs"] == 2


def test_expanded_overlapping_source_limit_precedes_canonical_text_supplement(service_client, monkeypatch):
    client, settings = service_client
    payload = source_input()
    content = payload["documents"][0]["content"]
    payload["documents"][0]["paragraphs"].append({"para_id": "段落_002", "char_start": 0, "char_end": len(content)})
    literal_bytes = len(import_service.canonical_json(content).encode("utf-8"))
    monkeypatch.setattr(import_service, "MAX_EXPANDED_SOURCE_BYTES", literal_bytes * 2)

    def must_not_supplement(*args, **kwargs):
        pytest.fail("Expanded source budget must be checked before validate_inputs supplements paragraph text")

    monkeypatch.setattr(import_service, "validate_inputs", must_not_supplement)
    response = client.post("/api/imports/preview", json=upload(payload))
    assert response.status_code == 413, response.text
    assert response.json()["detail"]["code"] == "import_expanded_source_too_large"
    assert "documents[0].paragraphs[1]" in response.json()["detail"]["message"]
    assert table_count(settings, "topics") == 0


def test_expansion_budget_counts_utf8_json_escape_bytes_and_accepts_boundary(service_client, monkeypatch):
    client, _ = service_client
    payload = source_input()
    content = "🦊\u0001合成原文"
    payload["documents"][0].update(content=content, paragraphs=[{"para_id": "段落_001", "char_start": 0, "char_end": len(content)}])
    literal_bytes = len(import_service.canonical_json(content).encode("utf-8"))
    assert literal_bytes > len(content.encode("utf-8"))
    monkeypatch.setattr(import_service, "MAX_EXPANDED_SOURCE_BYTES", literal_bytes * 2)
    assert client.post("/api/imports/preview", json=upload(payload)).status_code == 200
    monkeypatch.setattr(import_service, "MAX_EXPANDED_SOURCE_BYTES", literal_bytes * 2 - 1)
    response = client.post("/api/imports/preview", json=upload(payload))
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "import_expanded_source_too_large"


@pytest.mark.parametrize("content", ['{"title":"x","title":"y"}', '{"value":NaN}', '{"value":Infinity}', '{not json}', '[' * 1500])
def test_duplicate_json_keys_nonfinite_malformed_and_deep_inputs_are_rejected(service_client, content):
    client, _ = service_client
    response = client.post("/api/imports/preview", json={"entries": [{"name": "input.json", "content": content}]})
    assert response.status_code == 422
    assert "Traceback" not in response.text


def test_raw_request_utf8_bad_body_and_extra_keys_errors_never_echo_values(service_client):
    client, _ = service_client
    response = client.post("/api/imports/preview", content=b'{"entries":\xff}', headers={"Content-Type": "application/json"})
    assert response.json()["detail"]["code"] == "import_invalid_utf8"
    response = client.post("/api/imports/preview", json={"entries": [], "token": "DO_NOT_ECHO_THIS"})
    assert response.json()["detail"]["code"] == "import_unsupported_fields"
    assert "DO_NOT_ECHO_THIS" not in response.text


def test_invalid_commit_fingerprint_projection_ack_and_restart_safe(service_client):
    client, settings = service_client
    planned = preview(client)
    assert commit(client, {**planned, "fingerprint": "0" * 64}).json()["detail"]["code"] == "import_fingerprint_mismatch"
    assert commit(client, {**planned, "fingerprint": "非ASCII"}).status_code == 409
    assert commit(client, planned, acknowledge_excluded_fields="true").status_code == 422
    with TestClient(create_app(settings=settings)) as restarted:
        assert commit(restarted, planned).json()["detail"]["code"] == "import_preview_expired"
    assert table_count(settings, "topics") == 0


def test_preview_expires_and_capacity_is_bounded(service_client, monkeypatch):
    client, settings = service_client
    planned = preview(client)
    monkeypatch.setattr(import_service, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + import_service.PREVIEW_TTL_SECONDS + 1, time=time.time))
    assert commit(client, planned).json()["detail"]["code"] == "import_preview_expired"
    assert table_count(settings, "topics") == 0
    monkeypatch.setattr(import_service, "MAX_PREVIEWS", 1)
    assert client.post("/api/imports/preview", json=upload()).status_code == 200
    assert client.post("/api/imports/preview", json=upload()).status_code == 429


def test_file_request_and_count_limits_precede_import(service_client, monkeypatch):
    client, settings = service_client
    monkeypatch.setattr(import_service, "MAX_FILE_BYTES", 2)
    assert client.post("/api/imports/preview", json=upload()).status_code == 413
    monkeypatch.setattr(import_service, "MAX_FILE_BYTES", 10**9)
    monkeypatch.setattr(import_service, "MAX_REQUEST_BYTES", 2)
    assert client.post("/api/imports/preview", json=upload()).status_code == 413
    monkeypatch.setattr(import_service, "MAX_REQUEST_BYTES", 10**9)
    monkeypatch.setattr(import_service, "MAX_DOCUMENTS", 0)
    assert client.post("/api/imports/preview", json=upload()).status_code == 422
    assert table_count(settings, "topics") == 0


def test_commit_rollback_on_storage_failure_and_retry_idempotent(service_client):
    client, settings = service_client
    planned = preview(client)
    db = connect(settings.database_path)
    db.execute("CREATE TRIGGER reject_test_question BEFORE INSERT ON questions BEGIN SELECT RAISE(ABORT, 'DO_NOT_EXPOSE_DB_PATH_OR_PROVIDER_SECRET'); END")
    db.close()
    response = commit(client, planned)
    assert response.status_code == 503
    assert "DO_NOT_EXPOSE" not in response.text
    assert table_count(settings, "topics") == table_count(settings, "documents") == table_count(settings, "paragraphs") == 0
    db = connect(settings.database_path)
    db.execute("DROP TRIGGER reject_test_question")
    db.close()
    assert commit(client, planned).status_code == 200
    assert commit(client, planned).json()["idempotent"] is True


def test_parallel_same_preview_commit_is_content_idempotent(service_client):
    client, settings = service_client
    planned = preview(client)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: commit(client, planned).json(), range(2)))
    assert len({result["topic_id"] for result in results}) == 1
    assert sorted(result["idempotent"] for result in results) == [False, True]
    assert table_count(settings, "topics") == 1


def test_empty_and_mixed_file_selections_are_rejected(service_client):
    client, _ = service_client
    for entries in [[], upload()["entries"] * 2, [{"name": "documents.json", "content": "[]"}],
                    [{"name": "documents.json", "content": "[]"}, {"name": "raw_documents.json", "content": "[]"}]]:
        assert client.post("/api/imports/preview", json={"entries": entries}).status_code == 422


def test_question_preview_is_bounded(service_client):
    client, _ = service_client
    payload = source_input()
    payload["questions"] = [{"question_id": f"q_{index}", "question": "合成问题？"} for index in range(12)]
    planned = preview(client, upload(payload))
    assert len(planned["questions"]) == 10
    assert planned["truncated_questions_count"] == 2
