"""Synthetic, explicit test-provider integration of API, worker and LangGraph.

These cases are engineering checks, not blind competition evaluation. They do
not load competition gold data or make paid/real model requests.
"""

import asyncio
from collections import Counter
from contextlib import closing, contextmanager
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from zipfile import ZipFile

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
import pytest

from app.config import Settings
from app.data.db import connect
from app.data.ingest import import_topic


SOURCE_TEXTS = {
    "doc_001": "🦊通报称，截至当日失联3人。",
    "doc_002": "更正：失联4人，前条3人为初报。",
}


class IntegrationProvider:
    test_only_legacy_precision = True
    """Deterministic only because tests explicitly inject this object."""

    model_name = "explicit-integration-test-provider"
    prompt_version = "test-fixture-v1"

    def __init__(self, *, block_extraction=False, bad_quote=False):
        self.calls = Counter()
        self.block_extraction = block_extraction
        self.bad_quote = bad_quote
        self.extraction_started = threading.Event()
        self.release_extraction = threading.Event()
        self.analysis_inputs = []

    async def extract(self, question, paragraphs):
        self.calls["extract"] += 1
        self.extraction_started.set()
        while self.block_extraction and not self.release_extraction.is_set():
            await asyncio.sleep(0.01)
        result = []
        for paragraph in paragraphs:
            quote = paragraph["text"] if "更正" in paragraph["text"] else "失联3人"
            result.append({
                "paragraph_id": paragraph["id"], "quote": "伪造的失联5人" if self.bad_quote else quote,
                "fact_des": "更正公告提供当前口径" if "更正" in quote else "初次通报提供早期口径",
                "argument": {"时间": None, "主体": "失联人员", "地点": None, "内容": quote, "影响": None, "数值口径": "失联人数"},
                "entities": ["失联人员"],
            })
        return result

    @staticmethod
    def _answer(evidence):
        corrected = next(item for item in evidence if "更正" in item["quote"])
        initial = next(item for item in evidence if "更正" not in item["quote"])
        return {
            "decision": "asserted", "conclusion": "更正通报失联4人。",
            "judgments": [
                {"id": "J1", "statement": "更正公告明确将此前数字标为初报", "evidence_ids": [corrected["id"]], "prerequisite_judgment_ids": []},
                {"id": "J2", "statement": "当前按明确更正后的口径采用4人", "evidence_ids": [corrected["id"]], "prerequisite_judgment_ids": ["J1"]},
            ],
            "supporting_evidence_ids": [corrected["id"]], "conflicting_evidence_ids": [initial["id"]],
        }

    async def analyze(self, role, question, evidence, snapshot_id):
        self.calls[role] += 1
        self.analysis_inputs.append((role, deepcopy(question), deepcopy(evidence), snapshot_id))
        return self._answer(evidence)

    async def arbitrate(self, question, evidence, analyses):
        self.calls["arbitrate"] += 1
        assert {item["role"] for item in analyses} == {"authority_analyst", "cross_validation_analyst"}
        answer = self._answer(evidence)
        return {
            **answer,
            "conflicts": [{
                "id": "C1", "statement": "初报与更正公告的当前人数口径不同", "status": "resolved",
                "supporting_evidence_ids": answer["supporting_evidence_ids"],
                "conflicting_evidence_ids": answer["conflicting_evidence_ids"],
                "resolution_type": "recency", "basis": "后一段明确写明更正并指出此前3人为初报。",
                "accepted_reason": "更正文本明确给出4人及取代前条数字的关系。",
                "rejected_reason": "3人保留为初报记录，不作为更正后的当前口径。",
                "minority_opinion": "初报的3人仍是可定位的历史记录。",
                "reopen_conditions": "更正公告被撤回或出现针对同一口径的更新记录。",
            }],
        }


def seed_runtime(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="", model_premise_bank_enabled=False)
    documents = [
        {"doc_id": raw_id, "title": "合成工程测试资料", "content": content,
         "paragraphs": [{"para_id": "para_001", "char_start": 0, "char_end": len(content)}]}
        for raw_id, content in SOURCE_TEXTS.items()
    ]
    imported = import_topic(settings.database_path, documents,
                            [{"question_id": "q_001", "question": "通报中的当前失联人数是多少？"}], "合成集成测试")
    with closing(connect(settings.database_path)) as connection:
        question_id = connection.execute("SELECT id FROM questions WHERE topic_id=?", (imported["topic_id"],)).fetchone()["id"]
    return settings, imported["topic_id"], question_id


@contextmanager
def running_client(tmp_path, provider):
    # Import lazily: the parent implementation owns this application module.
    from legacy_fixture import create_legacy_app as create_app

    settings, topic_id, question_id = seed_runtime(tmp_path)
    with TestClient(create_app(settings=settings, provider=provider, embedded_worker=True)) as client:
        yield client, settings, topic_id, question_id


def run_body(question_id, **changes):
    return {"question_ids": [question_id], "include_baseline": False, "base_revision_id": None, "mode": "initial", **changes}


def start_run(client, topic_id, question_id, *, key="run-key", **changes):
    response = client.post(f"/api/topics/{topic_id}/runs", json=run_body(question_id, **changes), headers={"Idempotency-Key": key})
    assert response.status_code == 202, response.text
    return response.json()


def wait_terminal(client, path, *, timeout=10):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(path)
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in {"succeeded", "failed", "cancelled", "superseded"}:
            return last
        time.sleep(0.025)
    pytest.fail(f"Task did not reach a terminal state: {last}")


def event_ids(text):
    return [int(line.removeprefix("id:").strip()) for line in text.splitlines() if line.startswith("id:")]


def test_real_api_worker_langgraph_run_ready_and_four_json_export(tmp_path):
    provider = IntegrationProvider()
    with running_client(tmp_path, provider) as (client, settings, topic_id, question_id):
        created = start_run(client, topic_id, question_id, include_baseline=True)
        run = wait_terminal(client, f"/api/runs/{created['run_id']}")
        assert run["status"] == "succeeded", run
        revision_id = run["target_revision_id"]
        with closing(connect(settings.database_path)) as connection:
            topic = connection.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
            revision = connection.execute("SELECT * FROM graph_revisions WHERE id=?", (revision_id,)).fetchone()
            executions = connection.execute("SELECT role,evidence_snapshot_id,input_json FROM agent_executions WHERE run_id=?", (run["run_id"],)).fetchall()
        assert topic["published_revision_id"] == revision_id
        assert revision["state"] == "ready"
        independent = [row for row in executions if row["role"] in {"authority_analyst", "cross_validation_analyst"}]
        assert len(independent) == 2 and len({row["evidence_snapshot_id"] for row in independent}) == 1
        assert {row["role"] for row in executions} >= {"authority_analyst", "cross_validation_analyst", "arbitrate", "baseline"}
        assert settings.checkpoint_path.is_file()
        assert provider.calls == Counter({"extract": 1, "authority_analyst": 1, "cross_validation_analyst": 1, "arbitrate": 1, "baseline": 1})

        graph_response = client.get(f"/api/topics/{topic_id}/revisions/{revision_id}/graph")
        assert graph_response.status_code == 200, graph_response.text
        graph = graph_response.json()
        assert {node["kind"] for node in graph["nodes"]} == {"evidence", "judgment", "conclusion"}
        assert any(edge["relation"] == "prerequisite" for edge in graph["edges"])

        response = client.post(f"/api/topics/{topic_id}/exports", json={"revision_id": revision_id}, headers={"Idempotency-Key": "export-key"})
        assert response.status_code == 202, response.text
        export = wait_terminal(client, f"/api/exports/{response.json()['export_id']}")
        assert export["status"] == "succeeded", export
        download = client.get(f"/api/exports/{export['export_id']}/download")
        assert download.status_code == 200, download.text
        with ZipFile(BytesIO(download.content)) as archive:
            by_name = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
            required = {"evidence.json", "conclusions.json", "confidence.json", "conflicts.json", "manifest.json"}
            assert required <= by_name.keys()
            files = {name: json.loads(archive.read(by_name[name])) for name in required}
        schema = json.loads((Path(__file__).resolve().parents[1] / "contracts/competition-export.schema.json").read_text(encoding="utf-8-sig"))
        for filename, definition in {
            "evidence.json": "EvidenceExportArray", "conclusions.json": "ConclusionExportArray",
            "confidence.json": "ConfidenceExportArray", "conflicts.json": "ConflictExportArray",
        }.items():
            Draft202012Validator({"$ref": f"#/$defs/{definition}", "$defs": schema["$defs"]}).validate(files[filename])
        assert {row["doc_id"] for row in files["evidence.json"]} == {"doc_001", "doc_002"}
        assert all(row["para_id"] == "para_001" and row["quote"] in SOURCE_TEXTS[row["doc_id"]] for row in files["evidence.json"])
        assert [row["question_id"] for row in files["conclusions.json"]] == ["q_001"]
        assert files["conflicts.json"][0]["minority_opinions"]
        assert {row["object_type"] for row in files["confidence.json"]} == {"evidence", "conclusion"}


def test_terminal_events_are_ordered_and_last_event_id_resumes_without_replay(tmp_path):
    with running_client(tmp_path, IntegrationProvider()) as (client, settings, topic_id, question_id):
        created = start_run(client, topic_id, question_id)
        run = wait_terminal(client, f"/api/runs/{created['run_id']}")
        assert run["status"] == "succeeded", run
        response = client.get(f"/api/runs/{run['run_id']}/events")
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        ids = event_ids(response.text)
        assert len(ids) >= 8
        assert ids == sorted(set(ids))
        assert "node_started" in response.text and "node_completed" in response.text
        cursor = ids[len(ids) // 2]
        resumed = client.get(f"/api/runs/{run['run_id']}/events", headers={"Last-Event-ID": str(cursor)})
        assert event_ids(resumed.text) == [identity for identity in ids if identity > cursor]
        with closing(connect(settings.database_path)) as connection:
            assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1


def test_no_model_returns_503_without_creating_a_fake_run(tmp_path):
    with running_client(tmp_path, None) as (client, settings, topic_id, question_id):
        response = client.post(f"/api/topics/{topic_id}/runs", json=run_body(question_id), headers={"Idempotency-Key": "no-model"})
        assert response.status_code == 503, response.text
        assert "model" in response.text.lower() or "模型" in response.text
        with closing(connect(settings.database_path)) as connection:
            assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
            assert connection.execute("SELECT published_revision_id FROM topics WHERE id=?", (topic_id,)).fetchone()[0] is None


def test_run_idempotency_reuses_same_body_and_conflicts_on_changed_body(tmp_path):
    provider = IntegrationProvider()
    with running_client(tmp_path, provider) as (client, settings, topic_id, question_id):
        created = start_run(client, topic_id, question_id)
        run = wait_terminal(client, f"/api/runs/{created['run_id']}")
        assert run["status"] == "succeeded", run
        repeated = client.post(f"/api/topics/{topic_id}/runs", json=run_body(question_id), headers={"Idempotency-Key": "run-key"})
        assert repeated.status_code in {200, 202}, repeated.text
        assert repeated.json()["run_id"] == created["run_id"]
        changed = client.post(f"/api/topics/{topic_id}/runs", json=run_body(question_id, include_baseline=True), headers={"Idempotency-Key": "run-key"})
        assert changed.status_code == 409, changed.text
        assert provider.calls["extract"] == 1 and provider.calls["arbitrate"] == 1
        with closing(connect(settings.database_path)) as connection:
            assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1


def test_cancelled_worker_cannot_publish_late_model_results(tmp_path):
    provider = IntegrationProvider(block_extraction=True)
    try:
        with running_client(tmp_path, provider) as (client, settings, topic_id, question_id):
            created = start_run(client, topic_id, question_id)
            assert provider.extraction_started.wait(5), "worker did not start the model request"
            response = client.post(f"/api/runs/{created['run_id']}/cancel")
            assert response.status_code in {200, 202}, response.text
            provider.release_extraction.set()
            run = wait_terminal(client, f"/api/runs/{created['run_id']}")
            assert run["status"] == "cancelled", run
            with closing(connect(settings.database_path)) as connection:
                assert connection.execute("SELECT published_revision_id FROM topics WHERE id=?", (topic_id,)).fetchone()[0] is None
                assert connection.execute("SELECT state FROM graph_revisions WHERE id=?", (run["target_revision_id"],)).fetchone()[0] != "ready"
            assert provider.calls["arbitrate"] == 0
    finally:
        provider.release_extraction.set()


def test_invalid_model_quote_fails_run_without_publishing(tmp_path):
    provider = IntegrationProvider(bad_quote=True)
    with running_client(tmp_path, provider) as (client, settings, topic_id, question_id):
        created = start_run(client, topic_id, question_id)
        run = wait_terminal(client, f"/api/runs/{created['run_id']}")
        assert run["status"] == "failed", run
        assert provider.calls["authority_analyst"] == 0 and provider.calls["arbitrate"] == 0
        with closing(connect(settings.database_path)) as connection:
            assert connection.execute("SELECT published_revision_id FROM topics WHERE id=?", (topic_id,)).fetchone()[0] is None
