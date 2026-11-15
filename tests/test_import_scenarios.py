"""Source-only import-to-export scenarios using an explicitly injected test model.

These are engineering regressions, not blind model accuracy evaluation. The JSON
fixtures contain sources/questions only; no competition annotations are opened.
"""

from collections import Counter
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import time
from zipfile import ZipFile

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
import pytest

from app.agents import provider as provider_module
from app.config import Settings
from legacy_fixture import create_legacy_app as create_app


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/import_samples"
SCENARIOS = [path.parent.name for path in sorted(FIXTURES.glob("*/input.json"))]
SENTINEL = "SYNTHETIC_ANSWER_MUST_NEVER_ENTER_RUNTIME"


def load(name):
    return json.loads((FIXTURES / name / "input.json").read_text(encoding="utf-8"))


def entry(payload):
    return {"entries": [{"name": "input.json", "content": json.dumps(payload, ensure_ascii=False)}]}


def preview_and_commit(client, payload, *, acknowledge=False, paired=False):
    body = entry(payload)
    if paired:
        body = {"title": payload["title"], "entries": [
            {"name": "raw_documents.json", "content": json.dumps(payload["documents"], ensure_ascii=False)},
            {"name": "questions.json", "content": json.dumps(payload["questions"], ensure_ascii=False)},
        ]}
    response = client.post("/api/imports/preview", json=body)
    assert response.status_code == 200, response.text
    preview = response.json()
    assert "conclusion" not in {key for item in preview["questions"] for key in item}
    response = client.post("/api/imports/commit", json={
        "preview_id": preview["preview_id"], "fingerprint": preview["fingerprint"],
        "acknowledge_excluded_fields": acknowledge,
    })
    assert response.status_code == 200, response.text
    return response.json(), preview


def terminal(client, url, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in {"succeeded", "failed", "cancelled", "superseded"}:
            return result
        time.sleep(0.02)
    pytest.fail(f"task did not finish: {result}")


def new_run(client, topic_id, questions, key="synthetic-run"):
    response = client.post(f"/api/topics/{topic_id}/runs", headers={"Idempotency-Key": key}, json={
        "question_ids": [item["id"] for item in questions], "base_revision_id": None,
        "include_baseline": False, "mode": "initial",
    })
    assert response.status_code == 202, response.text
    return terminal(client, f"/api/runs/{response.json()['run_id']}")


class ScenarioProvider:
    test_only_legacy_precision = True
    """Canned source-bound outputs verify plumbing, not semantic model ability."""

    def __init__(self, *, defect=None, exhaust=False):
        self.defect, self.exhaust = defect, exhaust
        self.calls = Counter()
        self.inputs, self.repairs = [], []

    def account(self, operation):
        self.calls[operation] += 1
        provider_module._usage_records.get().append({
            "operation": operation, "input_tokens": 7, "output_tokens": 4,
            "total_tokens": 11, "usage_source": "provider", "cost": None,
        })

    @staticmethod
    def extraction(question, paragraphs):
        if "确切日期" in question["question"]:
            return []
        output = []
        for para in paragraphs:
            if "论坛留言" in para["text"] or "留言未提供" in para["text"]:
                continue
            quote = "设备已复位。" if "🦊记录员" in para["text"] else para["text"]
            start = para["text"].rindex(quote)
            output.append({
                "paragraph_id": para["id"], "quote": quote,
                "start_cp": start, "end_cp": start + len(quote),
                "fact_des": "合成资料中的可定位记录", "argument": {}, "entities": [],
            })
        return output

    async def extract(self, question, paragraphs):
        self.account("extract")
        self.inputs.append((deepcopy(question), deepcopy(paragraphs)))
        result = self.extraction(question, paragraphs)
        if self.defect == "ambiguous_quote":
            for row in result:
                if row["quote"] == "设备已复位。":
                    row.update(start_cp=None, end_cp=None)
        return result

    @staticmethod
    def answer(question, evidence, *, arbitration=False):
        text = question["question"]
        if "停运" in text or "确切日期" in text or "停机原因" in text:
            result = {"decision": "undetermined", "conclusion": "所给资料不足以确认所问事实。",
                      "judgments": [], "supporting_evidence_ids": [], "conflicting_evidence_ids": []}
            return {**result, "conflicts": []} if arbitration else result
        ids = lambda raw: [row["id"] for row in evidence if row["raw_doc_id"] == raw]
        conflicts = []
        if "最近一份明确更正" in text:
            support, opposing = ids("doc_002"), ids("doc_001")[:1]
            statement = "更正通报的青岚站登记失联人数为4人，3人保留为初报。"
            conflicts = [{
                "id": "C1", "statement": "同一站点初报与更正数字不同", "status": "resolved",
                "supporting_evidence_ids": support, "conflicting_evidence_ids": opposing,
                "resolution_type": "recency", "basis": "后文明确标明更正及前条为初报。",
                "accepted_reason": "更正通报明确取代初报的当前口径。",
                "rejected_reason": "初报保留为历史记录。", "minority_opinion": "初报描述较早时点。",
                "reopen_conditions": "出现同一站点更晚的更正通报。",
            }]
        elif "云港区" in text:
            support, opposing = ids("doc_002") + ids("doc_003"), []
            statement = "全区与单站统计范围不同，所给数字可以按各自范围并列保留。"
            conflicts = [{
                "id": "C1", "statement": "全区与单站人数不同", "status": "coexisting",
                "supporting_evidence_ids": ids("doc_002"), "conflicting_evidence_ids": ids("doc_003"),
                "resolution_type": "scope_separation", "basis": "两份原文明确分别为单站和全区。",
                "accepted_reason": "保留各自范围。", "rejected_reason": "不能用全区替代单站。",
                "minority_opinion": "两种口径并列。", "reopen_conditions": "统计范围被更正。",
            }]
        else:
            support, opposing = ids("doc_001") + ids("doc_002") + ids("doc_003"), []
            statement = "资料有设备复位记录；相同原文及相同来源链接的镜像不增加独立来源。"
        result = {
            "decision": "asserted", "conclusion": statement,
            "judgments": [
                {"id": "J1", "statement": "原文记载了相关事实或范围。", "evidence_ids": support,
                 "prerequisite_judgment_ids": []},
                {"id": "J2", "statement": "依据前述原文区分记录口径。", "evidence_ids": [],
                 "prerequisite_judgment_ids": ["J1"]},
            ], "supporting_evidence_ids": support, "conflicting_evidence_ids": opposing,
        }
        return {**result, "conflicts": conflicts} if arbitration else result

    async def analyze(self, role, question, evidence, snapshot_id):
        self.account(role)
        self.inputs.append((deepcopy(question), deepcopy(evidence)))
        result = self.answer(question, evidence)
        if role == "authority_analyst" and self.defect in {"ungrounded_judgment", "foreign_reference"}:
            if self.defect == "ungrounded_judgment":
                result["judgments"] = [{"id": "Jbad", "statement": "PRIVATE_INVALID_OUTPUT",
                                        "evidence_ids": [], "prerequisite_judgment_ids": []}]
            else:
                result["supporting_evidence_ids"] = ["FOREIGN_RESOURCE"]
        return result

    async def arbitrate(self, question, evidence, analyses):
        self.account("arbitrate")
        return self.answer(question, evidence, arbitration=True)

    async def repair_analysis(self, role, question, evidence, snapshot_id, invalid_output, diagnostics, *, analyses=None):
        self.account(role)
        self.repairs.append({"role": role, "diagnostics": deepcopy(diagnostics)})
        return deepcopy(invalid_output) if self.exhaust else self.answer(question, evidence)

    async def repair_extract(self, question, paragraphs, invalid_output, diagnostics):
        self.account("extract")
        self.repairs.append({"role": "extract", "diagnostics": deepcopy(diagnostics)})
        return deepcopy(invalid_output) if self.exhaust else self.extraction(question, paragraphs)


@pytest.mark.parametrize("name", SCENARIOS)
def test_new_source_import_all_questions_through_real_worker_graph_source_and_export(tmp_path, name):
    payload, provider = load(name), ScenarioProvider()
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="")
    with TestClient(create_app(settings, provider, embedded_worker=True)) as client:
        imported, preview = preview_and_commit(client, payload, paired=name.startswith("03"))
        topic_id = imported["topic_id"]
        assert preview["excluded_fields"] == [] and provider.calls == Counter()
        again, _ = preview_and_commit(client, deepcopy(payload))
        assert again["idempotent"] and again["topic_id"] == topic_id
        questions = client.get(f"/api/topics/{topic_id}/questions").json()
        run = new_run(client, topic_id, questions)
        assert run["status"] == "succeeded", run
        revision_id = run["target_revision_id"]
        graph = client.get(f"/api/topics/{topic_id}/revisions/{revision_id}/graph").json()
        artifacts = client.get(f"/api/topics/{topic_id}/revisions/{revision_id}/artifacts").json()
        assert graph["scope"]["is_complete_topic"] is True
        assert len(artifacts["conclusions"]) == len(payload["questions"])
        activity = client.get(f"/api/runs/{run['run_id']}/activity").json()
        assert activity["progress"]["completed_count"] == activity["progress"]["total_count"]
        assert activity["progress"]["failed_stages"] == []
        sources = {doc["doc_id"]: doc for doc in payload["documents"]}
        for node in graph["nodes"]:
            if node["kind"] != "evidence":
                continue
            evidence = client.get(f"/api/evidence/{node['id']}", params={"revision_id": revision_id}).json()
            paragraph = client.get(f"/api/paragraphs/{evidence['paragraph_id']}").json()
            assert paragraph["topic_id"] == topic_id
            assert paragraph["text"][evidence["start_cp"]:evidence["end_cp"]] == evidence["quote"]
            source = sources[evidence["raw_doc_id"]]
            assert source["content"][paragraph["start_cp"]:paragraph["end_cp"]] == paragraph["text"]
            if evidence["quote"] == "设备已复位。":
                assert evidence["start_cp"] == paragraph["text"].rindex(evidence["quote"])
                assert evidence["span_location_method"] == "model_explicit_occurrence"
        if name.startswith("01"):
            assert {item["result"] for item in artifacts["conflicts"]} == {"adopt_a", "coexist"}
        if name.startswith("02"):
            assert all(item["decision"] == "undetermined" for item in artifacts["conclusions"])
            assert all(item["supporting_evidence_ids"] == [] for item in artifacts["conclusions"])
        if name.startswith("03"):
            groups = [group for group in artifacts["source_groups"] if len(group["evidence_ids"]) >= 2]
            assert len(groups) == 1
            # Both mirror document identities are retained, but count as one source group.
            members = [n["content"]["raw_doc_id"] for n in graph["nodes"] if n["kind"] == "evidence" and n["id"] in groups[0]["evidence_ids"]]
            assert set(members) == {"doc_001", "doc_002"}
        response = client.post(f"/api/topics/{topic_id}/exports", json={"revision_id": revision_id}, headers={"Idempotency-Key": "complete-export"})
        assert response.status_code == 202, response.text
        export = terminal(client, f"/api/exports/{response.json()['export_id']}")
        assert export["status"] == "succeeded", export
        download = client.get(export["download_url"])
        with ZipFile(BytesIO(download.content)) as archive:
            files = {Path(p).name: json.loads(archive.read(p)) for p in archive.namelist() if p.endswith(".json")}
        schema = json.loads((ROOT / "contracts/competition-export.schema.json").read_text(encoding="utf-8-sig"))
        for filename, definition in {"evidence.json": "EvidenceExportArray", "conclusions.json": "ConclusionExportArray", "confidence.json": "ConfidenceExportArray", "conflicts.json": "ConflictExportArray"}.items():
            Draft202012Validator({"$ref": f"#/$defs/{definition}", "$defs": schema["$defs"]}).validate(files[filename])
        assert {row["question_id"] for row in files["conclusions.json"]} == {q["question_id"] for q in payload["questions"]}
        assert SENTINEL not in json.dumps(provider.inputs)
        assert sum(provider.calls.values()) == 4 * len(questions)


@pytest.mark.parametrize("name", [p.parent.name for p in (FIXTURES / "rejected").glob("*/input.json") if p.parent.name != "答案字段污染"])
def test_invalid_import_has_actionable_error_no_partial_sources_and_no_model_calls(tmp_path, name):
    provider = ScenarioProvider()
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="", model_api_key="")
    with TestClient(create_app(settings, provider)) as client:
        before = client.get("/api/topics").json()
        response = client.post("/api/imports/preview", json=entry(load("rejected/" + name)))
        assert response.status_code == 422, response.text
        assert response.json()["detail"]["code"].startswith("import_")
        assert "documents[" in response.json()["detail"]["message"]
        assert client.get("/api/topics").json() == before
        assert provider.calls == Counter()


def test_answer_projection_requires_confirmation_and_never_enters_database_or_provider(tmp_path):
    provider = ScenarioProvider()
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="", model_api_key="")
    with TestClient(create_app(settings, provider, embedded_worker=True)) as client:
        payload = load("rejected/答案字段污染")
        preview = client.post("/api/imports/preview", json=entry(payload)).json()
        assert SENTINEL not in json.dumps(preview)
        assert set(preview["excluded_fields"][0]["fields"]) == {"conclusion", "supporting_facts"}
        confirm = {"preview_id": preview["preview_id"], "fingerprint": preview["fingerprint"]}
        response = client.post("/api/imports/commit", json=confirm)
        assert response.status_code == 409 and response.json()["detail"]["code"] == "import_projection_ack_required"
        assert client.get("/api/topics").json() == []
        response = client.post("/api/imports/commit", json={**confirm, "acknowledge_excluded_fields": True})
        assert response.status_code == 200, response.text
        tid = response.json()["topic_id"]
        questions = client.get(f"/api/topics/{tid}/questions").json()
        run = new_run(client, tid, questions)
        assert run["status"] == "succeeded", run
        assert SENTINEL not in json.dumps(provider.inputs)
    assert SENTINEL.encode() not in settings.database_path.read_bytes()


@pytest.mark.parametrize("defect,scenario,exhaust", [
    ("ungrounded_judgment", SCENARIOS[0], False),
    ("foreign_reference", SCENARIOS[0], False),
    ("ambiguous_quote", SCENARIOS[2], False),
    ("ungrounded_judgment", SCENARIOS[0], True),
    ("ambiguous_quote", SCENARIOS[2], True),
])
def test_new_import_handles_previous_failure_families_with_one_strict_repair(tmp_path, defect, scenario, exhaust):
    provider = ScenarioProvider(defect=defect, exhaust=exhaust)
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="", model_api_key="")
    with TestClient(create_app(settings, provider, embedded_worker=True)) as client:
        imported, _ = preview_and_commit(client, load(scenario))
        tid = imported["topic_id"]
        questions = client.get(f"/api/topics/{tid}/questions").json()[:1]
        run = new_run(client, tid, questions)
        assert len(provider.repairs) == 1
        activity = client.get(f"/api/runs/{run['run_id']}/activity").json()
        assert "PRIVATE_INVALID_OUTPUT" not in json.dumps(activity)
        assert "FOREIGN_RESOURCE" not in json.dumps(activity)
        assert sum(row["total_tokens"] for records in run["usage"].values() for row in records) == sum(provider.calls.values()) * 11
        if not exhaust:
            assert run["status"] == "succeeded", run
            assert activity["progress"]["failed_stages"] == []
        else:
            assert run["status"] == "failed", run
            assert run["error"]["details"]["repair_exhausted"] is True
            stage = "verify" if defect == "ambiguous_quote" else "authority_analyst"
            assert run["error"]["stage"] == stage
            assert any(item["stage"] == stage for item in activity["progress"]["failed_stages"])
            assert client.get(f"/api/topics/{tid}").json()["published_revision_id"] is None
            assert provider.calls["arbitrate"] == 0
            revision_id = run["target_revision_id"]
            assert client.get(f"/api/topics/{tid}/revisions/{revision_id}/graph").status_code == 409
            response = client.post(f"/api/topics/{tid}/exports", json={"revision_id": revision_id}, headers={"Idempotency-Key": "blocked-export"})
            assert response.status_code == 409, response.text


def test_new_topic_with_reused_source_ids_cannot_overwrite_or_leak_previous_sources(tmp_path):
    provider = ScenarioProvider()
    settings = Settings(_env_file=None, data_dir=tmp_path, model_name="", model_api_key="")
    with TestClient(create_app(settings, provider, embedded_worker=True)) as client:
        first, _ = preview_and_commit(client, load(SCENARIOS[0]))
        first_questions = client.get(f"/api/topics/{first['topic_id']}/questions").json()
        first_run = new_run(client, first["topic_id"], first_questions)
        assert first_run["status"] == "succeeded", first_run
        graph_url = f"/api/topics/{first['topic_id']}/revisions/{first_run['target_revision_id']}/graph"
        before = client.get(graph_url).json()
        second, _ = preview_and_commit(client, load(SCENARIOS[2]))
        second_questions = client.get(f"/api/topics/{second['topic_id']}/questions").json()
        assert first["topic_id"] != second["topic_id"]
        assert {q["raw_question_id"] for q in first_questions} == {q["raw_question_id"] for q in second_questions}
        assert not {q["id"] for q in first_questions} & {q["id"] for q in second_questions}
        second_run = new_run(client, second["topic_id"], second_questions)
        assert second_run["status"] == "succeeded", second_run
        second_graph = client.get(f"/api/topics/{second['topic_id']}/revisions/{second_run['target_revision_id']}/graph").json()
        assert not {n["id"] for n in before["nodes"]} & {n["id"] for n in second_graph["nodes"]}
        assert client.get(graph_url).json() == before
        # Attempting to run topic B with a topic A internal question UUID is rejected.
        response = client.post(f"/api/topics/{second['topic_id']}/runs", headers={"Idempotency-Key": "cross-topic"}, json={
            "question_ids": [first_questions[0]["id"]], "base_revision_id": second_run["target_revision_id"],
            "include_baseline": False, "mode": "initial",
        })
        assert response.status_code in {404, 422}, response.text
