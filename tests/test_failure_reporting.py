"""End-to-end diagnostics and usage accounting with explicit test providers.

All input is synthetic and the real API/worker/LangGraph path is exercised.
No model endpoint or local credential file is used by these tests.
"""

from contextlib import closing
import json

from app.agents.provider import LangChainOpenAIProvider, ModelOutputInvalid
from app.data.db import connect
from test_integration import IntegrationProvider, running_client, start_run, wait_terminal


def event_payloads(response):
    assert response.status_code == 200, response.text
    return [json.loads(line.removeprefix("data:").strip())
            for line in response.text.splitlines() if line.startswith("data:")]


def test_truncated_extract_preserves_exact_error_usage_and_unpublished_state(tmp_path):
    class TruncatedProvider(IntegrationProvider):
        async def extract(self, question, paragraphs):
            self.calls["extract"] += 1
            LangChainOpenAIProvider._record_usage(
                self, "extract", {"input_tokens": 11, "output_tokens": 8192, "total_tokens": 8203}
            )
            raise ModelOutputInvalid(
                "模型输出达到长度上限，未采纳结果。", stage="extract",
                code="model_output_truncated", details={"finish_reason": "length"},
            )

    provider = TruncatedProvider()
    with running_client(tmp_path, provider) as (client, settings, topic_id, question_id):
        created = start_run(client, topic_id, question_id)
        run = wait_terminal(client, f"/api/runs/{created['run_id']}")
        expected_error = {
            "code": "model_output_truncated", "message": "模型输出达到长度上限，未采纳结果。",
            "stage": "extract", "details": {"finish_reason": "length"},
        }
        assert run["status"] == "failed", run
        assert run["error"] == expected_error
        assert len(run["usage"][question_id]) == 1
        usage = run["usage"][question_id][0]
        assert {key: usage[key] for key in ("operation", "input_tokens", "output_tokens", "total_tokens", "usage_source", "cost")} == {
            "operation": "extract", "input_tokens": 11, "output_tokens": 8192,
            "total_tokens": 8203, "usage_source": "provider", "cost": None,
        }
        events = event_payloads(client.get(f"/api/runs/{run['run_id']}/events"))
        assert [item["data"] for item in events if item["type"] == "run_failed"] == [expected_error]
        with closing(connect(settings.database_path)) as connection:
            stored = connection.execute("SELECT error_json,usage_json FROM runs WHERE id=?", (run["run_id"],)).fetchone()
            assert json.loads(stored["error_json"]) == expected_error
            assert json.loads(stored["usage_json"]) == run["usage"]
            assert connection.execute("SELECT published_revision_id FROM topics WHERE id=?", (topic_id,)).fetchone()[0] is None
            assert connection.execute("SELECT state FROM graph_revisions WHERE id=?", (run["target_revision_id"],)).fetchone()[0] == "failed"
        assert provider.calls["authority_analyst"] == 0
        assert provider.calls["cross_validation_analyst"] == 0
        assert provider.calls["arbitrate"] == 0


def test_untrusted_value_error_body_is_absent_from_api_events_and_persisted_error(tmp_path):
    sensitive_body = "sensitive-provider-body.test-only-secret-never-echo"

    class SensitiveFailureProvider(IntegrationProvider):
        async def extract(self, question, paragraphs):
            LangChainOpenAIProvider._record_usage(
                self, "extract", {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
            )
            raise ValueError(sensitive_body)

    with running_client(tmp_path, SensitiveFailureProvider()) as (client, settings, topic_id, question_id):
        created = start_run(client, topic_id, question_id)
        run = wait_terminal(client, f"/api/runs/{created['run_id']}")
        assert run["status"] == "failed", run
        assert run["error"]["code"] == "validation_failed"
        assert sensitive_body not in json.dumps(run)
        response = client.get(f"/api/runs/{run['run_id']}/events")
        assert sensitive_body not in response.text
        events = event_payloads(response)
        assert [item["data"] for item in events if item["type"] == "run_failed"] == [run["error"]]
        assert run["usage"][question_id][0]["total_tokens"] == 18
        with closing(connect(settings.database_path)) as connection:
            stored = connection.execute("SELECT error_json,usage_json FROM runs WHERE id=?", (run["run_id"],)).fetchone()
            assert sensitive_body not in stored["error_json"]
            assert sensitive_body not in stored["usage_json"]
            assert connection.execute("SELECT published_revision_id FROM topics WHERE id=?", (topic_id,)).fetchone()[0] is None


def test_successful_publication_keeps_all_four_completed_request_usage_records(tmp_path):
    class MeteredSuccessProvider(IntegrationProvider):
        async def extract(self, question, paragraphs):
            LangChainOpenAIProvider._record_usage(
                self, "extract", {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
            )
            return await super().extract(question, paragraphs)

        async def analyze(self, role, question, evidence, snapshot_id):
            values = {"authority_analyst": (13, 5, 18), "cross_validation_analyst": (17, 3, 20)}
            input_tokens, output_tokens, total_tokens = values[role]
            LangChainOpenAIProvider._record_usage(
                self, role, {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}
            )
            return await super().analyze(role, question, evidence, snapshot_id)

        async def arbitrate(self, question, evidence, analyses):
            LangChainOpenAIProvider._record_usage(
                self, "arbitrate", {"input_tokens": 19, "output_tokens": 4, "total_tokens": 23}
            )
            return await super().arbitrate(question, evidence, analyses)

    with running_client(tmp_path, MeteredSuccessProvider()) as (client, settings, topic_id, question_id):
        created = start_run(client, topic_id, question_id)
        run = wait_terminal(client, f"/api/runs/{created['run_id']}")
        assert run["status"] == "succeeded", run
        records = run["usage"][question_id]
        assert len(records) == 4
        assert {item["operation"] for item in records} == {"extract", "authority_analyst", "cross_validation_analyst", "arbitrate"}
        assert sum(item["total_tokens"] for item in records) == 79
        assert all(item["cost"] is None and item["usage_source"] == "provider" for item in records)
        with closing(connect(settings.database_path)) as connection:
            stored = connection.execute("SELECT usage_json FROM runs WHERE id=?", (run["run_id"],)).fetchone()[0]
            assert json.loads(stored) == run["usage"]
            assert connection.execute("SELECT published_revision_id FROM topics WHERE id=?", (topic_id,)).fetchone()[0] == run["target_revision_id"]
