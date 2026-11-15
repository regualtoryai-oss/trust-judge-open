"""One synthetic sealed-source lineage case; zero real model calls."""
from hashlib import sha256
import json

import pytest

from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
from app.repository import Repository
from app.worker import Worker
from test_context_workbench import _import
from test_lc_context_case import raw_case
from test_lc_local_rank_integration import model  # noqa: F401
from test_paragraph_quality_workbench import settings_for, patch_quality_http


@pytest.mark.asyncio
async def test_sealed_successful_front_from_failed_workflow_recovers_with_only_three_new_calls(tmp_path, monkeypatch, model):
    settings = settings_for(tmp_path / "runtime", model, compact=True, local_rank=False)
    repo = Repository(settings)
    topic = _import(repo, raw_case(["10:58首轮记录启动，随后下午时段仍待核验。", "另一条记录仍未最终确认。 "]))
    question = repo.questions(topic)[0]
    request = {"question_ids": [question["id"]], "base_revision_id": None, "include_baseline": False, "mode": "initial"}
    origin = repo.create_run(topic, request, "original-front", model_available=True)
    repo.claim()
    first = LCBundleWorkbenchProvider(settings)
    first.workbench_event = lambda kind, data: repo.event(origin["run_id"], kind, data)
    initial_http = patch_quality_http(monkeypatch, first, bad="disjoint_time")
    original_response = await first.select_source_bundles(question, repo.source_documents(topic), repo.paragraphs(topic))
    repo.record_usage(origin["run_id"], question["id"], first._front_records)
    repo.finish_failed(origin["run_id"], "model_output_invalid", "模拟后置工程校验失败，已付原文选证保留。", stage="source_bundle_selection")
    path = first.bundle_directory / "provider-response.json"
    original_bytes = path.read_bytes()
    recovered_settings = settings.model_copy(update={"source_bundle_recover_front_run_id": origin["run_id"],
        "source_bundle_recover_front_response": path.resolve(),
        "source_bundle_recover_front_response_sha256": sha256(original_bytes).hexdigest(),
        "strict_evidence_cost_parent_run_ids": [origin["run_id"]]})
    recovered_repo = Repository(recovered_settings)
    second = LCBundleWorkbenchProvider(recovered_settings)
    new_http = patch_quality_http(monkeypatch, second)
    current = recovered_repo.create_run(topic, {**request, "base_revision_id": recovered_repo.topic(topic)["head_revision_id"]},
        "recover-sealed-front", model_available=True)
    await Worker(recovered_repo, second).execute(recovered_repo.claim())
    run = recovered_repo.run(current["run_id"])
    assert run["status"] == "succeeded", run["error"]
    assert len(initial_http) == 1 and len(new_http) == 3 and second.front_provider is None
    assert second.prior_known_tokens == 120 and len(run["usage"][question["id"]]) == 3
    receipt_path = second.bundle_directory / "sealed-front-recovery.json"
    receipt = json.loads(receipt_path.read_text("utf8"))
    assert receipt["origin_run_id"] == origin["run_id"]
    assert receipt["new_model_calls"] == receipt["new_tokens"] == 0
    assert receipt["prior_known_tokens"] == 120 and receipt["original_response_modified"] is False
    snapshot = recovered_repo.snapshot(topic, run["target_revision_id"])
    result = snapshot["question_results"][question["id"]]
    assert result["source_bundle_proof"]["response"] == original_response
    assert path.read_bytes() == original_bytes
    assert len(recovered_repo.formal_submission_files(topic, run["target_revision_id"])["evidence.json"]) == 2
