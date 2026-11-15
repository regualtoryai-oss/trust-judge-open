"""V5 token-pack native four-call worker; explicit synthetic HTTP only."""
from copy import deepcopy

import pytest

from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
from app.config import Settings
from app.repository import Repository
from app.worker import Worker
from app.workflows.lc_bundle_workbench import replay
from test_context_workbench import _import
from test_lc_context_case import raw_case
from test_lc_local_rank_integration import model  # noqa: F401
from test_paragraph_quality_workbench import settings_for, patch_quality_http


def test_v5_requires_explicit_compact_and_excludes_local_rank():
    with pytest.raises(ValueError, match="parent token packing"):
        Settings(_env_file=None, source_bundle_parent_token_pack_enabled=True)


@pytest.mark.asyncio
async def test_v5_complete_native_worker_publication_export_and_proof_replay(tmp_path, monkeypatch, model):
    settings = settings_for(tmp_path / "runtime", model, compact=True, local_rank=False).model_copy(update={
        "source_bundle_parent_token_pack_enabled": True, "source_bundle_front_token_stop": 20000,
        "source_bundle_tail_reserved_tokens": 60000, "strict_evidence_token_stop": 90000,
        "strict_evidence_max_requests": 4})
    repo = Repository(settings)
    topic = _import(repo, raw_case([f"第{i}条程序记录仍未最终确认。完整原文包括条件和阶段。" for i in range(32)]))
    question = repo.questions(topic)[0]
    provider = LCBundleWorkbenchProvider(settings)
    observed = patch_quality_http(monkeypatch, provider)
    task = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "v5-native-four-calls", model_available=True)
    await Worker(repo, provider).execute(repo.claim())
    run = repo.run(task["run_id"])
    assert run["status"] == "succeeded", run["error"]
    assert run["source_bundle_parent_token_pack"] is True and len(observed) == 4
    assert observed[0]["tools"][0]["function"]["name"] == "LCParagraphQualityV5"
    assert observed[0]["max_tokens"] == 8192
    assert [row["operation"] for row in run["usage"][question["id"]]] == [
        "strict_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"]
    result = repo.snapshot(topic, run["target_revision_id"])["question_results"][question["id"]]
    parents = {row["id"]: row["text"] for row in repo.paragraphs(topic)}
    assert result["evidence"] and all(row["quote"] == parents[row["paragraph_id"]] for row in result["evidence"])
    response = result["source_bundle_proof"]["response"]
    assert response["proof"]["parent_token_pack"] is True
    assert response["proof"]["whole_chain_planning"]["version"] == "native-whole-chain-paragraph-quality-budget-planning-v5"
    artifact, scope = replay(question, repo.paragraphs(topic), repo.source_documents(topic), response, [])
    assert artifact == response["artifact"] and scope["parent_token_pack"] is True
    assert len(repo.formal_submission_files(topic, run["target_revision_id"])["evidence.json"]) == len(result["evidence"])
    changed = deepcopy(response)
    changed["proof"].pop("parent_token_pack")
    changed["proof"].pop("parent_token_pack_policy")
    with pytest.raises(ValueError):
        replay(question, repo.paragraphs(topic), repo.source_documents(topic), changed, [])
    assert len(observed) == 4
