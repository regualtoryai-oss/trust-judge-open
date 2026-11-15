"""Synthetic native failure -> sealed front + exact A/B cache -> one new arb.

Only in-process HTTPMock responses are used. The 120/360/480 counters test
accounting and are not real-provider costs or evidence-quality measurements.
"""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import httpx
import pytest

from app.agents import strict_transport
from app.agents import paid_response_cache
from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
from app.agents.provider import ModelOutputInvalid
from app.config import Settings
from app.domain import provisional_relations
from app.repository import Repository
from app.worker import Worker
from app.workflows.analysis import _project_question
from app.workflows.lc_bundle_workbench import replay
from scripts.capture_bundle_workbench_real import capture_episode
from scripts.score_bundle_workbench_real import _tail
from test_context_workbench import _import
from test_lc_context_case import raw_case
from test_paid_response_cache import save
from test_paragraph_quality_workbench import patch_quality_http, settings_for


def _native_http(monkeypatch, provider, *, overlap=False, only_arbitration=False):
    """Use the shared audited native mock, with an explicit no-fallback fence."""
    native_count_wire = strict_transport.count_wire
    observed = patch_quality_http(monkeypatch, provider)
    # The new paid-arbitration forecast must agree with the real pinned local
    # tokenizer at the actual HTTP guard. Provider usage remains synthetic120.
    monkeypatch.setattr(strict_transport, "count_wire", native_count_wire)
    underlying = strict_transport.httpx.AsyncHTTPTransport()

    async def handle(request):
        body = json.loads(request.content)
        tool = body["tools"][0]["function"]["name"]
        if only_arbitration:
            assert tool == "ScopedArbitrationWireOutput", "source/A/B must never repeat HTTP"
        response = await underlying.handle_async_request(request)
        value = response.json()
        if overlap and "独立研判B" in body["messages"][0]["content"]:
            call = value["choices"][0]["message"]["tool_calls"][0]["function"]
            output = json.loads(call["arguments"])
            assert len(output["supporting_evidence_ids"]) >= 2
            output["conflicting_evidence_ids"] = [output["supporting_evidence_ids"][0]]
            call["arguments"] = json.dumps(output, ensure_ascii=False)
        return httpx.Response(200, json=value)

    monkeypatch.setattr(strict_transport.httpx, "AsyncHTTPTransport", lambda **_kw: httpx.MockTransport(handle))
    return observed


async def _closed_three_call_failure(tmp_path, monkeypatch):
    settings = settings_for(tmp_path / "runtime", None, compact=True, local_rank=False).model_copy(update={
        "source_bundle_parent_token_pack_enabled": True,
        "source_bundle_front_token_stop": 20000,
        "source_bundle_tail_reserved_tokens": 60000,
        "strict_evidence_token_stop": 90000,
        "strict_evidence_max_requests": 4,
    })
    # Validate the complete production profile, rather than relying on model_copy.
    settings = Settings.model_validate(settings.model_dump())
    repo = Repository(settings)
    topic = _import(repo, raw_case([
        "首轮程序记录已经启动，最终结果仍待核验。",
        "另一份记录说明尚未最终确认，不能预断结论。",
        "补充材料尚待核验，保留条件，当前公开信息仍有缺口。",
    ]))
    question = repo.questions(topic)[0]
    options = {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}
    origin = repo.create_run(topic, options, "synthetic-old-overlap-failure", model_available=True)
    provider = LCBundleWorkbenchProvider(settings)
    observed = _native_http(monkeypatch, provider, overlap=True)
    fixed = provisional_relations.quarantine_provisional_relations

    def legacy_reject(output, role):
        # Simulate the previous contract failure after all three paid responses
        # have arrived; preserve their actual native wires and response bytes.
        if role == "cross_validation_analyst" and set(output["supporting_evidence_ids"]) & set(output["conflicting_evidence_ids"]):
            raise ModelOutputInvalid("同一证据不能同时作为结论的支持和冲突依据。", stage=role,
                details={"check": "overlapping_evidence_relations"})
        return fixed(output, role)

    with monkeypatch.context() as legacy:
        legacy.setattr(provisional_relations, "quarantine_provisional_relations", legacy_reject)
        await Worker(repo, provider).execute(repo.claim())
    run = repo.run(origin["run_id"])
    assert run["status"] == "failed" and len(observed) == 3, run["error"]
    assert run["error"]["stage"] == "cross_validation_analyst"
    usage = run["usage"][question["id"]]
    assert {row["operation"] for row in usage} == {"strict_selection", "authority_analyst", "cross_validation_analyst"}
    assert len(usage) == 3 and sum(row["total_tokens"] for row in usage) == 360
    native = capture_episode(provider.directory)
    assert len(native["tail"]["requests"]) == 2
    closure_path = tmp_path / "failed-native" / "failed-closure.json"
    save(closure_path, {"closed": True, "status": "failed", "references_read": False,
        "explicit_synthetic_HTTPMock_test": True,
        "artifacts": {"run": save(closure_path.parent / "run.json", run),
            "activity": save(closure_path.parent / "activity.json", repo.activity(run["id"])),
            "native_episode_manifest": save(closure_path.parent / "native.json", native)}})
    source_path = provider.bundle_directory / "provider-response.json"
    return settings, repo, topic, question, options, provider, run, native, closure_path, source_path


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_b", [False, True], ids=["one-new-arbitration", "missing-cache-no-HTTP-fallback"])
async def test_native_partial_tail_recovery_preserves_paid_inputs_and_whole_ledger(tmp_path, monkeypatch, missing_b):
    (settings, repo, topic, question, options, _first, origin, old_native,
        closure_path, source_path) = await _closed_three_call_failure(tmp_path, monkeypatch)
    original_source = source_path.read_bytes()
    original_responses = {row["response"]["file"]: Path(row["response"]["file"]).read_bytes()
        for row in old_native["tail"]["requests"]}
    if missing_b:
        prime = paid_response_cache.prime_paid_response
        def evict_B_after_verified_prime(*args, **kwargs):
            binding = prime(*args, **kwargs)
            if kwargs["operation"] == "cross_validation_analyst":
                # Explicitly evict the installed LC cache after full native
                # proof verification. Its attempted generation must meet the
                # cache-only HTTP guard, never re-bill the existing B output.
                args[0].clear()
            return binding
        monkeypatch.setattr(paid_response_cache, "prime_paid_response", evict_B_after_verified_prime)
    recovered_settings = settings.model_copy(update={
        "source_bundle_recover_front_run_id": origin["id"],
        "source_bundle_recover_front_response": source_path.resolve(),
        "source_bundle_recover_front_response_sha256": sha256(original_source).hexdigest(),
        "strict_evidence_cost_parent_run_ids": [origin["id"]],
        "strict_evidence_response_reuse_closures": [closure_path.resolve()],
        "strict_evidence_response_reuse_operations": ["authority_analyst", "cross_validation_analyst"],
    })
    recovered_settings = Settings.model_validate(recovered_settings.model_dump())
    recovered_repo = Repository(recovered_settings)
    second = LCBundleWorkbenchProvider(recovered_settings)
    new_http = _native_http(monkeypatch, second, only_arbitration=True)
    current = recovered_repo.create_run(topic, {**options,
        "base_revision_id": recovered_repo.topic(topic)["head_revision_id"]}, "sealed-plus-paid-AB", model_available=True)
    await Worker(recovered_repo, second).execute(recovered_repo.claim())
    run = recovered_repo.run(current["run_id"])
    assert source_path.read_bytes() == original_source
    assert all(Path(path).read_bytes() == data for path, data in original_responses.items())
    assert second.front_provider is None and second.prior_known_tokens == 360
    if missing_b:
        assert run["status"] == "failed", run
        assert run["error"]["code"] == "paid_response_cache_miss"
        assert run["error"]["stage"] == "cross_validation_analyst"
        assert not new_http and not run["usage"].get(question["id"])
        assert recovered_repo.topic(topic)["published_revision_id"] is None
        return

    assert run["status"] == "succeeded", run["error"]
    assert len(new_http) == 1
    usage = run["usage"][question["id"]]
    assert len(usage) == 1 and usage[0]["operation"] == "arbitrate" and usage[0]["total_tokens"] == 120
    seeded = [event["data"] for event in recovered_repo.events(run["id"]) if event["type"] == "cost_lineage_seeded"]
    assert len(seeded) == 1 and seeded[0]["prior_calls"] == 3 and seeded[0]["prior_known_tokens"] == 360
    snapshot = recovered_repo.snapshot(topic, run["target_revision_id"])
    result = snapshot["question_results"][question["id"]]
    assert len(result["evidence"]) == 3
    b = next(row for row in result["analyses"] if row["role"] == "cross_validation_analyst")
    quality = b["relation_quality"]
    assert quality["quarantined_count"] == 1 and quality["raw_model_response_modified"] is False
    assert quality["decision_changed"] is quality["evidence_removed"] is quality["semantic_verified"] is False
    overlap = quality["quarantined_evidence_ids"][0]
    assert quality["original_relations"]["supporting_evidence_ids"] == b["judgments"][0]["evidence_ids"]
    assert quality["original_relations"]["conflicting_evidence_ids"] == [overlap]
    assert overlap not in b["supporting_evidence_ids"] + b["conflicting_evidence_ids"]
    assert overlap in b["judgments"][0]["evidence_ids"]
    assert b["decision"] == "undetermined" and b["conclusion"] == "原始记录未能确定最终结果。"
    assert b["judgments"][0]["statement"] == "原始记录保留未决状态。"
    assert len(b["judgments"][0]["evidence_ids"]) == 3

    # Reconstruct source proof and strict tail from the archived native bytes.
    response = result["source_bundle_proof"]["response"]
    artifact, scope = replay(question, recovered_repo.paragraphs(topic), recovered_repo.source_documents(topic), response, [])
    assert artifact == response["artifact"] and scope["parent_token_pack"] is True
    native = capture_episode(second.directory)
    native["cost_parent_native_files"] = {origin["id"]: old_native["tail"]}
    original_forecasts = [*old_native["front"].get("budget_forecasts", []),
        *old_native["tail"].get("budget_forecasts", [])]
    native["ancestor_budget_forecasts"] = original_forecasts
    native["ancestor_budget_forecast_metadata"] = {row["file"]:
        old_native["tail"]["source-bundle-episode.json"] for row in original_forecasts}
    audit = _tail(result, _project_question(question), native, usage,
        {"previous_records": origin["usage"][question["id"]], "new_records": []},
        {"engineering_episode_stop": 90000, "max_total_requests": 4,
            "prior_known_tokens": sum(row["total_tokens"] for row in origin["usage"][question["id"]]),
            "prior_actual_requests": len(origin["usage"][question["id"]])}, [])
    assert audit["actual_tail_requests"] == 1 and audit["cached_tail_deliveries"] == 2
    assert audit["whole_episode_actual_requests"] == 4 and audit["whole_episode_known_tokens"] == 480
    assert audit["independent_A_B_inputs_verified"] is True

    files = recovered_repo.formal_submission_files(topic, run["target_revision_id"])
    assert len(files["evidence.json"]) == 3 and len(files["conclusions.json"]) == 1
    conclusion = files["conclusions.json"][0]
    assert not ({row["fact_id"] for row in conclusion["supporting_facts"]}
        & {row["fact_id"] for row in conclusion["conflicting_facts"]})
    export = recovered_repo.create_export(topic, run["target_revision_id"], "actual-worker-ZIP")
    assert Worker(recovered_repo, second).export_one() is True
    done = recovered_repo.export(export["id"])
    assert done["status"] == "succeeded", done["error"]
    zip_path = recovered_repo.one("SELECT artifact_path FROM export_jobs WHERE id=?", (export["id"],))["artifact_path"]
    with ZipFile(zip_path) as archive:
        for name, rows in files.items():
            assert json.loads(archive.read(name)) == rows
            assert sha256(archive.read(name)).hexdigest() == done["manifest"]["files"][name]["sha256"]
    assert len(new_http) == 1  # Publication/replay/export never spend new tokens.
