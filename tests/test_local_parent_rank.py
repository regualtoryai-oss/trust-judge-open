"""Synthetic source/model tests. No ONNX, provider request or reference data."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from app.domain import local_parent_rank as rank
from app.domain.source_bundles import paragraph_manifest_sha256
from test_source_bundle_tool_bridge import raw_case


def case(texts=None, question="项目获批还是仍需审核？"):
    raw = raw_case()
    raw["question"]["question"] = question
    texts = texts or ["源😀" * 200 + "完整尾部甲", "仍需审核乙", ""]
    content, rows, manifest = "源前缀|", [], []
    for index, text in enumerate(texts):
        start = len(content)
        content += text
        row = {"id": f"opaque-{20-index}", "document_id": "doc-internal", "raw_doc_id": "原文档",
            "raw_para_id": f"原{index}", "document_start_cp": start, "document_end_cp": len(content), "text": text}
        rows.append(row)
        manifest.append({"id": row["id"], "raw_para_id": row["raw_para_id"], "start_cp": start,
            "end_cp": len(content), "text_sha256": sha256(text.encode()).hexdigest()})
        content += "\r\n<实际gap😀>"
    raw["paragraphs"] = rows
    raw["documents"][0].update(content=content, content_sha256=sha256(content.encode()).hexdigest(),
        paragraph_manifest=manifest, paragraph_manifest_sha256=paragraph_manifest_sha256(manifest))
    return raw


@pytest.fixture
def model(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    files = []
    for name in sorted(rank.MODEL_FILES):
        path = folder / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(("synthetic model fixture:" + name).encode())
        files.append({"file": name, "bytes": path.stat().st_size, "sha256": rank._file_sha(path)})
    manifest = {"fastembed_supported_name": rank.MODEL_NAME, "original_model": "BAAI/bge-reranker-v2-m3",
        "revision": "synthetic-pinned-revision", "model_file": "onnx/model_int8.onnx",
        "production_auto_download": False, "files": files}
    bound = rank._save(folder, "local-model-manifest.json", manifest)
    return Path(bound["file"]), bound["sha256"]


@pytest.fixture
def infer(monkeypatch):
    calls = []
    def run(job, output_dir):
        calls.append(deepcopy(job))
        job_bound = rank._save(output_dir, "inference-job.json", job)
        scores = {key: float(len(row["document"]) + (3 if "乙" in row["document"] else 0))
            for key, row in job["pairs"].items()}
        result = {"format": "local-parent-inference-result-v1", "status": "CLOSED", "job_sha256": job_bound["sha256"],
            "scores": scores, "pair_token_counts": {key: 10 for key in scores}, "runtime_packages": rank.RUNTIME_PACKAGES,
            "truncation": False, "provider_calls": 0, "external_network_calls": 0}
        bound = rank._save(output_dir, "inference-result.json", result)
        return result, bound
    monkeypatch.setattr(rank, "_run_inference", run)
    return calls


def build(tmp_path, model, raw=None, name="run", cache="cache"):
    raw = raw or case()
    path, expected = model
    return rank.build_local_rank(raw, raw_case_sha256=rank.fingerprint(raw), model_manifest=path,
        expected_manifest_sha256=expected, cache_dir=tmp_path / cache, output_dir=tmp_path / name)


def test_complete_tail_empty_body_only_and_first_literal_question_facets(model):
    raw = case(question="时点😀甲还是乙还是丙？")
    plan, pairs = rank.prepare_pairs(raw, model[1])
    assert [q["text"] for q in plan["queries"]] == ["时点😀甲还是乙还是丙？", "时点😀甲还是乙还是丙？\n重点核验：时点😀甲",
        "时点😀甲还是乙还是丙？\n重点核验：乙还是丙？"]
    for row in raw["paragraphs"]:
        windows = [r for r in plan["records"] if r["paragraph_id"] == row["id"] and r["query_id"] == "F0"]
        assert windows[0]["start_cp"] == 0 and windows[-1]["end_cp"] == len(row["text"])
        assert all(w["text"] == row["text"][w["start_cp"]:w["end_cp"]] for w in windows)
        assert all(a["end_cp"] >= b["start_cp"] for a, b in zip(windows, windows[1:]))
    assert any(payload["document"] == "" for payload in pairs.values())
    assert all("实际gap" not in payload["document"] and "程序记录" not in payload["document"] for payload in pairs.values())
    assert len(rank.prepare_pairs(case(question="原题只有一问？"), model[1])[0]["queries"]) == 1


def test_parent_max_requires_every_facet_and_every_tail_without_invented_zero(model):
    raw = case()
    plan, pairs = rank.prepare_pairs(raw, model[1])
    scores = {key: -100.0 for key in pairs}
    final = [r for r in plan["records"] if r["paragraph_id"] == raw["paragraphs"][0]["id"] and r["query_id"] == "F2"][-1]
    scores[final["pair_sha256"]] = 7.0
    parents = rank.parent_scores(raw, plan, scores)
    assert len(parents) == len(raw["paragraphs"])
    assert parents[0]["facet_scores"][-1] == {"query_id": "F2", "max_window_logit": 7.0}
    assert parents[-1]["facet_scores"][0]["max_window_logit"] == -100.0
    scores.pop(final["pair_sha256"])
    with pytest.raises(ValueError, match="complete_finite"):
        rank.parent_scores(raw, plan, scores)


def test_closed_build_exact_cache_reuse_and_independent_verification(tmp_path, model, infer):
    raw = case()
    result = build(tmp_path, model, raw)
    assert len(infer) == 1
    audit = rank.verify_local_rank(raw, result["rank_plan"], result["proof"],
        expected_model_manifest_sha256=model[1], raw_case_sha256=rank.fingerprint(raw))
    assert audit["provider_calls"] == 0 and audit["parent_count"] == 3
    assert audit["query_count"] == 3 and audit["semantic_verified"] is False
    again = build(tmp_path, model, raw, name="again")
    assert len(infer) == 1
    assert rank._read(again["proof"])["cache_hit_pairs"] > 0
    rank.verify_local_rank(raw, again["rank_plan"], again["proof"], expected_model_manifest_sha256=model[1])
    assert result["rank_plan"]["selection"]["facet_aggregation"] == "equal-weight-rrf"


@pytest.mark.parametrize("mutation", ["weights", "tokenizer", "manifest", "source"])
def test_changed_inputs_fail_before_any_local_inference(tmp_path, model, infer, mutation):
    raw = case()
    if mutation == "source":
        raw["paragraphs"][0]["text"] += "伪造尾部"
    elif mutation == "manifest":
        model[0].write_bytes(model[0].read_bytes() + b" ")
    else:
        path = model[0].parent / ("tokenizer.json" if mutation == "tokenizer" else "onnx/model_int8.onnx")
        path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        build(tmp_path, model, raw)
    assert not infer
    assert json.loads((tmp_path / "run/failed.json").read_text())["provider_calls"] == 0


def test_cache_changed_score_artifact_fails_instead_of_silent_rescore(tmp_path, model, infer):
    result = build(tmp_path, model)
    closed = rank._read(result["proof"])
    Path(closed["pair_scores"]["file"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="artifact_changed"):
        build(tmp_path, model, name="again")
    assert len(infer) == 1


def test_exact_query_cache_key_new_question_requires_new_payloads(tmp_path, model, infer):
    build(tmp_path, model)
    build(tmp_path, model, case(question="另一个原始问题？"), name="new-question")
    assert len(infer) == 2
    assert not set(infer[0]["pairs"]) & set(infer[1]["pairs"])


def test_publish_verification_rejects_tampered_parent_max_and_declared_raw_sha(tmp_path, model, infer):
    raw = case()
    result = build(tmp_path, model, raw)
    with pytest.raises(ValueError, match="raw_source_changed"):
        rank.verify_local_rank(raw, result["rank_plan"], result["proof"], expected_model_manifest_sha256=model[1], raw_case_sha256="0" * 64)
    closed = rank._read(result["proof"])
    parents = rank._read(closed["parent_scores"])
    parents[0]["facet_scores"][0]["max_window_logit"] += 1
    closed["parent_scores"] = rank._save(tmp_path, "changed-parents.json", parents)
    proof = rank._save(tmp_path, "changed-closed.json", closed)
    with pytest.raises(ValueError, match="parent_max_changed"):
        rank.verify_local_rank(raw, result["rank_plan"], proof, expected_model_manifest_sha256=model[1])


def test_runner_changes_and_partial_cache_receipts_are_rejected(tmp_path, model, infer, monkeypatch):
    result = build(tmp_path, model)
    closed = rank._read(result["proof"])
    closed["status"] = "RUNNING"
    running = rank._save(tmp_path, "running.json", closed)
    with pytest.raises(ValueError):
        rank.seed_local_cache(tmp_path / "other-cache", running, expected_model_manifest_sha256=model[1])
    sources = rank._sources()
    sources["scripts/local_parent_rank.py"]["sha256"] = "0" * 64
    monkeypatch.setattr(rank, "_sources", lambda: sources)
    with pytest.raises(ValueError, match="runner_source_changed"):
        rank.verify_local_rank(case(), result["rank_plan"], result["proof"], expected_model_manifest_sha256=model[1])


def _next_codec_sources(tmp_path):
    """An equivalent producer revision; the cached model/pair payload is unchanged."""
    sources = rank._sources()
    name = "app/domain/lc_bundle_case.py"
    path = tmp_path / "next-codec-source.py"
    path.write_bytes(Path(sources[name]["file"]).read_bytes() + b"\n# synthetic equivalent codec revision\n")
    sources[name] = rank.descriptor(path)
    return sources


def test_exact_pairs_reuse_prior_codec_archive_but_publish_requires_current_sources(tmp_path, model, infer, monkeypatch):
    raw = case()
    old = build(tmp_path, model, raw)
    old_closed = rank._read(old["proof"])
    next_sources = _next_codec_sources(tmp_path)
    monkeypatch.setattr(rank, "_sources", lambda: next_sources)

    # Both an existing exact-pair cache and an explicit CLOSED-origin import
    # accept the old bound bytes, without re-running even local inference.
    reused = build(tmp_path, model, raw, name="next-version")
    rank.seed_local_cache(tmp_path / "imported-cache", old["proof"],
        expected_model_manifest_sha256=model[1])
    imported = build(tmp_path, model, raw, name="next-imported", cache="imported-cache")
    assert len(infer) == 1
    for result in (reused, imported):
        closed = rank._read(result["proof"])
        assert closed["new_local_pairs"] == 0
        assert closed["cache_hit_pairs"] == closed["unique_payload_pairs"]
        assert all(origin == old["proof"] for origin in closed["cache_origins"].values())
        assert closed["source_archives"]["app/domain/lc_bundle_case.py"]["sha256"] != old_closed[
            "source_archives"]["app/domain/lc_bundle_case.py"]["sha256"]
        rank.verify_local_rank(raw, result["rank_plan"], result["proof"],
            expected_model_manifest_sha256=model[1])

    # The prior producer is a cache origin, not a current publishable proof.
    with pytest.raises(ValueError, match="runner_source_changed"):
        rank.verify_local_rank(raw, old["rank_plan"], old["proof"],
            expected_model_manifest_sha256=model[1])


@pytest.mark.parametrize("mutation", ["changed", "missing"])
def test_prior_codec_cache_archive_tampering_fails_without_rescore(tmp_path, model, infer, monkeypatch, mutation):
    old = build(tmp_path, model)
    closed = rank._read(old["proof"])
    next_sources = _next_codec_sources(tmp_path)
    monkeypatch.setattr(rank, "_sources", lambda: next_sources)
    archive = Path(closed["source_archives"]["app/domain/lc_bundle_case.py"]["file"])
    if mutation == "missing":
        archive.unlink()
    else:
        archive.write_bytes(archive.read_bytes() + b"\n# changed after closure\n")

    with pytest.raises(ValueError, match="runner_source_changed"):
        build(tmp_path, model, name="rejected")
    with pytest.raises(ValueError, match="runner_source_changed"):
        rank.seed_local_cache(tmp_path / "rejected-import", old["proof"],
            expected_model_manifest_sha256=model[1])
    assert len(infer) == 1
