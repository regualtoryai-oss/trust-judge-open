"""Synthetic production lifecycle regression; no network or sample gold data."""
from copy import deepcopy
import json
import sqlite3
from zipfile import ZipFile

import pytest

from app.agents.scoped_arbitration import canonical_scoped_arbitration, validate_scoped_result
from app.config import Settings
from app.data.ingest import import_topic
from app.errors import ServiceError
from app.repository import Repository
from app.worker import Worker
from app.workflows.strict_workbench import BALANCED_VERSION, VERSION, compile_result, prepare_case
from test_strict_workbench import StrictFixture


QUESTION = "甲设备已经获得批准，还是尚未获得批准，公开说法是否已经核实？"
CLAIM = "甲设备尚未获得批准。"
STATEMENTS = (
    "经营方声明：甲设备已经获得批准。",
    "主管部门声明：甲设备尚未获得批准。",
    "批准状态说明：公开材料未提供批准文件，尚不足以消解两种说法。",
)


def documents():
    output = []
    for doc in range(6):
        parts = [statement + "完整合成原段😀。" * 4 for statement in STATEMENTS]
        content = "\n".join(parts)
        offset = 0; paragraphs = []
        for index, text in enumerate(parts):
            paragraphs.append({"para_id": f"synth-{doc}#p{index + 1}", "char_start": offset, "char_end": offset + len(text)})
            offset += len(text) + 1
        output.append({"doc_id": f"synth-{doc}", "title": "甲设备批准状态公开说明", "content": content,
                       "source_credibility": .9, "paragraphs": paragraphs})
    return output


def remap(value, mapping):
    if isinstance(value, dict): return {key: remap(child, mapping) for key, child in value.items()}
    if isinstance(value, list): return [remap(child, mapping) for child in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def source_selection(raw):
    def stance(text):
        if "经营方声明" in text: return "supports_hypothesis"
        if "主管部门声明" in text: return "opposes_hypothesis"
        return "qualification"
    return {"hypotheses": [{"id": "H1", "claim": "甲设备已经获得批准。"}],
            "decisions": [{"p": row["paragraph_ref"], "relevance": 3, "stance": stance(row["text"]), "hypothesis_id": "H1"}
                          for row in raw["catalog"]]}


def source_annotations(raw, retained=None):
    return {"annotations": [{"p": row["paragraph_ref"], "fact_des": row["text"],
             "argument": {"时间": "", "主体": "", "地点": "", "内容": row["text"], "影响": "", "数值口径": ""},
             "entities": ["甲设备"]} for row in raw["catalog"] if retained is None or row["paragraph_ref"] in retained]}


class BalancedFixture(StrictFixture):
    strict_balanced_enabled = True
    strict_balance_policy = {"target": 10, "per_doc_soft_cap": 2}
    strict_recall_strategy = "balanced"

    def __init__(self):
        super().__init__()
        self.selection_input = None
        self.annotation_input = None
        self.arbitration_scope = None

    async def strict_selection(self, raw):
        self.record("strict_selection"); self.selection_input = deepcopy(raw)
        return source_selection(raw)

    async def strict_annotation(self, raw, chosen):
        self.record("strict_annotation"); self.annotation_input = deepcopy((raw, chosen))
        assert {row["paragraph_ref"] for row in raw["catalog"]} == {row["p"] for row in chosen["decisions"]}
        assert all(row["paragraph_complete"] and row["parent_complete"] for row in raw["catalog"])
        return source_annotations(raw)

    async def arbitrate(self, question, evidence, analyses):
        self.record("arbitrate")
        assert {row["role"] for row in analyses} == {"authority_analyst", "cross_validation_analyst"}
        aliases = {f"E{index + 1}": row["id"] for index, row in enumerate(evidence)}
        by_alias = {alias: evidence[index] for index, alias in enumerate(aliases)}
        # Explicit synthetic model relations, checked against raw quote content.
        # The final claim is opposite to H1, so copying selection stances would
        # yield the wrong S/C inventory and the test would detect it.
        roles = {alias: ("oppose_principal_claim" if "经营方声明" in row["quote"]
                         else "support_principal_claim" if "主管部门声明" in row["quote"] else "qualification")
                 for alias, row in by_alias.items()}
        positive = [alias for alias, role in roles.items() if role == "support_principal_claim"]
        negative = [alias for alias, role in roles.items() if role == "oppose_principal_claim"]
        assert positive and negative
        wire = {"principal_claim": CLAIM, "relations": roles, "decision": "undetermined",
                "conclusion": "主管部门声明尚未批准，经营方声明已经获批；公开材料尚不足以消解两种说法。",
                "judgments": [{"id": "J1", "statement": "主管部门声明甲设备尚未获批。", "evidence_ids": positive,
                               "prerequisite_judgment_ids": []},
                              {"id": "J2", "statement": "经营方声明甲设备已经获批。", "evidence_ids": negative,
                               "prerequisite_judgment_ids": []}],
                "conflicts": [{"id": "C1", "statement": "甲设备是否已经获得批准的说法存在对立。", "status": "unresolved",
                    "supporting_evidence_ids": positive, "conflicting_evidence_ids": negative, "resolution_type": "other",
                    "basis": "公开材料包含两种相反说法。", "accepted_reason": "现有材料未能确定唯一采纳侧。",
                    "rejected_reason": "暂不排除任一侧。", "minority_opinion": "经营方的相反主张继续保留。",
                    "reopen_conditions": "获得可核验的批准程序记录后复核。"}]}
        base, scope = canonical_scoped_arbitration(wire, list(aliases))
        self.arbitration_scope = remap(scope, aliases)
        self.arbitration_scope["evidence_roles"] = {aliases[alias]: role for alias, role in scope["evidence_roles"].items()}
        return remap(base, aliases)


def setup(tmp_path, *, balanced=True, questions=1):
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="",
                        model_strict_evidence_enabled=True, strict_evidence_balanced_enabled=balanced)
    repo = Repository(settings)
    imported = import_topic(settings.database_path, documents(),
        [{"question_id": f"q{i}", "question": QUESTION} for i in range(questions)], "合成平衡工作台")
    return repo, imported["topic_id"]


def create(repo, topic, *, question_ids=None, mode="initial", key="balanced-fixture"):
    return repo.create_run(topic, {"base_revision_id": repo.topic(topic)["head_revision_id"],
        "question_ids": question_ids or [row["id"] for row in repo.questions(topic)],
        "mode": mode, "include_baseline": False}, key, model_available=True)


async def completed(tmp_path, *, questions=1):
    repo, topic = setup(tmp_path, questions=questions)
    provider = BalancedFixture(); worker = Worker(repo, provider)
    queued = create(repo, topic, question_ids=[repo.questions(topic)[0]["id"]])
    await worker.execute(repo.claim())
    done = repo.run(queued["id"])
    assert done["status"] == "succeeded", done["error"]
    return repo, topic, provider, worker, done


async def test_balanced_production_worker_publish_preview_and_zip_preserve_scoped_counterevidence(tmp_path):
    repo, topic, provider, worker, done = await completed(tmp_path)
    assert provider.calls == ["strict_selection", "strict_annotation", "authority_analyst", "cross_validation_analyst", "arbitrate"]
    assert provider.inputs[0][1:] == provider.inputs[1][1:]
    assert len(provider.selection_input["catalog"]) == 18
    assert len(provider.annotation_input[0]["catalog"]) == 10
    snapshot = repo.snapshot(topic, done["target_revision_id"])
    result = next(iter(snapshot["question_results"].values()))
    assert result["verification_protocol"] == BALANCED_VERSION
    assert len(result["strict_evidence_proof"]["selection"]["decisions"]) == 18
    assert len(result["evidence"]) == 10 and len(result["analyses"]) == 2
    scope, final = result["evidence_relation_scope"], result["final"]
    assert scope["principal_claim"] == CLAIM
    assert scope["principal_claim"] != result["evidence_selection"]["hypotheses"][0]["claim"]
    assert scope["selection_stances_used_for_mapping"] is False
    assert validate_scoped_result(scope, final, result["evidence"]) == scope
    assert final["decision"] == "undetermined" and final["conflicting_evidence_ids"]
    conflict_pairs = set()
    for row in result["evidence"]:
        if row["id"] in final["conflicting_evidence_ids"]:
            assert "经营方声明" in row["quote"] and row["selection_stance"] == "supports_hypothesis"
            conflict_pairs.add((row["raw_doc_id"], row["raw_para_id"]))
    assert snapshot["conclusions"][0]["evidence_relation_scope"] == scope
    preview = repo.submission_preview(topic, done["target_revision_id"])
    assert preview["partial"] is False
    conclusions = preview["files"]["conclusions.json"]
    assert conclusions[0]["conflicting_facts"]
    assert len(conclusions[0]["conflicting_facts"]) == len(final["conflicting_evidence_ids"])
    assert {(row["doc_id"], row["para_id"]) for row in conclusions[0]["conflicting_facts"]} == conflict_pairs
    files = repo.formal_submission_files(topic, done["target_revision_id"])
    assert files == preview["files"]
    job = repo.create_export(topic, done["target_revision_id"], "balanced-export")
    assert worker.export_one() and repo.export(job["id"])["status"] == "succeeded"
    with ZipFile(repo.settings.export_dir / (job["id"] + ".zip")) as archive:
        assert json.loads(archive.read("conclusions.json")) == files["conclusions.json"]
        assert len(json.loads(archive.read("evidence.json"))) == 10


async def test_new_v2_origin_cannot_downgrade_to_self_consistent_v1_proof(tmp_path):
    repo, topic, _, _, done = await completed(tmp_path)
    question = repo.questions(topic)[0]; paragraphs = repo.paragraphs(topic)
    raw = prepare_case(question, paragraphs)
    legitimate_v1 = compile_result(question, paragraphs, raw, source_selection(raw), source_annotations(raw))
    assert legitimate_v1["verification_protocol"] == VERSION
    with pytest.raises(ValueError):
        repo._validate_result_protocol(legitimate_v1, done["id"], topic, question["id"])


@pytest.mark.parametrize("mutation", ["missing", "role_mismatch"])
async def test_missing_or_tampered_scope_is_rejected_before_publication(tmp_path, monkeypatch, mutation):
    repo, topic = setup(tmp_path); provider = BalancedFixture(); worker = Worker(repo, provider)
    original = repo.publish
    def changed_publish(run_id, snapshot, fresh_results):
        changed_snapshot, changed_fresh = deepcopy(snapshot), deepcopy(fresh_results)
        for results in (changed_snapshot["question_results"], changed_fresh):
            for result in results.values():
                if mutation == "missing": result.pop("evidence_relation_scope")
                else:
                    ref = result["final"]["conflicting_evidence_ids"][0]
                    result["evidence_relation_scope"]["evidence_roles"][ref] = "support_principal_claim"
        for conclusion in changed_snapshot["conclusions"]:
            if mutation == "missing": conclusion.pop("evidence_relation_scope", None)
        return original(run_id, changed_snapshot, changed_fresh)
    monkeypatch.setattr(repo, "publish", changed_publish)
    queued = create(repo, topic); await worker.execute(repo.claim()); failed = repo.run(queued["id"])
    assert failed["status"] == "failed" and failed["error"]["stage"] == "publish"
    assert repo.topic(topic)["published_revision_id"] is None
    assert provider.calls[-1] == "arbitrate" and provider.calls.count("arbitrate") == 1
    assert sum(row["total_tokens"] for rows in failed["usage"].values() for row in rows) == 60


@pytest.mark.parametrize("mutation", ["missing", "role_mismatch", "principal_claim"])
async def test_persisted_scope_tampering_blocks_preview_and_formal_export(tmp_path, monkeypatch, mutation):
    repo, topic, _, _, done = await completed(tmp_path)
    snapshot = repo.snapshot(topic, done["target_revision_id"])
    result = next(iter(snapshot["question_results"].values()))
    if mutation == "missing": result.pop("evidence_relation_scope")
    elif mutation == "principal_claim": result["evidence_relation_scope"]["principal_claim"] = "不可判定"
    else:
        ref = result["final"]["conflicting_evidence_ids"][0]
        result["evidence_relation_scope"]["evidence_roles"][ref] = "support_principal_claim"
    with pytest.raises(sqlite3.IntegrityError, match="immutable_ready_revision"):
        with repo.connection() as db:
            db.execute("UPDATE revision_artifacts SET payload_json=? WHERE revision_id=?",
                       (json.dumps(snapshot, ensure_ascii=False), done["target_revision_id"]))
    # The database already blocks mutation. Inject a corrupted decoded artifact
    # at the read boundary to exercise the independent export checks as well.
    original_snapshot = repo.snapshot
    monkeypatch.setattr(repo, "snapshot", lambda tid, rid: deepcopy(snapshot)
        if (tid, rid) == (topic, done["target_revision_id"]) else original_snapshot(tid, rid))
    for action in (repo.submission_preview, repo.formal_submission_files):
        with pytest.raises(ServiceError) as failure: action(topic, done["target_revision_id"])
        assert failure.value.code == "export_validation_failed"


async def test_mixed_v1_and_v2_results_keep_original_protocol_and_export(tmp_path):
    repo, topic = setup(tmp_path, balanced=False, questions=2)
    legacy = StrictFixture()
    # Reuse the lifecycle fixture's legacy orchestration, with truthful source
    # annotations for this file's independent synthetic source collection.
    async def legacy_selection(raw): legacy.record("strict_selection"); return source_selection(raw)
    async def legacy_annotation(raw, chosen): legacy.record("strict_annotation"); return source_annotations(raw)
    legacy.strict_selection, legacy.strict_annotation = legacy_selection, legacy_annotation
    first = create(repo, topic, question_ids=[repo.questions(topic)[0]["id"]], key="first-v1")
    await Worker(repo, legacy).execute(repo.claim()); old = repo.run(first["id"])
    assert old["status"] == "succeeded", old["error"]
    before = deepcopy(repo.snapshot(topic, old["target_revision_id"])["question_results"])
    assert next(iter(before.values()))["verification_protocol"] == VERSION
    repo.settings = repo.settings.model_copy(update={"strict_evidence_balanced_enabled": True})
    provider = BalancedFixture(); worker = Worker(repo, provider)
    second = create(repo, topic, question_ids=[repo.questions(topic)[1]["id"]], mode="retry", key="second-v2")
    await worker.execute(repo.claim()); done = repo.run(second["id"])
    assert done["status"] == "succeeded", done["error"]
    results = repo.snapshot(topic, done["target_revision_id"])["question_results"]
    assert {result["verification_protocol"] for result in results.values()} == {VERSION, BALANCED_VERSION}
    for qid, result in before.items(): assert results[qid] == result
    files = repo.formal_submission_files(topic, done["target_revision_id"])
    assert len(files["conclusions.json"]) == 2
    job = repo.create_export(topic, done["target_revision_id"], "mixed-export")
    assert worker.export_one() and repo.export(job["id"])["status"] == "succeeded"
