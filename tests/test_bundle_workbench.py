"""Explicit synthetic source/native-wire fixtures; no paid/model requests."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from app.agents.provider import LangChainOpenAIProvider, ModelOutputInvalid
from app.agents.scoped_arbitration import canonical_scoped_arbitration
from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL
from app.workflows.analysis import AnalysisCancelled, run_analysis
from app.workflows.bundle_workbench import PROMPT_VERSION, REPLAY_SOURCES, _native_presence, compile_result, validate_result
from scripts.source_bundle_tool import execute
from test_source_bundles import corpus


Q = {"id": "question-internal", "raw_question_id": "q-synthetic", "question": "甲设备批准状态是否已经核实？"}


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def descriptor(path, value, *, raw=False, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if raw else b"\n".join(packed(row) for row in value) if lines else packed(value)
    path.write_bytes(data)
    return {"file": str(path.resolve()), "sha256": sha256(data).hexdigest()}


def sources(texts=None):
    docs, rows = corpus(texts if texts is not None else ["经营方表示甲设备已经批准。", "主管部门表示甲设备尚未批准。", "公开材料保留未确认限定。"])
    docs[0]["title"] = "甲设备批准状态"
    for p in rows:
        p.update(paragraph_id=p["id"], title=docs[0]["title"], content_sha256=docs[0]["content_sha256"],
            source_credibility_json="0.9", raw_publish_time=None,
            original_extra_metadata={"source_only_label": "保留合法仓储字段"})
    return docs, rows


def response_fixture(tmp_path, docs, rows, *, excluded=(), parent_tokens=0):
    question = {"question_id": Q["raw_question_id"], "question": Q["question"]}
    case = {"format": "frozen-source-bundle-case-v1", "question": question, "documents": docs, "paragraphs": rows, "limits": {}}
    case_d = descriptor(tmp_path / "case.json", case)
    case_bytes = Path(case_d["file"]).read_bytes()
    history = []
    for operation, arguments in (("search", {"query": Q["question"], "limit": 10, "exclude_paragraph_ids": []}),
            ("read", {"anchor_paragraph_ids": [rows[0]["id"]], "before": 0, "after": 2})):
        output = execute(case_bytes, case_d["sha256"], {"operation": operation, "arguments": arguments, "history": history})
        assert output["status"] == "ok"
        history.append(output["history_entry"])
    read_result = output["result"]
    annotation = {"start_paragraph_id": rows[0]["id"], "end_paragraph_id": rows[1]["id"],
        "fact_des": "经营方与主管部门对甲设备批准状态的说法不同。", "argument": {"时间": "", "主体": "", "地点": "", "内容": "公开说法尚有分歧", "影响": "", "数值口径": ""},
        "entities": ["甲设备"], "proposed_roles": ["L"]}
    request = {"operation": "submit", "arguments": {"selected": [annotation]}, "history": history}
    submitted = execute(case_bytes, case_d["sha256"], request)
    assert submitted["status"] == "ok"
    binding = submitted["scope"]["binding"]
    question_data = json.dumps({"original_question": json.loads(packed(question)), "corpus_document_count": len(docs),
        "corpus_paragraph_count": len(rows), "source_binding": json.loads(packed(binding)),
        "unseen_context": "unknown_not_irrelevant", "publishable": False}, ensure_ascii=False, separators=(",", ":"))
    rendered = packed(read_result).decode()
    body = json.loads(packed({"system": question_data, "messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "name": "read_source_bundles", "id": "read-native", "input": history[1]["arguments"]}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read-native", "content": [{"type": "text", "text": rendered}]}]}],
        "tools": [{"name": "submit_bundle_evidence"}], "max_tokens": 100, "thinking": {"type": "disabled"}}))
    native_d = descriptor(tmp_path / "native.json", body)
    presence = {"format": "dsh-source-bundle-native-presence-v1", "read_receipts": [{"tool_use_id": "read-native",
        "native_read_result_sha256": sha256(rendered.encode()).hexdigest(), "bundle_ids": [x["bundle_id"] for x in read_result["bundles"]],
        "source_bundle_set_sha256": read_result["source_bundle_set_sha256"], "complete_original_slices_present": True}],
        "native_messages_sha256": sha256(json.dumps(body["messages"], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "source_bytes_present_in_current_request": True, "semantic_verified": False, "publishable": False}
    artifact = {**submitted["result"], "native_model_delivery_proof": presence,
        "source_bytes_present_in_submission_request": True, "semantic_verified": False, "publishable": False}
    artifact_d = descriptor(tmp_path / "artifact.json", artifact)
    usage = {"new_request_count": 1, "new_tokens": 7, "parent_tokens": parent_tokens,
             "whole_episode_tokens": parent_tokens + 7, "known_usage_complete": True}
    closed = {"format": "dsh-source-bundle-closed-receipt-v1", "status": "succeeded", "episode_closed": True,
        "source_binding": binding, "raw_case_sha256": case_d["sha256"], "artifact_sha256": artifact_d["sha256"],
        "native_requests": [native_d], **{k: usage[k] for k in ("new_request_count", "new_tokens", "parent_tokens", "whole_episode_tokens")}}
    begun = {"status": "request_started", "request_sequence": 1, "wire_sha256": native_d["sha256"],
        "known_episode_tokens_before": parent_tokens, "reserved_output_tokens": 100}
    settled = {**begun, "status": "usage_settled", "provider_usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        "known_episode_tokens_after": parent_tokens + 7, "response_completed": True, "halted": False, "unknown_usage_requests": 0}
    root = Path(__file__).resolve().parents[1]
    proof = {"format": "dsh-source-bundle-workbench-proof-v1", "raw_case": case_d,
        "submit_request": descriptor(tmp_path / "submit-request.json", request), "tool_submission": descriptor(tmp_path / "submit-response.json", submitted),
        "artifact": artifact_d, "closed_receipt": descriptor(tmp_path / "closed.json", closed), "native_requests": [native_d],
        "presence": descriptor(tmp_path / "presence.jsonl", [presence], lines=True),
        "usage_ledger": descriptor(tmp_path / "usage.jsonl", [begun, settled], lines=True),
        "source_archives": {name: descriptor(tmp_path / "archive" / name, (root / name).read_bytes(), raw=True) for name in REPLAY_SOURCES},
        "excluded_spans": sorted(map(list, excluded)), "usage_report": usage}
    if parent_tokens:
        proof["previous_usage"] = descriptor(tmp_path / "parent.json", {"records": [{"usage_source": "provider",
            "input_tokens": parent_tokens, "output_tokens": 0, "total_tokens": parent_tokens}]})
    return {"artifact": artifact, "proof": proof, "usage_report": usage}, {"body": body, "presence": presence,
        "reads": [{"arguments": history[1]["arguments"], "result": read_result}], "question_data": question_data}


def test_compile_replays_source_wire_usage_and_separates_proposed_relations(tmp_path):
    docs, rows = sources()
    response, _ = response_fixture(tmp_path, docs, rows, parent_tokens=23)
    result = compile_result(Q, rows, docs, response)
    assert result["source_bundle_protocol"] == SOURCE_BUNDLE_PROTOCOL
    evidence = result["evidence"][0]
    assert evidence["source_paragraph_ids"] == [p["id"] for p in rows[:2]]
    assert evidence["start_cp"] == rows[0]["document_start_cp"]
    assert evidence["proposed_roles"] == ["L"] and evidence["proposed_roles_semantics_verified"] is False
    assert "evidence_relation_scope" not in result
    assert evidence["native_model_delivery_verified"]
    assert result["source_coverage"]["provided_full_paragraphs"] == 3
    assert validate_result(result, Q, rows, docs) == result
    Path(response["proof"]["native_requests"][0]["file"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file_sha256_changed"):
        validate_result(result, Q, rows, docs)


def numeric_annotation(rows):
    return {"start_paragraph_id": rows[0]["id"], "end_paragraph_id": rows[1]["id"],
        "fact_des": rows[0]["text"],
        "argument": {"时间": "", "主体": "甲设备", "地点": "", "内容": rows[0]["text"],
            "影响": "", "数值口径": "9名人员；１２元；９.５％"},
        "entities": ["甲设备", "AI171"], "proposed_roles": ["L"]}


def numeric_sources():
    return sources(["甲设备审批记录：\n9名人员参与。\n2026年开展验收，型号AI171，金额１２元，比例９.５％。😀",
        "主管部门尚未发布批准结果。", "公开材料保留未确认限定。"])


def numeric_replayed_response(monkeypatch, docs, rows, annotation):
    """Isolate the post-replay compiler gate; do not imitate a sealed episode.

    Legacy DSH submit has its own frozen JSON-based check. The real failing
    quality codec already replayed successfully; this tests that compiler
    boundary with actual document geometry, not a claimed native success.
    """
    from app.domain.source_bundles import assemble_source_bundle
    source = assemble_source_bundle(docs, rows, rows[0]["id"], rows[1]["id"])
    evidence = {**source, **{key: deepcopy(annotation[key]) for key in ("fact_des", "argument", "entities", "proposed_roles")},
        "bundle_purpose": "citation_source"}
    artifact, scope = {"evidence": [evidence]}, {"returned_complete_paragraph_ids": [r["id"] for r in rows]}
    monkeypatch.setattr("app.workflows.bundle_workbench._replay", lambda *args: deepcopy((artifact, scope)))
    return {"proof": {"format": "explicit-unit-only-post-replay-boundary"}}


def test_numeric_validation_uses_literal_fields_not_json_escaped_newlines(monkeypatch):
    from app.domain.numeric_literals import unsupported_numeric_claims
    docs, rows = numeric_sources()
    annotation = numeric_annotation(rows)
    # Demonstrate the actual regression: JSON syntax invents n9/n2026 IDs.
    assert unsupported_numeric_claims(packed([annotation["fact_des"], annotation["argument"], annotation["entities"]]).decode(), rows[0]["text"])
    response = numeric_replayed_response(monkeypatch, docs, rows, annotation)
    result = compile_result(Q, rows, docs, response)
    assert result["evidence"][0]["fact_des"] == annotation["fact_des"]
    assert result["evidence"][0]["argument"] == annotation["argument"]
    assert validate_result(result, Q, rows, docs) == result


@pytest.mark.parametrize("field,bad", [("fact_des", "\n10名人员参与。"),
    ("fact_des", "型号为AI172。"), ("数值口径", "12人"), ("数值口径", "95％")])
def test_numeric_validation_still_rejects_changed_value_unit_ratio_or_identifier(monkeypatch, field, bad):
    docs, rows = numeric_sources(); annotation = numeric_annotation(rows)
    if field == "fact_des": annotation[field] = bad
    else: annotation["argument"][field] = bad
    response = numeric_replayed_response(monkeypatch, docs, rows, annotation)
    with pytest.raises(ValueError, match="source_bundle_annotation_literal_grounding"):
        compile_result(Q, rows, docs, response)


@pytest.mark.parametrize("mutation", ["truncate", "duplicate", "unknown_read", "question"])
def test_actual_native_presence_is_replayed_not_a_boolean(mutation, tmp_path):
    docs, rows = sources(); _, native = response_fixture(tmp_path, docs, rows)
    body, presence = deepcopy(native["body"]), deepcopy(native["presence"])
    if mutation == "truncate": body["messages"][1]["content"][0]["content"][0]["text"] = "preview only"
    elif mutation == "duplicate": body["messages"].append(deepcopy(body["messages"][1]))
    elif mutation == "unknown_read": body["messages"][0]["content"][0]["input"]["anchor_paragraph_ids"] = [rows[2]["id"]]
    else: body["system"] = "question removed"
    presence["native_messages_sha256"] = sha256(json.dumps(body["messages"], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="source_bundle_native_"):
        _native_presence(body, presence, native["reads"], native["question_data"], final=True)


def test_exclusions_are_original_local_spans_projected_to_document_scope(tmp_path):
    docs, rows = sources()
    excluded = [(rows[1]["id"], 0, 3, rows[1]["text"][:3])]
    response, _ = response_fixture(tmp_path, docs, rows, excluded=excluded)
    with pytest.raises(ValueError, match="selected_operator_excluded"):
        compile_result(Q, rows, docs, response, excluded_spans=excluded)


class BundleFixture:
    model_name = "explicit-synthetic-bundle-fixture"
    prompt_version = PROMPT_VERSION
    source_bundle_enabled = strict_balanced_enabled = True
    source_bundle_protocol = SOURCE_BUNDLE_PROTOCOL
    strict_evidence_enabled = paragraph_ranked_enabled = premise_bank_enabled = False
    repair_analysis = None
    _record_usage = LangChainOpenAIProvider._record_usage

    def __init__(self, response, pause=None):
        self.response, self.pause, self.calls, self.inputs = response, pause, [], []
        self.arbitration_scope = None

    async def select_source_bundles(self, question, documents, paragraphs, *, excluded_spans):
        self.calls.append("source_bundle_selection")
        self.inputs.append(("selection", deepcopy((question, documents, paragraphs))))
        self._record_usage("source_bundle_selection", {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7})
        if self.pause is not None: self.pause["paused"] = True
        return deepcopy(self.response)

    def answer(self, evidence):
        ids = [r["id"] for r in evidence]
        return {"decision": "undetermined", "conclusion": "公开说法仍有分歧。", "supporting_evidence_ids": ids, "conflicting_evidence_ids": [],
            "judgments": [{"id": "J1", "statement": "原文保留不同主体的说法。", "evidence_ids": ids, "prerequisite_judgment_ids": []}] if ids else []}

    async def analyze(self, role, question, evidence, snapshot_id):
        self.calls.append(role); self.inputs.append((role, deepcopy((question, evidence, snapshot_id))))
        self._record_usage(role, {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4})
        result = self.answer(evidence)
        evidence[0]["fact_des"] = "Peer must not see this mutation"
        return result

    async def arbitrate(self, question, evidence, analyses):
        self.calls.append("arbitrate")
        assert {a["role"] for a in analyses} == {"authority_analyst", "cross_validation_analyst"}
        self._record_usage("arbitrate", {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4})
        aliases = {f"E{i+1}": row["id"] for i, row in enumerate(evidence)}
        answer, scope = canonical_scoped_arbitration({
            "principal_claim": "公开说法仍有分歧。", "relations": {ref: "qualification" for ref in aliases},
            **{k: v for k, v in self.answer([{"id": alias} for alias in aliases]).items() if k not in {"supporting_evidence_ids", "conflicting_evidence_ids"}},
            "conflicts": []}, list(aliases))
        def remap(value):
            if isinstance(value, dict): return {key: remap(item) for key, item in value.items()}
            if isinstance(value, list): return [remap(item) for item in value]
            return aliases.get(value, value) if isinstance(value, str) else value
        self.arbitration_scope = remap(scope)
        self.arbitration_scope["evidence_roles"] = {aliases[k]: v for k, v in scope["evidence_roles"].items()}
        return remap(answer)


async def graph(tmp_path, docs, rows, provider, *, thread="run", resume=None, pause=None):
    return await run_analysis(Q, rows, provider, source_documents=docs, checkpoint_path=tmp_path / "checkpoints.sqlite",
        thread_id=thread, resume_from_thread_id=resume, cancel_check=lambda: bool(pause and pause["paused"]), emit=lambda *_: None)


@pytest.mark.asyncio
async def test_actual_graph_has_bundle_freeze_two_independent_analysts_and_scoped_arbitration(tmp_path):
    docs, rows = sources(); response, _ = response_fixture(tmp_path / "proof", docs, rows)
    provider = BundleFixture(response)
    result = await graph(tmp_path, docs, rows, provider)
    assert set(provider.calls) == {"source_bundle_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"}
    assert provider.inputs[0][1][2] == rows  # all original CP and metadata fields
    inputs = [data for role, data in provider.inputs if role.endswith("analyst")]
    assert len(inputs) == 2 and inputs[0] == inputs[1]
    assert inputs[0][1] == result["evidence"]
    assert result["evidence_relation_scope"]["selection_stances_used_for_mapping"] is False
    assert sum(r["total_tokens"] for r in result["usage"]) == 19
    validate_result(result, Q, rows, docs)


@pytest.mark.asyncio
async def test_resume_reuses_verified_bundle_source_and_preserves_independent_inputs(tmp_path):
    docs, rows = sources(); response, _ = response_fixture(tmp_path / "proof", docs, rows)
    pause = {"paused": False}; first = BundleFixture(response, pause)
    with pytest.raises(AnalysisCancelled): await graph(tmp_path, docs, rows, first, thread="first", pause=pause)
    assert first.calls == ["source_bundle_selection"]
    second = BundleFixture(response)
    result = await graph(tmp_path, docs, rows, second, thread="second", resume="first")
    assert "source_bundle_selection" not in second.calls
    assert "source_bundle_selection" in result["execution_provenance"]["reused_nodes"]
    assert result["source_bundle_proof"]["response"]["usage_report"]["whole_episode_tokens"] == 7
    assert sum(r["total_tokens"] for r in result["usage"]) == 12
    assert all(a["evidence_snapshot_id"] == result["evidence_snapshot_id"] for a in result["analyses"])
    validate_result(result, Q, rows, docs)


@pytest.mark.parametrize("mutation", ["unknown_usage", "reset_parent", "not_closed"])
def test_unknown_usage_or_reset_episode_budget_never_passes_replay(tmp_path, mutation):
    docs, rows = sources(); response, _ = response_fixture(tmp_path, docs, rows, parent_tokens=11)
    proof = response["proof"]
    if mutation == "unknown_usage":
        events = [json.loads(line) for line in Path(proof["usage_ledger"]["file"]).read_text("utf-8").splitlines()]
        events[-1]["status"] = "usage_unknown"
        proof["usage_ledger"] = descriptor(tmp_path / "usage.jsonl", events, lines=True)
    elif mutation == "reset_parent":
        response["usage_report"]["parent_tokens"] = 0
    else:
        closed = json.loads(Path(proof["closed_receipt"]["file"]).read_text("utf-8"))
        closed["episode_closed"] = False
        proof["closed_receipt"] = descriptor(tmp_path / "closed.json", closed)
    with pytest.raises(ValueError, match="source_bundle_"):
        compile_result(Q, rows, docs, response)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["source_metadata", "native_cache"])
async def test_resume_refuses_changed_source_or_native_cache_before_new_calls(tmp_path, mutation):
    docs, rows = sources(); response, _ = response_fixture(tmp_path / "proof", docs, rows)
    pause = {"paused": False}; first = BundleFixture(response, pause)
    with pytest.raises(AnalysisCancelled): await graph(tmp_path, docs, rows, first, thread="first", pause=pause)
    if mutation == "source_metadata":
        docs[0]["title"] = "Changed original metadata"
        for row in rows: row["title"] = docs[0]["title"]
    else:
        Path(response["proof"]["native_requests"][0]["file"]).write_text("{}", encoding="utf-8")
    second = BundleFixture(response)
    with pytest.raises(ModelOutputInvalid, match="续接检查点"):
        await graph(tmp_path, docs, rows, second, thread="second", resume="first")
    assert second.calls == []


@pytest.mark.asyncio
async def test_bundle_resume_keeps_successful_parallel_analyst_and_rebinds_snapshot(tmp_path):
    docs, rows = sources(); response, _ = response_fixture(tmp_path / "proof", docs, rows)
    class FailCross(BundleFixture):
        async def analyze(self, role, question, evidence, snapshot_id):
            value = await super().analyze(role, question, evidence, snapshot_id)
            if role == "cross_validation_analyst":
                raise ModelOutputInvalid("Synthetic cross failure", stage=role)
            return value
    first = FailCross(response)
    with pytest.raises(ModelOutputInvalid): await graph(tmp_path, docs, rows, first, thread="first")
    second = BundleFixture(response)
    result = await graph(tmp_path, docs, rows, second, thread="second", resume="first")
    assert second.calls == ["cross_validation_analyst", "arbitrate"]
    assert "authority_analyst" in result["execution_provenance"]["reused_nodes"]
    assert all(row["evidence_snapshot_id"] == result["evidence_snapshot_id"] for row in result["analyses"])
    validate_result(result, Q, rows, docs)


@pytest.mark.asyncio
async def test_bundle_branch_requires_explicit_final_arbitration_scope(tmp_path):
    docs, rows = sources(); response, _ = response_fixture(tmp_path / "proof", docs, rows)
    class MissingScope(BundleFixture):
        async def arbitrate(self, question, evidence, analyses):
            result = await super().arbitrate(question, evidence, analyses)
            self.arbitration_scope = None
            return result
    with pytest.raises(ModelOutputInvalid) as error:
        await graph(tmp_path, docs, rows, MissingScope(response))
    assert error.value.details["check"] == "principal_claim_relations"
