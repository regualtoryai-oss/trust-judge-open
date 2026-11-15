from copy import deepcopy
import json

import pytest

from scripts import score_real_quality_episode as scorer
from scripts.final_evidence_scoring import normalize_dsh

Q = {"question_id": "q_demo", "question": "方案是已确认还是仍在评估？"}


def fact(identifier, doc, para="p1", text="仍在评估，尚未确认。"):
    return {"fact_id": identifier, "doc_id": doc, "para_id": para, "quote": text,
            "fact_des": text, "argument": {}, "entities": []}


def normalized(rows, supporting=(), conflicting=()):
    by_id = {row["fact_id"]: row for row in rows}
    ref = lambda identifier: {key: by_id[identifier][key] for key in ("fact_id", "doc_id", "para_id", "quote")}
    return normalize_dsh({"format": "dsh-proposed-evidence-v1", "question_id": Q["question_id"], "evidence": rows,
        "conclusion": {**Q, "conclusion": "未决。", "supporting_facts": [ref(i) for i in supporting],
            "conflicting_facts": [ref(i) for i in conflicting]}}, Q)


def reference(supporting, conflicting):
    return {**Q, "supporting_facts": [{"fact_id": i} for i in supporting],
            "conflicting_facts": [{"fact_id": i} for i in conflicting]}


def test_question_union_not_topic_denominator_and_relations_not_inferred():
    refs = [fact("g1", "a"), fact("g2", "b"), fact("g3", "topic_only")]
    selected = normalized([fact("f1", "a"), fact("f2", "b")], ["f1"])
    contexts = {(r["doc_id"], r["para_id"]): r["quote"] for r in refs}
    scope = {("a", "p1"), ("b", "p1")}
    result = scorer.score_normalized(Q, selected, refs, reference(["g1", "g2"], ["g2"]), contexts, scope, scope)
    m = result["question_selection"]["A_id_only_proxy"]
    assert (m["matches"], m["submitted_records"], m["reference_records"]) == (2, 2, 2)
    assert m["precision"] == m["recall"] == 1
    assert result["supporting"]["A_id_only_proxy"]["recall"] == .5
    assert result["conflicting"]["A_id_only_proxy"]["recall"] == 0
    assert result["single_question_vs_complete_topic_diagnostic"]["score"]["A_id_only_proxy"]["reference_records"] == 3
    assert result["relation_errors"]["supporting_facts"]["missed_reference_sources"][0]["reason"] == "selected_parent_not_linked_to_this_final_relation"
    assert result["question_selection"]["semantic_interpretations"]["II"]["lower"]["matches"] == 0
    assert result["question_selection"]["semantic_B_point_score"] is None


def test_pipeline_attrition_retains_unlocated_and_mirror_uncertainty():
    reference_rows = [fact("a", "outside"), fact("b", "unread"), fact("c", "read"), {"fact_id": "unknown"}]
    contexts = {(name, "p1"): "相同原段" for name in ("outside", "unread", "read", "mirror")}
    errors = scorer.source_errors([fact("f", "mirror")], reference_rows,
        {("unread", "p1"), ("read", "p1"), ("mirror", "p1")}, {("read", "p1"), ("mirror", "p1")}, contexts)
    assert [r["reason"] for r in errors["missed_reference_sources"]] == [
        "outside_candidate_pool", "candidate_not_completely_delivered", "complete_parent_delivered_not_selected", "unlocated_reference_retained"]
    assert errors["nonreference_selected_sources"][0]["literal_parent_mirrors_of_reference"]
    assert errors["nonreference_selected_sources"][0]["semantic_false_fact_asserted"] is False


def test_unread_final_selection_is_not_a_valid_quality_scope():
    row = fact("f", "a")
    with pytest.raises(ValueError, match="completely_delivered"):
        scorer.score_normalized(Q, normalized([row]), [fact("g", "a")], reference(["g"], []), {}, {("a", "p1")}, set())


def test_unlocated_reference_stays_in_union_denominator():
    refs = [fact("g", "a")]
    result = scorer.score_normalized(Q, normalized([]), refs, reference(["g", "missing"], []), {}, set(), set())
    assert result["pipeline"]["question_reference_union_units"] == 2
    assert result["question_selection"]["A_id_only_proxy"]["reference_records"] == 2
    assert result["question_selection"]["A_id_only_proxy"]["precision"] is None


def test_invalid_closure_never_opens_reference_or_baseline(monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("closed_bytes_changed")
    monkeypatch.setattr(scorer, "audit_closed_episode", reject)
    monkeypatch.setattr(scorer, "read_bound", lambda *a, **k: pytest.fail("read_before_closure"))
    with pytest.raises(ValueError, match="closed_bytes_changed"):
        scorer.score_episode("unused", "a" * 64, "unused", raw_input="unused", raw_sha256="a" * 64,
                             sample_root="unused", baseline_path="unused")


def test_no_final_is_quality_unavailable_not_zero(monkeypatch):
    monkeypatch.setattr(scorer, "audit_closed_episode", lambda *a, **k: {"input_guards": [], "has_final_artifact": False})
    monkeypatch.setattr(scorer, "read_bound", lambda *a, **k: {"status": "budget_blocked"})
    monkeypatch.setattr(scorer, "read_historical_baselines", lambda *a, **k: pytest.fail("baseline_before_final"))
    with pytest.raises(ValueError, match="complete_proposed_submission"):
        scorer.score_episode("unused", "a" * 64, "unused", raw_input="unused", raw_sha256="a" * 64, sample_root="unused")


def test_source_binding_checks_question_identity_titles_and_unicode_span():
    text = "甲😀乙。"
    raw = {"questions": [Q], "documents": [{"doc_id": "d", "title": "原题名", "content": text,
        "paragraphs": [{"para_id": "p", "text": text, "char_start": 0, "char_end": len(text)}]}]}
    contexts, documents = scorer.original_sources(raw)
    case = {"question": Q, "catalog": [{"paragraph_ref": "P1", "doc_id": "d", "para_id": "p", "title": "原题名", "text": text, "paragraph_complete": True}],
        "source_metadata": {"P1": {"source_text_sha256_utf8": scorer.sha256(text.encode()).hexdigest(),
            "raw_source_coordinate_binding_verified": True, "document_start_cp": 0, "document_end_cp": len(text)}}}
    assert scorer.validate_case(case, raw, contexts, documents) == {"P1": ("d", "p")}
    bad = deepcopy(case)
    bad["catalog"][0]["title"] = "别的题名"
    with pytest.raises(ValueError, match="binding_changed"):
        scorer.validate_case(bad, raw, contexts, documents)


@pytest.fixture
def recovery_fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_bytes((json.dumps(value, ensure_ascii=False) + "\n").encode())
        return scorer.bound(path)[1]
    row = {"p": "P1", "roles": ["L"], "fact_des": "尚未确认。", "argument": {"时间": "本月"}, "entities": []}
    extra = {"p": "P2", "roles": [], "fact_des": "", "argument": {"时间": ""}, "entities": []}
    original = {"selected": [row, extra], "supporting": ["P1"], "conflicting": [],
        "conclusion": "尚未确认。", "basis": "公开记录有限。", "verdict_label": "undetermined"}
    accepted = deepcopy(original)
    accepted["selected"] = [deepcopy(row)]
    accepted["selected"][0]["argument"]["时间"] = ""
    receipt = {"status": "failed_model_turn", "manifest_sha256": ""}
    manifest_d = write("manifest.json", {"casePath": "raw.json"})
    receipt["manifest_sha256"] = manifest_d["sha256"]
    receipt_d = write("closed-receipt.json", receipt)
    sdk_d = write("sdk-summary.json", {"trace": [{"event_type": "tool/call", "tool": "submit_evidence", "arguments": json.dumps(original, ensure_ascii=False)}]})
    raw_d = write("raw-args.json", original)
    accepted_d = write("accepted-args.json", accepted)
    artifact = {"evidence": [{"fact_id": "fact_001", "paragraph_ref": "P1", "proposed_roles": ["L"],
        **{key: accepted["selected"][0][key] for key in ("fact_des", "argument", "entities")}}],
        "conclusion": {"conclusion": accepted["conclusion"], "basis": accepted["basis"], "verdict_label": "undetermined",
            "supporting_facts": [{"fact_id": "fact_001"}], "conflicting_facts": []}}
    recovery = {"format": "offline-recovered-real-tool-v1", "source_kind": "offline_recovered_from_failed_real_tool",
        "closed": True, "references_read_during_recovery": False,
        "known_episode_tokens": 1000, "published": False, "semantic_verified": False,
        "transformation": {"program_only": True, "model_calls": 0}, "original_closed_receipt": receipt_d,
        "manifest": manifest_d, "sdk_summary": sdk_d, "raw_tool_arguments": raw_d,
        "accepted_tool_arguments": accepted_d, "quarantine": write("quarantine.json", [{"p": "P2"}]),
        "recovery_implementation": write("recovery-code.json", {"program": "synthetic-test"}),
        "recovered_submission": write("recovered-submission.json", artifact)}
    case = {"catalog": [{"paragraph_ref": "P1", "doc_id": "d1", "para_id": "p1", "text": "尚未确认。"},
                        {"paragraph_ref": "P2", "doc_id": "d2", "para_id": "p2", "text": "评估中。"}]}
    def validate(value):
        descriptor = write("recovery-receipt.json", value)
        return scorer.validate_offline_recovery(descriptor["file"], descriptor["sha256"], receipt_d["file"],
            receipt, manifest_d["file"], {"has_final_artifact": False, "known_total_tokens": 1000}, case, [])
    return recovery, accepted, write, validate


def test_mechanical_recovery_preserves_failure_usage_and_initial_selection(recovery_fixture):
    recovery, accepted, write, validate = recovery_fixture
    artifact, audit = validate(recovery)
    assert audit["original_end_to_end_success"] is False
    assert audit["known_episode_tokens_not_reset"] == 1000
    assert audit["original_model_selected_count"] == 2
    assert audit["replayed_accepted_count"] == 1
    assert audit["excluded_unlinked_original_refs"] == ["P2"]
    assert audit["cleared_nonliteral_time_refs"] == ["P1"]


@pytest.mark.parametrize("mutation", ["conclusion", "relation", "annotation", "budget", "initial_args"])
def test_recovery_cannot_silently_change_model_judgment_or_reset_cost(recovery_fixture, mutation):
    recovery, accepted, write, validate = recovery_fixture
    if mutation == "conclusion":
        accepted["conclusion"] = "已经确认。"
    elif mutation == "relation":
        accepted["selected"] = []
    elif mutation == "annotation":
        accepted["selected"][0]["fact_des"] = "其他段落断言。"
    elif mutation == "budget":
        recovery["known_episode_tokens"] = 0
    else:
        recovery["raw_tool_arguments"] = write("raw-changed.json", {})
    recovery["accepted_tool_arguments"] = write("accepted-mutated.json", accepted)
    with pytest.raises(ValueError):
        validate(recovery)


@pytest.fixture
def staged_delivery_fixture(tmp_path):
    def write(name, value, lines=False):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "\n".join(json.dumps(row, ensure_ascii=False) for row in value) if lines else json.dumps(value, ensure_ascii=False)
        path.write_bytes((data + "\n").encode())
        return scorer.bound(path)[1]
    case = {"catalog": [{"paragraph_ref": "P1", "title": "原题名", "text": "甲😀乙。"},
                        {"paragraph_ref": "P2", "title": "另一题名", "text": "完整未读原段。"}]}
    by_ref = {"P1": ("d1", "p1"), "P2": ("d2", "p2")}
    result = {"status": "complete_selected_original_paragraphs", "paragraph_columns": ["p", "title", "text"],
        "paragraphs": [["P1", "原题名", "甲😀乙。"]], "paragraph_complete": True, "candidate_scope_only": True,
        "source_ready": False, "model_context_delivery_verified": False, "model_comprehension_verified": False,
        "semantic_verified": False, "publishable": False}
    literal = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    proofs = [{"phase": "preview", "context_sha256": "a" * 64, "full_read_result_sha256": None,
        "read_refs": [], "native_presence_verified_before_admission": True, "semantic_verified": False},
        {"phase": "final", "context_sha256": "b" * 64, "full_read_result_sha256": scorer.sha256(literal.encode()).hexdigest(),
        "read_refs": ["P1"], "native_presence_verified_before_admission": True, "semantic_verified": False}]
    sdk = {"sessionId": "session-test", "trace": [
        {"event_type": "tool/call", "tool": "read_selected_sources", "arguments": '{"refs":["P1"]}'},
        {"event_type": "tool/result", "call_id": "read-call", "is_error": False},
        {"event_type": "tool/call", "tool": "submit_evidence", "arguments": {"selected": []}}]}
    session = [{"type": "session", "id": "session-test"},
        {"type": "tool/call", "seq": 10, "data": {"name": "read_selected_sources", "callId": "read-call", "arguments": '{"refs":["P1"]}'}},
        {"type": "tool/result", "seq": 11, "data": {"message": {"toolCallId": "read-call", "isError": False,
            "content": [{"type": "text", "text": literal}]}}},
        {"type": "tool/call", "seq": 18, "data": {"name": "submit_evidence", "callId": "submit-call", "arguments": {"selected": []}}}]
    scope = {"context_mode": "staged-short-window", "source_count": 2, "candidate_preview_refs": ["P1", "P2"],
        "local_fetched_full_refs": ["P1"], "source_context_prepared_refs": ["P1"], "source_context_prepared": True}
    def validate(*, change_proofs=None, change_session=None, change_scope=None, change_sdk=None, bind=True):
        ps, log, sc, sk = deepcopy(proofs), deepcopy(session), deepcopy(scope), deepcopy(sdk)
        if change_proofs:
            change_proofs(ps)
        if change_session:
            change_session(log)
        if change_scope:
            change_scope(sc)
        if change_sdk:
            change_sdk(sk)
        pd = write("staged-presence.jsonl", ps, True)
        sd = write("sdk-summary.json", sk)
        ld = write("sessions/session-test/session.v4.jsonl", log, True)
        receipt = {"model_requests": 2, "staged_presence_audit_sha256": pd["sha256"],
            "sdk_summary_sha256": sd["sha256"], "official_session_log": ld}
        if not bind:
            receipt.pop("official_session_log")
        return scorer.verify_staged_delivery(tmp_path / "closed-receipt.json", receipt, case, by_ref, {"scope": sc}, [])
    return validate


def test_staged_read_scope_is_actual_full_original_delivery_not_all_previews(staged_delivery_fixture):
    read, audit = staged_delivery_fixture()
    assert read == {("d1", "p1")}
    assert audit["candidate_previews"] == 2
    assert audit["complete_original_parents_delivered"] == 1
    assert audit["model_comprehension_or_entailment_verified"] is False


@pytest.mark.parametrize("mutation", ["unbound", "no_final_proof", "read_different", "altered_text", "failed_read", "scope_claims_unread", "duplicate_proof", "sdk_disagrees"])
def test_staged_cannot_score_unbound_or_not_delivered_originals(staged_delivery_fixture, mutation):
    options = {
        "unbound": {"bind": False},
        "no_final_proof": {"change_proofs": lambda rows: rows.pop()},
        "read_different": {"change_session": lambda rows: rows[1]["data"].update(arguments={"refs": ["P2"]})},
        "altered_text": {"change_session": lambda rows: rows[2]["data"]["message"]["content"][0].update(text="伪造原段")},
        "failed_read": {"change_session": lambda rows: rows[2]["data"]["message"].update(isError=True)},
        "scope_claims_unread": {"change_scope": lambda scope: scope.update(local_fetched_full_refs=["P1", "P2"])},
        "duplicate_proof": {"change_proofs": lambda rows: rows.append(deepcopy(rows[-1]))},
        "sdk_disagrees": {"change_sdk": lambda sdk: sdk["trace"][0].update(arguments={"refs": ["P2"]})},
    }
    with pytest.raises(ValueError):
        staged_delivery_fixture(**options[mutation])
