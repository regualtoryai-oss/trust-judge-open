from copy import deepcopy
import json

import pytest

from scripts import score_workbench_real_episode as scorer


def test_known_usage_never_turns_unknown_or_inconsistent_counters_into_zero():
    row = {"usage_source": "provider", "input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
    assert scorer.known_usage([row])["total_tokens"] == 15
    for bad in ({**row, "input_tokens": None}, {**row, "total_tokens": 14},
                {**row, "usage_source": "unavailable"}, {**row, "output_tokens": True}):
        with pytest.raises(ValueError, match="actual_provider_usage"):
            scorer.known_usage([bad])


def candidate_fixture():
    question = {"question_id": "q", "question": "该方案是否确认？"}
    source = {"id": "uuid-source", "raw_doc_id": "d", "raw_para_id": "p", "title": "原题名", "text": "甲😀\n尚未确认。"}
    catalog = {"question": question, "catalog": [{"paragraph_ref": "P1", "doc_id": "d", "para_id": "p",
        "title": source["title"], "text": source["text"], "paragraph_complete": True}],
        "source_metadata": {"P1": {"paragraph_id": source["id"]}}}
    payload = {"question": question["question"], "paragraphs": [{"p": "P1", "title": source["title"], "text": source["text"]}],
        "complete_parents": True, "outside_and_adjacent_context": "unknown"}
    wire = {"messages": [{"role": "system", "content": "只读资料。"},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
    return question, source, catalog, payload, wire


def test_complete_candidate_payload_preserves_unicode_newlines_titles_and_question():
    question, source, catalog, payload, wire = candidate_fixture()
    assert scorer.verify_candidates(catalog, question, {("d", "p"): source}, wire) == {("d", "p")}


@pytest.mark.parametrize("change", ["window", "title", "question", "duplicate"])
def test_candidate_preview_or_changed_identity_cannot_claim_complete_delivery(change):
    question, source, catalog, payload, wire = candidate_fixture()
    if change == "window":
        payload["paragraphs"][0]["text"] = "尚未确认。"
    elif change == "title":
        catalog["catalog"][0]["title"] = "改写标题"
    elif change == "question":
        payload["question"] = "只问一方的新问题"
    else:
        catalog["catalog"].append({**catalog["catalog"][0], "paragraph_ref": "P2"})
        catalog["source_metadata"]["P2"] = catalog["source_metadata"]["P1"]
    wire["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    with pytest.raises(ValueError):
        scorer.verify_candidates(catalog, question, {("d", "p"): source}, wire)


def annotation_fixture():
    _q, source, catalog, _p, _w = candidate_fixture()
    selection = {"hypotheses": [{"id": "h", "claim": "已确认"}], "decisions": [
        {"p": "P1", "relevance": 2, "stance": "qualification", "hypothesis_id": "h"}]}
    row = {"p": "P1", "fact_des": "原文说明尚未确认。", "argument": {"内容": "尚未确认"}, "entities": []}
    annotation = {"annotations": [row]}
    result = {"strict_evidence_proof": {"selection": deepcopy(selection), "annotation": deepcopy(annotation)},
        "evidence": [{"paragraph_id": source["id"], **{k: deepcopy(row[k]) for k in ("fact_des", "argument", "entities")}}]}
    responses = [{"tool_calls": [{"args": selection}]}, {"tool_calls": [{"args": annotation}]}]
    return catalog, result, responses


def test_actual_visible_annotation_fields_must_match_published_fields():
    catalog, result, responses = annotation_fixture()
    assert scorer.verify_annotations(result, catalog, responses)["metadata_fabricated"] is False


@pytest.mark.parametrize("change", ["empty", "fabricated", "missing_parent", "source_only"])
def test_source_only_or_placeholder_annotations_cannot_qualify_as_complete(change):
    catalog, result, responses = annotation_fixture()
    if change == "empty":
        responses[1]["tool_calls"][0]["args"]["annotations"][0]["fact_des"] = ""
        result["strict_evidence_proof"]["annotation"] = deepcopy(responses[1]["tool_calls"][0]["args"])
    elif change == "fabricated":
        result["evidence"][0]["fact_des"] = "程序编造的新注释"
    elif change == "missing_parent":
        responses[1]["tool_calls"][0]["args"]["annotations"] = []
        result["strict_evidence_proof"]["annotation"] = {"annotations": []}
    else:
        responses[1]["tool_calls"][0]["args"] = {"annotations": ["P1"]}
    with pytest.raises(ValueError):
        scorer.verify_annotations(result, catalog, responses)


def test_reference_or_credential_path_rejected_before_file_open(monkeypatch):
    monkeypatch.setattr(scorer, "read_bound", lambda *a, **k: pytest.fail("must_reject_before_read"))
    for name in ("evidence_samples.json", "conclusions.json", "gold_answers.json", ".env"):
        with pytest.raises(ValueError, match="reference_or_credential"):
            scorer.descriptor({"file": name, "sha256": "a" * 64}, [])


def test_nonclosed_real_episode_never_reaches_reference_access(monkeypatch):
    monkeypatch.setattr(scorer, "prepare_closed", lambda *a, **k: (_ for _ in ()).throw(ValueError("not_closed")))
    monkeypatch.setattr(scorer, "read_bound", lambda *a, **k: pytest.fail("reference_before_closure"))
    with pytest.raises(ValueError, match="not_closed"):
        scorer.score_episode("unused", "a" * 64, "unused")


def test_native_whole_episode_budget_and_actual_usage_binding(tmp_path):
    def store(name, value):
        path = tmp_path / name
        path.write_bytes((json.dumps(value) + "\n").encode())
        return scorer.bound(path)[1]
    episode = store("episode.json", {"strict": True, "thinking": "disabled", "temperature": 0,
        "automatic_retries": 0, "model": "deepseek-v4-pro", "experimental_stop_threshold": 40000, "max_requests": 5, "sources": {}})
    wire = store("wire.json", {"model": "deepseek-v4-pro", "thinking": {"type": "disabled"}, "max_tokens": 4096,
        "tools": [{"function": {"name": "Select", "strict": True}}], "tool_choice": {"function": {"name": "Select"}}})
    pre = store("pre.json", {"wire": wire, "operation": "strict_selection", "known_tokens_before": 0,
        "serialized_json_token_estimate": 100, "reserved_output_tokens": 4096, "safety_tokens": 512,
        "experimental_stop_threshold": 40000, "admitted": True})
    usage = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
    settlement = store("settlement.json", {**usage, "operation": "strict_selection", "usage_known": True,
        "actual_request_started": True, "known_tokens_before": 0, "known_total_tokens_after": 150})
    response = store("response.json", {"usage": usage})
    finish = store("finished.json", {"operation": "strict_selection", "request_started": True,
        "unknown_usage_requests": 0, "known_total_tokens": 150})
    manifest = {"closed": True, "real_provider": True, "fixture": False, "run_id": "r", "episode": episode,
        "known_usage": usage, "requests": [{"wire": wire, "preflight": pre, "settlement": settlement,
            "response": response, "finished": finish}]}
    run_usage = [{**usage, "operation": "strict_selection", "usage_source": "provider"}]
    audit, _wire, _response = scorer.audit_native(manifest, [], run_id="r", run_usage=run_usage)
    assert audit["actual_requests"] == 1
    assert audit["known_usage"]["total_tokens"] == 150
    assert audit["remote_service_signed_receipt"] is False
    with pytest.raises(ValueError, match="usage_or_request_order"):
        scorer.audit_native(manifest, [], run_id="r", run_usage=[{**run_usage[0], "input_tokens": 121, "total_tokens": 151}])
    parent = store("parent-manifest.json", manifest)
    parent_run = store("parent-run.json", {"id": "r", "usage": {"q": run_usage}})
    child_pre = store("child-pre.json", {**json.loads(scorer.bound(pre["file"])[0]), "known_tokens_before": 150})
    child_finish = store("child-finished.json", {"operation": "strict_selection", "request_started": True,
        "unknown_usage_requests": 0, "known_total_tokens": 150, "known_episode_tokens_with_ancestors": 300})
    seed = store("seed.json", {"records": run_usage, "prior_known_tokens": 150, "prior_calls": 1, "not_new_usage": True})
    child = {**manifest, "run_id": "new-r", "costparents": [{"run": parent_run, "native_manifest": parent}],
        "resume_prior_usage": seed, "requests": [{"wire": wire, "preflight": child_pre,
            "settlement": settlement, "response": response, "finished": child_finish}]}
    audit, _wire, _response = scorer.audit_native(child, [], run_id="new-r", run_usage=run_usage)
    assert audit["known_usage"]["total_tokens"] == 150
    assert audit["prior_known_usage"]["total_tokens"] == 150
    assert audit["whole_episode_known_usage"]["total_tokens"] == 300
    assert audit["whole_episode_actual_requests"] == 2
    reset = deepcopy(child)
    reset.pop("costparents")
    with pytest.raises(ValueError, match="whole_cost"):
        scorer.audit_native(reset, [], run_id="new-r", run_usage=run_usage)


def keyed_fixture():
    q, source, catalog, _p, _w = candidate_fixture()
    catalog["catalog"][0]["parent_complete"] = True
    argument = {key: "" for key in ("时间", "主体", "地点", "内容", "影响", "数值口径")}
    argument["内容"] = "尚未确认"
    selection_wire = {"hypothesis": "该方案已经确认", "decisions": {"P1": "2_qualifier"}}
    annotation_wire = {"annotations": {"P1": {"fact_des": "原文说明尚未确认。", "argument": argument, "entities": []}}}
    from app.agents.strict_evidence import canonical_selection_wire, canonical_annotation_wire
    selection = canonical_selection_wire(catalog, selection_wire).model_dump()
    annotation = canonical_annotation_wire(catalog, selection, annotation_wire).model_dump()
    result = {"strict_evidence_proof": {"selection": selection, "annotation": annotation},
        "evidence": [{"paragraph_id": source["id"], **deepcopy(annotation_wire["annotations"]["P1"])}]}
    responses = [{"tool_calls": [{"args": selection_wire}]}, {"tool_calls": [{"args": annotation_wire}]}]
    decoder_sha = scorer.bound(scorer.ROOT / "app/agents/strict_evidence.py")[1]["sha256"]
    return catalog, result, responses, decoder_sha


def test_keyed_wire_fixed_decoder_preserves_canonical_model_fields():
    catalog, result, responses, decoder_sha = keyed_fixture()
    audit = scorer.verify_annotations(result, catalog, responses, decoder_sha256=decoder_sha)
    assert audit["fixed_source_bound_keyed_decoder_used"] is True
    assert audit["metadata_fabricated"] is False


@pytest.mark.parametrize("change", ["missing_p", "missing_metadata", "wrong_decoder_sha", "altered_field"])
def test_keyed_decoder_never_fills_missing_decisions_or_annotations(change):
    catalog, result, responses, decoder_sha = keyed_fixture()
    if change == "missing_p":
        responses[0]["tool_calls"][0]["args"]["decisions"].pop("P1")
    elif change == "missing_metadata":
        responses[1]["tool_calls"][0]["args"]["annotations"]["P1"].pop("fact_des")
    elif change == "wrong_decoder_sha":
        decoder_sha = "0" * 64
    else:
        responses[1]["tool_calls"][0]["args"]["annotations"]["P1"]["argument"]["主体"] = "模型原本没有的主体"
    with pytest.raises(ValueError):
        scorer.verify_annotations(result, catalog, responses, decoder_sha256=decoder_sha)


def test_nonliteral_time_projection_is_explicitly_audited_and_preserves_raw_response():
    catalog, result, responses, decoder_sha = keyed_fixture()
    raw = responses[1]["tool_calls"][0]["args"]
    raw["annotations"]["P1"]["argument"]["时间"] = "9月10日下午"
    saved = deepcopy(responses)
    change = {"p": "P1", "field": "argument.时间", "before": "9月10日下午", "after": "",
        "reason": "time_must_be_literal_or_empty"}
    log = {"operation": "strict_annotation", "changes": [change], "original_model_response_preserved": True,
        "allowed_projection": "nonliteral_argument_time_to_empty_only"}
    audit = scorer.verify_annotations(result, catalog, responses, decoder_sha256=decoder_sha, projection_audit=log)
    assert audit["optional_time_projection"]["changes"] == [change]
    assert audit["optional_time_projection"]["all_other_model_fields_unchanged"] is True
    assert responses == saved
    with pytest.raises(ValueError, match="requires_bound_native_audit"):
        scorer.verify_annotations(result, catalog, responses, decoder_sha256=decoder_sha)
    with pytest.raises(ValueError, match="independent_raw_field_diff"):
        scorer.verify_annotations(result, catalog, responses, decoder_sha256=decoder_sha,
            projection_audit={**log, "changes": []})


@pytest.mark.parametrize("field", ["literal_time", "fact_des", "entities", "numeric_argument"])
def test_independent_projection_rejects_any_other_raw_to_canonical_change(field):
    catalog, result, responses, _decoder = keyed_fixture()
    raw = responses[1]["tool_calls"][0]["args"]
    canonical = deepcopy(result["strict_evidence_proof"]["annotation"])
    if field == "literal_time":
        raw["annotations"]["P1"]["argument"]["时间"] = "尚未确认"
    elif field == "fact_des":
        canonical["annotations"][0]["fact_des"] = "程序新写描述"
    elif field == "entities":
        canonical["annotations"][0]["entities"] = ["新实体"]
    else:
        canonical["annotations"][0]["argument"]["数值口径"] = "2人"
    with pytest.raises(ValueError, match="other_or_literal"):
        scorer.verify_annotation_projection(catalog, raw, canonical, None)


def test_raw_provider_tool_wire_must_match_sdk_args_and_have_unique_keys():
    parsed = {"name": "Select", "args": {"decisions": {"P1": "2_support"}}}
    raw = {"function": {"name": "Select", "arguments": json.dumps(parsed["args"])}}
    assert scorer.verify_visible_tool_wire({"tool_calls": [parsed], "tool_calls_wire": [raw]}) is True
    raw["function"]["arguments"] = '{"decisions":{"P1":"2_support","P1":"0"}}'
    with pytest.raises(ValueError, match="duplicate_key"):
        scorer.verify_visible_tool_wire({"tool_calls": [parsed], "tool_calls_wire": [raw]})
    raw["function"]["arguments"] = '{"decisions":{"P1":"0"}}'
    with pytest.raises(ValueError, match="SDK_view"):
        scorer.verify_visible_tool_wire({"tool_calls": [parsed], "tool_calls_wire": [raw]})


def cached_native_fixture(tmp_path):
    def store(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return scorer.bound(path)[1]
    def episode(name, operations, prior=0):
        requests, records = [], []
        for i, operation in enumerate(operations):
            prefix = name + "-" + str(i)
            wire = store(prefix + "-wire.json", {"model": "deepseek-v4-pro", "thinking": {"type": "disabled"},
                "max_tokens": 4096, "messages": [{"role": "user", "content": "完整原文😀"}],
                "tools": [{"function": {"name": operation, "strict": True}}],
                "tool_choice": {"function": {"name": operation}}})
            usage = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
            record = {**usage, "operation": operation, "usage_source": "provider"}
            pre = store(prefix + "-pre.json", {"wire": wire, "operation": operation,
                "known_tokens_before": prior + i * 150, "serialized_json_token_estimate": 100,
                "reserved_output_tokens": 4096, "safety_tokens": 512,
                "experimental_stop_threshold": 60000, "admitted": True})
            raw_call = {"name": operation, "args": {"key": "原始模型字段"}}
            response = store(prefix + "-response.json", {"usage": usage, "tool_calls": [raw_call],
                "tool_calls_wire": [{"function": {"name": operation, "arguments": json.dumps(raw_call["args"])}}]})
            settlement = store(prefix + "-settle.json", {**usage, "operation": operation, "usage_known": True,
                "actual_request_started": True, "known_tokens_before": i * 150,
                "known_total_tokens_after": (i + 1) * 150})
            finish = store(prefix + "-finish.json", {"operation": operation, "request_started": True,
                "unknown_usage_requests": 0, "known_total_tokens": (i + 1) * 150,
                "known_episode_tokens_with_ancestors": prior + (i + 1) * 150})
            requests.append({"wire": wire, "preflight": pre, "settlement": settlement,
                "response": response, "finished": finish})
            records.append(record)
        meta = store(name + "-episode.json", {"strict": True, "thinking": "disabled", "temperature": 0,
            "automatic_retries": 0, "model": "deepseek-v4-pro", "experimental_stop_threshold": 60000,
            "max_requests": 5, "sources": {}})
        return {"closed": True, "fixture": False, "real_provider": True, "run_id": name,
            "episode": meta, "requests": requests, "known_usage": scorer.known_usage(records)}, records
    parent, old_usage = episode("parent", ["strict_selection", "strict_annotation"])
    child, new_usage = episode("child", ["authority_analyst", "cross_validation_analyst", "arbitrate"], prior=300)
    child["costparents"] = [{"run": store("parent-run.json", {"id": "parent", "usage": {"q": old_usage}}),
        "native_manifest": store("parent-manifest.json", parent)}]
    child["resume_prior_usage"] = store("seed.json", {"records": old_usage, "prior_known_tokens": 300,
        "prior_calls": 2, "not_new_usage": True})
    child["cached_requests"] = []
    for i, operation in enumerate(("strict_selection", "strict_annotation")):
        original = parent["requests"][i]
        hit = {"parent_run_id": "parent", "parent_request_ordinal": i + 1,
            "parent_wire_sha256": original["wire"]["sha256"],
            "parent_response_sha256": original["response"]["sha256"],
            "exact_request_JSON_values_equal": True, "no_provider_request_started": True}
        child["cached_requests"].append({"operation": operation, "wire": original["wire"],
            "response": original["response"], "cache_hit": store("hit-" + str(i) + ".json", hit),
            "finished": store("cache-finished-" + str(i) + ".json", {"operation": operation,
                "request_started": False, "cache_hit": True, "outcome": "success", "unknown_usage_requests": 0})})
    return parent, child, new_usage, store


def test_cached_two_paid_responses_plus_three_new_requests_are_not_double_charged(tmp_path):
    _parent, child, usage, _store = cached_native_fixture(tmp_path)
    audit, wires, responses = scorer.audit_native(child, [], run_id="child", run_usage=usage)
    assert audit["actual_requests"] == 3
    assert audit["cached_requests"] == 2
    assert audit["prior_actual_requests"] == 2
    assert audit["whole_episode_actual_requests"] == 5
    assert audit["known_usage"]["total_tokens"] == 450
    assert audit["whole_episode_known_usage"]["total_tokens"] == 750
    assert audit["operations"] == ["authority_analyst", "cross_validation_analyst", "arbitrate"]
    assert audit["logical_operations"] == ["strict_selection", "strict_annotation", *audit["operations"]]
    assert len(wires) == len(responses) == 5
    assert all(row["new_cost"] is False for row in audit["cached_request_bindings"])


@pytest.mark.parametrize("change", ["messages", "schema", "response", "unknown_parent", "started", "ordinal", "invoke_failed"])
def test_cache_requires_exact_original_wire_unmodified_response_and_explicit_paid_parent(tmp_path, change):
    parent, child, usage, store = cached_native_fixture(tmp_path)
    row = child["cached_requests"][0]
    if change in {"messages", "schema"}:
        wire = json.loads(scorer.bound(row["wire"]["file"])[0])
        if change == "messages":
            wire["messages"][0]["content"] = "删减后的原文"
        else:
            wire["tools"][0]["function"]["parameters"] = {"new": "schema"}
        row["wire"] = store("changed-wire.json", wire)
    elif change == "response":
        response = json.loads(scorer.bound(row["response"]["file"])[0])
        response["tool_calls"][0]["args"]["key"] = "程序改写"
        row["response"] = store("changed-response.json", response)
    elif change == "invoke_failed":
        finished = json.loads(scorer.bound(row["finished"]["file"])[0])
        finished["outcome"] = "invalid_output"
        row["finished"] = store("changed-cache-finished.json", finished)
    else:
        hit = json.loads(scorer.bound(row["cache_hit"]["file"])[0])
        hit[{"unknown_parent": "parent_run_id", "started": "no_provider_request_started", "ordinal": "parent_request_ordinal"}[change]] = {
            "unknown_parent": "unbound", "started": False, "ordinal": 2}[change]
        row["cache_hit"] = store("changed-hit.json", hit)
    with pytest.raises(ValueError, match="cache_"):
        scorer.audit_native(child, [], run_id="child", run_usage=usage)


def four_cached_native_fixture(tmp_path):
    parent, middle, middle_usage, store = cached_native_fixture(tmp_path)
    new_request = deepcopy(middle["requests"][2])
    middle["requests"] = middle["requests"][:2]
    middle_usage = middle_usage[:2]
    middle["known_usage"] = scorer.known_usage(middle_usage)
    for i, request in enumerate(middle["requests"], 3):
        wire = json.loads(scorer.bound(request["wire"]["file"])[0])
        response = json.loads(scorer.bound(request["response"]["file"])[0])
        finish = json.loads(scorer.bound(request["finished"]["file"])[0])
        settlement = json.loads(scorer.bound(request["settlement"]["file"])[0])
        if i == 4:
            wire["tools"][0]["function"]["parameters"] = {"type": "object", "properties": {"key": {"type": "string"}},
                "required": ["key"], "additionalProperties": False}
            response["tool_calls"] = []
            response["tool_calls_wire"][0]["type"] = "function"
            response["tool_calls_wire"][0]["function"]["arguments"] = '{"key": "他说"原话"。"}'
            finish["outcome"] = settlement["outcome"] = "invalid_output"
        request["wire"] = store(f"request-{i:02d}-wire.json", wire)
        request["response"] = store(f"request-{i:02d}-response.json", response)
        request["finished"] = store(f"request-{i:02d}-finished.json", finish)
        request["settlement"] = store(f"request-{i:02d}-settlement.json", settlement)
        pre = json.loads(scorer.bound(request["preflight"]["file"])[0])
        pre["wire"] = request["wire"]
        request["preflight"] = store(f"request-{i:02d}-preflight.json", pre)
    root = deepcopy(middle)
    root["run_id"] = "current"
    root["requests"] = [new_request]
    pre = json.loads(scorer.bound(new_request["preflight"]["file"])[0])
    pre["known_tokens_before"] = 600
    new_request["preflight"] = store("request-05-preflight.json", pre)
    settled = json.loads(scorer.bound(new_request["settlement"]["file"])[0])
    settled["known_tokens_before"], settled["known_total_tokens_after"] = 0, 150
    new_request["settlement"] = store("request-05-settlement.json", settled)
    finished = json.loads(scorer.bound(new_request["finished"]["file"])[0])
    finished["known_total_tokens"], finished["known_episode_tokens_with_ancestors"] = 150, 750
    new_request["finished"] = store("request-05-finished.json", finished)
    new_usage = [{"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
        "operation": "arbitrate", "usage_source": "provider"}]
    root["known_usage"] = scorer.known_usage(new_usage)
    root["costparents"] = [*middle["costparents"], {
        "run": store("middle-run.json", {"id": "child", "usage": {"q": middle_usage}}),
        "native_manifest": store("middle-manifest.json", middle)}]
    old_rows = [{"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
        "operation": op, "usage_source": "provider"} for op in ("strict_selection", "strict_annotation")]
    root["resume_prior_usage"] = store("current-seed.json", {"records": old_rows + middle_usage,
        "prior_known_tokens": 600, "prior_calls": 4, "not_new_usage": True})
    decoder_sha = scorer.bound(scorer.ROOT / "app/agents/strict_json_syntax.py")[1]["sha256"]
    episode = json.loads(scorer.bound(root["episode"]["file"])[0])
    episode["sources"] = {"app/agents/strict_json_syntax.py": decoder_sha}
    root["episode"] = store("current-episode.json", episode)
    root["cached_requests"] = deepcopy(middle["cached_requests"])
    for i, request in enumerate(middle["requests"], 3):
        op = middle_usage[i - 3]["operation"]
        row = {"operation": op, "wire": request["wire"], "response": request["response"],
            "cache_hit": store(f"cache-{i:02d}-hit.json", {"parent_run_id": "child", "parent_request_ordinal": i,
                "parent_wire_sha256": request["wire"]["sha256"], "parent_response_sha256": request["response"]["sha256"],
                "no_provider_request_started": True, "exact_request_JSON_values_equal": True}),
            "finished": store(f"cache-{i:02d}-finished.json", {"operation": op, "request_started": False,
                "cache_hit": True, "outcome": "success", "unknown_usage_requests": 0})}
        if i == 4:
            from app.agents.strict_json_syntax import decode_visible_arguments
            response = json.loads(scorer.bound(request["response"]["file"])[0])
            _decoded, audit = decode_visible_arguments(response["tool_calls_wire"][0]["function"]["arguments"])
            row["syntax-recovery"] = store("cache-04-syntax-recovery.json", {"operation": op, **audit,
                "original_response_sha256": request["response"]["sha256"], "original_response_preserved": True,
                "schema_validation_passed": True, "new_model_calls": 0, "scope": "quote_escape_insertions_only"})
        root["cached_requests"].append(row)
    root["logical_operations"] = ["strict_selection", "strict_annotation", "authority_analyst", "cross_validation_analyst", "arbitrate"]
    return root, new_usage, store


def test_four_cached_stages_one_new_arbitration_deduplicates_nested_parent_and_audits_quote_syntax(tmp_path):
    root, usage, _store = four_cached_native_fixture(tmp_path)
    audit, _wires, responses = scorer.audit_native(root, [], run_id="current", run_usage=usage)
    assert audit["actual_requests"] == 1 and audit["cached_requests"] == 4
    assert audit["prior_actual_requests"] == 4 and audit["whole_episode_actual_requests"] == 5
    assert audit["prior_known_usage"]["total_tokens"] == 600
    assert audit["whole_episode_known_usage"]["total_tokens"] == 750
    assert set(audit["cost_ancestor_run_ids"]) == {"parent", "child"}
    syntax = audit["cached_request_bindings"][3]["syntax_recovery"]
    assert syntax["only_quote_escape_insertions"] is True and len(syntax["insertions"]) == 2
    assert syntax["wire_schema_independently_validated"] is True
    assert responses[3]["tool_calls"] == []  # Preserve the original failed SDK view.


@pytest.mark.parametrize("change", ["missing_syntax_audit", "hash", "offset", "new_calls", "wrong_decoder", "different_duplicate_parent"])
def test_syntax_recovery_or_duplicate_cost_identity_cannot_be_forged(tmp_path, change):
    root, usage, store = four_cached_native_fixture(tmp_path)
    row = root["cached_requests"][3]
    if change == "missing_syntax_audit":
        row.pop("syntax-recovery")
    elif change == "wrong_decoder":
        episode = json.loads(scorer.bound(root["episode"]["file"])[0])
        episode["sources"]["app/agents/strict_json_syntax.py"] = "0" * 64
        root["episode"] = store("wrong-decoder-episode.json", episode)
    elif change == "different_duplicate_parent":
        parent = json.loads(scorer.bound(root["costparents"][0]["native_manifest"]["file"])[0])
        parent["known_usage"]["total_tokens"] += 1
        root["costparents"][0]["native_manifest"] = store("changed-parent-manifest.json", parent)
    else:
        audit = json.loads(scorer.bound(row["syntax-recovery"]["file"])[0])
        if change == "hash": audit["original_response_sha256"] = "0" * 64
        elif change == "offset": audit["insertions"][0]["offset"] += 1
        else: audit["new_model_calls"] = 1
        row["syntax-recovery"] = store("changed-syntax-audit.json", audit)
    with pytest.raises((ValueError, AssertionError)):
        scorer.audit_native(root, [], run_id="current", run_usage=usage)


def archived_analysis_wire_fixture():
    decision = {"decision": "asserted", "conclusion": "材料支持部分判断。", "judgments": [
        {"id": "J1", "statement": "这是材料直接说明的事实。", "evidence_ids": ["E1"], "prerequisite_judgment_ids": []}],
        "supporting_evidence_ids": ["E1"], "conflicting_evidence_ids": []}
    judgment = {"type": "object", "properties": {
        "id": {"type": "string"}, "statement": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "prerequisite_judgment_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["id", "statement", "evidence_ids", "prerequisite_judgment_ids"], "additionalProperties": False,
        "anyOf": [{"properties": {key: {}}, "required": [key], "additionalProperties": False}
            for key in ("evidence_ids", "prerequisite_judgment_ids")]}
    schema = {"type": "object", "properties": {"decision": {"enum": ["asserted", "undetermined"]},
        "conclusion": {"type": "string"}, "judgments": {"type": "array", "items": judgment},
        "supporting_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "conflicting_evidence_ids": {"type": "array", "items": {"type": "string"}}},
        "required": list(decision), "additionalProperties": False}
    wire = {"tools": [{"function": {"name": "AnalysisOutput", "parameters": schema}}]}
    source_sha = scorer.bound(scorer.ROOT / "app/agents/schemas.py")[1]["sha256"]
    return decision, wire, source_sha


def test_archived_empty_predicate_anomaly_is_disclosed_and_local_complete_model_is_bound():
    decoded, wire, source_sha = archived_analysis_wire_fixture()
    original = deepcopy(wire)
    audit = scorer.verify_tail_schema(decoded, wire, operation="cross_validation_analyst",
        local_schema_sha256=source_sha, allow_archived_predicates=True)
    assert audit["wire_schema_independently_validated"] is False
    assert audit["authoritative_source_bound_local_schema_validated"] is True
    assert len(audit["archived_wire_schema_anomaly"]) == 1
    assert wire == original
    from jsonschema import ValidationError
    with pytest.raises(ValidationError):
        scorer.verify_tail_schema(decoded, wire, operation="arbitrate", local_schema_sha256=source_sha)


@pytest.mark.parametrize("change", ["substantive_predicate", "other_constraint", "missing_local_field", "wrong_source"])
def test_archived_predicate_exception_never_relaxes_other_rules_or_completes_fields(change):
    decoded, wire, source_sha = archived_analysis_wire_fixture()
    schema = wire["tools"][0]["function"]["parameters"]
    if change == "substantive_predicate":
        schema["properties"]["judgments"]["items"]["anyOf"][0]["properties"]["evidence_ids"] = {"minItems": 1}
    elif change == "other_constraint":
        schema["properties"]["decision"]["enum"] = ["undetermined"]
    elif change == "missing_local_field":
        decoded.pop("conflicting_evidence_ids")
        schema["required"].remove("conflicting_evidence_ids")
    else:
        source_sha = "0" * 64
    from jsonschema import ValidationError
    with pytest.raises((ValueError, ValidationError, AssertionError)):
        scorer.verify_tail_schema(decoded, wire, operation="authority_analyst", local_schema_sha256=source_sha,
            allow_archived_predicates=True)
