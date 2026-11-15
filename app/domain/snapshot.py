"""Build and validate a complete business snapshot from verified workflow results.

This boundary receives evidence already verified against source text by the
workflow. It checks canonical identity, reference integrity, graph structure and
export contracts. The repository must recheck source spans before publication;
without source paragraphs this module cannot independently prove exact quotes.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from uuid import NAMESPACE_URL, uuid5

from jsonschema import Draft202012Validator

from app.domain.confidence import compute_conclusion_confidence, compute_evidence_confidence
from app.domain.graph import validate_graph
from app.domain.source_bundles import SOURCE_BUNDLE_CORE_FIELDS, validate_source_bundle
from app.domain.source_passages import SOURCE_GEOMETRY, SOURCE_PASSAGE_CORE_FIELDS, validate_source_passage

SOURCE_BUNDLE_PROTOCOL = "document-source-bundle-v1"
SOURCE_CONTEXT_PROTOCOL = "document-source-context-v1"
SOURCE_DOCUMENT_PROTOCOLS = (SOURCE_BUNDLE_PROTOCOL, SOURCE_CONTEXT_PROTOCOL)


def source_protocol_for_results(results):
    """Union geometry only after each question's own protocol is checked."""
    protocols = {row.get("source_bundle_protocol") for row in results.values()}
    if not protocols <= {None, *SOURCE_DOCUMENT_PROTOCOLS}:
        raise ValueError("unsupported source bundle protocol")
    return (SOURCE_CONTEXT_PROTOCOL if SOURCE_CONTEXT_PROTOCOL in protocols else
            SOURCE_BUNDLE_PROTOCOL if SOURCE_BUNDLE_PROTOCOL in protocols else None)


def source_core_fields(row):
    return SOURCE_PASSAGE_CORE_FIELDS if "source_geometry" in row else SOURCE_BUNDLE_CORE_FIELDS

_ARGUMENT_KEYS = ("时间", "主体", "地点", "内容", "影响", "数值口径")
_RESOLUTIONS = {"majority_vote", "source_priority", "recency", "semantic_credibility", "scope_separation", "insufficient_evidence", "other"}
_NUMBERS = re.compile(r"\d+(?:\.\d+)?")


def _uuid(key):
    return str(uuid5(NAMESPACE_URL, "trust-judge:" + key))


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _refs(values, known, label, *, nonempty=False):
    if not isinstance(values, list) or any(not isinstance(ref, str) or ref not in known for ref in values):
        raise ValueError(f"{label} contains an unknown reference")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate references")
    if nonempty and not values:
        raise ValueError(f"{label} requires at least one qualified supporting evidence")
    return list(values)


def validate_source_bundle_rows(evidence, *, protocol=None, source_documents=None, source_paragraphs=None):
    """Recheck explicit document citations; source validity never attests reading.

    Legacy paragraph-local rows keep their existing source validator. A read
    context is not a final citation, even when its bytes are authentic.
    """
    if protocol not in (None, *SOURCE_DOCUMENT_PROTOCOLS):
        raise ValueError("unsupported source bundle protocol")
    if not isinstance(evidence, list) or any(not isinstance(row, dict) for row in evidence):
        raise ValueError("evidence source rows must be objects")
    anchor_scopes = {}
    for row in evidence:
        passage_fields = set(SOURCE_PASSAGE_CORE_FIELDS) - set(SOURCE_BUNDLE_CORE_FIELDS)
        passage = bool(passage_fields & set(row))
        if passage and (protocol != SOURCE_CONTEXT_PROTOCOL or row.get("source_geometry") != SOURCE_GEOMETRY):
            raise ValueError("sentence context geometry requires explicit source context protocol")
        scope = row.get("span_scope")
        if scope not in (None, "paragraph", "document"):
            raise ValueError("unsupported evidence span scope")
        if protocol == SOURCE_CONTEXT_PROTOCOL and scope != "document":
            raise ValueError("source context protocol requires document coordinates")
        if protocol in SOURCE_DOCUMENT_PROTOCOLS:
            # The competition format permits only one evidence entry per
            # original anchor. Do not silently widen a citation or export two
            # ranges under the same (doc_id, para_id) to satisfy that contract.
            anchor = (row.get("raw_doc_id"), row.get("raw_para_id"))
            signature = ("document", row.get("bundle_id")) if scope == "document" else ("paragraph", None)
            if anchor in anchor_scopes and anchor_scopes[anchor] != signature:
                raise ValueError("one original anchor cannot declare different citation ranges")
            anchor_scopes[anchor] = signature
        bundle = scope == "document" or any(field in row for field in
            ("bundle_id", "source_paragraph_ids", "document_content_sha256", "paragraph_manifest_sha256"))
        if not bundle:
            continue
        if protocol not in SOURCE_DOCUMENT_PROTOCOLS or scope != "document":
            raise ValueError("document citation requires explicit source bundle protocol")
        if row.get("bundle_purpose") == "read_context":
            raise ValueError("read context is not an adopted evidence citation")
        if source_documents is None or source_paragraphs is None:
            raise ValueError("document citation requires original documents and complete paragraphs")
        (validate_source_passage if passage else validate_source_bundle)(row, source_documents, source_paragraphs)


def _canonical_evidence(evidence):
    if not isinstance(evidence, dict):
        raise ValueError("canonical evidence must be an object")
    row = deepcopy(evidence)
    for key in ("id", "paragraph_id", "raw_doc_id", "raw_para_id", "quote", "fact_des", "source_group_id", "source_group_basis"):
        _text(row.get(key), f"evidence.{key}")
    start, end = row.get("start_cp"), row.get("end_cp")
    if type(start) is not int or type(end) is not int or start < 0 or end - start != len(row["quote"]):
        raise ValueError("canonical evidence must contain a valid codepoint-length span")
    argument = row.get("argument")
    if not isinstance(argument, dict) or set(argument) - set(_ARGUMENT_KEYS):
        raise ValueError("invalid evidence argument fields")
    if any(value is not None and not isinstance(value, str) for value in argument.values()):
        raise ValueError("evidence argument values must be strings or null")
    if not isinstance(row.get("entities"), list) or any(not isinstance(entity, str) or not entity for entity in row["entities"]):
        raise ValueError("evidence entities must be nonempty strings")
    # Preserve unknown source grades; they are explicitly mapped to 0 only for
    # calculation, with an unknown-grade basis rather than an invented ranking.
    grade = row.get("grade_score")
    if grade is not None:
        compute_evidence_confidence(grade, 0, True)
    return row


def _source_ref(evidence):
    return {"doc_id": evidence["raw_doc_id"], "para_id": evidence["raw_para_id"]}


def _unique_refs(ids, evidence):
    output = []
    for identity in ids:
        ref = _source_ref(evidence[identity])
        if ref not in output:
            output.append(ref)
    return output


def _confidence_groups(ids, evidence):
    # The confidence helper merges same-group entries and uses their highest
    # established source grade. Origin detection itself remains upstream.
    groups = []
    for identity in ids:
        row = evidence[identity]
        grade = row.get("grade_score")
        groups.append({"group_id": row["source_group_id"], "grade_score": 0 if grade is None else grade, "grade_basis": "unknown source grade; numeric 0 is conservative, not a calibrated probability" if grade is None else row.get("grade_basis", "caller-established source grade; not calibrated"), "evidence_refs": [{**_source_ref(row), "quote": row["quote"]}]})
    return groups


def _node(identity, kind, question_id, content, *, evidence_id=None):
    return {"id": identity, "node_id": identity, "kind": kind, "evidence_id": evidence_id, "question_id": question_id, "content": deepcopy(content), "status": "valid", "authority": "machine", "locked": False, "node_version": 1}


def _conflict_side_position(ids, evidence):
    # The issue describes the comparison, not either side's position. Preserve
    # only this side's verified factual statements, including attribution and
    # uncertainty. Fact IDs matter even when both sides quote the same paragraph.
    return "来源陈述：" + "\n".join(evidence[identity]["fact_des"] for identity in ids)


def _conflict(raw, question, final_support, evidence, topic_id):
    if not isinstance(raw, dict):
        raise ValueError("conflict must be an object")
    identity = _text(raw.get("id"), "conflict.id")
    statement = _text(raw.get("statement"), "conflict.statement")
    side_a = _refs(raw.get("supporting_evidence_ids"), evidence, "conflict supporting refs", nonempty=True)
    side_b = _refs(raw.get("conflicting_evidence_ids"), evidence, "conflict opposing refs", nonempty=True)
    if set(side_a) & set(side_b):
        raise ValueError("conflict sides must contain distinct evidence identities")
    status = raw.get("status")
    if status == "resolved":
        adopted_a, adopted_b = bool(set(final_support) & set(side_a)), bool(set(final_support) & set(side_b))
        if adopted_a == adopted_b:
            raise ValueError("resolved conflict must identify exactly one adopted side in final supporting refs")
        result = "adopt_a" if adopted_a else "adopt_b"
    elif status in {"coexisting", "unresolved"}:
        result = "coexist" if status == "coexisting" else "unresolved"
    else:
        raise ValueError("unsupported conflict status")
    resolution_type = _text(raw.get("resolution_type"), "conflict.resolution_type")
    basis = _text(raw.get("basis"), "conflict.basis")
    accepted = _text(raw.get("accepted_reason"), "conflict.accepted_reason")
    excluded = _text(raw.get("rejected_reason"), "conflict.rejected_reason")
    minority = _text(raw.get("minority_opinion"), "conflict.minority_opinion")
    reopen = _text(raw.get("reopen_conditions"), "conflict.reopen_conditions")
    minority_ids = side_b if result == "adopt_a" else side_a
    return {"id": identity, "topic_id": topic_id, "question_id": question["id"], "raw_question_id": question["raw_question_id"], "conflict_id": identity, "conflict_kind": "source_vs_source", "issue": statement, "side_a": {"position": _conflict_side_position(side_a, evidence), "agent_role": None, "evidence_refs": _unique_refs(side_a, evidence)}, "side_b": {"position": _conflict_side_position(side_b, evidence), "agent_role": None, "evidence_refs": _unique_refs(side_b, evidence)}, "result": result, "resolution_type": resolution_type if resolution_type in _RESOLUTIONS else "other", "applicability": basis + ("; original resolution type=" + resolution_type if resolution_type not in _RESOLUTIONS else ""), "adopted_reason": accepted, "excluded_reason": excluded, "minority_opinions": [{"position": minority, "evidence_refs": _unique_refs(minority_ids, evidence), "retention_reason": "保留模型显式记录的未采纳立场，供人工核对。", "reopen_condition": reopen}] if result in {"adopt_a", "adopt_b"} else [], "reopen_condition": reopen}


def build_snapshot(topic: dict, questions: list[dict], results: dict, *, revision_id: str, graph_revision: int, run_id: str, source_paragraphs=None, source_documents=None) -> dict:
    topic_id = _text(topic.get("id") if isinstance(topic, dict) else None, "topic.id")
    _text(revision_id, "revision_id")
    _text(run_id, "run_id")
    if type(graph_revision) is not int or graph_revision < 0 or not isinstance(questions, list) or not questions or not isinstance(results, dict):
        raise ValueError("snapshot requires questions, result mapping and a nonnegative graph revision")
    question_map = {}
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("question must be an object")
        for field in ("id", "raw_question_id", "question"):
            _text(question.get(field), "question." + field)
        if question.get("topic_id", topic_id) != topic_id or question["id"] in question_map:
            raise ValueError("questions must be unique and belong to the snapshot topic")
        question_map[question["id"]] = question
    if set(question_map) != set(results):
        raise ValueError("snapshot results must cover every requested question exactly once")
    if len({q["raw_question_id"] for q in questions}) != len(questions):
        raise ValueError("duplicate raw question ID within topic")
    evidence = {}
    question_evidence = {}
    for qid, result in results.items():
        if not isinstance(result, dict) or not isinstance(result.get("evidence"), list):
            raise ValueError("workflow result requires canonical evidence")
        validate_source_bundle_rows(result["evidence"], protocol=result.get("source_bundle_protocol"),
            source_documents=source_documents, source_paragraphs=source_paragraphs)
        local = {}
        for raw in result["evidence"]:
            row = _canonical_evidence(raw)
            if row["id"] in local:
                raise ValueError("duplicate evidence identity in question snapshot")
            local[row["id"]] = row
            if row["id"] in evidence:
                previous = evidence[row["id"]]
                keys = ("paragraph_id", "raw_doc_id", "raw_para_id", "start_cp", "end_cp", "quote", "source_group_id", *SOURCE_PASSAGE_CORE_FIELDS)
                if any(row.get(key) != previous.get(key) for key in keys):
                    raise ValueError("same evidence identity has incompatible source metadata")
            else:
                evidence[row["id"]] = row
        question_evidence[qid] = local
    nodes = [_node(identity, "evidence", None, row, evidence_id=identity) for identity, row in evidence.items()]
    edges, conclusions, confidence, conflicts, agent_outputs = [], [], [], [], []
    for identity, row in evidence.items():
        quality = compute_evidence_confidence(row.get("grade_score") or 0, 0, True)
        quality["object_ref"] = {**_source_ref(row), "quote": row["quote"]}
        quality["inputs"].update({"source_level": row.get("source_level") or "unknown", "source_group_id": row["source_group_id"], "support_strength_basis": "semantic support strength has not been calibrated; initial conservative numeric 0"})
        confidence.append(quality)
    for qid, question in question_map.items():
        result, local = results[qid], question_evidence[qid]
        final = result.get("final")
        if not isinstance(final, dict) or final.get("decision") not in {"asserted", "undetermined"}:
            raise ValueError("workflow requires an explicit final decision")
        relation_scope = result.get("evidence_relation_scope")
        if relation_scope is not None:
            from app.agents.scoped_arbitration import validate_scoped_result
            validate_scoped_result(relation_scope, final, list(local.values()),
                **({"question": question} if relation_scope.get("format") in {"issue-frame-evidence-relations-v2", "conclusion-aligned-evidence-relations-v3"} else {}))
        statement = _text(final.get("conclusion"), "final conclusion")
        supporting = _refs(
            final.get("supporting_evidence_ids"), local, "final supporting refs",
            nonempty=final["decision"] == "asserted",
        )
        opposing = _refs(final.get("conflicting_evidence_ids"), local, "final conflicting refs")
        if set(supporting) & set(opposing):
            raise ValueError("one final evidence relation cannot be both supporting and conflicting")
        if final["decision"] == "asserted":
            # A conclusion can explicitly reject a cited opposing claim and
            # must be allowed to identify that claim's numbers. Uncited
            # candidates and question wording remain outside this boundary.
            cited_numbers = set(_NUMBERS.findall(" ".join(local[identity]["quote"] for identity in supporting + opposing)))
            from app.domain.program_inventory import strip_verified_inventory_statements
            remaining = strip_verified_inventory_statements(statement, final.get("program_source_inventory"), set(supporting + opposing))
            if not set(_NUMBERS.findall(remaining)).issubset(cited_numbers):
                raise ValueError("asserted conclusion introduces a numeric literal absent from cited evidence")
        judgments = final.get("judgments")
        if not isinstance(judgments, list):
            raise ValueError("judgments must be a list")
        judgment_map = {}
        for judgment in judgments:
            if not isinstance(judgment, dict):
                raise ValueError("judgment must be an object")
            identity = _text(judgment.get("id"), "judgment.id")
            if identity in judgment_map or identity in evidence:
                raise ValueError("duplicate judgment or evidence node ID")
            _text(judgment.get("statement"), "judgment.statement")
            judgment_map[identity] = judgment
        referenced_prerequisites = set()
        for identity, judgment in judgment_map.items():
            refs = _refs(judgment.get("evidence_ids"), local, "judgment evidence refs")
            prerequisites = _refs(judgment.get("prerequisite_judgment_ids"), judgment_map, "judgment prerequisites")
            if not refs and not prerequisites:
                raise ValueError("judgment requires a grounded evidence path")
            referenced_prerequisites.update(prerequisites)
            nodes.append(_node(identity, "judgment", qid, judgment))
            edges.extend({"from_node_id": ref, "to_node_id": identity, "relation": "supports"} for ref in refs)
            edges.extend({"from_node_id": ref, "to_node_id": identity, "relation": "prerequisite"} for ref in prerequisites)
        conclusion_id = _uuid(qid + ":conclusion")
        if conclusion_id in evidence or conclusion_id in judgment_map:
            raise ValueError("conclusion identity collides with another node")
        labels = final.get("evidence_verdict_labels", {})
        if (not isinstance(labels, dict) or any(identity not in local or value not in {"full_support", "partial_support"}
                                               for identity, value in labels.items())):
            raise ValueError("invalid evidence verdict labels")
        conclusion = {"question_id": qid, "raw_question_id": question["raw_question_id"], "question": question["question"], "conclusion": statement, "decision": final["decision"], "supporting_evidence_ids": supporting, "conflicting_evidence_ids": opposing, "conclusion_node_id": conclusion_id,
                      "evidence_verdict_labels": deepcopy(labels)}
        if relation_scope is not None:
            conclusion["evidence_relation_scope"] = deepcopy(relation_scope)
        nodes.append(_node(conclusion_id, "conclusion", qid, conclusion))
        edges.extend({"from_node_id": ref, "to_node_id": conclusion_id, "relation": "supports"} for ref in supporting)
        edges.extend({"from_node_id": ref, "to_node_id": conclusion_id, "relation": "contradicts"} for ref in opposing)
        terminal = set(judgment_map) - referenced_prerequisites
        edges.extend({"from_node_id": ref, "to_node_id": conclusion_id, "relation": "prerequisite"} for ref in sorted(terminal))
        raw_conflicts = final.get("conflicts")
        if not isinstance(raw_conflicts, list):
            raise ValueError("arbitration requires an explicit conflict list")
        unresolved_ids = []
        for raw in raw_conflicts:
            conflicts.append(_conflict(raw, question, supporting, local, topic_id))
            if raw["status"] == "unresolved":
                unresolved_ids.extend(raw["conflicting_evidence_ids"])
        # DAG validation below proves a finite path. Every judgment requires
        # evidence or prerequisites; no ungrounded leaf can enter the chain.
        chain = list(judgment_map) + [conclusion_id]
        quality = compute_conclusion_confidence(_confidence_groups(supporting, local), _confidence_groups(list(dict.fromkeys(unresolved_ids)), local), chain, chain)
        quality["object_ref"] = {"question_id": question["raw_question_id"]}
        confidence.append(quality)
        conclusion["confidence"] = quality["score"]
        nodes[-1]["content"] = deepcopy(conclusion)
        conclusions.append(conclusion)
        analyses = result.get("analyses")
        if not isinstance(analyses, list) or len(analyses) != 2:
            raise ValueError("snapshot requires two independent analyst outputs")
        snapshot_id = _text(result.get("evidence_snapshot_id"), "evidence_snapshot_id")
        roles = {analysis.get("role") for analysis in analyses if isinstance(analysis, dict)}
        if roles != {"authority_analyst", "cross_validation_analyst"} or any(analysis.get("evidence_snapshot_id") != snapshot_id for analysis in analyses):
            raise ValueError("independent analysts must have different roles and the same frozen evidence snapshot")
        for analysis in analyses:
            agent_outputs.append({"question_id": qid, **deepcopy(analysis)})
        agent_outputs.append({"question_id": qid, "role": "arbitrate", "evidence_snapshot_id": snapshot_id, **deepcopy(final)})
        if result.get("baseline") is not None:
            baseline = result["baseline"]
            if not isinstance(baseline, dict) or baseline.get("role") != "baseline" or baseline.get("evidence_snapshot_id") != snapshot_id:
                raise ValueError("baseline must use the same evidence snapshot")
            agent_outputs.append({"question_id": qid, **deepcopy(baseline)})
    validate_graph(nodes, edges)
    if len({conflict["id"] for conflict in conflicts}) != len(conflicts):
        raise ValueError("conflict identities must be unique within a topic snapshot")
    source_groups = {}
    for identity, row in evidence.items():
        group = source_groups.setdefault(row["source_group_id"], {"group_id": row["source_group_id"], "basis": row["source_group_basis"], "evidence_ids": []})
        group["evidence_ids"].append(identity)
    snapshot = {"topic_id": topic_id, "revision_id": revision_id, "graph_revision": graph_revision, "run_id": run_id, "question_results": deepcopy(results), "nodes": nodes, "edges": edges, "evidence": list(evidence.values()), "conclusions": conclusions, "confidence": confidence, "conflicts": conflicts, "agent_outputs": agent_outputs, "source_groups": list(source_groups.values())}
    competition_files(snapshot, source_paragraphs=source_paragraphs, source_documents=source_documents)
    return snapshot


def competition_files(snapshot: dict, *, source_paragraphs=None, source_documents=None) -> dict[str, list]:
    """Project four schema-validated files, keeping raw official IDs unchanged."""
    evidence = {row["id"]: row for row in snapshot["evidence"]}
    if len(evidence) != len(snapshot["evidence"]):
        raise ValueError("snapshot repeats an evidence identity")
    bundle_ids = set()
    authorized_bundles = {}
    for result in snapshot["question_results"].values():
        validate_source_bundle_rows(result["evidence"], protocol=result.get("source_bundle_protocol"),
            source_documents=source_documents, source_paragraphs=source_paragraphs)
        bundle_ids.update(row["id"] for row in result["evidence"] if row.get("span_scope") == "document")
        for row in result["evidence"]:
            if row.get("span_scope") == "document":
                authorized_bundles.setdefault(row["id"], []).append(row)
    validate_source_bundle_rows(snapshot["evidence"], protocol=source_protocol_for_results(snapshot["question_results"]),
        source_documents=source_documents, source_paragraphs=source_paragraphs)
    if any(row.get("span_scope") == "document" and row["id"] not in bundle_ids for row in evidence.values()):
        raise ValueError("document citation absent from authorized question result")
    for row in evidence.values():
        if row.get("span_scope") == "document" and not any(
            all(row.get(field) == original.get(field) for field in SOURCE_PASSAGE_CORE_FIELDS)
            for original in authorized_bundles[row["id"]]):
            raise ValueError("document citation differs from authorized question source range")
    if not snapshot["conclusions"] or {row["question_id"] for row in snapshot["conclusions"]} != set(snapshot["question_results"]):
        raise ValueError("export must cover exactly the snapshot question set")
    # Validate the complete pre-projection quality list before merging or
    # dropping any repeated paragraph row. Set membership alone would allow a
    # tampered score or a swapped quote to be concealed by consolidation.
    expected_quality = []
    for item in evidence.values():
        quality = compute_evidence_confidence(item.get("grade_score") or 0, 0, True)
        quality["object_ref"] = {**_source_ref(item), "quote": item["quote"]}
        quality["inputs"].update({"source_level": item.get("source_level") or "unknown", "source_group_id": item["source_group_id"],
                                 "support_strength_basis": "semantic support strength has not been calibrated; initial conservative numeric 0"})
        expected_quality.append(quality)
    for conclusion in snapshot["conclusions"]:
        _refs(conclusion["supporting_evidence_ids"], evidence, "export supporting refs",
              nonempty=conclusion.get("decision") == "asserted")
        _refs(conclusion["conflicting_evidence_ids"], evidence, "export conflicting refs")
        final = snapshot["question_results"][conclusion["question_id"]]["final"]
        scope = snapshot["question_results"][conclusion["question_id"]].get("evidence_relation_scope")
        if scope != conclusion.get("evidence_relation_scope"):
            raise ValueError("conclusion changed its evidence relation target")
        if scope is not None:
            from app.agents.scoped_arbitration import validate_scoped_result
            local = snapshot["question_results"][conclusion["question_id"]]["evidence"]
            validate_scoped_result(scope, final, local,
                **({"question": conclusion["question"]} if scope.get("format") in {"issue-frame-evidence-relations-v2", "conclusion-aligned-evidence-relations-v3"} else {}))
            if (conclusion["supporting_evidence_ids"] != final["supporting_evidence_ids"]
                    or conclusion["conflicting_evidence_ids"] != final["conflicting_evidence_ids"]):
                raise ValueError("conclusion changed its scoped evidence relations")
        unresolved = list(dict.fromkeys(identity for conflict in final["conflicts"] if conflict["status"] == "unresolved"
                                        for identity in conflict["conflicting_evidence_ids"]))
        chain = [j["id"] for j in final["judgments"]] + [conclusion["conclusion_node_id"]]
        quality = compute_conclusion_confidence(_confidence_groups(conclusion["supporting_evidence_ids"], evidence),
            _confidence_groups(unresolved, evidence), chain, chain)
        quality["object_ref"] = {"question_id": conclusion["raw_question_id"]}
        expected_quality.append(quality)
    if snapshot["confidence"] != expected_quality:
        raise ValueError("quality projection cannot be reproduced from canonical evidence or references absent evidence")
    # Preserve every document/paragraph identity, including mirror sources.
    # Consolidation changes only this export projection, never the saved
    # question proofs, evidence IDs or original source text.
    grouped = {}
    def export_key(item):
        key = (item["raw_doc_id"], item["raw_para_id"])
        # A document citation is its exact original range, even when another
        # range has the same first parent. Never expand it to that one parent.
        return (*key, item["bundle_id"]) if item.get("span_scope") == "document" else key
    for item in evidence.values():
        grouped.setdefault(export_key(item), []).append(item)
    official_evidence, paragraph_export = [], {}
    source_map = {}
    for p in source_paragraphs or []:
        key = (p["raw_doc_id"], p["raw_para_id"])
        if key in source_map:
            raise ValueError("duplicate export source paragraph")
        source_map[key] = p
    for key, peers in grouped.items():
        anchor = max(peers, key=lambda item: len(item["quote"]))
        if key in source_map:
            source = source_map[key]
            for peer in peers:
                start, end = peer["start_cp"], peer["end_cp"]
                if (peer["paragraph_id"] != source["id"] or not 0 <= start < end <= len(source["text"])
                        or source["text"][start:end] != peer["quote"]):
                    raise ValueError("export source paragraph differs from canonical evidence")
            anchor = {**anchor, "quote": source["text"], "start_cp": 0, "end_cp": len(source["text"])}
        for peer in peers:
            offset = peer["start_cp"] - anchor["start_cp"]
            if (offset < 0 or peer["end_cp"] > anchor["end_cp"]
                    or anchor["quote"][offset:offset+len(peer["quote"])] != peer["quote"]):
                raise ValueError("paragraph consolidation requires an original continuous source span")
        identity = "fact_" + hashlib.sha256(json.dumps(list(key), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:20]
        def join(values):
            return "；".join(dict.fromkeys(value for value in values if isinstance(value, str) and value.strip()))
        exported = {"fact_id": identity, "doc_id": key[0], "para_id": key[1], "quote": anchor["quote"],
                    "fact_des": join(peer["fact_des"] for peer in peers),
                    "argument": {field: join(peer.get("argument", {}).get(field) for peer in peers) for field in _ARGUMENT_KEYS},
                    "entities": list(dict.fromkeys(entity for peer in peers for entity in peer["entities"]))}
        official_evidence.append(exported)
        paragraph_export[key] = exported
    resolution_codes = {"majority_vote": "Majority_Vote", "source_priority": "Source_Priority", "recency": "Most_Recent",
                        "semantic_credibility": "Credibility", "scope_separation": "Other", "insufficient_evidence": "Other", "other": "Other"}
    def rich_refs(ids, conclusion, *, conflicting=False):
        result, seen = [], set()
        labels = conclusion.get("evidence_verdict_labels", {})
        for identity in ids:
            source = evidence[identity]
            key = export_key(source)
            if key in seen:
                continue
            seen.add(key)
            exported = paragraph_export[key]
            # Full support is never fabricated from an asserted conclusion.
            # A missing per-evidence model label stays conservative/partial.
            local_ids = [i for i in ids if export_key(evidence[i]) == key]
            def signature(item):
                return json.dumps({field: item.get(field) for field in ("fact_des", "argument", "entities")},
                                  ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            covered = {signature(evidence[i]) for i in local_ids if labels.get(i) == "full_support"}
            label = "full_support" if covered and all(signature(p) in covered for p in grouped[key]) else "partial_support"
            ref = {field: deepcopy(exported[field]) for field in ("doc_id", "para_id", "fact_id", "quote", "fact_des")}
            ref["fact_des"] = "；".join(dict.fromkeys(evidence[i]["fact_des"] for i in local_ids))
            ref["verdict_label"] = label
            if conflicting:
                types = [resolution_codes.get(c["resolution_type"], "Other") for c in snapshot["conflicts"]
                         if c["question_id"] == conclusion["question_id"]
                         and any((r["doc_id"], r["para_id"]) == key[:2] for side in ("side_a", "side_b") for r in c[side]["evidence_refs"])]
                ref["resolution_type"] = types[0] if types and len(set(types)) == 1 else "Other"
            result.append(ref)
        return result
    official_conclusions = []
    for row in snapshot["conclusions"]:
        if row.get("decision") not in {"asserted", "undetermined"}:
            raise ValueError("export conclusion requires an explicit decision")
        supporting = _refs(
            row["supporting_evidence_ids"], evidence, "export supporting refs",
            nonempty=row["decision"] == "asserted",
        )
        official_conclusions.append({
            "question_id": row["raw_question_id"],
            "question": row["question"],
            "conclusion": row["conclusion"],
            "supporting_facts": rich_refs(supporting, row),
            "conflicting_facts": rich_refs(_refs(row["conflicting_evidence_ids"], evidence, "export conflicting refs"), row, conflicting=True),
        })
    conflict_keys = ("conflict_id", "conflict_kind", "side_a", "side_b", "result", "resolution_type", "applicability", "adopted_reason", "excluded_reason", "minority_opinions", "reopen_condition")
    official_conflicts = [{"question_id": row["raw_question_id"], **{key: deepcopy(row[key]) for key in conflict_keys}} for row in snapshot["conflicts"]]
    exported_confidence, confidence_seen = [], set()
    original_refs = {(item["raw_doc_id"], item["raw_para_id"], item["quote"]) for item in evidence.values()}
    projected_quotes = {}
    for item in evidence.values():
        ref = (item["raw_doc_id"], item["raw_para_id"], item["quote"])
        projected_quotes.setdefault(ref, set()).add(paragraph_export[export_key(item)]["quote"])
    def export_quote(ref):
        values = projected_quotes.get((ref["doc_id"], ref["para_id"], ref["quote"]), set())
        if len(values) != 1:
            raise ValueError("ambiguous or absent exported source range")
        return next(iter(values))
    for raw in snapshot["confidence"]:
        score = deepcopy(raw)
        if score["object_type"] == "evidence":
            if tuple(score["object_ref"][field] for field in ("doc_id", "para_id", "quote")) not in original_refs:
                raise ValueError("quality score references absent evidence")
            projected_quote = export_quote(score["object_ref"])
            key = (score["object_ref"]["doc_id"], score["object_ref"]["para_id"], projected_quote)
            if key in confidence_seen:
                continue
            confidence_seen.add(key)
            score["object_ref"]["quote"] = projected_quote
        else:
            for group_name in ("support_groups", "unresolved_conflict_groups"):
                for group in score["inputs"][group_name]:
                    refs = []
                    for ref in group["evidence_refs"]:
                        if (ref["doc_id"], ref["para_id"], ref["quote"]) not in original_refs:
                            raise ValueError("quality source group references absent evidence")
                        ref["quote"] = export_quote(ref)
                        if ref not in refs:
                            refs.append(ref)
                    group["evidence_refs"] = refs
        exported_confidence.append(score)
    files = {"evidence.json": official_evidence, "conclusions.json": official_conclusions, "confidence.json": exported_confidence, "conflicts.json": official_conflicts}
    schema_path = Path(__file__).resolve().parents[2] / "contracts/competition-export.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    definitions = {"evidence.json": "EvidenceExportArray", "conclusions.json": "ConclusionExportArray", "confidence.json": "ConfidenceExportArray", "conflicts.json": "ConflictExportArray"}
    for filename, data in files.items():
        validator = Draft202012Validator({"$ref": f"#/$defs/{definitions[filename]}", "$defs": schema["$defs"]})
        errors = sorted(validator.iter_errors(data), key=lambda error: str(list(error.path)))
        if errors:
            raise ValueError(f"{filename} failed export schema: {errors[0].message}")
    raw_refs = {(row["doc_id"], row["para_id"]) for row in official_evidence}
    for row in official_conclusions:
        if any((ref["doc_id"], ref["para_id"]) not in raw_refs for ref in row["supporting_facts"] + row["conflicting_facts"]):
            raise ValueError("conclusion export references evidence absent from the file")
    raw_questions = {row["question_id"] for row in official_conclusions}
    if len(raw_questions) != len(official_conclusions):
        raise ValueError("conclusion export repeats a question ID")
    for row in official_conflicts:
        if row["question_id"] not in raw_questions or any((ref["doc_id"], ref["para_id"]) not in raw_refs for side in (row["side_a"], row["side_b"], *row["minority_opinions"]) for ref in side["evidence_refs"]):
            raise ValueError("conflict export contains an absent question or evidence reference")
    exact_refs = {(row["doc_id"], row["para_id"], row["quote"]) for row in official_evidence}
    for row in files["confidence.json"]:
        if row["object_type"] == "conclusion":
            if row["object_ref"]["question_id"] not in raw_questions:
                raise ValueError("quality score references an absent question")
            grouped_refs = [ref for key in ("support_groups", "unresolved_conflict_groups") for group in row["inputs"][key] for ref in group["evidence_refs"]]
            if any((ref["doc_id"], ref["para_id"], ref["quote"]) not in exact_refs for ref in grouped_refs):
                raise ValueError("quality source group references absent evidence")
            recalculated = compute_conclusion_confidence(**row["inputs"])
        elif tuple(row["object_ref"][key] for key in ("doc_id", "para_id", "quote")) not in exact_refs:
            raise ValueError("quality score references absent evidence")
        else:
            recalculated = compute_evidence_confidence(row["inputs"]["grade_score"], row["inputs"]["support_strength"], row["inputs"]["quote_verified"])
        if any(row[key] != recalculated[key] for key in ("score", "formula_version", "parameter_version", "interpretation", "components", "calculation")):
            raise ValueError("quality score cannot be reproduced from its recorded inputs")
    return files
