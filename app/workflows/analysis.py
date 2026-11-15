"""Question-scoped LangGraph workflow with a shared immutable evidence snapshot."""

import copy
import hashlib
import json
import asyncio
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict
from uuid import NAMESPACE_URL, uuid4, uuid5

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agents.provider import AnalysisProvider, ModelNotConfigured, ModelOutputInvalid, capture_usage, request_output_budget
from app.agents.schemas import AnalysisOutput, ArbitrationOutput, EvidenceCandidate, ExtractionOutput
from app.domain.coverage import (
    CoverageLimitExceeded,
    merge_exact_candidates, plan_source_batches,
)
from app.domain.evidence_selection import validate_evidence_selection
from app.domain.spans import validate_span
from app.workflows.checkpoint_resume import fork_checkpoint


class AnalysisCancelled(RuntimeError):
    """Cancellation is checked at every node and after external model requests."""


class WorkflowState(TypedDict, total=False):
    source_bundle_enabled: bool
    source_bundle_protocol: str
    source_documents: list[dict]
    source_bundle_excluded_spans: list
    source_bundle_response: dict
    source_bundle_proof: dict
    strict_evidence_enabled: bool
    strict_candidate_limit: int
    strict_excluded_spans: list
    strict_case: dict
    strict_selection: dict
    strict_artifact: dict
    strict_evidence_proof: dict
    evidence_relation_scope: dict
    paragraph_ranked_enabled: bool
    paragraph_ranked_config: dict
    paragraph_ranked_excluded_spans: list
    ranked_query_catalog: dict
    ranked_recall: dict
    ranked_rank_batches: list[list[str]]
    ranked_rank_index: int
    ranked_ranks: list[dict]
    ranked_selected_refs: list[str]
    ranked_annotations: list[dict]
    ranked_field_reviews: list[dict]
    paragraph_ranked_proof: dict
    verification_protocol: str
    candidate_artifact: dict | None
    candidate_import_provenance: dict
    scope_phase: str
    scope_batches: list[list[str]]
    scope_batch_index: int
    scope_assessments: list[dict]
    scope_selected_view_ids: list[str]
    scope_narrowing_view_ids: list[str]
    scope_narrowed_parent_ids: list[str]
    scope_pending_narrow: dict | None
    scope_narrowing_ledger: list[dict]
    scope_working_evidence: list[dict]
    scope_annotation_versions: list[dict]
    scope_reverified_view_ids: list[str]
    recall_recovery_plan: dict
    quote_faithful_bank: dict
    purpose_batches: list[list[str]]
    purpose_batch_index: int
    purpose_ledgers: list[dict]
    purpose_review: dict
    strict_scope_review: dict
    material_eligibility: dict
    operator_exclusion_audit: list[dict]
    workflow_input_sha256: str
    question: dict
    paragraphs: list[dict]
    retrieved: list[dict]
    source_batches: list[dict]
    source_batch_index: int
    source_coverage: dict
    extraction_reviews: list[dict]
    candidates: list[dict]
    extract_repair_attempted: bool
    manual_candidates: list[dict]
    interpretation_overrides: dict[str, dict]
    evidence: list[dict]
    candidate_evidence: list[dict]
    evidence_selection: dict
    precision_batches: list[list[str]]
    precision_batch_index: int
    precision_selected_ids: list[str]
    precision_assessments: list[dict]
    output_capacity_floors: dict[str, int]
    evidence_snapshot_id: str
    authority_analysis: dict
    cross_analysis: dict
    baseline_analysis: dict
    final: dict


def _uuid(key: str) -> str:
    return str(uuid5(NAMESPACE_URL, "trust-judge:" + key))


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))



def _filter_operator_exclusions(evidence, excluded):
    """Whole immutable candidate windows yield to any explicit span overlap.

    Never crop/relocate a quote. The original extraction checkpoint and imported
    candidate artifact stay unchanged; only a metadata audit is added here.
    """
    from app.domain.strict_verification import build_fact_views
    kept, audit = [], []
    for row in evidence:
        matches = [(start, end, quote) for pid, start, end, quote in excluded
                   if pid == row["paragraph_id"] and row["start_cp"] < end and start < row["end_cp"]]
        if not matches:
            kept.append(copy.deepcopy(row))
            continue
        audit.append({"evidence_id": row["id"], "paragraph_id": row["paragraph_id"],
            "start_cp": row["start_cp"], "end_cp": row["end_cp"],
            "quote_sha256": hashlib.sha256(row["quote"].encode("utf-8")).hexdigest(),
            "fact_view_ids": [view["id"] for view in build_fact_views([row])],
            "interpretation_origin": row.get("interpretation_origin"), "reason": "operator_excluded_overlap",
            "overlapping_exclusions": [{"start_cp": start, "end_cp": end,
                "quote_sha256": hashlib.sha256(quote.encode("utf-8")).hexdigest()}
                for start, end, quote in sorted(matches)]})
    return kept, audit


def _plan_precision_batches(evidence: list[dict], *, max_candidates=48, max_codepoints=40_000) -> list[list[str]]:
    """Bound each review request without dropping any source-verified candidate.

    Source-local windows are ordered together before grouping exact quotations,
    so overlapping context and a retained representative can share a batch.
    This is a review convenience,
    not proof of common origin or permission to discard different sources.
    One candidate larger than the character target is retained in its own batch.
    """
    if type(max_candidates) is not int or max_candidates < 1 or type(max_codepoints) is not int or max_codepoints < 1:
        raise ValueError("invalid_precision_batch_limits")
    groups = {}
    seen = set()
    def source_order(item):
        text_key = lambda field: item.get(field) if isinstance(item.get(field), str) else ""
        return (text_key("raw_doc_id"), text_key("raw_para_id"), text_key("paragraph_id"),
                item.get("start_cp") if type(item.get("start_cp")) is int else 0,
                item.get("end_cp") if type(item.get("end_cp")) is int else 0)

    for item in sorted(evidence, key=source_order):
        identity = item["id"]
        if identity in seen:
            raise ValueError("duplicate_precision_candidate")
        seen.add(identity)
        groups.setdefault(item["quote"], []).append(item)
    batches, pending, codepoints = [], [], 2

    def append_pending():
        nonlocal pending, codepoints
        if pending:
            batches.append(pending)
            pending, codepoints = [], 2

    for group in groups.values():
        group_sizes = [len(_canonical_json(item)) for item in group]
        group_codepoints = sum(group_sizes) + max(0, len(group) - 1)
        fits_alone = len(group) <= max_candidates and group_codepoints + 2 <= max_codepoints
        if fits_alone and pending and (len(pending) + len(group) > max_candidates or codepoints + group_codepoints + 1 > max_codepoints):
            append_pending()
        for item, size in zip(group, group_sizes):
            if pending and (len(pending) >= max_candidates or codepoints + size + 1 > max_codepoints):
                append_pending()
            codepoints += size + bool(pending)
            pending.append(item["id"])
    append_pending()
    planned = [identity for batch in batches for identity in batch]
    if len(planned) != len(evidence) or set(planned) != seen:
        raise ValueError("incomplete_precision_batches")
    return batches


def _project_question(question: dict) -> dict:
    # Defense in depth: runtime imports already strip answers. Never forward
    # conclusion/supporting_facts or other unknown fields to a model.
    keys = ("id", "topic_id", "raw_question_id", "question_id", "question")
    projected = {key: copy.deepcopy(question[key]) for key in keys if key in question}
    if not isinstance(projected.get("question"), str) or not projected["question"].strip():
        raise ValueError("A projected question must contain nonempty question text")
    if not projected.get("id"):
        raise ValueError("A projected question must contain its internal id")
    return projected


def _source_groups(paragraphs: list[dict]) -> dict[str, tuple[str, str]]:
    """Conservatively join documents by exact hash or an explicit shared URL.

    This is deterministic grouping of known identity signals, not semantic
    repost/original-origin detection. Missing metadata leaves separate groups.
    """
    parents = {paragraph["document_id"]: paragraph["document_id"] for paragraph in paragraphs}

    def root(document):
        while parents[document] != document:
            parents[document] = parents[parents[document]]
            document = parents[document]
        return document

    signals = {}
    doc_signals = {document: set() for document in parents}
    for paragraph in paragraphs:
        document = paragraph["document_id"]
        for field, prefix in (("content_sha256", "content"), ("content_url", "explicit_url")):
            if not paragraph.get(field):
                continue
            signal = prefix + ":" + paragraph[field]
            doc_signals[document].add(signal)
            if signal in signals:
                left, right = sorted((root(document), root(signals[signal])))
                parents[right] = left
            else:
                signals[signal] = document
    groups = {}
    for document in parents:
        groups.setdefault(root(document), set()).update(doc_signals[document])
    return {
        document: (
            _uuid("source:" + min(groups[root(document)] or {"document:" + root(document)})),
            "known_exact_hash_or_explicit_url_only" if groups[root(document)] else "same_document_only_not_complete_origin_detection",
        )
        for document in parents
    }


def _project_paragraph(paragraph: dict) -> dict:
    keys = (
        "id", "document_id", "raw_doc_id", "raw_para_id", "text", "start_cp", "end_cp",
        "content_sha256", "content_url", "title", "raw_publish_time", "parsed_publish_time", "time_precision",
        "source_credibility_json", "source_credibility",
        "document_start_cp", "document_end_cp",
    )
    return {key: copy.deepcopy(paragraph[key]) for key in keys if key in paragraph}


def _source_location_choices(candidates, paragraphs, *, candidate_index=None):
    """Offer bounded literal locations, never choose an occurrence for a model.

    These source snippets are private repair input, not public diagnostics. Only
    paragraphs actually supplied to extraction are eligible. Coordinates use
    Python/Unicode codepoints and still go through the ordinary span verifier.
    """
    if not isinstance(candidates, list):
        return []
    available = {row["id"]: row["text"] for row in paragraphs}
    choices = []
    indexed = enumerate(candidates[:12])
    if type(candidate_index) is int and 0 <= candidate_index < len(candidates):
        # The invalid item may occur after an old diagnostic display window;
        # offer its bounded literal choices without dropping earlier evidence.
        indexed = [(candidate_index, candidates[candidate_index])]
    for index, candidate in indexed:
        if not isinstance(candidate, dict):
            continue
        paragraph_id, quote = candidate.get("paragraph_id"), candidate.get("quote")
        if not isinstance(paragraph_id, str) or paragraph_id not in available or not isinstance(quote, str) or not quote:
            continue
        text, spans = available[paragraph_id], []
        start = text.find(quote)
        while start >= 0 and len(spans) < 8:
            end = start + len(quote)
            spans.append({"start_cp": start, "end_cp": end,
                          "context_before": text[max(0, start - 32):start],
                          "context_after": text[end:end + 32]})
            start = text.find(quote, start + 1)
        if len(spans) > 1:
            choices.append({"candidate_index": index, "paragraph_id": paragraph_id,
                            "quote": quote, "valid_spans": spans})
    return choices


def _project_manual_candidate(candidate: dict) -> dict:
    if not isinstance(candidate, dict):
        raise ValueError("人工新增证据必须是字段对象。")
    allowed = ("paragraph_id", "quote", "fact_des", "argument", "entities", "start_cp", "end_cp", "stance", "id", "source_evidence_id", "fact_view_id", "manual_identity")
    output = {key: copy.deepcopy(candidate[key]) for key in allowed if key in candidate}
    if "stance" in output:
        output.update(_project_interpretation({"stance": output["stance"]}))
    if isinstance(output.get("argument"), dict):
        output["argument"] = {
            key: value for key, value in output["argument"].items()
            if key in {"时间", "主体", "地点", "内容", "影响", "数值口径"}
        }
    return output


def _project_interpretation(patch: dict) -> dict:
    """Annotations can change interpretation only; immutable fields are ignored."""
    if not isinstance(patch, dict):
        raise ValueError("人工解释修订必须是字段对象。")
    allowed = {key: copy.deepcopy(patch[key]) for key in ("fact_des", "argument", "entities", "stance") if key in patch}
    for field in ("fact_des", "stance"):
        if field in allowed and (not isinstance(allowed[field], str) or not allowed[field].strip()):
            raise ValueError("人工事实描述和立场必须是非空文本。")
    if "entities" in allowed and (
        not isinstance(allowed["entities"], list) or any(not isinstance(value, str) for value in allowed["entities"])
    ):
        raise ValueError("人工实体字段必须是文本列表。")
    if "argument" in allowed:
        if not isinstance(allowed["argument"], dict):
            raise ValueError("人工论元字段必须是对象。")
        allowed["argument"] = {
            key: value for key, value in allowed["argument"].items()
            if key in {"时间", "主体", "地点", "内容", "影响", "数值口径"}
        }
        if any(value is not None and not isinstance(value, str) for value in allowed["argument"].values()):
            raise ValueError("人工论元值必须是文本或未知null。")
    return allowed


def _validate_analysis(payload: dict, evidence: list[dict], question_id: str, role: str, *, arbitration=False, source_inventory_resolver=None) -> dict:
    def invalid(message: str, check: str, **location) -> ModelOutputInvalid:
        # Use fixed diagnostic labels only. Returned prose, quotes and model
        # identifiers must not leak through error messages or validation input.
        return ModelOutputInvalid(message, stage=role, details={"question_id": question_id, "check": check, **location})

    schema = ArbitrationOutput if arbitration else AnalysisOutput
    try:
        output = schema.model_validate(payload).model_dump()
    except (ValidationError, TypeError, ValueError):
        raise invalid(f"{role} 输出未通过结构校验。", "schema") from None
    evidence_ids = {item["id"] for item in evidence}
    local_ids = [item["id"] for item in output["judgments"]]
    if len(local_ids) != len(set(local_ids)):
        raise invalid("判断局部ID不能重复。", "duplicate_judgment_id")
    mapping = {local_id: _uuid(f"judgment:{question_id}:{role}:{local_id}") for local_id in local_ids}
    refs_to_check = [output["supporting_evidence_ids"], output["conflicting_evidence_ids"]]
    prerequisites = {}
    for index, judgment in enumerate(output["judgments"]):
        refs_to_check.append(judgment["evidence_ids"])
        dependencies = judgment["prerequisite_judgment_ids"]
        if not judgment["evidence_ids"] and not dependencies:
            raise invalid("判断必须具有证据或已声明的前提。", "ungrounded_judgment",
                          judgment_index=index, field_path=f"judgments[{index}]",
                          evidence_count=0, prerequisite_count=0)
        if len(dependencies) != len(set(dependencies)):
            raise invalid("判断前提引用不能重复。", "duplicate_prerequisite",
                          judgment_index=index, field_path=f"judgments[{index}].prerequisite_judgment_ids")
        if any(dependency not in mapping for dependency in dependencies):
            raise invalid("判断引用了未声明的前提。", "unknown_prerequisite")
        prerequisites[judgment["id"]] = dependencies
    if arbitration and set(output["supporting_evidence_ids"]) & set(output["conflicting_evidence_ids"]):
        raise invalid("同一证据不能同时作为结论的支持和冲突依据。", "overlapping_evidence_relations")
    # Validate the model's local DAG before allocating any business identifiers.
    visiting, visited = set(), set()

    def visit(node):
        if node in visiting:
            raise invalid("判断前提存在循环。", "prerequisite_cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in prerequisites[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in prerequisites:
        visit(node)
    if arbitration:
        conflict_ids = [conflict["id"] for conflict in output["conflicts"]]
        if len(conflict_ids) != len(set(conflict_ids)):
            raise invalid("冲突局部ID不能重复。", "duplicate_conflict_id")
        for conflict in output["conflicts"]:
            refs_to_check.extend([conflict["supporting_evidence_ids"], conflict["conflicting_evidence_ids"]])
            conflict["id"] = _uuid(f"conflict:{question_id}:{role}:{conflict['id']}")
    for refs in refs_to_check:
        if len(refs) != len(set(refs)) or any(ref not in evidence_ids for ref in refs):
            raise invalid("证据引用重复或不属于本轮冻结快照。", "invalid_evidence_reference")
    # Only provisional A/B labels can be ambiguous. Validate the original
    # references first, then isolate overlap in a derived view without choosing
    # a direction, changing any model prose, or altering the sealed response.
    if not arbitration and role in {"authority_analyst", "cross_validation_analyst"}:
        from app.domain.provisional_relations import quarantine_provisional_relations
        output, relation_quality = quarantine_provisional_relations(output, role)
        if relation_quality is not None:
            output["relation_quality"] = relation_quality
    elif set(output["supporting_evidence_ids"]) & set(output["conflicting_evidence_ids"]):
        raise invalid("同一证据不能同时作为结论的支持和冲突依据。", "overlapping_evidence_relations")
    if output["decision"] == "asserted" and not output["supporting_evidence_ids"]:
        raise invalid("确定性结论必须引用支持证据。", "assertion_without_support")
    if arbitration:
        # Check model-owned final relations before leaving the repair boundary.
        # These are the same invariants enforced again by the snapshot builder.
        for index, conflict in enumerate(output["conflicts"]):
            left, right = set(conflict["supporting_evidence_ids"]), set(conflict["conflicting_evidence_ids"])
            if left & right:
                raise invalid("冲突双方必须引用不同证据。", "overlapping_conflict_sides",
                              field_path=f"conflicts[{index}]")
            if conflict["status"] == "resolved":
                support = set(output["supporting_evidence_ids"])
                if bool(support & left) == bool(support & right):
                    raise invalid("已消解冲突必须明确唯一采纳侧。", "ambiguous_conflict_resolution",
                                  field_path=f"conflicts[{index}]")
        if output["decision"] == "asserted":
            cited = set(output["supporting_evidence_ids"] + output["conflicting_evidence_ids"])
            numbers = re.compile(r"\d+(?:\.\d+)?")
            known = set(numbers.findall(" ".join(item["quote"] for item in evidence if item["id"] in cited)))
            remaining = output["conclusion"]
            if source_inventory_resolver is not None:
                from app.domain.program_inventory import strip_verified_inventory_statements
                proof = source_inventory_resolver(cited)
                remaining = strip_verified_inventory_statements(remaining, proof, cited)
                if remaining != output["conclusion"]:
                    output["program_source_inventory"] = proof
            if not set(numbers.findall(remaining)).issubset(known):
                raise invalid("结论包含已引用证据中不存在的数字。", "uncited_numeric_literal",
                              field_path="conclusion")
    for judgment in output["judgments"]:
        judgment["id"] = mapping[judgment["id"]]
        judgment["prerequisite_judgment_ids"] = [mapping[ref] for ref in judgment["prerequisite_judgment_ids"]]
    return output


async def run_analysis(
    question: dict,
    paragraphs: list[dict],
    provider: AnalysisProvider,
    *,
    checkpoint_path: Path,
    thread_id: str,
    cancel_check: Callable[[], bool],
    emit: Callable[[str, dict], None],
    excluded_spans: set[tuple[str, int, int, str]] | None = None,
    include_baseline: bool = False,
    retrieval_limit: int = 16,
    source_batch_codepoints: int = 20_000,
    source_batch_spans: int = 128,
    manual_evidence: list[dict] | None = None,
    interpretation_overrides: dict[str, dict] | None = None,
    usage_callback: Callable[[list[dict]], None] | None = None,
    resume_from_thread_id: str | None = None,
    strict_scope_enabled: bool = True,
    candidate_artifact: dict | None = None,
    test_only_legacy_precision: bool = False,
    source_documents: list[dict] | None = None,
) -> dict:
    """Run real nodes, checkpointing each superstep to a separate SQLite file.

    This builds execution state only; publishing business revisions is the
    caller's responsibility and must perform its own cancellation/CAS checks.
    """
    if provider is None:
        raise ModelNotConfigured("真实研判需要显式配置模型。")
    bundle_enabled = getattr(provider, "source_bundle_enabled", False)
    if type(bundle_enabled) is not bool:
        raise ValueError("source_bundle_enabled must be explicit boolean")
    if bundle_enabled:
        from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL, SOURCE_CONTEXT_PROTOCOL
        from app.workflows.bundle_workbench import PROMPT_VERSION
        bundle_framework = getattr(provider, "source_bundle_config", {}).get("front_framework", "dsh")
        context_passages = getattr(provider, "source_bundle_config", {}).get("context_passages", False)
        bundle_protocol = SOURCE_CONTEXT_PROTOCOL if context_passages else SOURCE_BUNDLE_PROTOCOL
        expected_bundle_prompt = ("trusted-analysis-langchain-source-bundles-v1" if bundle_framework == "langchain" else PROMPT_VERSION)
        if (type(context_passages) is not bool or context_passages and bundle_framework != "langchain"
                or getattr(provider, "source_bundle_protocol", None) != bundle_protocol
                or bundle_framework not in {"dsh", "langchain"}
                or getattr(provider, "prompt_version", None) != expected_bundle_prompt
                or getattr(provider, "strict_evidence_enabled", False)
                or getattr(provider, "paragraph_ranked_enabled", False) or getattr(provider, "premise_bank_enabled", False)
                or test_only_legacy_precision or candidate_artifact is not None or manual_evidence or interpretation_overrides
                or include_baseline or not callable(getattr(provider, "select_source_bundles", None))
                or not isinstance(source_documents, list) or not source_documents):
            raise ModelNotConfigured("原文段落束模式的完整源文、协议或适配器配置不一致，未调用模型。")
        strict_scope_enabled = False
    strict_enabled = getattr(provider, "strict_evidence_enabled", False)
    if type(strict_enabled) is not bool:
        raise ValueError("strict_evidence_enabled must be explicit boolean")
    strict_candidate_limit = getattr(provider, "strict_candidate_limit", 60)
    if strict_enabled:
        from app.workflows.strict_workbench import PROMPT_VERSION
        if (getattr(provider, "premise_bank_enabled", False) or getattr(provider, "paragraph_ranked_enabled", False)
                or test_only_legacy_precision or candidate_artifact is not None or manual_evidence or interpretation_overrides
                or include_baseline or getattr(provider, "prompt_version", None) != PROMPT_VERSION
                or any(not callable(getattr(provider, method, None)) for method in ("strict_selection", "strict_annotation"))):
            raise ModelNotConfigured("严格选证模式的适配器、策略或人工覆盖配置不一致，未调用模型。")
        strict_scope_enabled = False
    ranked_enabled = getattr(provider, "paragraph_ranked_enabled", False)
    if type(ranked_enabled) is not bool:
        raise ValueError("paragraph_ranked_enabled must be explicit boolean")
    if ranked_enabled:
        from app.domain.paragraph_ranked import config_values, PROMPT_VERSION
        if (getattr(provider, "premise_bank_enabled", False) or test_only_legacy_precision or candidate_artifact is not None
                or manual_evidence or interpretation_overrides):
            raise ModelNotConfigured("v19不能混用旧候选/人工解释覆盖或旧严格账本；请使用独立段落策略。")
        if (getattr(provider, "prompt_version", None) != PROMPT_VERSION or
                any(not callable(getattr(provider, method, None)) for method in ("paragraph_queries", "paragraph_rank", "paragraph_annotate", "paragraph_fields"))):
            raise ModelNotConfigured("v19模型适配器与段落策略配置不一致，未调用模型。")
        ranked_config = config_values(getattr(provider, "paragraph_ranked_config", None))
        strict_scope_enabled = False
    if not strict_scope_enabled and not test_only_legacy_precision and not ranked_enabled and not strict_enabled and not bundle_enabled:
        raise ValueError("Production workflows require strict scope verification")
    if test_only_legacy_precision:
        # Only explicitly injected test doubles may exercise the legacy graph.
        # There is no API option or missing-capability production fallback.
        if provider.__class__.__module__.startswith("app."):
            raise ValueError("Legacy precision compatibility is test-only")
        strict_scope_enabled = False
    if strict_scope_enabled and not callable(getattr(provider, "verify_evidence_scope", None)):
        raise ModelNotConfigured("模型适配器未实现严格事实复核，未开始模型任务。")
    if candidate_artifact is not None and not strict_scope_enabled:
        raise ValueError("Candidate artifact reuse requires strict scope verification")
    if not thread_id:
        raise ValueError("A nonempty checkpoint thread_id is required")
    if type(retrieval_limit) is not int or retrieval_limit < 1:
        raise ValueError("retrieval_limit must be a positive integer")
    if manual_evidence is not None and not isinstance(manual_evidence, list):
        raise ValueError("manual_evidence must be a list")
    if interpretation_overrides is not None and not isinstance(interpretation_overrides, dict):
        raise ValueError("interpretation_overrides must be an evidence-id mapping")
    clean_question = _project_question(question)
    premise_bank_enabled = getattr(provider, "premise_bank_enabled", False)
    if type(premise_bank_enabled) is not bool:
        raise ValueError("premise_bank_enabled must be an explicit boolean")
    if premise_bank_enabled and (not strict_scope_enabled or not callable(getattr(provider, "review_fact_purpose", None))):
        raise ModelNotConfigured("v18模型适配器必须支持严格复核与独立用途审核，不回退旧策略。")
    paragraph_map = {row["id"]: copy.deepcopy(row) if bundle_enabled else _project_paragraph(row) for row in paragraphs}
    if len(paragraph_map) != len(paragraphs):
        raise ValueError("Duplicate paragraph identifiers are not accepted")
    if bundle_enabled:
        from app.domain.source_bundles import build_source_bundles
        build_source_bundles(source_documents, list(paragraph_map.values()), [])
    excluded = excluded_spans or set()
    source_groups = _source_groups(list(paragraph_map.values()))
    manual_candidates = [_project_manual_candidate(candidate) for candidate in (manual_evidence or [])]
    overrides = {}
    for evidence_id, patch in (interpretation_overrides or {}).items():
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("Interpretation override requires an evidence id")
        overrides[evidence_id] = _project_interpretation(patch)

    def cancelled():
        if cancel_check():
            raise AnalysisCancelled("研判已请求取消。")

    def begin(node: str):
        cancelled()
        emit("node_started", {"node": node, "question_id": clean_question["id"]})

    def complete(node: str, detail: dict | None = None):
        # Preserve a returned paid response even when pause arrives during it.
        # The next node checks cancellation before doing any new work.
        emit("node_completed", {"node": node, "question_id": clean_question["id"], **(detail or {})})

    def snapshot_id(evidence):
        digest = hashlib.sha256(_canonical_json(evidence).encode("utf-8")).hexdigest()
        prior = getattr(provider, "reusable_frozen_snapshot", None) if strict_enabled or bundle_enabled else None
        if prior is not None:
            if prior.get("snapshot_sha256") != digest or not prior.get("evidence_snapshot_id"):
                raise ModelOutputInvalid("本次证据与已付研判的冻结快照不同，未复用或调用模型。",
                    stage="freeze", code="source_integrity_failed", details={"check":"frozen_snapshot_reuse_binding"})
            return prior["evidence_snapshot_id"]
        return _uuid(f"snapshot:{thread_id}:{digest}")

    reported_usage = None

    def report_usage(*, allow_empty=False):
        nonlocal reported_usage
        if usage_callback is None or (not usage and not allow_empty):
            return
        serialized = _canonical_json(usage)
        if serialized != reported_usage:
            reported_usage = serialized
            usage_callback(copy.deepcopy(usage))

    # Learned output limits are local to this question execution and exact
    # workflow stage. A confirmed length failure establishes that later batches
    # and content repairs of that stage should not repeat the smaller limit.
    # This is a requested-capacity high water mark, never a usage counter.
    capacity_floors: dict[str, int] = {}

    async def request_with_capacity(stage, callback):
        """Re-request only this stage after a provider-confirmed length finish.

        These are fresh requests with the identical source/snapshot inputs, not
        continuation text. Per-task ContextVars isolate the concurrent analysts;
        content-repair counts and completed graph stages remain untouched.
        An expanded stage limit is reused by later batches and its repairs;
        separate analysts, extraction and precision review never share floors.
        Providers without this optional capability preserve their old behavior.
        """
        policy_for = getattr(provider, "capacity_policy", None)
        if not callable(policy_for):
            return await callback()
        policy = policy_for(stage)
        initial, limit, retries = policy["initial_tokens"], policy["max_tokens"], policy["retries"]
        if (type(initial) is not int or type(limit) is not int or type(retries) is not int
                or not 1 <= initial <= limit <= 393216 or not 0 <= retries <= 3):
            raise ValueError("Invalid model capacity policy")
        if (strict_enabled or bundle_enabled) and retries != 0:
            raise ValueError("Strict workflows forbid automatic capacity retries")
        budget, attempt = min(limit, max(initial, capacity_floors.get(stage, initial))), 0

        def annotate(error):
            if isinstance(error, ModelOutputInvalid):
                if error.stage is None:
                    error.stage = stage
                actual_capacity = error.details.get("last_max_output_tokens")
                # The official DSH front can make several native requests with
                # different validated wire ceilings inside one graph node.
                # Preserve that failed request's actual ceiling, not the
                # enclosing stage's maximum submission capacity.
                if not (bundle_enabled and stage == "source_bundle_selection"
                        and type(actual_capacity) is int and 1 <= actual_capacity <= limit):
                    actual_capacity = budget
                error.details = {**error.details, "capacity_attempts": attempt,
                                 "last_max_output_tokens": actual_capacity, "max_output_tokens_limit": limit}
            elif isinstance(error, Exception) and getattr(error, "stage", None) is None:
                error.stage = stage

        async def invoke():
            cancelled()
            emit("model_capacity_selected", {
                "node": stage, "stage": stage, "question_id": clean_question["id"],
                "attempt": attempt, "max_output_tokens": budget,
                "max_output_tokens_limit": limit, "policy_initial_tokens": initial,
                "stage_floor_tokens": min(limit, capacity_floors.get(stage, initial)),
                "budget_origin": "learned_stage_floor" if budget > initial and attempt == 0 else (
                    "length_retry" if attempt else "stage_policy"),
                "is_usage": False,
            })
            with request_output_budget(budget):
                return await callback()

        try:
            return await invoke()
        except ModelOutputInvalid as error:
            annotate(error)
            failure = error
        except BaseException as error:
            annotate(error)
            raise
        while True:
            # The exact provider finish reason is required: no capacity retry
            # for schema errors, filtering, transport failures or guessed causes.
            if (failure.code != "model_output_truncated" or failure.details.get("finish_reason") != "length"
                    or attempt >= retries or budget >= limit):
                raise failure
            report_usage()
            cancelled()
            previous = budget
            budget = min(budget * 2, limit)
            capacity_floors[stage] = max(capacity_floors.get(stage, initial), budget)
            attempt += 1
            started = time.monotonic()
            summary = {"attempt": attempt, "max_output_tokens": budget, "previous_max_output_tokens": previous}
            base = {"call_id": str(uuid4()), "node": stage, "stage": stage,
                    "name": f"{stage}_capacity_retry", "kind": "model", "engine": "LangChain",
                    "question_id": clean_question["id"], **summary, "summary": summary,
                    "started_at": datetime.now(timezone.utc).isoformat()}
            emit("model_capacity_retry_started", base)
            emit("tool_started", base)
            try:
                result = await invoke()
            except BaseException as error:
                report_usage()
                annotate(error)
                cancelled_error = isinstance(error, (AnalysisCancelled, asyncio.CancelledError))
                info = {"code": "cancelled" if cancelled_error else (
                    error.code if isinstance(error, ModelOutputInvalid) else "model_capacity_request_failed")}
                if isinstance(error, ModelOutputInvalid):
                    info.update({"message": str(error), "details": copy.deepcopy(error.details)})
                ended = {**base, **info, "completed_at": datetime.now(timezone.utc).isoformat(),
                         "elapsed_ms": round((time.monotonic() - started) * 1000)}
                emit("tool_failed", ended)
                emit("model_capacity_retry_failed", ended)
                if not isinstance(error, ModelOutputInvalid):
                    raise
                failure = error
            else:
                report_usage()
                ended = {**base, "completed_at": datetime.now(timezone.utc).isoformat(),
                         "elapsed_ms": round((time.monotonic() - started) * 1000)}
                emit("tool_completed", ended)
                emit("model_capacity_retry_completed", ended)
                return result

    async def repair_once(stage, error, callback, validate, *, name=None, capacity_stage=None):
        """One explicit correction, always revalidated against unchanged inputs.

        No repair is allowed for transport errors, truncation, filtering, human
        annotations or unavailable repair providers. Failed content stays local
        to this request: events contain only the fixed validation diagnostics.
        """
        cancelled()
        if strict_enabled or bundle_enabled:
            raise error
        if error.code not in {"validation_failed", "model_output_invalid"} or not callable(callback):
            raise error
        started = time.monotonic()
        base = {
            "call_id": str(uuid4()), "node": stage, "stage": stage,
            "name": name or f"{stage}_repair", "kind": "model", "engine": "LangChain",
            "question_id": clean_question["id"], "attempt": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        diagnostic = {"check": error.details.get("check", "schema"), "details": copy.deepcopy(error.details)}
        report_usage()
        emit("model_repair_started", {**base, **diagnostic})
        emit("tool_started", {**base, "summary": "针对本阶段校验错误进行一次定向修正"})
        try:
            response = await request_with_capacity(capacity_stage or ("extract" if stage == "verify" else stage), callback)
            result = validate(response)
        except BaseException as failure:
            report_usage()
            info = {}
            if isinstance(failure, ModelOutputInvalid):
                if failure.stage is None:
                    failure.stage = stage
                failure.details = {**failure.details, "repair_attempt": 1, "repair_exhausted": True}
                info = {"code": failure.code, "message": str(failure), "details": copy.deepcopy(failure.details)}
            else:
                if isinstance(failure, Exception) and getattr(failure, "stage", None) is None:
                    failure.stage = stage
                info = {"code": "cancelled" if isinstance(failure, (AnalysisCancelled, asyncio.CancelledError)) else "model_repair_request_failed"}
            ended = {**base, **info, "completed_at": datetime.now(timezone.utc).isoformat(),
                     "elapsed_ms": round((time.monotonic() - started) * 1000)}
            emit("tool_failed", ended)
            emit("model_repair_failed", {**ended, "check": info.get("details", {}).get("check", "schema")})
            raise
        report_usage()
        ended = {**base, "completed_at": datetime.now(timezone.utc).isoformat(),
                 "elapsed_ms": round((time.monotonic() - started) * 1000)}
        emit("tool_completed", {**ended, "summary": "定向修正响应通过本阶段校验"})
        emit("model_repair_completed", ended)
        return result

    def instrument(node, engine, kind, function):
        async def traced(state):
            cancelled()
            started = time.monotonic()
            effective_engine, effective_kind = engine, kind
            if strict_enabled and node == "strict_annotation":
                from app.workflows.strict_workbench import annotation_inputs
                subset, _selection, _ledger = annotation_inputs(state["strict_case"], state["strict_selection"],
                    getattr(provider,"strict_balance_policy",None))
                if not subset["catalog"]:
                    effective_engine, effective_kind = "Python", "local"
            base = {
                "call_id": str(uuid4()), "node": node, "name": node, "tool": node, "kind": effective_kind,
                "engine": effective_engine, "question_id": clean_question["id"],
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            emit("tool_started", base)
            try:
                result = await function(state)
            except BaseException as error:
                report_usage()
                if getattr(error, "stage", None) is None and hasattr(error, "code"):
                    error.stage = node
                emit("tool_failed", {
                    **base, "completed_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "code": "cancelled" if isinstance(error, (AnalysisCancelled, asyncio.CancelledError)) else (
                        error.code if isinstance(getattr(error, "code", None), str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error.code) else "tool_execution_failed"
                    ),
                    **({"message": str(error), "details": copy.deepcopy(error.details)} if isinstance(error, ModelOutputInvalid) else {}),
                })
                raise
            report_usage()
            counts = {
                "paragraph_count": len(result.get("retrieved", [])),
                "candidate_count": len(result.get("candidates", [])),
                "evidence_count": len(result.get("evidence", [])),
            }
            emit("tool_completed", {
                **base, "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "summary": {key: value for key, value in counts.items() if value},
            })
            return result
        return traced

    async def retrieve_node(state):
        begin("retrieve")
        try:
            plan = plan_source_batches(
                state["question"]["question"], state["paragraphs"],
                max_codepoints=source_batch_codepoints, max_catalog_spans=source_batch_spans,
            )
        except CoverageLimitExceeded:
            raise ModelOutputInvalid("单段原文超过完整目录容量，尚未调用模型；请保留原段标识并分窗口导入。",
                stage="retrieve", code="model_input_too_large", details={"check": "source_catalog_limit"}) from None
        batches = plan["batches"]
        coverage = {
            "strategy": "full_source_scan", "batch_count": len(batches),
            "total_paragraphs": plan["coverage"]["paragraph_count"],
            "total_codepoints": plan["coverage"]["codepoint_count"],
            "completed_batches": 0, "scanned_paragraphs": 0, "scanned_codepoints": 0,
            "candidates_count": 0, "complete": not batches,
            "source_fingerprint": plan["coverage"]["source_fingerprint"],
            "plan_fingerprint": plan["coverage"]["plan_fingerprint"],
        }
        emit("source_coverage_planned", {"question_id": clean_question["id"], **coverage})
        complete("retrieve", {"paragraph_count": coverage["total_paragraphs"], "batch_count": len(batches)})
        return {
            "retrieved": [row for batch in batches for row in batch["paragraphs"]],
            "source_batches": batches, "source_batch_index": 0, "source_coverage": coverage,
            "extract_repair_attempted": False,
        }

    def invalid(message: str, check: str) -> ModelOutputInvalid:
        return ModelOutputInvalid(
            message, stage="verify", details={"question_id": clean_question["id"], "check": check}
        )

    def canonical_candidate(raw, *, human=False, allowed_ids=None):
        annotation = {}
        span_annotation = {}
        payload = raw
        if human:
            payload = {key: value for key, value in raw.items() if key not in {"id", "source_evidence_id", "fact_view_id", "manual_identity"}}
            if "stance" in payload:
                # Stance is a human annotation, not a model extraction field.
                # Keep strict model-schema validation while preserving the
                # separately validated annotation in the frozen evidence.
                annotation = _project_interpretation({"stance": payload.pop("stance")})
        elif isinstance(raw, dict):
            # A literal quote that occurs exactly once has an unambiguous,
            # deterministic coordinate. The program owns character counting;
            # model offsets are merely hints and cannot override the source.
            # Canonicalize before validating the coordinate pair so even an
            # inaccurate pair cannot reject an otherwise exact unique quote.
            source_id, literal = raw.get("paragraph_id"), raw.get("quote")
            source = paragraph_map.get(source_id) if isinstance(source_id, str) else None
            if source is not None and isinstance(literal, str) and literal:
                found = source["text"].find(literal)
                if found >= 0 and source["text"].find(literal, found + 1) < 0:
                    end_found = found + len(literal)
                    has_hint = raw.get("start_cp") is not None or raw.get("end_cp") is not None
                    span_annotation = {
                        "span_location_method": "exact_unique_quote",
                        "model_span_corrected": has_hint and (
                            raw.get("start_cp") != found or raw.get("end_cp") != end_found
                        ),
                    }
                    payload = {**raw, "start_cp": found, "end_cp": end_found}
        try:
            candidate = EvidenceCandidate.model_validate(payload).model_dump()
        except (ValidationError, ValueError, TypeError):
            raise invalid("候选证据未通过结构校验。", "candidate_schema") from None
        paragraph_id = candidate["paragraph_id"]
        if paragraph_id not in paragraph_map:
            raise invalid("候选证据段落不属于本轮来源资料。", "unknown_paragraph")
        if not human and paragraph_id not in allowed_ids:
            raise invalid("候选引用了未提供给模型的段落。", "unseen_paragraph")
        paragraph = paragraph_map[paragraph_id]
        text, quote = paragraph["text"], candidate["quote"]
        start, end = candidate["start_cp"], candidate["end_cp"]
        if start is None:
            start = text.find(quote)
            if start < 0 or text.find(quote, start + 1) >= 0:
                raise invalid("引用不存在或重复出现；重复引用必须提供明确码点坐标。", "quote_occurrence")
            end = start + len(quote)
        try:
            validate_span(text, quote, start, end)
        except (ValueError, TypeError):
            raise invalid("候选引用与原文码点位置不一致。", "quote_span") from None
        if not human and not span_annotation:
            # Repeated quotes are accepted only after the explicit occurrence
            # passes exact validation. Never choose one occurrence silently.
            span_annotation = {
                "span_location_method": "model_explicit_occurrence", "model_span_corrected": False,
            }
        group_id, group_basis = source_groups[paragraph["document_id"]]
        evidence_id = _uuid("evidence:" + _canonical_json([paragraph_id, start, end, quote]))
        from app.domain.source_metadata import source_credibility_value
        grade, grade_basis = source_credibility_value(paragraph)
        row = {
            "id": evidence_id,
            "paragraph_id": paragraph_id,
            "raw_doc_id": paragraph["raw_doc_id"],
            "raw_para_id": paragraph["raw_para_id"],
            "start_cp": start,
            "end_cp": end,
            "quote": quote,
            "fact_des": candidate["fact_des"],
            "argument": candidate["argument"],
            "entities": candidate["entities"],
            "source_group_id": group_id,
            "source_group_basis": group_basis,
            "grade_score": grade, "grade_basis": grade_basis,
            "source_level": "unknown",
            "interpretation_origin": "human" if human else "model",
            **span_annotation,
            **annotation,
        }
        if human and strict_scope_enabled:
            from app.domain.manual_identity import manual_identity
            try:
                row.update(manual_identity(raw, row))
            except ValueError:
                raise invalid("人工事实身份与其固定原文不一致。", "manual_identity") from None
        return row

    def canonical_model_candidates(candidates, allowed_ids):
        output = {}
        if not isinstance(candidates, list):
            raise invalid("候选证据必须是列表。", "candidate_schema")
        for index, raw in enumerate(candidates):
            try:
                item = canonical_candidate(raw, allowed_ids=allowed_ids)
            except ModelOutputInvalid as error:
                field = {"quote_occurrence": "quote", "quote_span": "quote",
                         "unknown_paragraph": "paragraph_id", "unseen_paragraph": "paragraph_id"}.get(error.details.get("check"))
                error.details = {**error.details, "candidate_index": index,
                                 "field_path": f"candidates[{index}]" + (f".{field}" if field else "")}
                raise
            previous = output.get(item["id"])
            if previous is None:
                output[item["id"]] = item
            else:
                # An immutable quote may carry more than one atomic factual
                # interpretation. Keep one source identity while preserving
                # those distinct model annotations for review and audit.
                def interpretation(row):
                    return {field: copy.deepcopy(row[field]) for field in ("fact_des", "argument", "entities")}
                views = [interpretation(previous), *previous.get("additional_interpretations", [])]
                new_view = interpretation(item)
                if _canonical_json(new_view) not in {_canonical_json(view) for view in views}:
                    previous.setdefault("additional_interpretations", []).append(new_view)
        return {"candidates": candidates, "evidence": output}


    def validated_batch(value, allowed_ids, *, error_stage="extract"):
        try:
            candidates = ExtractionOutput.model_validate({"evidence": value}).model_dump()["evidence"]
        except (ValidationError, TypeError, ValueError):
            raise ModelOutputInvalid("候选证据未通过结构校验。", stage=error_stage,
                                     code="model_output_invalid", details={"check": "schema"}) from None
        accepted = canonical_model_candidates(candidates, allowed_ids)
        # A provider-owned list can carry the validated fixed-paragraph
        # dispositions. They are audit data, never an acceptance shortcut.
        source_review = getattr(value, "source_review", None)
        if source_review is not None:
            accepted["source_review"] = copy.deepcopy(source_review)
        return accepted

    async def extract_node(state):
        # Each loop iteration is a separate LangGraph checkpoint. A successful
        # paid batch survives a later pause/failure without replaying its model
        # request; an in-flight batch is never reported as already scanned.
        index, batches = state.get("source_batch_index", 0), state["source_batches"]
        if index == 0:
            begin("extract")
        else:
            cancelled()
        if index >= len(batches):
            complete("extract", {"candidate_count": len(state.get("candidates", [])), "batch_count": len(batches)})
            return {"candidates": state.get("candidates", []),
                    "output_capacity_floors": copy.deepcopy(capacity_floors)}
        batch = batches[index]
        sources = batch["paragraphs"]
        allowed_ids = set(batch["paragraph_ids"])
        started = time.monotonic()
        summary = {"batch_index": index + 1, "batch_count": len(batches),
                   "paragraph_count": len(sources), "codepoint_count": batch["codepoints"]}
        base = {"call_id": str(uuid4()), "node": "extract", "name": "evidence_batch_extraction",
                "tool": "evidence_batch_extraction", "kind": "model", "engine": "LangChain",
                "question_id": clean_question["id"], "started_at": datetime.now(timezone.utc).isoformat(),
                **summary, "summary": summary}
        emit("tool_started", base)
        candidates, repaired = None, False
        try:
            try:
                candidates = await request_with_capacity("extract", lambda: provider.extract(
                    copy.deepcopy(state["question"]), copy.deepcopy(sources)))
                accepted = validated_batch(candidates, allowed_ids)
            except ModelOutputInvalid as error:
                repair = getattr(provider, "repair_extract", None)
                if not callable(repair):
                    raise
                diagnostics = copy.deepcopy(error.details)
                if candidates is not None:
                    diagnostics["source_location_choices"] = _source_location_choices(
                        candidates, sources, candidate_index=error.details.get("candidate_index"))
                invalid_output = ({"evidence": copy.deepcopy(candidates)} if candidates is not None
                                  else copy.deepcopy(getattr(error, "candidate", None)))
                accepted = await repair_once(
                    error.stage or "extract", error,
                    lambda: repair(copy.deepcopy(state["question"]), copy.deepcopy(sources),
                                   invalid_output, diagnostics),
                    lambda value: validated_batch(value, allowed_ids, error_stage=error.stage or "extract"), name="extract_repair",
                )
                repaired = True
        except BaseException as error:
            report_usage()
            emit("tool_failed", {
                **base, "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "code": "cancelled" if isinstance(error, (AnalysisCancelled, asyncio.CancelledError)) else (
                    error.code if isinstance(error, ModelOutputInvalid) else "tool_execution_failed"),
                **({"message": str(error), "details": copy.deepcopy(error.details)} if isinstance(error, ModelOutputInvalid) else {}),
            })
            raise
        report_usage()
        combined = merge_exact_candidates([state.get("candidates", []), accepted["candidates"]])
        coverage = {
            **state["source_coverage"], "completed_batches": index + 1,
            "scanned_paragraphs": state["source_coverage"]["scanned_paragraphs"] + len(sources),
            "scanned_codepoints": state["source_coverage"]["scanned_codepoints"] + batch["codepoints"],
            "candidates_count": len(combined), "complete": index + 1 == len(batches),
        }
        emit("tool_completed", {
            **base, "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "summary": {**summary, "candidate_count": len(accepted["candidates"])},
        })
        emit("source_coverage_progress", {"question_id": clean_question["id"], **coverage})
        if coverage["complete"]:
            complete("extract", {"candidate_count": len(combined), "batch_count": len(batches)})
        return {
            "candidates": combined, "source_batch_index": index + 1, "source_coverage": coverage,
            "extraction_reviews": [*state.get("extraction_reviews", []),
                *([accepted["source_review"]] if "source_review" in accepted else [])],
            "extract_repair_attempted": state.get("extract_repair_attempted", False) or repaired,
            "output_capacity_floors": copy.deepcopy(capacity_floors),
        }

    def after_extract(state):
        return "extract" if state.get("source_batch_index", 0) < len(state["source_batches"]) else "verify"

    async def verify_node(state):
        begin("verify")
        # All automatic candidates were already validated/repaired against the
        # bounded batch actually supplied to the model. No repair is allowed
        # here against an oversized concatenation of the whole topic.
        candidates = state["candidates"]
        evidence = canonical_model_candidates(candidates, {row["id"] for row in state["retrieved"]})["evidence"]
        # Manual sources can come from any paragraph in this immutable pool.
        manual_aliases = {}
        for raw in state["manual_candidates"]:
            item = canonical_candidate(raw, human=True)
            evidence[item["id"]] = item
            if isinstance(raw.get("id"), str):
                previous = manual_aliases.get(raw["id"])
                if previous is not None and previous != item["id"]:
                    raise invalid("人工解释目标对应多个版本，需明确指定事实身份。", "manual_identity")
                manual_aliases[raw["id"]] = item["id"]
        for evidence_id, patch in state["interpretation_overrides"].items():
            target_id = manual_aliases.get(evidence_id, evidence_id)
            if target_id not in evidence:
                raise invalid("人工解释目标未出现在本轮证据中；必须同时提供其已定位引用。", "unknown_interpretation_target")
            if not patch:
                continue
            item = evidence[target_id]
            for field, value in patch.items():
                if field == "argument":
                    item["argument"].update(value)
                else:
                    item[field] = value
            item["interpretation_origin"] = "human"
        # Explicit exclusion takes priority over both model and human inclusion.
        canonical_pool = sorted(evidence.values(), key=lambda item: item["id"])
        verified, operator_exclusion_audit = _filter_operator_exclusions(canonical_pool, excluded)
        automatic = [row for row in verified if row["interpretation_origin"] != "human"]
        if strict_scope_enabled:
            complete("verify", {"verified_count": len(verified), "excluded_count": len(excluded)})
        return {"operator_exclusion_audit": operator_exclusion_audit,
                "evidence": verified,
                # Keep the full canonical source/annotation proof, including
                # explicitly excluded views. The scope entry removes them from
                # execution again; bank replay must still be able to recompute
                # their IDs rather than trust an absent-ID audit summary.
                "candidate_evidence": copy.deepcopy(canonical_pool if strict_scope_enabled else verified),
                "precision_batches": _plan_precision_batches(automatic), "precision_batch_index": 0,
                "precision_selected_ids": [], "precision_assessments": []}

    async def precision_node(state):
        """Review one bounded batch and checkpoint every successful paid result.

        Manual interpretations retain their explicit user intent. The model may
        select only from automatic source-verified evidence and must account for
        every candidate, so an unrelated/duplicate dismissal remains auditable.
        """
        cancelled()
        select = getattr(provider, "select_evidence", None)
        pool = state["candidate_evidence"]
        automatic = [row for row in pool if row["interpretation_origin"] != "human"]
        manual = [row for row in pool if row["interpretation_origin"] == "human"]
        if not callable(select) or not automatic:
            complete("verify", {"verified_count": len(pool), "excluded_count": len(excluded)})
            return {"evidence_selection": {"applied": False, "candidate_count": len(automatic),
                        "selected_count": len(automatic), "rejected_count": 0,
                        "manual_preserved_count": len(manual)}, "precision_batch_index": len(state["precision_batches"]),
                    "output_capacity_floors": copy.deepcopy(capacity_floors)}
        batches, index = state["precision_batches"], state["precision_batch_index"]
        by_id = {row["id"]: row for row in automatic}
        batch = [by_id[identity] for identity in batches[index]]
        summary = {"batch_index": index + 1, "batch_count": len(batches), "candidate_count": len(batch)}
        started = time.monotonic()
        base = {"call_id": str(uuid4()), "node": "verify", "name": "evidence_precision_review",
                "tool": "evidence_precision_review", "kind": "model", "engine": "LangChain",
                "question_id": clean_question["id"], "started_at": datetime.now(timezone.utc).isoformat(),
                **summary, "summary": summary}
        emit("tool_started", base)
        response = None
        def validate_selection(value):
            try:
                # Duplicate references must resolve to a retained representative
                # inside the same bounded review, never an unseen other batch.
                return validate_evidence_selection(value, batch)
            except (ValidationError, TypeError, ValueError):
                raise ModelOutputInvalid("候选证据的相关性审查未通过完整性校验。", stage="verify",
                    code="model_output_invalid", details={"check": "evidence_selection"}) from None
        try:
            try:
                response = await request_with_capacity("verify", lambda: select(
                    copy.deepcopy(state["question"]), copy.deepcopy(batch)))
                selection = validate_selection(response)
            except ModelOutputInvalid as error:
                repair = getattr(provider, "repair_evidence_selection", None)
                if not callable(repair):
                    raise
                selection = await repair_once(
                    "verify", error,
                    lambda: repair(copy.deepcopy(state["question"]), copy.deepcopy(batch),
                        copy.deepcopy(response if response is not None else getattr(error, "candidate", None)),
                        copy.deepcopy(error.details)),
                    validate_selection, name="evidence_precision_repair", capacity_stage="verify",
                )
        except BaseException as error:
            report_usage()
            emit("tool_failed", {
                **base, "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "code": "cancelled" if isinstance(error, (AnalysisCancelled, asyncio.CancelledError)) else (
                    error.code if isinstance(error, ModelOutputInvalid) else "tool_execution_failed"),
                **({"message": str(error), "details": copy.deepcopy(error.details)} if isinstance(error, ModelOutputInvalid) else {}),
            })
            raise
        report_usage()
        selected_ids = [*state["precision_selected_ids"], *selection["selected_evidence_ids"]]
        assessments = [*state["precision_assessments"], *selection["assessments"]]
        selected = set(selected_ids)
        partial = {"precision_batch_index": index + 1, "precision_selected_ids": selected_ids,
                   "precision_assessments": assessments, "output_capacity_floors": copy.deepcopy(capacity_floors)}
        batch_counts = {"candidate_count": len(batch), "selected_count": len(selection["selected_evidence_ids"]),
                        "rejected_count": len(batch) - len(selection["selected_evidence_ids"])}
        emit("tool_completed", {
            **base, "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.monotonic() - started) * 1000), "summary": {**summary, **batch_counts},
        })
        progress = {"question_id": clean_question["id"], "batch_count": len(batches), "completed_batches": index + 1,
                    "candidate_count": len(automatic), "reviewed_count": len(assessments),
                    "selected_count": len(selected), "rejected_count": len(assessments) - len(selected),
                    "manual_preserved_count": len(manual), "complete": index + 1 == len(batches)}
        emit("evidence_precision_progress", progress)
        if index + 1 < len(batches):
            return partial
        try:
            aggregate = validate_evidence_selection({"selected_evidence_ids": selected_ids, "assessments": assessments}, automatic)
        except (ValidationError, TypeError, ValueError):
            raise ModelOutputInvalid("分批证据精筛汇总未通过完整性校验。", stage="verify",
                code="model_output_invalid", details={"check": "evidence_selection"}) from None
        kept = sorted([row for row in automatic if row["id"] in selected] + manual, key=lambda row: row["id"])
        counts = {"candidate_count": len(automatic), "selected_count": len(selected),
                  "rejected_count": len(automatic) - len(selected), "manual_preserved_count": len(manual),
                  "batch_count": len(batches), "completed_batches": index + 1}
        complete("verify", {"verified_count": len(kept), "excluded_count": len(excluded), **counts})
        return {**partial, "evidence": kept, "evidence_selection": {"applied": True, **aggregate, **counts}}

    def after_precision(state):
        return "precision_review" if state["precision_batch_index"] < len(state["precision_batches"]) else "freeze"

    # Scope decisions are independent of the previous precision reasons or
    # final answer. All original first/additional views are supplied, including
    # candidates the historical precision pass may have labelled background.
    from app.domain.strict_verification import (
        build_fact_views, validate_scope_ledger, validate_narrowing_ledger,
    )
    from app.domain.scope_batches import (
        plan_scope_batches, subset_evidence, merge_corrections, verified_fact_rows, deduplicate_verified_facts,
    )
    authorized_narrow_attempts = set()  # Process-local, deliberately not restored.

    async def candidate_import_node(state):
        begin("candidate_import")
        from app.domain.refine_input import validate_candidates, source_manifest, digest, RefineInputInvalid
        artifact = copy.deepcopy(state["candidate_artifact"])
        try:
            if (artifact.get("artifact_sha256") != digest({key: value for key, value in artifact.items() if key != "artifact_sha256"})
                    or artifact.get("question") != clean_question
                    or artifact.get("paragraph_manifest") != source_manifest(paragraphs)
                    or not artifact.get("source_coverage", {}).get("complete")):
                raise RefineInputInvalid("refine_artifact_hash")
            validate_candidates(artifact["candidate_evidence"], paragraphs)
        except RefineInputInvalid as error:
            raise ModelOutputInvalid("候选工件未通过当前原文与问题校验，尚未调用模型。", stage="candidate_import",
                code="source_integrity_failed", details={"check": error.check}) from None
        pool = artifact["candidate_evidence"]
        provenance = {"kind": "candidate_artifact_import", "artifact_sha256": artifact["artifact_sha256"],
            "source_revision_id": artifact["source_revision_id"], "source_run_id": artifact["source_run_id"],
            "new_source_scan_performed": False, "historical_source_coverage": artifact["source_coverage"],
            "historical_source_usage": artifact["source_usage"]}
        emit("candidate_artifact_imported", {"question_id": clean_question["id"],
            **{key: value for key, value in provenance.items() if key != "historical_source_usage"},
            "candidate_count": len(pool)})
        complete("candidate_import", {"candidate_count": len(pool)})
        return {"candidate_evidence": copy.deepcopy(pool), "evidence": copy.deepcopy(pool),
                "source_coverage": copy.deepcopy(artifact["source_coverage"]),
                "candidate_import_provenance": provenance,
                "evidence_selection": {"applied": False, "strategy": "strict_scope_replacement"}}

    async def scope_prepare_node(state):
        begin("scope_review")
        # The shared entry also protects imported NEW quality-refine jobs,
        # which intentionally bypass verify_node and its initial-run filter.
        pool, newly_excluded = _filter_operator_exclusions(state["candidate_evidence"], excluded)
        exclusion_audit, seen_exclusions = [], set()
        for item in [*state.get("operator_exclusion_audit", []), *newly_excluded]:
            identity = _canonical_json(item)
            if identity not in seen_exclusions:
                exclusion_audit.append(item)
                seen_exclusions.add(identity)
        batches = plan_scope_batches(pool, question=clean_question)
        emit("scope_review_planned", {"question_id": clean_question["id"], "batch_count": len(batches),
            "fact_view_count": len(build_fact_views(pool)), "candidate_count": len(pool)})
        return {"operator_exclusion_audit": exclusion_audit,
                "scope_phase": "original" if batches else "done", "scope_batches": batches,
            "scope_batch_index": 0, "scope_assessments": [], "scope_selected_view_ids": [],
            "scope_narrowing_view_ids": [], "scope_narrowed_parent_ids": [], "scope_pending_narrow": None,
            "scope_narrowing_ledger": [], "scope_working_evidence": pool,
            "scope_annotation_versions": [], "scope_reverified_view_ids": []}

    async def scope_request(name, summary, callback, validator, repair_callback=None):
        cancelled()
        started = time.monotonic()
        base = {"call_id": str(uuid4()), "node": "scope_review", "name": name, "tool": name,
                "kind": "model", "engine": "LangChain", "question_id": clean_question["id"], "summary": summary}
        emit("tool_started", base)
        response = None
        try:
            try:
                response = await request_with_capacity("verify", callback)
                result = validator(response)
            except ModelOutputInvalid as error:
                if repair_callback is None:
                    raise
                result = await repair_once("scope_review", error,
                    lambda: repair_callback(response if response is not None else getattr(error, "candidate", None), error.details),
                    validator, name=name + "_repair", capacity_stage="verify")
        except BaseException as error:
            report_usage()
            emit("tool_failed", {**base, "elapsed_ms": round((time.monotonic() - started) * 1000),
                "code": "cancelled" if isinstance(error, (AnalysisCancelled, asyncio.CancelledError)) else getattr(error, "code", "model_request_failed"),
                **({"details": copy.deepcopy(error.details), "message": str(error)} if isinstance(error, ModelOutputInvalid) else {})})
            if isinstance(error, ModelOutputInvalid):
                error.stage = "scope_review"
            raise
        report_usage()
        emit("tool_completed", {**base, "elapsed_ms": round((time.monotonic() - started) * 1000),
            "summary": {**summary, "verified_fact_count": len(result.get("selected_fact_view_ids", [])),
                        "narrowing_count": len(result.get("narrowing_view_ids", []))}})
        return result

    def scope_validate(value, batch, *, narrowing_ids=None):
        try:
            if narrowing_ids is not None:
                return validate_narrowing_ledger(value, clean_question, batch, narrowing_ids)
            return validate_scope_ledger(value, clean_question, batch)
        except (ValueError, TypeError, KeyError):
            raise ModelOutputInvalid("严格事实逐项复核未通过身份与完整性校验。", stage="scope_review",
                code="model_output_invalid", details={"check": "scope_decision_contract"}, candidate=copy.deepcopy(value)) from None

    def advance_narrow(state, *, working=None, ledger=None, versions=None):
        working = working if working is not None else state["scope_working_evidence"]
        ledger = ledger if ledger is not None else state["scope_narrowing_ledger"]
        versions = versions if versions is not None else state["scope_annotation_versions"]
        index = state["scope_batch_index"] + 1
        patch = {"scope_working_evidence": working, "scope_narrowing_ledger": ledger,
                 "scope_annotation_versions": versions, "scope_pending_narrow": None,
                 "scope_batch_index": index, "output_capacity_floors": copy.deepcopy(capacity_floors)}
        if index == len(state["scope_batches"]):
            corrected_ids = sorted({fid for item in ledger for row in item.get("corrections", [])
                                    for fid in row.get("corrected_fact_view_ids", [])})
            batches = plan_scope_batches(working, question=clean_question, view_ids=corrected_ids)
            patch.update({"scope_phase": "reverify" if batches else "done", "scope_batches": batches,
                          "scope_batch_index": 0})
        return patch

    async def scope_loop_node(state):
        cancelled()
        phase, index = state["scope_phase"], state["scope_batch_index"]
        if phase == "done":
            return {}
        ids = state["scope_batches"][index]
        batch = subset_evidence(state["scope_working_evidence"], ids)
        summary = {"batch_index": index + 1, "batch_count": len(state["scope_batches"]), "fact_view_count": len(ids)}
        if phase == "narrow":
            pending = state.get("scope_pending_narrow")
            if pending is None:
                if set(ids) & set(state["scope_narrowed_parent_ids"]):
                    raise ModelOutputInvalid("原解释已使用过一次内容修正，不能静默循环。", stage="scope_review",
                        code="model_output_invalid", details={"check": "scope_annotation_narrowing"})
                attempt_id = str(uuid4())
                authorized_narrow_attempts.add(attempt_id)
                # Reserve in a separate paid-free checkpoint. A process restart
                # cannot unknowingly repeat an interrupted content correction.
                return {"scope_pending_narrow": {"attempt_id": attempt_id, "fact_view_ids": ids},
                        "scope_narrowed_parent_ids": [*state["scope_narrowed_parent_ids"], *ids]}
            if pending["attempt_id"] not in authorized_narrow_attempts:
                emit("scope_narrowing_unresolved", {"question_id": clean_question["id"],
                    "unresolved_count": len(ids), "code": "narrow_attempt_interrupted"})
                interrupted = {"status": "interrupted_unknown", "source_fact_view_ids": ids, "corrections": []}
                return advance_narrow(state, ledger=[*state["scope_narrowing_ledger"], interrupted])
            authorized_narrow_attempts.remove(pending["attempt_id"])
            narrow = getattr(provider, "narrow_evidence_annotations", None)
            if not callable(narrow):
                raise ModelNotConfigured("模型适配器未实现定向解释修正。")
            repair = getattr(provider, "repair_narrow_evidence_annotations", None)
            try:
                result = await scope_request("narrow_evidence_annotations", summary,
                    lambda: narrow(copy.deepcopy(clean_question), copy.deepcopy(batch), copy.deepcopy(ids)),
                    lambda value: scope_validate(value, batch, narrowing_ids=ids),
                    (lambda invalid, details: repair(copy.deepcopy(clean_question), copy.deepcopy(batch), copy.deepcopy(ids),
                        copy.deepcopy(invalid), copy.deepcopy(details))) if callable(repair) else None)
            except ModelOutputInvalid:
                # Keep the original immutable pool and explicitly unresolved
                # opinions. Never replace a failed correction with a fake fact.
                emit("scope_narrowing_unresolved", {"question_id": clean_question["id"],
                    "unresolved_count": len(ids), "code": "narrow_annotation_invalid"})
                return advance_narrow(state, ledger=[*state["scope_narrowing_ledger"],
                    {"status": "invalid", "source_fact_view_ids": ids, "corrections": []}])
            working = merge_corrections(state["scope_working_evidence"], result["corrected_evidence"])
            return advance_narrow(state, working=working,
                ledger=[*state["scope_narrowing_ledger"], result],
                versions=[*state["scope_annotation_versions"], *result["annotation_versions"]])
        review = provider.verify_evidence_scope
        repair = getattr(provider, "repair_evidence_scope", None)
        result = await scope_request("verify_evidence_scope" if phase == "original" else "reverify_evidence_scope", summary,
            lambda: review(copy.deepcopy(clean_question), copy.deepcopy(batch)),
            lambda value: scope_validate(value, batch),
            (lambda invalid, details: repair(copy.deepcopy(clean_question), copy.deepcopy(batch),
                copy.deepcopy(invalid), copy.deepcopy(details))) if callable(repair) else None)
        previous = set(state["scope_selected_view_ids"])
        if phase == "reverify":
            previous.difference_update(ids)  # An unchanged corrected view must obey its latest independent verdict.
        selected = sorted(previous | set(result["selected_fact_view_ids"]))
        assessments = [*state["scope_assessments"], *[{**row, "phase": phase} for row in result["assessments"]]]
        narrowing = sorted(set(state["scope_narrowing_view_ids"]) | (set(result["narrowing_view_ids"]) if phase == "original" else set()))
        patch = {"scope_assessments": assessments, "scope_selected_view_ids": selected,
                 "scope_narrowing_view_ids": narrowing, "scope_batch_index": index + 1,
                 "output_capacity_floors": copy.deepcopy(capacity_floors)}
        if phase == "reverify":
            patch["scope_reverified_view_ids"] = sorted(set(state["scope_reverified_view_ids"]) | set(ids))
        emit("scope_review_progress", {"question_id": clean_question["id"], "phase": phase,
            "batch_count": len(state["scope_batches"]), "completed_batches": index + 1,
            "reviewed_count": len(assessments), "verified_fact_count": len(selected), "narrowing_count": len(narrowing)})
        if index + 1 == len(state["scope_batches"]):
            if phase == "original":
                from app.domain.recall_recovery import plan_recall_recovery
                recovery = plan_recall_recovery(state["scope_working_evidence"], selected, narrowing)
                patch["recall_recovery_plan"] = recovery
                narrowing = recovery["parent_view_ids"]
                patch["scope_narrowing_view_ids"] = narrowing
                emit("recall_recovery_planned", {"question_id": clean_question["id"],
                    "plan_id": recovery["plan_id"], **recovery["audit"]["counts"]})
                batches = plan_scope_batches(state["scope_working_evidence"], question=clean_question,
                    view_ids=narrowing)
            else:
                batches = []
            patch.update({"scope_phase": "narrow" if batches else "done", "scope_batches": batches, "scope_batch_index": 0})
        return patch

    async def scope_finish_node(state):
        cancelled()
        accepted = set(state["scope_selected_view_ids"])
        corrections = [row for item in state["scope_narrowing_ledger"] for row in item.get("corrections", [])]
        children = {row["source_fact_view_id"]: row.get("corrected_fact_view_ids", []) for row in corrections}
        unresolved = {parent for parent in state["scope_narrowing_view_ids"] if not (set(children.get(parent, [])) & accepted)}
        unresolved.update(row["fact_view_id"] for row in state["scope_assessments"] if row["phase"] == "reverify"
            and (row["fact_entailment"] == "needs_narrowing"
                 or (row["fact_entailment"] == "contradicted" and row["target_relation"] in {
                     "direct_answer", "explicit_counterclaim", "necessary_qualifier"})
                 or row.get("target_fact_present_but_annotation_incomplete")))
        rows = verified_fact_rows(state["scope_working_evidence"], accepted, state["scope_assessments"])
        from app.domain.material_eligibility import apply_workflow_material_eligibility, Invalid as MaterialEligibilityInvalid
        try:
            rows, material_eligibility = apply_workflow_material_eligibility(
                clean_question, list(paragraph_map.values()), rows)
        except MaterialEligibilityInvalid as error:
            raise ModelOutputInvalid("材料发布时间资格未通过来源校验，未发布结果。", stage="scope_review",
                code="source_integrity_failed", details={"check": error.check}) from None
        rows, duplicate_map = deduplicate_verified_facts(rows)
        for row in rows:
            versions = [version for version in state["scope_annotation_versions"] if version["fact_view_id"] == row.get("fact_view_id")]
            if versions:
                row["annotation_versions"] = copy.deepcopy(versions)
        manual_pending = sum(row.get("manual_review_required") is True for row in rows)
        audit = {"applied": True, "strategy": "strict_all_candidate_fact_views", "complete": True,
            "candidate_anchor_count": len(state["candidate_evidence"]),
            "original_fact_view_count": len(build_fact_views(state["candidate_evidence"])),
            "verified_fact_view_count": len(accepted), "selected_fact_view_ids": sorted(accepted),
            "assessments": copy.deepcopy(state["scope_assessments"]),
            "narrowed_parent_ids": copy.deepcopy(state["scope_narrowed_parent_ids"]),
            "narrowing_ledger": copy.deepcopy(state["scope_narrowing_ledger"]),
            "annotation_versions": copy.deepcopy(state["scope_annotation_versions"]),
            "recall_recovery_plan": copy.deepcopy(state.get("recall_recovery_plan")),
            "unresolved_view_ids": sorted(unresolved), "unresolved_count": len(unresolved),
            "manual_unverified_count": manual_pending, "formal_export_ready": not manual_pending,
            "operator_excluded_candidate_count": len(state.get("operator_exclusion_audit", [])),
            "operator_excluded_fact_view_ids": sorted({identity for item in state.get("operator_exclusion_audit", []) for identity in item["fact_view_ids"]}),
            "exact_annotation_duplicate_map": duplicate_map,
            "material_eligibility_applied": material_eligibility["applied"],
            "publication_qualification_counts": copy.deepcopy(material_eligibility["counts"]),
            "meaning": "Complete review coverage is not semantic precision/recall or proof of complete extraction."}
        complete("scope_review", {"verified_fact_count": len(accepted), "unresolved_count": len(unresolved),
                                  "manual_preserved_count": sum(row.get("interpretation_origin") == "human" for row in rows)})
        emit("scope_review_finished", {"question_id": clean_question["id"], "verified_fact_count": len(accepted),
            "unresolved_count": len(unresolved), "manual_preserved_count": sum(row.get("interpretation_origin") == "human" for row in rows)})
        emit("material_eligibility_checked", {"question_id": clean_question["id"],
            "applied": material_eligibility["applied"], **material_eligibility["counts"]})
        result = {"evidence": rows, "strict_scope_review": audit, "material_eligibility": material_eligibility,
                  "evidence_selection": {"applied": False, "strategy": "strict_scope_replacement"}}
        if premise_bank_enabled:
            from app.domain.premise_bank import build_quote_faithful_bank, plan_fact_purpose_batches
            bank = build_quote_faithful_bank(clean_question, state["candidate_evidence"], audit, list(paragraph_map.values()))
            batches = plan_fact_purpose_batches(clean_question, bank)
            emit("purpose_review_planned", {"question_id": clean_question["id"], "bank_id": bank["bank_id"],
                "bank_fact_count": len(bank["rows"]), "batch_count": len(batches), "formal_evidence": False})
            result.update(evidence=[], quote_faithful_bank=bank, purpose_batches=batches,
                          purpose_batch_index=0, purpose_ledgers=[])
        return result

    async def purpose_loop_node(state):
        cancelled()
        index, batches = state["purpose_batch_index"], state["purpose_batches"]
        if index == 0:
            begin("purpose_review")
        if index >= len(batches):
            return {}
        from app.domain.premise_bank import validate_fact_purpose_ledger, PremiseBankInvalid
        bank, ids = state["quote_faithful_bank"], batches[index]
        summary = {"batch_index": index + 1, "batch_count": len(batches), "bank_fact_count": len(ids)}
        base = {"call_id": str(uuid4()), "node": "purpose_review", "name": "review_fact_purpose",
            "tool": "review_fact_purpose", "kind": "model", "engine": "LangChain", "question_id": clean_question["id"],
            "summary": summary, "started_at": datetime.now(timezone.utc).isoformat()}
        started = time.monotonic()
        emit("tool_started", base)
        def validate(value):
            try:
                if value.get("fact_ids") != ids:
                    raise PremiseBankInvalid("premise_purpose_batch_identity")
                return validate_fact_purpose_ledger(value, clean_question, bank)
            except (ValueError, TypeError, AttributeError) as error:
                raise ModelOutputInvalid("用途审核未完整覆盖当前固定批次。", stage="purpose_review",
                    code="model_output_invalid", details={"check": getattr(error, "check", "premise_purpose_contract")}) from None
        response = None
        try:
            try:
                response = await request_with_capacity("purpose_review", lambda: provider.review_fact_purpose(
                    copy.deepcopy(clean_question), copy.deepcopy(bank), copy.deepcopy(ids)))
                result = validate(response)
            except ModelOutputInvalid as error:
                repair = getattr(provider, "repair_fact_purpose", None)
                if not callable(repair):
                    raise
                result = await repair_once("purpose_review", error,
                    lambda: repair(copy.deepcopy(clean_question), copy.deepcopy(bank), copy.deepcopy(ids),
                        None, copy.deepcopy(error.details)), validate,
                    name="purpose_review_repair", capacity_stage="purpose_review")
        except BaseException as error:
            report_usage()
            emit("tool_failed", {**base, "elapsed_ms": round((time.monotonic() - started) * 1000),
                "code": "cancelled" if isinstance(error, (AnalysisCancelled, asyncio.CancelledError)) else getattr(error, "code", "purpose_request_failed"),
                **({"details": copy.deepcopy(error.details)} if isinstance(error, ModelOutputInvalid) else {})})
            raise
        report_usage()
        emit("tool_completed", {**base, "elapsed_ms": round((time.monotonic() - started) * 1000),
            "summary": {**summary, "permitted_fact_count": len(result["selected_evidence_ids"])}})
        emit("purpose_review_progress", {"question_id": clean_question["id"], **summary,
            "completed_batches": index + 1, "permitted_fact_count": len(result["selected_evidence_ids"])})
        return {"purpose_batch_index": index + 1, "purpose_ledgers": [*state["purpose_ledgers"], result],
                "output_capacity_floors": copy.deepcopy(capacity_floors)}

    async def purpose_finish_node(state):
        cancelled()
        from app.domain.premise_bank import compile_purpose_projection, PremiseBankInvalid
        try:
            rows, review, material = compile_purpose_projection(clean_question, state["candidate_evidence"],
                state["strict_scope_review"], state["quote_faithful_bank"], state["purpose_ledgers"], list(paragraph_map.values()))
            if review["manual_pending_ids"]:
                raise PremiseBankInvalid("premise_manual_purpose_unverified")
        except PremiseBankInvalid as error:
            raise ModelOutputInvalid("用途许可、原文或人工观点未通过完整重放校验。", stage="purpose_review",
                code="model_output_invalid", details=error.details) from None
        complete("purpose_review", {"bank_fact_count": len(state["quote_faithful_bank"]["rows"]), "formal_fact_count": len(rows)})
        return {"evidence": rows, "purpose_review": review, "material_eligibility": material}

    async def freeze_node(state):
        begin("freeze")
        evidence = state["evidence"]
        if bundle_enabled:
            from app.workflows.bundle_workbench import validate_result
            try:
                validate_result(state, clean_question, list(paragraph_map.values()), source_documents, excluded_spans=excluded)
            except (KeyError, TypeError, ValueError, OSError):
                raise ModelOutputInvalid("原文段落束的来源、送达与注释证明重放失败。", stage="freeze", code="source_integrity_failed",
                    details={"check": "source_bundle_replay"}) from None
        elif strict_enabled:
            from app.workflows.strict_workbench import validate_result
            try:
                validate_result(state, clean_question, list(paragraph_map.values()),
                    candidate_limit=strict_candidate_limit, excluded_spans=excluded)
            except (KeyError, TypeError, ValueError):
                raise ModelOutputInvalid("严格选证的原文与结构证明重放失败。", stage="freeze", code="source_integrity_failed",
                    details={"check": "strict_evidence_replay"}) from None
        elif ranked_enabled:
            from app.domain.paragraph_ranked import validate_ranked_result
            try:
                validate_ranked_result({k: state[k] for k in ("evidence", "verification_protocol", "paragraph_ranked_proof", "material_eligibility")},
                    clean_question, list(paragraph_map.values()), expected_config=ranked_config, excluded_spans=excluded)
            except (KeyError, TypeError, ValueError):
                raise ModelOutputInvalid("段落证据证明未通过冻结重放校验。", stage="freeze", code="source_integrity_failed",
                    details={"check": "paragraph_ranked_freeze_replay"}) from None
        elif strict_scope_enabled:
            from app.domain.material_eligibility import validate_workflow_material_eligibility, Invalid as MaterialEligibilityInvalid
            try:
                validate_workflow_material_eligibility({"evidence": evidence,
                    "material_eligibility": state.get("material_eligibility"),
                    "strict_scope_review": state["strict_scope_review"],
                    **({"quote_faithful_bank": state["quote_faithful_bank"], "purpose_review": state["purpose_review"],
                        "candidate_evidence": state["candidate_evidence"]} if premise_bank_enabled else {})},
                    clean_question, list(paragraph_map.values()))
            except MaterialEligibilityInvalid as error:
                raise ModelOutputInvalid("材料资格重放校验失败，未发布结果。", stage="freeze",
                    code="source_integrity_failed", details={"check": error.check}) from None
            from app.domain.source_lineage import attach_source_group_members
            from app.domain.program_inventory import attach_inventory_basis
            if not state["material_eligibility"]["applied"]:
                # The existing inventory describes the complete corpus. A
                # publication-qualified question must not receive late/unknown
                # members or counts through otherwise opaque source metadata.
                evidence = attach_source_group_members(evidence, state["candidate_evidence"],
                    list(paragraph_map.values()), excluded_spans=excluded)
                evidence = attach_inventory_basis(evidence, state["candidate_evidence"],
                    list(paragraph_map.values()), excluded_spans=excluded)
        digest = hashlib.sha256(_canonical_json(evidence).encode("utf-8")).hexdigest()
        identifier = snapshot_id(evidence)
        if strict_enabled:
            provider.selection_hypotheses = copy.deepcopy(state["strict_evidence_proof"]["selection"]["hypotheses"])
        if strict_enabled and getattr(provider, "reusable_frozen_snapshot", None) is not None:
            emit("frozen_snapshot_reused", {"question_id":clean_question["id"],
                **provider.reusable_frozen_snapshot, "content_sha256_verified":True, "new_model_calls":0})
        complete("freeze", {"evidence_snapshot_id": identifier, "snapshot_sha256": digest})
        return {"evidence_snapshot_id": identifier, "evidence": evidence}

    async def analyst_node(state, role):
        begin(role)
        # Never pass the workflow state or peer result to independent analysts.
        output = None
        try:
            output = await request_with_capacity(role, lambda: provider.analyze(
                role, copy.deepcopy(state["question"]), copy.deepcopy(state["evidence"]), state["evidence_snapshot_id"]
            ))
            validated = _validate_analysis(output, state["evidence"], clean_question["id"], role)
        except ModelOutputInvalid as error:
            repair = getattr(provider, "repair_analysis", None)
            if not callable(repair):
                raise
            validated = await repair_once(
                role, error,
                lambda: repair(role, copy.deepcopy(state["question"]), copy.deepcopy(state["evidence"]),
                               state["evidence_snapshot_id"], copy.deepcopy(output if output is not None else getattr(error, "candidate", None)),
                               copy.deepcopy(error.details)),
                lambda value: _validate_analysis(value, state["evidence"], clean_question["id"], role),
            )
        validated.update({"role": role, "evidence_snapshot_id": state["evidence_snapshot_id"]})
        complete(role, {"evidence_snapshot_id": state["evidence_snapshot_id"]})
        key = {
            "authority_analyst": "authority_analysis",
            "cross_validation_analyst": "cross_analysis",
            "baseline": "baseline_analysis",
        }[role]
        return {key: validated}

    async def analyst_a(state):
        return await analyst_node(state, "authority_analyst")

    async def analyst_b(state):
        return await analyst_node(state, "cross_validation_analyst")

    async def baseline_node(state):
        return await analyst_node(state, "baseline")

    async def finish_node(state):
        cancelled()
        return {}

    def inventory_resolver(state):
        if strict_enabled or ranked_enabled or not strict_scope_enabled or state.get("material_eligibility", {}).get("applied"):
            return None
        from app.domain.program_inventory import inventory_proof
        return lambda cited: inventory_proof([row for row in state["evidence"] if row["id"] in cited],
            state["candidate_evidence"], list(paragraph_map.values()), excluded_spans=excluded)

    async def arbitrate_node(state):
        begin("arbitrate")
        analyses = [state["authority_analysis"], state["cross_analysis"]]
        output = None
        try:
            output = await request_with_capacity("arbitrate", lambda: provider.arbitrate(
                copy.deepcopy(state["question"]), copy.deepcopy(state["evidence"]), copy.deepcopy(analyses)
            ))
            validated = _validate_analysis(output, state["evidence"], clean_question["id"], "arbitrate", arbitration=True, source_inventory_resolver=inventory_resolver(state))
        except ModelOutputInvalid as error:
            repair = getattr(provider, "repair_analysis", None)
            if not callable(repair):
                raise
            validated = await repair_once(
                "arbitrate", error,
                lambda: repair("arbitrate", copy.deepcopy(state["question"]), copy.deepcopy(state["evidence"]),
                               state["evidence_snapshot_id"], copy.deepcopy(output if output is not None else getattr(error, "candidate", None)),
                               copy.deepcopy(error.details), analyses=copy.deepcopy(analyses)),
                lambda value: _validate_analysis(value, state["evidence"], clean_question["id"], "arbitrate", arbitration=True, source_inventory_resolver=inventory_resolver(state)),
            )
        scope = getattr(provider,"arbitration_scope",None) if strict_enabled or bundle_enabled else None
        if (bundle_enabled or (strict_enabled and getattr(provider, "strict_balanced_enabled", False))) and scope is None:
            raise ModelOutputInvalid("仲裁缺少可重放的主张与证据关系。", stage="arbitrate",
                code="validation_failed", details={"check":"principal_claim_relations"})
        if scope is not None:
            from app.agents.scoped_arbitration import validate_scoped_result
            try:
                validate_scoped_result(scope,validated,state["evidence"],
                    **({"question": state["question"]} if scope.get("format") in {"issue-frame-evidence-relations-v2", "conclusion-aligned-evidence-relations-v3"} else {}))
            except ValueError:
                raise ModelOutputInvalid("发布前的主张与证据角色重放失败。",stage="arbitrate",
                    code="validation_failed",details={"check":"principal_claim_relations"}) from None
        complete("arbitrate")
        return {"final": validated, **({"evidence_relation_scope":copy.deepcopy(scope)} if scope is not None else {})}

    # Human coordinates are deliberate selections, never repairable hints.
    # Reject an invalid selection before any paid automatic corpus scan.
    for raw in manual_candidates:
        canonical_candidate(raw, human=True)

    async def strict_selection_node(state):
        from app.agents.strict_evidence import validate_selection
        from app.workflows.strict_workbench import prepare_case
        begin("strict_selection")
        raw_case = prepare_case(clean_question, list(paragraph_map.values()),
            candidate_limit=strict_candidate_limit, excluded_spans=excluded,
            recall_strategy=getattr(provider,"strict_recall_strategy",None))
        response = await request_with_capacity("strict_selection", lambda: provider.strict_selection(copy.deepcopy(raw_case)))
        try:
            selection = validate_selection(raw_case, response).model_dump()
        except (TypeError, ValueError):
            raise ModelOutputInvalid("严格选证输出未覆盖全部来源或含无效字段。", stage="strict_selection",
                code="model_output_invalid", details={"check": "strict_selection_contract"}) from None
        complete("strict_selection", {"candidate_count": len(raw_case["catalog"]),
            "selected_count": sum(row["relevance"] >= 2 for row in selection["decisions"])})
        return {"strict_case": raw_case, "strict_selection": selection}

    async def source_bundle_selection_node(state):
        from app.workflows.bundle_workbench import compile_result
        begin("source_bundle_selection")
        response = await request_with_capacity("source_bundle_selection", lambda: provider.select_source_bundles(
            copy.deepcopy(clean_question), copy.deepcopy(source_documents), copy.deepcopy(list(paragraph_map.values())),
            excluded_spans=sorted(map(list, excluded))))
        try:
            projection = compile_result(clean_question, list(paragraph_map.values()), source_documents, response, excluded_spans=excluded)
            if projection.get("source_bundle_protocol") != bundle_protocol:
                raise ValueError("source_context_protocol_changed_after_selection")
        except (KeyError, TypeError, ValueError, OSError):
            raise ModelOutputInvalid("原文段落束选择缺少可重放的完整读取、原文或注释证明。", stage="source_bundle_selection",
                code="model_output_invalid", details={"check": "source_bundle_selection_contract"}) from None
        complete("source_bundle_selection", {"evidence_count": len(projection["evidence"])})
        return {**projection, "source_bundle_response": copy.deepcopy(response)}

    async def strict_annotation_node(state):
        from app.workflows.strict_workbench import compile_result, annotation_inputs
        begin("strict_annotation")
        policy = getattr(provider,"strict_balance_policy",None)
        raw, selected, allocation = annotation_inputs(state["strict_case"],state["strict_selection"],policy)
        if allocation is not None:
            emit("evidence_balanced", {"question_id":clean_question["id"],
                "candidate_count":allocation["audit"]["candidate_count"],
                "qualified_count":allocation["audit"]["qualified_count"],
                "retained_count":allocation["audit"]["retained_count"],
                "document_count":allocation["audit"]["source_counts"]["retained_document_ids"],
                "target":allocation["policy"]["target"],"semantic_quality_verified":False})
        if raw["catalog"]:
            response = await request_with_capacity("strict_annotation", lambda: provider.strict_annotation(
                copy.deepcopy(raw), copy.deepcopy(selected)))
        else:
            response = {"annotations":[]}
            emit("annotation_skipped_no_selection", {"question_id":clean_question["id"],"new_model_calls":0})
        try:
            projection = compile_result(clean_question, list(paragraph_map.values()), state["strict_case"],
                state["strict_selection"], response, candidate_limit=strict_candidate_limit, excluded_spans=excluded,
                balance_policy=policy,recall_strategy=getattr(provider,"strict_recall_strategy",None))
        except (KeyError, TypeError, ValueError):
            raise ModelOutputInvalid("选中原段的注释、字段或原文定位未通过校验。", stage="strict_annotation",
                code="model_output_invalid", details={"check": "strict_annotation_contract"}) from None
        complete("strict_annotation", {"evidence_count": len(projection["evidence"])})
        return {**projection, "strict_artifact": copy.deepcopy(response)}

    graph = StateGraph(WorkflowState)
    for name, node in (
        ("retrieve", retrieve_node), ("extract", extract_node), ("verify", verify_node), ("freeze", freeze_node),
        ("analyst_a", analyst_a), ("analyst_b", analyst_b), ("arbitrate", arbitrate_node),
    ):
        label = {"analyst_a": "authority_analyst", "analyst_b": "cross_validation_analyst"}.get(name, name)
        external = name in {"extract", "analyst_a", "analyst_b", "arbitrate"}
        engine = "LangChain" if external else ("BM25/jieba" if name == "retrieve" else "Python")
        if strict_enabled and label in getattr(provider, "paid_response_operations", set()):
            external, engine = False, "LangChain Cache"
        graph.add_node(name, node if name == "extract" else instrument(label, engine, "model" if external else "local", node))
    graph.add_node("candidate_import", instrument("candidate_import", "Python", "local", candidate_import_node))
    if bundle_enabled:
        graph.add_node("source_bundle_selection", instrument("source_bundle_selection", "LangChain" if bundle_framework == "langchain" else "DSH", "model", source_bundle_selection_node))
        graph.add_edge(START, "source_bundle_selection")
        graph.add_edge("source_bundle_selection", "freeze")
    elif strict_enabled:
        reuse = getattr(provider, "paid_response_operations", set())
        for stage, node in (("strict_selection", strict_selection_node), ("strict_annotation", strict_annotation_node)):
            graph.add_node(stage, instrument(stage, "LangChain Cache" if stage in reuse else "LangChain strict",
                "local" if stage in reuse else "model", node))
        graph.add_edge(START, "strict_selection")
        graph.add_edge("strict_selection", "strict_annotation")
        graph.add_edge("strict_annotation", "freeze")
    elif ranked_enabled:
        from app.workflows.paragraph_ranked_nodes import build_nodes
        nodes = build_nodes(question=clean_question, paragraphs=list(paragraph_map.values()), provider=provider,
            config=ranked_config, excluded=excluded, begin=begin, complete=complete, emit=emit,
            request_with_capacity=request_with_capacity, repair_once=repair_once, instrument=instrument)
        for name, node in nodes.items():
            graph.add_node(name, node)
        graph.add_edge(START, "paragraph_queries")
        graph.add_edge("paragraph_queries", "paragraph_recall")
        graph.add_edge("paragraph_recall", "paragraph_rank")
        graph.add_conditional_edges("paragraph_rank", lambda state: "paragraph_rank"
            if state["ranked_rank_index"] < len(state["ranked_rank_batches"]) else "paragraph_annotate",
            {"paragraph_rank": "paragraph_rank", "paragraph_annotate": "paragraph_annotate"})
        graph.add_edge("paragraph_annotate", "paragraph_fields")
        graph.add_edge("paragraph_fields", "freeze")
    else:
        graph.add_edge(START, "candidate_import" if candidate_artifact is not None else "retrieve")
        graph.add_edge("retrieve", "extract")
        graph.add_conditional_edges("extract", after_extract, {"extract": "extract", "verify": "verify"})
    graph.add_node("precision_review", precision_node)
    graph.add_edge("verify", "scope_prepare" if strict_scope_enabled else "precision_review")
    if strict_scope_enabled:
        graph.add_node("scope_prepare", scope_prepare_node)
        graph.add_node("scope_loop", scope_loop_node)
        graph.add_node("scope_finish", scope_finish_node)
        graph.add_edge("candidate_import", "scope_prepare")
        graph.add_edge("scope_prepare", "scope_loop")
        graph.add_conditional_edges("scope_loop", lambda state: "scope_finish" if state["scope_phase"] == "done" else "scope_loop",
                                    {"scope_finish": "scope_finish", "scope_loop": "scope_loop"})
        if premise_bank_enabled:
            graph.add_node("purpose_loop", purpose_loop_node)
            graph.add_node("purpose_finish", purpose_finish_node)
            graph.add_edge("scope_finish", "purpose_loop")
            graph.add_conditional_edges("purpose_loop", lambda state: "purpose_loop"
                if state["purpose_batch_index"] < len(state["purpose_batches"]) else "purpose_finish",
                {"purpose_loop": "purpose_loop", "purpose_finish": "purpose_finish"})
            graph.add_edge("purpose_finish", "freeze")
        else:
            graph.add_edge("scope_finish", "freeze")
    else:
        graph.add_edge("candidate_import", "freeze")
    graph.add_conditional_edges("precision_review", after_precision, {"precision_review": "precision_review", "freeze": "freeze"})
    graph.add_edge("freeze", "analyst_a")
    graph.add_edge("freeze", "analyst_b")
    graph.add_edge(["analyst_a", "analyst_b"], "arbitrate")
    if include_baseline:
        graph.add_node("baseline", instrument("baseline", "LangChain", "model", baseline_node))
        graph.add_node("finish", finish_node)
        graph.add_edge("freeze", "baseline")
        graph.add_edge(["arbitrate", "baseline"], "finish")
        graph.add_edge("finish", END)
    else:
        graph.add_edge("arbitrate", END)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with capture_usage() as usage:
        try:
            cancelled()
            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
                app = graph.compile(checkpointer=saver)
                initial = {
                    "question": clean_question, "paragraphs": list(paragraph_map.values()),
                    "manual_candidates": manual_candidates, "interpretation_overrides": overrides,
                    "candidate_artifact": copy.deepcopy(candidate_artifact),
                    "paragraph_ranked_enabled": ranked_enabled,
                    "strict_evidence_enabled": strict_enabled,
                    **({"source_bundle_enabled": True, "source_bundle_protocol": bundle_protocol,
                        "source_documents": copy.deepcopy(source_documents), "source_bundle_excluded_spans": sorted(map(list, excluded))} if bundle_enabled else {}),
                    **({"strict_candidate_limit": strict_candidate_limit, "strict_excluded_spans": sorted(map(list, excluded))} if strict_enabled else {}),
                    **({"paragraph_ranked_config": ranked_config, "paragraph_ranked_excluded_spans": sorted(map(list, excluded))} if ranked_enabled else {}),
                }
                fingerprint = hashlib.sha256(_canonical_json({
                    **initial, "excluded_spans": sorted(excluded), "include_baseline": include_baseline,
                    "retrieval_limit": retrieval_limit, "source_batch_codepoints": source_batch_codepoints,
                    "source_batch_spans": source_batch_spans, "strict_scope_enabled": strict_scope_enabled,
                    "premise_bank_enabled": premise_bank_enabled,
                    "paragraph_ranked_enabled": ranked_enabled,
                    "strict_evidence_enabled": strict_enabled,
                    **({"strict_candidate_limit": strict_candidate_limit,
                        **({"strict_balance_policy": getattr(provider, "strict_balance_policy", None),
                            "strict_recall_strategy": getattr(provider, "strict_recall_strategy", None),
                            "strict_balanced_enabled": True} if getattr(provider, "strict_balanced_enabled", False) else {})} if strict_enabled else {}),
                    **({"paragraph_ranked_config": ranked_config} if ranked_enabled else {}),
                }).encode("utf-8")).hexdigest()
                initial["workflow_input_sha256"] = fingerprint
                provenance = None
                if resume_from_thread_id is not None:
                    provenance = await fork_checkpoint(
                        saver, source_thread_id=resume_from_thread_id, target_thread_id=thread_id,
                        initial_state=initial, input_sha256=fingerprint, snapshot_id=snapshot_id,
                    )
                    emit("checkpoint_reused", {"question_id": clean_question["id"], **provenance})
                    resumed_state = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
                    resumed_channels = resumed_state.checkpoint.get("channel_values", {})
                    saved_floors = resumed_channels.get("output_capacity_floors", {})
                    if (not isinstance(saved_floors, dict)
                            or any(stage not in {"extract", "verify", "purpose_review", "paragraph_queries", "paragraph_rank", "paragraph_annotate", "paragraph_fields", "strict_selection", "strict_annotation", "source_bundle_selection", "authority_analyst", "cross_validation_analyst", "arbitrate", "baseline"}
                                   or type(budget) is not int or not 1 <= budget <= 393216
                                   for stage, budget in saved_floors.items())):
                        raise ModelOutputInvalid("续接检查点的阶段容量记录无效，未调用模型。", stage="resume",
                            code="resume_checkpoint_invalid", details={"check": "output_capacity_floors"})
                    capacity_floors.update(saved_floors)
                    resumed_coverage = resumed_channels.get("source_coverage")
                    if isinstance(resumed_coverage, dict) and candidate_artifact is None:
                        emit("source_coverage_progress", {"question_id": clean_question["id"],
                                                           **copy.deepcopy(resumed_coverage), "reused": True})
                result = await app.ainvoke(
                    None if provenance else initial,
                    config={"configurable": {"thread_id": thread_id}, "recursion_limit": max(1000, len(paragraph_map) * 16 + 100)},
                    durability="sync",
                )
            cancelled()
            output = {
                "evidence": result["evidence"],
                "candidate_evidence": copy.deepcopy(result["candidate_evidence"]),
                "operator_exclusion_audit": copy.deepcopy(result.get("operator_exclusion_audit", [])),
                "analyses": [result["authority_analysis"], result["cross_analysis"]],
                "final": result["final"],
                "usage": copy.deepcopy(usage),
                "evidence_snapshot_id": result["evidence_snapshot_id"],
                "source_coverage": copy.deepcopy(result["source_coverage"]),
                "extraction_reviews": copy.deepcopy(result.get("extraction_reviews", [])),
                "evidence_selection": copy.deepcopy(result["evidence_selection"]),
            }
            if strict_scope_enabled:
                output["strict_scope_review"] = copy.deepcopy(result["strict_scope_review"])
                output["material_eligibility"] = copy.deepcopy(result["material_eligibility"])
            if premise_bank_enabled:
                from app.domain.premise_bank import VERSION as premise_version
                output.update(quote_faithful_bank=copy.deepcopy(result["quote_faithful_bank"]),
                    purpose_review=copy.deepcopy(result["purpose_review"]), verification_protocol=premise_version)
            if ranked_enabled:
                output.update(verification_protocol=result["verification_protocol"],
                    paragraph_ranked_proof=copy.deepcopy(result["paragraph_ranked_proof"]),
                    material_eligibility=copy.deepcopy(result["material_eligibility"]))
            if strict_enabled:
                output.update(verification_protocol=result["verification_protocol"],
                    strict_evidence_proof=copy.deepcopy(result["strict_evidence_proof"]),
                    material_eligibility=copy.deepcopy(result["material_eligibility"]))
                if result.get("evidence_relation_scope") is not None:
                    output["evidence_relation_scope"] = copy.deepcopy(result["evidence_relation_scope"])
            if bundle_enabled:
                output.update(source_bundle_protocol=bundle_protocol, verification_protocol=bundle_protocol,
                    source_bundle_proof=copy.deepcopy(result["source_bundle_proof"]), material_eligibility=copy.deepcopy(result["material_eligibility"]),
                    evidence_relation_scope=copy.deepcopy(result["evidence_relation_scope"]))
            if candidate_artifact is not None:
                output["candidate_import_provenance"] = copy.deepcopy(result["candidate_import_provenance"])
            if include_baseline:
                output["baseline"] = result["baseline_analysis"]
            if provenance:
                output["execution_provenance"] = provenance
                for analysis in [*output["analyses"], output["final"], *([output["baseline"]] if include_baseline else [])]:
                    role = analysis.get("role", "arbitrate")
                    analysis["execution_provenance"] = {
                        **copy.deepcopy(provenance), "reused": role in provenance["reused_nodes"],
                    }
            return output
        finally:
            # Completed requests can be billable even when validation or a
            # cancellation prevents publication. Forward counters on every exit,
            # isolated from the workflow's own mutable accounting list.
            report_usage(allow_empty=True)
