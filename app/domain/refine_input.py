"""Immutable candidate-artifact inputs for NEW quality review jobs.

This module never reads source answers, checkpoints, model clients or filesystem
paths. The repository supplies a fixed ready snapshot in one transaction. The
worker repeats these checks before any new model call. Reuse is not a scan and
is not a checkpoint continuation across versions.
"""
from copy import deepcopy
import hashlib
import json

from app.domain.coverage import plan_source_batches
from app.domain.spans import validate_span
from app.domain.strict_verification import build_fact_views, project_question


class RefineInputInvalid(ValueError):
    def __init__(self, check):
        super().__init__(check)
        self.check = check


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def source_manifest(paragraphs):
    fields = ("id", "document_id", "raw_doc_id", "raw_para_id", "content_sha256", "content_url",
              "raw_publish_time", "source_credibility_json", "document_start_cp", "document_end_cp")
    rows = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict) or not isinstance(paragraph.get("text"), str):
            raise RefineInputInvalid("refine_source_manifest")
        rows.append({**{key: deepcopy(paragraph.get(key)) for key in fields},
                     "text_sha256": hashlib.sha256(paragraph["text"].encode("utf-8")).hexdigest(),
                     "codepoints": len(paragraph["text"])})
    if len({row["id"] for row in rows}) != len(rows):
        raise RefineInputInvalid("refine_source_manifest")
    return sorted(rows, key=lambda row: row["id"])


def _coverage(question, paragraphs, coverage, source_config):
    if not isinstance(coverage, dict) or not isinstance(source_config, dict):
        raise RefineInputInvalid("refine_incomplete_coverage")
    try:
        plan = plan_source_batches(question["question"], paragraphs,
            max_codepoints=source_config["source_batch_codepoints"],
            max_catalog_spans=source_config["source_batch_spans"])
    except (KeyError, ValueError, TypeError):
        raise RefineInputInvalid("refine_source_manifest") from None
    expected = {"strategy": "full_source_scan", "complete": True,
                "total_paragraphs": len(paragraphs), "scanned_paragraphs": len(paragraphs),
                "total_codepoints": sum(len(row["text"]) for row in paragraphs),
                "scanned_codepoints": sum(len(row["text"]) for row in paragraphs),
                "batch_count": len(plan["batches"]), "completed_batches": len(plan["batches"]),
                "source_fingerprint": plan["coverage"]["source_fingerprint"],
                "plan_fingerprint": plan["coverage"]["plan_fingerprint"]}
    for key, value in expected.items():
        if coverage.get(key) != value or (type(value) is int and type(coverage.get(key)) is not int):
            raise RefineInputInvalid("refine_incomplete_coverage")


def validate_candidates(candidates, paragraphs):
    if not isinstance(candidates, list):
        raise RefineInputInvalid("refine_missing_candidate_ledger")
    by_id = {row["id"]: row for row in paragraphs}
    try:
        build_fact_views(candidates)  # Complete first/additional annotation schema.
        for row in candidates:
            paragraph = by_id[row["paragraph_id"]]
            if (row["raw_doc_id"] != paragraph["raw_doc_id"]
                    or row["raw_para_id"] != paragraph["raw_para_id"]
                    or row.get("interpretation_origin") not in {"human", "model"}):
                raise ValueError("identity")
            validate_span(paragraph["text"], row["quote"], row["start_cp"], row["end_cp"])
    except (KeyError, ValueError, TypeError):
        raise RefineInputInvalid("refine_candidate_source") from None


def safe_usage(usage):
    """Source costs are history, never merged into this new job's counters."""
    allowed = {"extract", "select_evidence", "verify_evidence_scope", "narrow_evidence_annotations",
               "authority_analyst", "cross_validation_analyst", "arbitrate", "baseline", "review_fact_purpose", "unknown"}
    if not isinstance(usage, list):
        raise RefineInputInvalid("refine_source_usage")
    return [{"operation": row.get("operation") if isinstance(row.get("operation"), str) and row.get("operation") in allowed else "unknown",
             **{key: row.get(key) if type(row.get(key)) is int and row[key] >= 0 else None
                for key in ("input_tokens", "output_tokens", "total_tokens")},
             "usage_source": "provider" if row.get("usage_source") == "provider" else "unavailable",
             "cost": None} for row in usage if isinstance(row, dict)]


def create_candidate_artifact(*, topic_id, source_revision_id, source_artifact_sha256,
                              source_manifest_sha256, question, paragraphs, result,
                              source_config, overlays, created_at):
    if "candidate_evidence" not in result:
        raise RefineInputInvalid("refine_missing_candidate_ledger")
    candidates = result["candidate_evidence"]
    question = project_question(question)
    validate_candidates(candidates, paragraphs)
    _coverage(question, paragraphs, result.get("source_coverage"), source_config)
    if not result.get("origin_run_id"):
        raise RefineInputInvalid("refine_source_origin")
    config_keys = ("prompt_version", "prompt_sha256", "code_version", "model_provider", "model_name",
                   "source_strategy", "source_batch_codepoints", "source_batch_spans")
    material = {"format": "candidate-artifact-v1", "topic_id": topic_id,
        "source_revision_id": source_revision_id, "source_artifact_sha256": source_artifact_sha256,
        "source_run_id": result["origin_run_id"], "source_config": {key: source_config.get(key) for key in config_keys},
        "source_manifest_sha256": source_manifest_sha256, "paragraph_manifest": source_manifest(paragraphs),
        "question": question, "question_sha256": digest(question), "overlays_sha256": digest(overlays),
        "candidate_evidence": deepcopy(candidates), "candidate_sha256": digest(candidates),
        "source_coverage": deepcopy(result["source_coverage"]), "source_usage": safe_usage(result.get("usage", [])),
        "created_at": created_at}
    return {**material, "artifact_sha256": digest(material)}


def validate_candidate_artifact(artifact, *, topic_id, question, paragraphs,
                                source_manifest_sha256, overlays):
    try:
        material = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        if artifact.get("format") != "candidate-artifact-v1" or digest(material) != artifact.get("artifact_sha256"):
            raise RefineInputInvalid("refine_artifact_hash")
        if (artifact.get("topic_id") != topic_id or artifact.get("question") != project_question(question)
                or artifact.get("question_sha256") != digest(project_question(question))):
            raise RefineInputInvalid("refine_question_binding")
        if (artifact.get("source_manifest_sha256") != source_manifest_sha256
                or artifact.get("paragraph_manifest") != source_manifest(paragraphs)):
            raise RefineInputInvalid("refine_source_manifest")
        if artifact.get("overlays_sha256") != digest(overlays):
            raise RefineInputInvalid("refine_overlay_binding")
        if digest(artifact.get("candidate_evidence")) != artifact.get("candidate_sha256"):
            raise RefineInputInvalid("refine_artifact_hash")
        validate_candidates(artifact["candidate_evidence"], paragraphs)
        _coverage(artifact["question"], paragraphs, artifact["source_coverage"], artifact["source_config"])
    except RefineInputInvalid:
        raise
    except (KeyError, ValueError, TypeError, AttributeError):
        raise RefineInputInvalid("refine_artifact_hash") from None
    return deepcopy(artifact)
