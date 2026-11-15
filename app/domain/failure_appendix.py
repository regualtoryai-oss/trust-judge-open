"""Nonformal failure outputs; retrieval candidates are never factual support."""
from copy import deepcopy

from app.domain.paragraph_ranked import source_material_policy
from app.domain.retrieval import retrieve


def build_failure_appendix(run, questions, paragraphs, *, source_manifest_sha256, excluded_spans=()):
    if run.get("status") != "failed":
        raise ValueError("failure_appendix_requires_failed_run")
    by_id = {p["id"]: p for p in paragraphs}
    excluded = set()
    for pid, start, end, quote in excluded_spans:
        paragraph = by_id.get(pid)
        if (paragraph is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(paragraph["text"])
                or paragraph["text"][start:end] != quote):
            raise ValueError("failure_appendix_exclusion_source_changed")
        excluded.add(pid)
    requested = set(run["question_ids"])
    selected = [q for q in questions if q["id"] in requested]
    if len(selected) != len(requested):
        raise ValueError("failure_appendix_question_source_changed")
    outputs = []
    for question in selected:
        policy = source_material_policy(question, paragraphs)
        allowed = set(policy["eligible_paragraph_ids"]) - excluded
        candidates = retrieve(question["question"], [p for p in paragraphs if p["id"] in allowed], limit=5)
        outputs.append({"question_id": question["raw_question_id"], "question": question["question"],
            "conclusion": "公开信息层面不可判定", "model_generated_conclusion": False,
            "conclusion_explanation": "生成任务失败后的固定兜底文字；未完成模型研判，不能据此认定公开信息确实不可判定。",
            "supporting_facts": [], "conflicting_facts": [],
            "retrieval_candidates": [{"rank": i, "doc_id": p["raw_doc_id"], "para_id": p["raw_para_id"],
                "quote": p["text"], "start_cp": 0, "end_cp": len(p["text"]),
                "candidate_only": True, "verification_status": "unverified_not_support"}
                for i, p in enumerate(candidates, 1)],
            "retrieval_method": "BM25 body text; no semantic verification",
            "eligible_source_paragraph_count": len(allowed), "matching_candidate_count": len(candidates)})
    error = run.get("error") or {}
    return {"format": "nonformal-failure-appendix-v1", "formal_submission": False,
        "run_id": run["run_id"], "topic_id": run["topic_id"], "original_run_status": "failed",
        "source_manifest_sha256": source_manifest_sha256,
        "error_code": error.get("code") if isinstance(error.get("code"), str) else "unknown",
        "original_usage": deepcopy(run.get("usage")), "extra_model_calls": 0, "extra_model_tokens": 0,
        "notice": "失败附录，不属于比赛提交包；候选原文未核验，不能当作支持或冲突证据。",
        "questions": outputs}
