import json
from pathlib import Path

import pytest

from scripts.audit_historical_final_baseline import audit, descriptor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return descriptor(path)


@pytest.fixture
def archive(tmp_path):
    root = tmp_path
    inp = {"question": {"question_id": "q_2", "question": "某程序是否已经确认？"},
           "paragraphs": [{"text": "尚未确认。"}]}
    input_d = write(root / "frozen-inputs/inputs/test.json", inp)
    plan = {"cases": [{"case_id": "test", "input_file": "inputs/test.json", "input_sha256": input_d["sha256"],
                       "title": "虚构题", "source_input_sha256": "a" * 64, "source_paragraphs": 1}]}
    plan_d = write(root / "frozen-inputs/plan.json", plan)
    row = {"id": "f1", "raw_doc_id": "d1", "raw_para_id": "p1", "quote": "尚未确认。", "fact_des": "尚未确认。",
           "argument": {"时间": None}, "entities": []}
    result = {"evidence": [row], "final": {"conclusion": "无法确认。", "supporting_evidence_ids": ["f1"], "conflicting_evidence_ids": []}}
    result_d = write(root / "generation/test/result.json", result)
    standard = [{"doc_id": "d1", "para_id": "p1", "quote": row["quote"], "fact_des": row["fact_des"], "argument": {"时间": ""}, "entities": []}]
    standard_d = write(root / "generation/test/standard_evidence.json", standard)
    usage = [{"operation": "sample", "usage_source": "provider", "model": "test-model", "input_tokens": 2, "output_tokens": 3, "total_tokens": 5}]
    summary = {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5, "unknown_usage_records": 0, "records": 1}
    terminal = {"status": "succeeded", "input_sha256": input_d["sha256"], "artifacts": {"result": result_d, "standard_evidence": standard_d},
                "usage": usage, "usage_summary": summary, "elapsed_seconds": 1}
    terminal_d = write(root / "generation/test/terminal.json", terminal)
    receipt = {"plan_sha256": plan_d["sha256"], "all_cases_succeeded": True, "model_name": "test-model", "provider": "test-provider", "prompt_version": "test",
               "cases": [{"case_id": "test", "status": "succeeded", "terminal": terminal_d, "usage_summary": summary}]}
    write(root / "generation/final-receipt.json", receipt)
    return root


def test_valid_chain_preserves_actual_returned_operations(archive):
    result = audit(archive, ["test"])
    assert result["cases"][0]["actual_usage_totals"]["total_tokens"] == 5
    assert result["references_read"] is False and result["new_model_calls"] == 0


def test_modified_input_rejected(archive):
    with (archive / "frozen-inputs/inputs/test.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="sha_changed"):
        audit(archive, ["test"])


def test_modified_final_artifact_rejected(archive):
    with (archive / "generation/test/result.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="artifact_changed"):
        audit(archive, ["test"])


def test_unsucceeded_receipt_is_not_final_quality_baseline(archive):
    path = archive / "generation/final-receipt.json"
    receipt = json.loads(path.read_text("utf-8"))
    receipt["all_cases_succeeded"] = False
    write(path, receipt)
    with pytest.raises(ValueError, match="complete_generation"):
        audit(archive, ["test"])
