from copy import deepcopy
from hashlib import sha256
import json

import pytest

from scripts.prepare_wider_candidate_inputs import digest, prepare, select_case


def synthetic_case():
    catalog, metadata, ranks = [], {}, []
    for number in range(1, 4):
        ref, doc, para = f"P{number}", f"d{number}", f"p{number}"
        # Identical bodies must remain independently identifiable sources.
        text = "完整原段😀\r\n尾部限定：尚未确认。"
        text_sha = sha256(text.encode()).hexdigest()
        row = {"paragraph_ref": ref, "doc_id": doc, "para_id": para, "title": "原题名",
               "text": text, "paragraph_complete": True, "parent_complete": True}
        catalog.append(row)
        metadata[ref] = {"doc_id": doc, "para_id": para, "source_text_sha256_utf8": text_sha,
                         "source_credibility": 0.7, "raw_publish_time": "3天前"}
        ranks.append({"paragraph_ref": ref, "doc_id": doc, "para_id": para, "title": "原题名",
                      "source_text_sha256_utf8": text_sha, "local_logit": 4.0-number})
    common = {"topic_id": "topic", "question_id": "question"}
    return ({**common, "question": "原始对象与时点条件：甲还是乙？", "topic_title": "原专题",
             "catalog": catalog, "source_metadata": metadata},
            {**common, "question": "原始对象与时点条件：甲还是乙？", "title": "原专题", "candidates": ranks})


def test_full_parents_original_question_titles_metadata_and_mirrors_preserved():
    source, ranking = synthetic_case()
    frozen = deepcopy(source)
    out, provenance = select_case(source, ranking, limit=2)
    assert source == frozen
    assert out["catalog"] == source["catalog"][:2]
    assert out["source_metadata"] == {key: source["source_metadata"][key] for key in ("P1", "P2")}
    assert out["question"] == source["question"]
    assert len(out["catalog"]) == 2
    assert out["catalog"][0]["text"] == out["catalog"][1]["text"]
    assert out["catalog"][0]["doc_id"] != out["catalog"][1]["doc_id"]
    assert provenance[0]["full_parent_codepoints"] == len(source["catalog"][0]["text"])
    assert out["publishable"] is False


def test_original_projected_question_object_is_preserved_without_rewriting():
    source, ranking = synthetic_case()
    source["question"] = {"question": source["question"], "question_id": source["question_id"]}
    selected, _ = select_case(source, ranking, limit=2)
    assert selected["question"] == source["question"]
    assert selected["question"] is not source["question"]


@pytest.mark.parametrize("change", ["question", "title", "text", "identity", "metadata", "partial", "nan", "order", "missing", "duplicate", "answer"])
def test_wrong_source_or_reference_contamination_is_rejected(change):
    source, ranking = synthetic_case()
    if change == "question": ranking["question"] += " changed"
    elif change == "title": ranking["candidates"][0]["title"] = "改写标题"
    elif change == "text": source["catalog"][0]["text"] += "新文"
    elif change == "identity": ranking["candidates"][0]["para_id"] = "另一段"
    elif change == "metadata": source["source_metadata"]["P1"]["source_text_sha256_utf8"] = "wrong"
    elif change == "partial": source["catalog"][0]["parent_complete"] = False
    elif change == "nan": ranking["candidates"][0]["local_logit"] = float("nan")
    elif change == "order": ranking["candidates"].reverse()
    elif change == "missing": ranking["candidates"].pop()
    elif change == "duplicate": ranking["candidates"][1] = deepcopy(ranking["candidates"][0])
    elif change == "answer": source["hidden"] = {"supporting_facts": []}
    with pytest.raises(ValueError):
        select_case(source, ranking, limit=2)


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False) + "\n").encode())


def freeze_files(tmp_path):
    source, ranking = synthetic_case()
    original = tmp_path / "original.json"
    write_json(original, source)
    write_json(tmp_path / "preparation.json", {"cases": [{"topic_id": "topic", "question_id": "question",
               "case_input": str(original), "case_sha256": digest(original)}]})
    write_json(tmp_path / "all-parent-ranking.json", {"cases": [ranking]})
    freeze = tmp_path / "candidate-freeze.json"
    write_json(freeze, {"generation_complete": True, "cases": 1, "reference_reads": [], "paid_api_calls": 0,
               "all_parent_ranking_sha256": digest(tmp_path / "all-parent-ranking.json"),
               "preparation_sha256": digest(tmp_path / "preparation.json")})
    return freeze, original


def test_frozen_inputs_provenance_and_exclusive_output(tmp_path):
    freeze, original = freeze_files(tmp_path)
    output = tmp_path / "out"
    receipt = prepare(freeze_path=freeze, freeze_sha=digest(freeze), output=output, limit=2)
    assert receipt["status"] == "CLOSED"
    assert receipt["paid_api_calls"] == receipt["new_local_model_calls"] == 0
    assert receipt["budget_status"] == "not_measured_requires_official_native_preflight"
    index = json.loads((output / "input-index.json").read_text("utf-8"))
    item = index["cases"][0]
    assert item["paragraphs"] == 2
    assert digest(item["file"]) == item["sha256"]
    provenance = json.loads((output / "ranking-provenance.json").read_text("utf-8"))
    assert [row["rank"] for row in provenance["cases"][0]["rows"]] == [1, 2]
    with pytest.raises(FileExistsError):
        prepare(freeze_path=freeze, freeze_sha=digest(freeze), output=output, limit=2)


@pytest.mark.parametrize("mutate", ["source", "ranking", "freeze"])
def test_any_frozen_binding_change_prevents_output(tmp_path, mutate):
    freeze, original = freeze_files(tmp_path)
    freeze_sha = digest(freeze)
    changed = {"source": original, "ranking": tmp_path / "all-parent-ranking.json", "freeze": freeze}[mutate]
    changed.write_bytes(changed.read_bytes() + b" ")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="sha_changed"):
        prepare(freeze_path=freeze, freeze_sha=freeze_sha, output=output, limit=2)
    assert not output.exists()
