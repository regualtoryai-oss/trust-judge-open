"""Compact v19 model sources; no model/network and no reference answers."""
from copy import deepcopy
import json

import pytest

from app.domain import paragraph_ranked as p


def catalog_fixture():
    sources = [{"id": f"internal-p-{i}", "document_id": f"internal-d-{i}", "raw_doc_id": f"raw-doc-{i}",
        "raw_para_id": f"raw-doc-{i}#p8", "text": f"第{i}段：甲方称已批准；主管部门表示尚待核查。🦊\n仍未确认。",
        "title": "甲设备进展", "raw_publish_time": "2038年5月9日", "source_credibility_json": "0.95"}
        for i in range(1, 4)]
    recall = {"candidate_catalog": [{"paragraph_ref": f"P{i * 10}", "paragraph_id": row["id"],
        "raw_doc_id": row["raw_doc_id"], "raw_para_id": row["raw_para_id"], "source_text_sha256": p.digest(row["text"]),
        "fusion_score": .95, "retrieval_hits": [{"group": "X", "field": "text", "rank": 1, "bm25_score": 11}],
        "candidate_origin": "mirror_expansion", "mirror_origins": [{"similarity": .95, "paragraph_id": "internal-other"}],
        "unknown_gold": "DO_NOT_FORWARD"} for i, row in enumerate(sources, 1)]}
    return recall, {row["id"]: row for row in sources}


def test_stable_aliases_literal_source_and_proof_metadata_stay_separate():
    recall, sources = catalog_fixture(); before = deepcopy((recall, sources))
    model = p.rank_batch_catalog(recall, sources, ["P30", "P10"])
    assert [r["paragraph_ref"] for r in model] == ["P10", "P30"]  # never renumber a subset
    assert all(set(row) == {"paragraph_ref", "text", "title", "publication_time"} for row in model)
    assert model[0]["text"] == sources["internal-p-1"]["text"]
    assert model[1]["text"] == sources["internal-p-3"]["text"]
    assert all(row["publication_time"] == "2038-05-09" for row in model)
    wire = json.dumps(model, ensure_ascii=False)
    assert all(forbidden not in wire for forbidden in ("internal-p-", "raw-doc-", "0.95", "similarity", "DO_NOT_FORWARD", "sha256", "retrieval_hits"))
    assert (recall, sources) == before
    assert recall["candidate_catalog"][0]["source_text_sha256"] == p.digest(model[0]["text"])


@pytest.mark.parametrize("stage", ["annotate", "fields"])
def test_annotation_and_fields_receive_only_alias_and_literal_text(stage):
    recall, sources = catalog_fixture()
    expected = [{"paragraph_ref": r["paragraph_ref"], "text": r["text"]} for r in p.rank_batch_catalog(recall, sources)]
    assert p.rank_batch_catalog(recall, sources, stage=stage) == expected
    assert p.paragraph_model_catalog(p.rank_batch_catalog(recall, sources), stage=stage) == expected


@pytest.mark.parametrize("time,expected", [
    ("昨天", None), ("2038-05", None), ("人民日报", None), (None, None),
    ("2038年5月9日", "2038-05-09"), ("2038-05-09 12:30", "2038-05-09T12:30:00"),
    ("2038-05-09T12:30:00+08:00", "2038-05-09T12:30:00+08:00"),
    ("2038-02-30", None),
])
def test_unknown_raw_time_never_forwarded_and_explicit_time_idempotent(time, expected):
    row = {"paragraph_ref": "P2", "text": "原文。", "raw_publish_time": time,
        "publish_time_metadata": {"normalized": "FORGED", "status": "explicit_datetime"}}
    result = p.paragraph_model_catalog([row])
    assert result == [{"paragraph_ref": "P2", "text": "原文。", **({"publication_time": expected} if expected else {})}]
    assert p.paragraph_model_catalog(result) == result


@pytest.mark.parametrize("damage", ["text", "raw_doc_id", "raw_para_id", "id", "hash", "missing_source", "duplicate_alias"])
def test_internal_source_binding_is_checked_before_discarding_metadata(damage):
    recall, sources = catalog_fixture()
    if damage == "text": sources["internal-p-1"]["text"] += "后加内容。"
    elif damage in {"raw_doc_id", "raw_para_id", "id"}: sources["internal-p-1"][damage] = "different"
    elif damage == "hash": recall["candidate_catalog"][0]["source_text_sha256"] = "0" * 64
    elif damage == "missing_source": sources.pop("internal-p-1")
    else: recall["candidate_catalog"][1]["paragraph_ref"] = "P10"
    with pytest.raises(ValueError, match="source_binding"):
        p.rank_batch_catalog(recall, sources)


@pytest.mark.parametrize("refs", [["P404"], ["P10", "P10"], ["P10", {}], "P10"])
def test_unknown_or_duplicate_requested_aliases_are_not_silently_ignored(refs):
    recall, sources = catalog_fixture()
    with pytest.raises(ValueError, match="source_binding"):
        p.rank_batch_catalog(recall, sources, refs)


@pytest.mark.parametrize("catalog", [[{"paragraph_ref": "P0", "text": "原文"}],
    [{"paragraph_ref": "RAW-PRIVATE-ID", "text": "原文"}],
    [{"paragraph_ref": "P1", "text": "原文"}] * 2,
    [{"paragraph_ref": "P1", "text": ""}], [{"paragraph_ref": "P1", "text": None}]])
def test_direct_provider_projection_rejects_invalid_aliases_and_empty_sources(catalog):
    with pytest.raises(ValueError, match="model_catalog"):
        p.paragraph_model_catalog(catalog)


def test_direct_provider_projection_drops_untrusted_metadata_and_is_fresh():
    original = [{"paragraph_ref": "P1", "text": "原文不变。", "title": "标题", "publication_time": "2038-05-09",
        "paragraph_id": "PRIVATE", "source_credibility": .95, "candidate_origin": "official", "score": 3,
        "mirror_origins": [{"similarity": .99}], "answer": "DO_NOT_FORWARD"}]
    result = p.paragraph_model_catalog(original)
    assert result == [{"paragraph_ref": "P1", "text": "原文不变。", "title": "标题", "publication_time": "2038-05-09"}]
    result[0]["title"] = "changed"
    assert original[0]["title"] == "标题"
    with pytest.raises(ValueError): p.paragraph_model_catalog(original, stage="unknown")
