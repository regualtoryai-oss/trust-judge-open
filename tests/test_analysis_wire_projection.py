"""Synthetic arbitration-only aliases; no provider, gold or model output fixture."""
from copy import deepcopy
from uuid import NAMESPACE_URL, uuid5

import pytest

from app.domain.analysis_wire_projection import project_analyses, restore_analyses, validate_projection


def uid(value):
    return str(uuid5(NAMESPACE_URL, value))


def analyses(count=20):
    rows = []
    for prefix, role in (("a", "authority_analyst"), ("b", "cross_validation_analyst")):
        judgments = []
        for index in range(count):
            judgments.append({"id": uid(f"{prefix}:{index}"),
                "statement": f"完整判断{prefix}{index}😀。限定：不是已核实结论。\n原ID文字也保留：{uid(prefix + ':0')}",
                "evidence_ids": [uid(prefix + ":evidence")] if index == 0 else [],
                "prerequisite_judgment_ids": [row["id"] for row in judgments]})
        rows.append({"role": role, "evidence_snapshot_id": uid("frozen-snapshot"),
            "decision": "asserted" if prefix == "a" else "undetermined",
            "conclusion": f"{prefix}独立结论全文。\n包括2025年时点、50人、引文与原ID：{uid(prefix + ':0')}。",
            "summary": f"{prefix}完整summary：限定和不同立场不能消失。",
            "judgments": judgments, "supporting_evidence_ids": [uid(prefix + ":evidence")],
            "conflicting_evidence_ids": [uid(prefix + ":opposing-evidence")]})
    return rows


def test_high_fanout_forty_judgments_reconstruct_every_prose_reference_and_edge():
    original = analyses(20)  # 40 judgments, 380 edges, without reducing either analyst.
    untouched = deepcopy(original)
    result = project_analyses(original)
    projected, audit = result["projected"], result["audit"]
    assert original == untouched
    assert restore_analyses(projected, audit, trusted_original=original) == original
    receipt = validate_projection(original, projected, audit)
    assert sum(r["judgment_count"] for r in receipt["roles"]) == 40
    assert sum(r["dependency_count"] for r in receipt["roles"]) == 380
    assert receipt["projected_bytes"] < receipt["original_bytes"]
    for old, new in zip(original, projected):
        assert "evidence_snapshot_id" not in new
        for key in ("role", "decision", "conclusion", "summary", "supporting_evidence_ids", "conflicting_evidence_ids"):
            assert new[key] == old[key]
        for old_judgment, new_judgment in zip(old["judgments"], new["judgments"]):
            assert new_judgment["statement"] == old_judgment["statement"]
            assert new_judgment["evidence_ids"] == old_judgment["evidence_ids"]
    assert projected[0]["judgments"][-1]["id"] == "A20"
    assert projected[1]["judgments"][-1]["prerequisite_judgment_ids"] == [f"B{i}" for i in range(1, 20)]


def test_roles_define_aliases_regardless_of_input_order_and_empty_judgments_remain_empty():
    original = analyses(2)[::-1]
    result = project_analyses(original)
    assert result["projected"][0]["judgments"][0]["id"] == "B1"
    assert result["projected"][1]["judgments"][0]["id"] == "A1"
    for row in original:
        row.update(decision="undetermined", judgments=[], supporting_evidence_ids=[], conflicting_evidence_ids=[], summary="")
    result = project_analyses(original)
    assert restore_analyses(**result, trusted_original=original) == original


@pytest.mark.parametrize("change,reason", [
    ("cross_role", "foreign_prerequisite"), ("unknown", "foreign_prerequisite"),
    ("cycle", "prerequisite_cycle"), ("self_cycle", "prerequisite_cycle"),
    ("duplicate_id", "duplicate_judgment_identity"), ("alias_identity", "alias_collision"),
    ("alias_evidence", "alias_collision"), ("duplicate_dependency", "reference_list"),
    ("snapshot", "snapshot_binding"), ("unknown_field", "analysis_keys"),
])
def test_foreign_cycles_alias_collisions_and_unlisted_metadata_fail_closed(change, reason):
    original = analyses(2)
    first, second = original[0]["judgments"]
    if change == "cross_role":
        second["prerequisite_judgment_ids"] = [original[1]["judgments"][0]["id"]]
    elif change == "unknown":
        second["prerequisite_judgment_ids"] = [uid("foreign")]
    elif change == "cycle":
        first["prerequisite_judgment_ids"] = [second["id"]]
    elif change == "self_cycle":
        first["prerequisite_judgment_ids"] = [first["id"]]
    elif change == "duplicate_id":
        second["id"] = first["id"]
    elif change == "alias_identity":
        first["id"] = "A1"
    elif change == "alias_evidence":
        first["evidence_ids"] = ["B2"]
    elif change == "duplicate_dependency":
        second["prerequisite_judgment_ids"] *= 2
    elif change == "snapshot":
        original[1]["evidence_snapshot_id"] = uid("other-snapshot")
    elif change == "unknown_field":
        original[0]["api_key"] = "synthetic-secret-must-not-reflect"
    with pytest.raises(ValueError, match=reason) as caught:
        project_analyses(original)
    assert "synthetic-secret" not in str(caught.value)


@pytest.mark.parametrize("change", ["statement", "summary", "evidence", "edge", "alias", "role", "audit", "trusted_original"])
def test_replay_requires_trusted_original_not_a_self_asserted_alias_audit(change):
    original = analyses(2)
    result = project_analyses(original)
    projected, audit = result["projected"], result["audit"]
    if change == "statement":
        projected[0]["judgments"][0]["statement"] += "被篡改"
    elif change == "summary":
        projected[0]["summary"] = "被删减"
    elif change == "evidence":
        projected[0]["judgments"][0]["evidence_ids"] = []
    elif change == "edge":
        projected[0]["judgments"][1]["prerequisite_judgment_ids"] = []
    elif change == "alias":
        projected[0]["judgments"][0]["id"] = "B1"
    elif change == "role":
        projected[0]["role"] = "cross_validation_analyst"
    elif change == "audit":
        audit["role_bindings"][0]["judgment_ids"][0]["original_id"] = uid("forged")
    elif change == "trusted_original":
        original[0]["conclusion"] += "不同trusted source"
    with pytest.raises(ValueError, match="trusted_binding_changed"):
        restore_analyses(projected, audit, trusted_original=original)
