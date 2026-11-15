"""Black-box regression for private repeated-quote location choices.

Model corrections must explicitly choose a literal occurrence or return a longer
unique quotation. Source text remains immutable; candidate choices are never
published as events/errors, and there is still only one correction budget.
"""

from copy import deepcopy
import json

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import pytest

from app.agents.provider import ModelOutputInvalid
from app.workflows.analysis import run_analysis


TEXT = "🦊甲记录：设备已复位。🚦复核备注：设备已复位。检测尚未开始。"
QUOTE = "设备已复位"
UNIQUE = "检测尚未开始"
QUESTION = {"id": "synthetic-location-choice-question", "question": "记录中设备状态是否有复核，检测是否开始？"}
PARAGRAPHS = [{
    "id": "synthetic-location-choice-paragraph", "document_id": "synthetic-location-choice-document",
    "raw_doc_id": "source_emoji", "raw_para_id": "p_001", "text": TEXT,
    "start_cp": 0, "end_cp": len(TEXT),
}]


def unresolved(*, arbitration=False):
    result = {"decision": "undetermined", "conclusion": "合成资料仅记录状态，不能推断检测结果。",
              "judgments": [], "supporting_evidence_ids": [], "conflicting_evidence_ids": []}
    return {**result, "conflicts": []} if arbitration else result


class ChoiceProvider:
    def __init__(self, selection="second"):
        self.selection = selection
        self.repairs = []
        self.calls = []
        self.analysis_inputs = []

    async def extract(self, question, paragraphs):
        self.calls.append("extract")
        return [{"paragraph_id": paragraphs[0]["id"], "quote": quote,
                 "fact_des": "合成原文状态", "argument": {}, "entities": [],
                 "start_cp": None, "end_cp": None} for quote in (UNIQUE, QUOTE)]

    async def repair_extract(self, question, paragraphs, invalid_output, diagnostics):
        self.calls.append("repair_extract")
        self.repairs.append({"question": deepcopy(question), "paragraphs": deepcopy(paragraphs),
                             "invalid_output": deepcopy(invalid_output), "diagnostics": deepcopy(diagnostics)})
        rows = deepcopy(invalid_output["evidence"])
        if self.selection == "second":
            choices = diagnostics["source_location_choices"]
            choice = next(item for item in choices if item["candidate_index"] == 1)
            rows[1].update(choice["valid_spans"][1])
            # Only explicitly chosen coordinates are output. Context remains
            # private repair input and is not a field in the output contract.
            rows[1] = {key: value for key, value in rows[1].items() if key not in {"context_before", "context_after"}}
        elif self.selection == "longer_unique":
            rows[1].update(quote="复核备注：设备已复位。", start_cp=None, end_cp=None)
        elif self.selection == "wrong_utf16":
            start = TEXT.rindex(QUOTE)
            utf16_start = len(TEXT[:start].encode("utf-16-le")) // 2
            rows[1].update(start_cp=utf16_start, end_cp=utf16_start + len(QUOTE))
        return rows

    async def analyze(self, role, question, evidence, snapshot_id):
        self.calls.append(role)
        self.analysis_inputs.append((role, deepcopy(evidence), snapshot_id))
        return unresolved()

    async def arbitrate(self, question, evidence, analyses):
        self.calls.append("arbitrate")
        return unresolved(arbitration=True)


async def execute(tmp_path, provider, events):
    return await run_analysis(
        QUESTION, PARAGRAPHS, provider, test_only_legacy_precision=True, checkpoint_path=tmp_path / "locations.sqlite3",
        thread_id="private-location-choices", cancel_check=lambda: False,
        emit=lambda kind, data: events.append((kind, deepcopy(data))),
    )


def assert_private_choices(provider, events):
    assert len(provider.repairs) == 1
    request = provider.repairs[0]
    assert request["question"] == QUESTION
    assert request["paragraphs"] == PARAGRAPHS
    assert [row["quote"] for row in request["invalid_output"]["evidence"]] == [UNIQUE, QUOTE]
    diagnostic = request["diagnostics"]
    assert diagnostic["check"] == "quote_occurrence"
    assert diagnostic["candidate_index"] == 1
    assert "candidates[1]" in diagnostic["field_path"]
    choices = diagnostic["source_location_choices"]
    row = next(item for item in choices if item["candidate_index"] == 1)
    assert row["paragraph_id"] == PARAGRAPHS[0]["id"] and row["quote"] == QUOTE
    starts = [TEXT.index(QUOTE), TEXT.rindex(QUOTE)]
    assert [(span["start_cp"], span["end_cp"]) for span in row["valid_spans"]] == [(start, start + len(QUOTE)) for start in starts]
    for span in row["valid_spans"]:
        start, end = span["start_cp"], span["end_cp"]
        assert TEXT[start:end] == QUOTE
        assert TEXT[:start].endswith(span["context_before"])
        assert TEXT[end:].startswith(span["context_after"])
        assert span["context_before"] and span["context_after"]
    assert "🦊" in row["valid_spans"][0]["context_before"]
    assert len(TEXT[:starts[0]].encode("utf-16-le")) // 2 != starts[0]
    public = json.dumps(events, ensure_ascii=False)
    for private in ("source_location_choices", "valid_spans", "context_before", "context_after", TEXT, QUOTE):
        assert private not in public
    return row


async def test_repeated_quote_receives_literal_unicode_choices_and_explicit_second_location_is_valid(tmp_path):
    original = deepcopy(PARAGRAPHS)
    provider, events = ChoiceProvider("second"), []
    result = await execute(tmp_path, provider, events)
    assert_private_choices(provider, events)
    selected = next(row for row in result["evidence"] if row["quote"] == QUOTE)
    assert (selected["start_cp"], selected["end_cp"]) == (TEXT.rindex(QUOTE), TEXT.rindex(QUOTE) + len(QUOTE))
    assert selected["span_location_method"] == "model_explicit_occurrence"
    assert selected["model_span_corrected"] is False
    assert PARAGRAPHS == original
    assert len(provider.analysis_inputs) == 2
    assert all(rows == result["evidence"] and sid == result["evidence_snapshot_id"] for _, rows, sid in provider.analysis_inputs)
    assert provider.calls.count("repair_extract") == 1


async def test_repair_can_choose_a_longer_unique_continuous_original_quote_without_counting_characters(tmp_path):
    provider, events = ChoiceProvider("longer_unique"), []
    result = await execute(tmp_path, provider, events)
    assert_private_choices(provider, events)
    selected = next(row for row in result["evidence"] if row["quote"] == "复核备注：设备已复位。")
    assert selected["start_cp"] == TEXT.index(selected["quote"])
    assert selected["span_location_method"] == "exact_unique_quote"
    assert TEXT[selected["start_cp"]:selected["end_cp"]] == selected["quote"]


@pytest.mark.parametrize("selection,check", [("keep_null", "quote_occurrence"), ("wrong_utf16", "quote_span")])
async def test_second_output_still_requires_explicit_valid_location_and_never_reaches_analysts(tmp_path, selection, check):
    provider, events = ChoiceProvider(selection), []
    with pytest.raises(ModelOutputInvalid) as failure:
        await execute(tmp_path, provider, events)
    assert_private_choices(provider, events)
    assert failure.value.stage == "verify"
    assert failure.value.details["check"] == check
    assert failure.value.details["candidate_index"] == 1
    assert failure.value.details["repair_exhausted"] is True
    assert "source_location_choices" not in failure.value.details
    assert QUOTE not in str(failure.value)
    assert provider.calls == ["extract", "repair_extract"]
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "locations.sqlite3")) as saver:
        checkpoints = [item async for item in saver.alist({"configurable": {"thread_id": "private-location-choices"}})]
    assert checkpoints
    assert all("final" not in item.checkpoint["channel_values"] for item in checkpoints)

async def test_private_choices_are_bounded_but_other_exact_explicit_occurrence_remains_legal(tmp_path):
    text = "🦊" + "；".join([QUOTE] * 10) + "。" + UNIQUE
    paragraphs = [{**PARAGRAPHS[0], "text": text, "end_cp": len(text)}]

    class LateExplicitChoice(ChoiceProvider):
        async def repair_extract(self, question, sources, invalid_output, diagnostics):
            self.calls.append("repair_extract")
            self.repairs.append({"diagnostics": deepcopy(diagnostics), "paragraphs": deepcopy(sources)})
            choice = next(item for item in diagnostics["source_location_choices"] if item["candidate_index"] == 1)
            assert len(choice["valid_spans"]) == 8
            assert all(len(span["context_before"]) <= 32 and len(span["context_after"]) <= 32 for span in choice["valid_spans"])
            result = deepcopy(invalid_output["evidence"])
            # The source remains authoritative: an explicit later occurrence
            # can be valid even when the bounded diagnostic does not list it.
            start = sources[0]["text"].rindex(QUOTE)
            assert start > choice["valid_spans"][-1]["start_cp"]
            result[1].update(start_cp=start, end_cp=start + len(QUOTE))
            return result

    provider, events = LateExplicitChoice(), []
    result = await run_analysis(
        QUESTION, paragraphs, provider, test_only_legacy_precision=True, checkpoint_path=tmp_path / "bounded.sqlite3",
        thread_id="bounded-locations", cancel_check=lambda: False,
        emit=lambda kind, data: events.append((kind, deepcopy(data))),
    )
    selected = next(row for row in result["evidence"] if row["quote"] == QUOTE)
    assert selected["start_cp"] == text.rindex(QUOTE)
    assert selected["span_location_method"] == "model_explicit_occurrence"
    assert provider.calls.count("repair_extract") == 1
    assert provider.repairs[0]["paragraphs"] == paragraphs
    assert "source_location_choices" not in json.dumps(events)
