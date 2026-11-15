"""Actual LangChain serialization with MockTransport; no real account calls."""
import json

import pytest

from app.agents.provider import capture_usage, request_output_budget
from app.domain.model_cost_guard import CostGuardStopped, message_size_proxy
from app.domain.paragraph_ranked import ARGUMENTS, digest
from tests.test_deepseek_provider import ANALYSIS, QUESTION, intercepted_adapter


async def test_cost_admission_sees_final_schema_and_can_stop_before_any_transport():
    observed = []
    async with intercepted_adapter(content=json.dumps(ANALYSIS)) as (adapter, requests):
        async def deny(**value):
            observed.append({"operation":value["operation"],"budget":value["max_output_tokens"],
                "size":message_size_proxy(value["messages"]),"system":value["messages"][0].content})
            raise CostGuardStopped("cost_message_size_limit")
        adapter.request_preinvoke_hook = deny
        with capture_usage() as usage, request_output_budget(8192):
            with pytest.raises(CostGuardStopped, match="cost_message_size_limit"):
                await adapter.analyze("authority_analyst", QUESTION, [], "test-snapshot")
        assert requests == [] and usage == []
    assert observed[0]["operation"] == "authority_analyst" and observed[0]["budget"] == 8192
    assert "JSON Schema:" in observed[0]["system"]
    assert "source_inventory_basis" in observed[0]["system"]
    assert observed[0]["size"]["message_content_utf8_bytes"] > 1000


async def test_admitted_request_counts_only_provider_returned_usage():
    observed = []
    async with intercepted_adapter(content=json.dumps(ANALYSIS)) as (adapter, requests):
        async def admit(**value): observed.append(message_size_proxy(value["messages"]))
        adapter.request_preinvoke_hook = admit
        with capture_usage() as usage:
            assert await adapter.analyze("authority_analyst", QUESTION, [], "test-snapshot") == ANALYSIS
    assert len(requests) == len(observed) == 1
    assert usage[0]["total_tokens"] == 18
    assert "input_tokens_estimate" in observed[0] and observed[0]["input_tokens_estimate"] is None


@pytest.mark.parametrize("stage", ["rank", "annotate", "fields"])
async def test_compact_model_request_has_no_internal_proof_and_leaf_reasoning_disabled(stage):
    annotation={"paragraph_ref":"P3","fact_des":"甲方称尚待核查。", "argument":{k:None for k in ARGUMENTS},"entities":["甲方"]}
    outputs={"rank":{"decisions":[{"paragraph_ref":"P3","score":2,"critical_qualifier":True,"stance":"qualifier","reason":"尚待核查限定。"}]},
       "annotate":{"annotations":[annotation]},
       "fields":{"decisions":[{"paragraph_ref":"P3","annotation_sha256":digest(annotation),"fact_des_supported":True,
          "field_checks":{k:True for k in ARGUMENTS},"entities_supported":True,"attribution_preserved":True,"reason":"符合给定原段。"}]}}
    catalog=[{"paragraph_ref":"P3","text":"甲方称尚待核查。","title":"资料标题","raw_publish_time":"2038-05-09",
       "paragraph_id":"PRIVATE-ID","raw_doc_id":"RAW-PRIVATE","source_text_sha256":"PRIVATE-HASH",
       "retrieval_hits":[{"group":"X","bm25_score":123}],"mirror_origins":[{"similarity":.99}],"unknown":"DO-NOT-FORWARD"}]
    async with intercepted_adapter(content=json.dumps(outputs[stage],ensure_ascii=False)) as (adapter,requests):
        if stage=="rank": await adapter.paragraph_rank(QUESTION,catalog)
        elif stage=="annotate": await adapter.paragraph_annotate(QUESTION,catalog)
        else: await adapter.paragraph_fields(QUESTION,catalog,[annotation])
    assert len(requests)==1
    payload=json.loads(requests[0].content)
    row=json.loads(payload["messages"][1]["content"])["catalog"][0]
    assert set(row)==({"paragraph_ref","text","title","publication_time"} if stage=="rank" else {"paragraph_ref","text"})
    assert row["paragraph_ref"]=="P3" and row["text"]==catalog[0]["text"]
    wire=payload["messages"][1]["content"]
    assert all(value not in wire for value in ("PRIVATE-ID","RAW-PRIVATE","PRIVATE-HASH","DO-NOT-FORWARD","retrieval_hits","mirror_origins"))
    assert payload["thinking"]=={"type":"disabled"} and payload["reasoning_effort"]=="none"
