import test from "node:test";
import assert from "node:assert/strict";
import { missionStages, paragraphRankCoverage, paragraphRankCoverageText } from "../src/runtime.ts";
import type { RunEvent } from "../src/types.ts";

function event(seq: number, type: string, data: Record<string, unknown>, question = "q", run = "r"): RunEvent {
  return { run_id: run, seq, type, created_at: "2026-10-19T00:00:00Z", data: { question_id: question, ...data } };
}
const plan = event(1,"paragraph_recall_planned",{source_paragraph_count:596,candidate_count:60,batch_count:5,complete_source_scan:false});
test("ranked strategy uses its actual stages and legacy strategies keep their stages", () => {
  assert.equal(missionStages({run_id:"r",status:"running",paragraph_ranked_enabled:true})[0]!.title,"问题与召回");
  assert.equal(missionStages({run_id:"r",status:"running"})[0]!.title,"检索资料");
  assert.equal(missionStages({run_id:"r",status:"running",question_ids:["q"]},[plan])[1]!.title,"逐段排序");
  assert.equal(missionStages({run_id:"different",status:"running"},[plan])[0]!.title,"检索资料");
});
test("candidate ranking counts never imply full-source model scanning", () => {
  const coverage = paragraphRankCoverage([plan,event(2,"paragraph_rank_progress",{candidate_count:60,batch_count:5,ranked_paragraph_count:24,completed_batches:2})],"r","q")!;
  assert.equal(coverage.rankedParagraphs,24);
  assert.equal(paragraphRankCoverageText(coverage),"从 596 段原文召回 60 段 · 已评分 24 / 60 段 · 2 / 5 批");
});
test("foreign and invalid events cannot invent candidate completion", () => {
  const coverage = paragraphRankCoverage([plan,event(2,"paragraph_rank_progress",{candidate_count:60,batch_count:5,ranked_paragraph_count:24,completed_batches:2},"another"),
    event(3,"paragraph_rank_progress",{candidate_count:60,batch_count:5,ranked_paragraph_count:61,completed_batches:6})],"r","q")!;
  assert.equal(coverage.rankedParagraphs,0);
  assert.equal(paragraphRankCoverage([event(1,"paragraph_recall_planned",{source_paragraph_count:2,candidate_count:60,batch_count:5,complete_source_scan:false})],"r","q"),null);
});
test("an explicit new recall plan resets its own scoring counts", () => {
  const coverage = paragraphRankCoverage([plan,event(2,"paragraph_rank_progress",{candidate_count:60,batch_count:5,ranked_paragraph_count:24,completed_batches:2}),
    {...plan,seq:3}],"r","q")!;
  assert.equal(coverage.rankedParagraphs,0);
});
