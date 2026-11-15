import test from "node:test";
import assert from "node:assert/strict";
import { refineRequest, scopeEventSummary, scopeProgressText, historicalCoverageText, purposeReviewCoverage, purposeCoverageText } from "../src/quality-refine.ts";
import { extractionBatchSummary, tokenTotal } from "../src/runtime.ts";
import type { Activity, Run, RunEvent } from "../src/types.ts";
const event = (seq: number, data: Record<string, unknown> = {}, type = "scope_review_progress"): RunEvent => ({ run_id: "r", seq, type, created_at: "", data: { question_id: "q", phase: "original", batch_count: 4, completed_batches: 2, reviewed_count: 20, verified_fact_count: 8, narrowing_count: 3, ...data } });
const activity = (events: RunEvent[]) => ({run:{run_id:"r",status:"running",question_ids:["q"]},events}) as Activity;
test("refine explicitly creates a new task from a pinned candidate revision", () => {
  assert.deepEqual(refineRequest("q","current","source"), {question_ids:["q"],base_revision_id:"current",source_revision_id:"source",include_baseline:false,mode:"quality-refine"});
  assert.throws(()=>refineRequest("q",null,""));
});
test("strict progress is isolated by run and question and ordered by durable sequence", () => {
  const newer = event(7,{phase:"reverify",completed_batches:1,batch_count:2,reviewed_count:25});
  const other = {...event(9),run_id:"other"};
  assert.match(scopeProgressText(activity([newer,event(2),other,event(11,{question_id:"other"})]),"q"),/修正后二验 1\/2 批 · 累计复核 25 次/);
  assert.equal(scopeProgressText(activity([newer]),"missing"),"");
  assert.equal(scopeProgressText(activity([newer]),"q","selected-other-run"),"");
  assert.equal(scopeEventSummary(newer,"other","q"),"");
});
test("finished event replaces phase progress and preserves unresolved and manual counts", () => {
  const finish=event(8,{verified_fact_count:11,unresolved_count:2,manual_preserved_count:1},"scope_review_finished");
  assert.equal(scopeProgressText(activity([finish,event(3)]),"q"),"事实复核完成 · 通过 11 个 · 待处理 2 个 · 人工保留 1 个");
});
test("malformed phase and counts cannot manufacture a progress measure", () => {
  for (const data of [{completed_batches:5},{batch_count:0},{reviewed_count:1.5},{narrowing_count:-1},{verified_fact_count:Infinity},{phase:"invented"}]) assert.equal(scopeEventSummary(event(2,data),"r","q"),"");
  assert.equal(scopeEventSummary(event(-1),"r","q"),"");
  assert.equal(scopeEventSummary(event(2,{question_id:""}),"r"),"");
});
test("plan labels complete candidates separately from fact versions",()=>{
  assert.equal(scopeEventSummary(event(1,{candidate_count:323,fact_view_count:400,batch_count:15},"scope_review_planned"),"r","q"),"完整候选 323 条 · 400 个事实版本 · 初审 15 批");
});
const historicalRun = (): Run => ({run_id:"r",status:"running",question_ids:["q"],mode:"quality-refine",derived_inputs:[{question_id:"q",artifact_sha256:"hash",source_revision_id:"fixed",source_run_id:"old",candidate_count:323,historical_source_usage:[{total_tokens:900}],historical_source_coverage:{strategy:"full_source_scan",complete:true,scanned_paragraphs:372,total_paragraphs:372,completed_batches:28,batch_count:28}}]});
test("historical source scan is clearly reused and never added to current usage",()=>{
  const run=historicalRun();
  assert.equal(historicalCoverageText(run,"q"),"历史扫描 372/372 段 · 28/28 批 · 本次复用 323 条完整候选（本次未重新扫描）");
  assert.equal(tokenTotal([{total_tokens:100}]),100);
  assert.equal(historicalCoverageText({...run,mode:"initial"},"q"),"");
  assert.equal(historicalCoverageText(run,"other"),"");
});
test("incomplete or duplicate historical inputs are not shown as a complete scan",()=>{
  const run=historicalRun();
  (run.derived_inputs![0].historical_source_coverage as Record<string,unknown>).scanned_paragraphs=371;
  assert.equal(historicalCoverageText(run,"q"),"");
  const duplicate=historicalRun();duplicate.derived_inputs!.push(duplicate.derived_inputs![0]);
  assert.equal(historicalCoverageText(duplicate,"q"),"");
});
test("batch summaries accept only bounded program grammar for fact verification",()=>{
  assert.equal(extractionBatchSummary("第 1 / 4 批 · 20 个事实版本 · 8 个通过事实 · 3 个待缩窄"),"第 1 / 4 批 · 20 个事实版本 · 8 个通过事实 · 3 个待缩窄");
  assert.equal(extractionBatchSummary("第 5 / 4 批 · 8 个通过事实"),"");
  assert.equal(extractionBatchSummary("第 1 / 4 批 · 模型声称精确率100%"),"");
  assert.match(extractionBatchSummary({batch_index:1,batch_count:4,reviewed_count:20,verified_fact_count:8,secret:"hidden"}),/累计复核次数 20 · 通过事实 8$/);
});

const purposeEvent = (seq: number, data: Record<string, unknown>, type = "purpose_review_progress"): RunEvent => ({ run_id: "r", seq, type, created_at: "", data: { question_id: "q", ...data } });
const purposePlan = () => purposeEvent(1, { bank_id: "binding", bank_fact_count: 22, batch_count: 2, formal_evidence: false }, "purpose_review_planned");
const purposeBatch = (seq = 2, extra: Record<string, unknown> = {}) => purposeEvent(seq, { batch_index: 1, batch_count: 2, completed_batches: 1, bank_fact_count: 13, permitted_fact_count: 4, ...extra });

test("purpose progress distinguishes the internal bank, batch permission and unknown formal total", () => {
  const coverage = purposeReviewCoverage([purposeBatch(), purposePlan()], "r", "q")!;
  assert.equal(coverage.bankFactCount, 22);
  assert.equal(coverage.currentBatchFactCount, 13);
  assert.equal(coverage.currentBatchPermittedCount, 4);
  assert.equal(coverage.formalFactCount, null);
  assert.equal(purposeCoverageText(coverage), "内部事实库 22 个 · 用途审核 1/2 批 · 本批用途许可 4/13 个 · 正式许可总数尚未形成");
  assert.equal(scopeEventSummary(purposeBatch(), "r", "q"), "用途审核 1/2 批 · 本批内部事实 13 个 · 本批用途许可 4 个（非正式总数）");
  assert.match(scopeProgressText(activity([purposePlan(), purposeBatch()]), "q"), /本批用途许可 4 个（非正式总数）/);
});

test("only the actual completed purpose node provides the final formal count, including zero", () => {
  const finish = purposeEvent(4, { node: "purpose_review", bank_fact_count: 22, formal_fact_count: 0 }, "node_completed");
  const coverage = purposeReviewCoverage([purposePlan(), purposeBatch(), finish], "r", "q")!;
  assert.equal(coverage.complete, true);
  assert.equal(coverage.formalFactCount, 0);
  assert.equal(coverage.currentBatchPermittedCount, null);
  assert.match(purposeCoverageText(coverage), /内部事实库 22 个 · 正式许可 0 个 · 无获准事实，不能作确定性判断/);
  assert.doesNotMatch(purposeCoverageText(coverage), /已发布|失败|通过率|召回/);
  assert.match(scopeProgressText(activity([finish, purposeBatch()]), "q"), /用途审核完成.*正式许可 0 个/);
  assert.equal(purposeReviewCoverage([finish], "r", "q")?.formalFactCount, 0);
});

test("purpose progress does not sum replayed batch counts or manufacture a formal total", () => {
  const second = purposeBatch(4, { batch_index: 2, completed_batches: 2, bank_fact_count: 9, permitted_fact_count: 3 });
  const coverage = purposeReviewCoverage([purposePlan(), purposeBatch(), purposeBatch(3), second], "r", "q")!;
  assert.equal(coverage.completedBatches, 2);
  assert.equal(coverage.currentBatchPermittedCount, 3);
  assert.equal(coverage.formalFactCount, null);
  assert.match(purposeCoverageText(coverage), /本批用途许可 3\/9 个.*总数尚未形成/);
});

test("purpose counters never borrow from a different run, question or historical scope stage", () => {
  assert.equal(purposeReviewCoverage([purposePlan()], "another-run", "q"), null);
  assert.equal(purposeReviewCoverage([purposePlan()], "r", "another-question"), null);
  assert.equal(purposeReviewCoverage([event(1)], "r", "q"), null);
  assert.equal(purposeReviewCoverage([purposePlan()], undefined, "q"), null);
  assert.equal(scopeEventSummary(purposeBatch(), "r", "other"), "");
});

test("malformed purpose counters cannot manufacture work or permission", () => {
  for (const patch of [{ bank_fact_count: -1 }, { bank_fact_count: 1.5 }, { permitted_fact_count: 14 }, { completed_batches: 3 }, { batch_index: 2 }, { permitted_fact_count: "4" }]) {
    assert.equal(scopeEventSummary(purposeBatch(2, patch), "r", "q"), "");
    assert.equal(purposeReviewCoverage([purposePlan(), purposeBatch(2, patch)], "r", "q")?.completedBatches, 0);
  }
  for (const patch of [{ formal_evidence: true }, { bank_fact_count: 0, batch_count: 1 }, { bank_fact_count: 2, batch_count: 0 }]) {
    const invalid = { ...purposePlan(), data: { ...purposePlan().data, ...patch } };
    assert.equal(purposeReviewCoverage([invalid], "r", "q"), null);
  }
  const invalidFinish = purposeEvent(4, { node: "purpose_review", bank_fact_count: 22, formal_fact_count: 23 }, "node_completed");
  assert.equal(purposeReviewCoverage([invalidFinish], "r", "q"), null);
});

test("empty bank completion is explicit and never inferred from an empty plan", () => {
  const plan = purposeEvent(1, { bank_fact_count: 0, batch_count: 0, formal_evidence: false }, "purpose_review_planned");
  assert.equal(purposeReviewCoverage([plan], "r", "q")?.complete, false);
  const done = purposeEvent(2, { node: "purpose_review", bank_fact_count: 0, formal_fact_count: 0 }, "node_completed");
  assert.equal(purposeReviewCoverage([plan, done], "r", "q")?.complete, true);
});

const reusedPurpose = (extra: Record<string, unknown> = {}) => purposeEvent(1, {
  kind: "checkpoint_continuation", source_thread_id: "original-thread", checkpoint_id: "stored-checkpoint",
  reused_nodes: ["scope_review", "purpose_review"], reused_purpose_coverage: {
    bank_id: "stored-bank", bank_fact_count: 22, formal_fact_count: 0, complete: true, reused: true,
  }, ...extra,
}, "checkpoint_reused");

test("completed checkpoint purpose counts are displayed as reuse, never new execution", () => {
  const coverage = purposeReviewCoverage([reusedPurpose()], "r", "q")!;
  assert.equal(coverage.reused, true);
  assert.equal(coverage.complete, true);
  assert.equal(coverage.batchCount, null);
  assert.equal(coverage.formalFactCount, 0);
  assert.equal(purposeCoverageText(coverage), "复用已完成用途审核 · 内部事实库 22 个 · 正式许可 0 个 · 无获准事实，不能作确定性判断");
  assert.match(scopeEventSummary(reusedPurpose(), "r", "q"), /^复用已完成用途审核（检查点）/);
  assert.doesNotMatch(scopeEventSummary(reusedPurpose(), "r", "q"), /新调用|已发布|召回率/);
  assert.equal(purposeReviewCoverage([reusedPurpose()], "other", "q"), null);
  assert.equal(purposeReviewCoverage([reusedPurpose()], "r", "other"), null);
});

test("partial or unbound reuse metadata cannot manufacture a completed purpose card", () => {
  const base = reusedPurpose().data.reused_purpose_coverage as Record<string, unknown>;
  for (const extra of [
    { kind: "checkpoint_recovery" }, { source_thread_id: "" }, { checkpoint_id: "" }, { reused_nodes: ["scope_review"] },
    { reused_purpose_coverage: undefined },
    ...[{ complete: false }, { reused: false }, { bank_id: "" }, { bank_fact_count: -1 }, { formal_fact_count: 23 }, { formal_fact_count: "0" }]
      .map(patch => ({ reused_purpose_coverage: { ...base, ...patch } })),
  ]) {
    assert.equal(purposeReviewCoverage([reusedPurpose(extra)], "r", "q"), null);
    assert.equal(scopeEventSummary(reusedPurpose(extra), "r", "q"), "");
  }
});
