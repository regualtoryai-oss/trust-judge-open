import test from "node:test";
import assert from "node:assert/strict";
import { analystStatus, attachRun, duration, isRunning, stageState, tokenTotal, toolSummary, viewStatus, eventLabel, toolName, toolCapacity, extractionCoverage, extractionCoverageText, extractionBatchSummary, missionStages, stageLabel } from "../src/runtime.ts";

test("source framework labels follow each frozen run rather than the current server", () => {
  assert.match(missionStages({ run_id: "lc", status: "running", source_bundle_enabled: true, source_bundle_front_framework: "langchain" })[0]!.sub, /LangChain/);
  assert.match(missionStages({ run_id: "old", status: "failed", source_bundle_enabled: true })[0]!.sub, /DSH/);
  assert.equal(toolName("langchain_source_bundle_select"), "LangChain 完整候选选证");
});
import type { Activity, Run, RunEvent } from "../src/types.ts";
const old: Run = { run_id: "old", status: "failed", created_at: "2026-10-17T10:00:00Z" };
const live: Run = { run_id: "live", status: "running" };

test("bundle workbench shows only its recorded native selection and existing independent tail", () => {
  const run: Run = { run_id: "bundle", status: "running", source_bundle_enabled: true, question_ids: ["q"] };
  assert.deepEqual(missionStages(run).flatMap(stage => stage.stages), ["source_bundle_selection", "freeze", "authority_analyst", "cross_validation_analyst", "arbitrate", "publish"]);
  assert.equal(toolName("read_source_bundles"), "连续原文全文读取");
  assert.equal(stageLabel("source_bundle_selection"), "连续原文检索与选证");
  const original: Run = { run_id: "legacy", status: "running", question_ids: ["q"] };
  const foreign: RunEvent = { run_id: "bundle", seq: 1, type: "node_started", data: { node: "source_bundle_selection", question_id: "q" }, created_at: "2026-10-20T00:00:00Z" };
  assert.equal(missionStages(original, [foreign])[0]!.title, "检索资料");
});
test("strict workbench shows its real two front stages and independent analyst tail", () => {
  const stages = missionStages({run_id:"strict",status:"running",strict_evidence_enabled:true});
  assert.deepEqual(stages.flatMap(stage => stage.stages), ["strict_selection","strict_annotation","freeze","authority_analyst","cross_validation_analyst","arbitrate","publish"]);
  assert.equal(toolName("strict_selection"), "完整原段选证");
});
test("refresh reconnects a live durable run ahead of a selected terminal history", () => {
  assert.equal(attachRun([old, live], "old"), live);
  assert.equal(attachRun([old]), old);
  assert.equal(attachRun([], "missing"), null);
});
test("paused and pausing presentation follows explicit durable view status", () => {
  assert.equal(viewStatus({ run_id: "x", status: "cancelled", view_status: "paused" }), "paused");
  assert.equal(isRunning({ run_id: "x", status: "running", view_status: "pausing" }), true);
  assert.equal(isRunning({ run_id: "x", status: "cancelled", view_status: "paused" }), false);
});
test("a pipeline group is complete only when all its real stages completed", () => {
  const activity = { progress: { completed_stages: [{ question_id: "q", stage: "extract" }], active_stages: [] } } as unknown as Activity;
  assert.equal(stageState(activity, ["extract", "verify", "freeze"]), "partial");
  activity.progress.active_stages.push({ question_id: "q", stage: "verify" });
  assert.equal(stageState(activity, ["extract", "verify", "freeze"]), "active");
  assert.equal(stageState(null, ["extract"]), "pending");
});
test("a multi-question stage remains partial until every requested question completes", () => {
  const activity = { run: { question_ids: ["q1","q2"] }, progress: { completed_stages: [{ question_id: "q1", stage: "retrieve" }], active_stages: [] } } as unknown as Activity;
  assert.equal(stageState(activity,["retrieve"]), "partial");
  activity.progress.completed_stages.push({question_id:"q2",stage:"retrieve"});
  assert.equal(stageState(activity,["retrieve"]), "done");
});
test("usage sums only recorded counters and never invents token values", () => {
  assert.equal(tokenTotal(null), null);
  assert.equal(tokenTotal({ q: [] }), null);
  assert.equal(tokenTotal({ q: [{ total_tokens: 123, input_tokens: 100 }, { total_tokens: 77 }] }), 200);
  assert.equal(tokenTotal({ q: [{ total_tokens: 0 }] }), 0);
});
test("tool count summaries are readable and exclude arbitrary diagnostic fields", () => {
  assert.equal(toolSummary({ verified_count: 12, candidate_count: 14, secret: "hidden" }), "核验证据 12 · 候选证据 14");
  assert.equal(toolSummary({ secret: "hidden" }), "");
  assert.equal(toolSummary("已冻结证据快照"), "已冻结证据快照");
});
test("elapsed and event labels reflect supplied facts rather than forecasts", () => {
  assert.equal(duration(125.9), "2 分 05 秒");
  assert.equal(duration(-5), "0 秒");
  assert.equal(eventLabel({run_id:"r",seq:1,type:"node_completed",data:{node:"verify"},created_at:""}), "原文核验完成");
});
test("failed A stays failed, completed B remains complete and downstream is blocked", () => {
  const activity = { run: { status:"failed", question_ids:["q"] }, progress:{ completed_stages:[{question_id:"q",stage:"freeze"},{question_id:"q",stage:"cross_validation_analyst"}], active_stages:[], failed_stages:[{question_id:"q",stage:"authority_analyst"}] } } as unknown as Activity;
  assert.equal(stageState(activity,["authority_analyst"]), "failed");
  assert.notEqual(analystStatus(activity,"authority_analyst"),"等待执行");
  assert.equal(analystStatus(activity,"cross_validation_analyst"),"已完成");
  assert.equal(stageState(activity,["arbitrate","publish"]),"blocked");
});
test("a successful repaired branch is complete and actual queued/paused states differ", () => {
  const activity = { run:{status:"running",question_ids:["q"]},progress:{completed_stages:[{question_id:"q",stage:"authority_analyst"}],active_stages:[],failed_stages:[{question_id:"q",stage:"authority_analyst"}]} } as unknown as Activity;
  assert.equal(stageState(activity,["authority_analyst"]),"done");
  activity.run.status="queued";
  assert.equal(stageState(activity,["cross_validation_analyst"]),"queued");
  activity.run={run_id:"r",status:"cancelled",view_status:"paused"};
  activity.progress.paused_stages=[{question_id:"q",stage:"cross_validation_analyst"}];
  assert.equal(stageState(activity,["cross_validation_analyst"]),"paused");
  assert.equal(stageState(activity,["arbitrate"]),"blocked");
});
test("snapshot failure belongs to final pipeline group without invented completion", () => {
  const activity = {run:{status:"failed"},progress:{completed_stages:[],active_stages:[],failed_stages:[{question_id:null,stage:"snapshot"}]}} as unknown as Activity;
  assert.equal(stageState(activity,["arbitrate","publish"]),"failed");
});
test("targeted repair events name the real stage and expose the extra call", () => {
  const base={run_id:"r",seq:1,data:{node:"authority_analyst",attempt:1},created_at:""};
  assert.match(eventLabel({...base,type:"model_repair_started"}),/来源研判 A定向修正开始/);
  assert.match(eventLabel({...base,type:"model_repair_completed"}),/定向修正通过/);
  assert.match(eventLabel({...base,type:"model_repair_failed"}),/定向修正失败/);
});
test("analyst failure labels distinguish requests, validation and execution", () => {
  const activity = {run:{status:"failed"},progress:{completed_stages:[],active_stages:[],failed_stages:[{question_id:"q",stage:"authority_analyst",code:"model_request_failed"}]}} as unknown as Activity;
  assert.equal(analystStatus(activity,"authority_analyst"),"请求失败");
  activity.progress.failed_stages![0].code="validation_failed";
  assert.equal(analystStatus(activity,"authority_analyst"),"校验失败");
  activity.progress.failed_stages![0].code="internal_error";
  assert.equal(analystStatus(activity,"authority_analyst"),"执行失败");
  activity.progress.failed_stages![0].details={check:"ungrounded_judgment"};
  assert.equal(analystStatus(activity,"authority_analyst"),"校验失败");
});

test("capacity retry labels distinguish real additional requests from one content repair", () => {
  const base = {run_id:"r",seq:1,created_at:"",data:{stage:"arbitrate",attempt:2,max_output_tokens:131072,previous_max_output_tokens:65536}};
  assert.equal(toolName("arbitrate_capacity_retry"), "综合仲裁 · 输出扩容重试");
  assert.equal(toolName("authority_analyst_repair_capacity_retry"), "来源研判 A · 定向修正 · 输出扩容重试");
  assert.match(eventLabel({...base,type:"model_capacity_retry_started"}), /综合仲裁输出扩容重试开始（第 2 次）/);
  assert.match(eventLabel({...base,type:"model_capacity_retry_completed"}), /响应完成/);
  assert.doesNotMatch(eventLabel({...base,type:"model_capacity_retry_completed"}), /校验通过|版本已发布/);
  assert.match(eventLabel({...base,type:"model_capacity_retry_failed"}), /未完成/);
  assert.doesNotMatch(eventLabel({...base,data:{stage:"arbitrate"},type:"model_capacity_retry_started"}), /第.*次/);
});

test("capacity metadata is an explicit ceiling and never counted as consumed tokens", () => {
  const summary = {attempt:1,previous_max_output_tokens:32768,max_output_tokens:65536,raw_response:"private",reasoning:"private"};
  assert.equal(toolSummary(summary), "第 1 次输出扩容重试 · 输出额度 32,768 → 65,536 tokens（上限，非已用）");
  assert.equal(tokenTotal(summary), null);
  assert.equal(tokenTotal({usage:{total_tokens:123},summary}), 123);
  assert.equal(toolSummary({attempt:-1,previous_max_output_tokens:NaN,max_output_tokens:Infinity,secret:"private"}), "");
});

test("call cards never borrow capacity from a concurrent or preceding retry", () => {
  const events = [
    {run_id:"r",seq:1,type:"model_capacity_retry_started",created_at:"",data:{call_id:"a",stage:"arbitrate",attempt:1,previous_max_output_tokens:32768,max_output_tokens:65536}},
    {run_id:"r",seq:2,type:"model_capacity_retry_started",created_at:"",data:{call_id:"b",stage:"arbitrate",attempt:2,previous_max_output_tokens:65536,max_output_tokens:131072}},
  ] satisfies RunEvent[];
  assert.match(toolCapacity({call_id:"a"},events), /32,768 → 65,536/);
  assert.match(toolCapacity({call_id:"b"},events), /65,536 → 131,072/);
  assert.equal(toolCapacity({call_id:"unknown"},events), "");
  assert.equal(toolCapacity({call_id:"a"},[{...events[0]!,data:{stage:"arbitrate",attempt:1,max_output_tokens:65536}}]), "");
});

function coverageEvent(seq: number, type: string, data: Record<string, unknown>, questionId = "q1", runId = "r"): RunEvent {
  return { run_id: runId, seq, type, created_at: "2026-10-18T10:00:00Z", data: { question_id: questionId, ...data } };
}

test("recorded full-source coverage is scoped to the exact task and selected question", () => {
  const events = [
    coverageEvent(1, "extract_coverage_planned", { strategy: "full_source_scan", total_paragraphs: 80, batch_count: 4 }),
    coverageEvent(2, "extract_coverage_progress", { total_paragraphs: 80, batch_count: 4, scanned_paragraphs: 20, completed_batches: 1, candidates_count: 6 }),
    coverageEvent(3, "extract_coverage_progress", { total_paragraphs: 80, batch_count: 4, scanned_paragraphs: 80, completed_batches: 4, candidates_count: 22 }, "q2"),
    coverageEvent(4, "extract_coverage_progress", { total_paragraphs: 80, batch_count: 4, scanned_paragraphs: 80, completed_batches: 4, candidates_count: 24 }, "q1", "other"),
  ];
  const actual = extractionCoverage(events, "r", "q1");
  assert.deepEqual(actual, { totalParagraphs: 80, batchCount: 4, scannedParagraphs: 20, completedBatches: 1, candidatesCount: 6 });
  assert.equal(extractionCoverageText(actual!), "已扫描 20 / 80 段 · 已完成 1 / 4 批");
  assert.equal(extractionCoverage(events, "r", undefined), null);
  assert.equal(extractionCoverage(events, "missing", "q1"), null);
});

test("coverage replay uses recorded cumulative counts without summing repeated events or tools", () => {
  const plan = coverageEvent(1, "extract_coverage_planned", { strategy: "full_source_scan", total_paragraphs: 12, batch_count: 3 });
  const progress = coverageEvent(3, "extract_coverage_progress", { total_paragraphs: 12, batch_count: 3, scanned_paragraphs: 4, completed_batches: 1, candidates_count: 2 });
  const done = coverageEvent(4, "extract_coverage_progress", { total_paragraphs: 12, batch_count: 3, scanned_paragraphs: 12, completed_batches: 3, candidates_count: 9 });
  assert.deepEqual(extractionCoverage([plan], "r", "q1"), { totalParagraphs: 12, batchCount: 3, scannedParagraphs: 0, completedBatches: 0, candidatesCount: null });
  assert.equal(extractionCoverage([progress, plan, progress], "r", "q1")?.scannedParagraphs, 4);
  const events = [plan, progress, done, coverageEvent(5, "run_paused", {}), coverageEvent(6, "tool_completed", { name: "evidence_batch_extraction", scanned_paragraphs: 100 })];
  assert.equal(extractionCoverage(events, "r", "q1")?.scannedParagraphs, 12);
  // Full input scanning does not announce a completed extraction stage or semantic recall.
  const activity = { run: { run_id: "r", status: "paused", question_ids: ["q1"] }, events, progress: { completed_stages: [], active_stages: [{ question_id: "q1", stage: "extract" }] } } as unknown as Activity;
  assert.notEqual(stageState(activity, ["extract"]), "done");
});

test("coverage ignores malformed counters instead of guessing or clamping progress", () => {
  const base = { strategy: "full_source_scan", total_paragraphs: 10, batch_count: 2, scanned_paragraphs: 5, completed_batches: 1 };
  const valid = coverageEvent(1, "extract_coverage_progress", base);
  for (const patch of [
    { scanned_paragraphs: 11 }, { completed_batches: 3 }, { scanned_paragraphs: -1 },
    { total_paragraphs: Infinity }, { batch_count: 1.5 }, { completed_batches: "1" },
  ]) assert.equal(extractionCoverage([valid, coverageEvent(2, "extract_coverage_progress", { ...base, ...patch })], "r", "q1")?.scannedParagraphs, 5);
  assert.equal(extractionCoverage([coverageEvent(1, "extract_coverage_planned", { ...base, strategy: "bm25" })], "r", "q1"), null);
  assert.equal(extractionCoverage([coverageEvent(1, "extract_coverage_progress", { ...base, scanned_paragraphs: undefined })], "r", "q1"), null);
});

test("batch call labels expose only recorded bounded batch counts", () => {
  const summary = { batch_index: 2, batch_count: 4, paragraph_count: 20, source_codepoints: 6000, candidates_count: 7, hidden_reasoning: "never display" };
  assert.equal(toolName("evidence_batch_extraction"), "原文分批抽证");
  assert.equal(extractionBatchSummary(summary), "第 2 / 4 批 · 本批 20 段 · 原文 6,000 字符 · 候选证据 7");
  assert.equal(toolSummary(summary), extractionBatchSummary(summary));
  assert.equal(extractionBatchSummary({ ...summary, batch_index: 0 }), "");
  assert.equal(extractionBatchSummary({ ...summary, batch_index: 5 }), "");
  assert.equal(extractionBatchSummary({ batch_index: 1, batch_count: 1, source_codepoints: "secret" }), "第 1 / 1 批");
  assert.equal(eventLabel(coverageEvent(1, "extract_coverage_planned", {})), "全量原文扫描已规划");
  assert.equal(eventLabel(coverageEvent(2, "extract_coverage_progress", {})), "原文扫描进度已记录");
});

test("current source coverage event names and workflow codepoint fields stay compatible", () => {
  const events = [
    coverageEvent(1, "source_coverage_planned", { strategy: "full_source_scan", total_paragraphs: 6, batch_count: 2 }),
    coverageEvent(2, "source_coverage_progress", { total_paragraphs: 6, batch_count: 2, scanned_paragraphs: 3, completed_batches: 1, candidates_count: 2 }),
  ];
  assert.equal(extractionCoverage(events, "r", "q1")?.scannedParagraphs, 3);
  assert.equal(eventLabel(events[0]!), "全量原文扫描已规划");
  assert.equal(eventLabel(events[1]!), "原文扫描进度已记录");
  assert.equal(extractionBatchSummary({ batch_index: 1, batch_count: 2, paragraph_count: 3, codepoint_count: 1200 }), "第 1 / 2 批 · 本批 3 段 · 原文 1,200 字符");
});

test("precision review summaries distinguish selected and rejected real candidates", () => {
  assert.equal(toolName("evidence_precision_review"), "分批证据精筛");
  assert.equal(toolName("evidence_precision_repair"), "证据精筛 · 定向修正");
  assert.equal(toolSummary({ candidate_count: 30, selected_count: 19, rejected_count: 11, manual_preserved_count: 0, hidden_reasoning: "never" }), "候选证据 30 · 精筛保留 19 · 精筛排除 11 · 人工保留 0");
  assert.equal(toolSummary({ batch_index: 2, batch_count: 4, candidate_count: 48, selected_count: 35, rejected_count: 13 }), "第 2 / 4 批 · 候选证据 48 · 精筛保留 35 · 精筛排除 13");
  assert.equal(extractionBatchSummary("第 2 / 4 批 · 48 条候选 · 35 条精筛保留 · 13 条精筛排除"), "第 2 / 4 批 · 48 条候选 · 35 条精筛保留 · 13 条精筛排除");
  assert.equal(extractionBatchSummary("第 0 / 4 批 · 48 条候选"), "");
  assert.equal(extractionBatchSummary("第 2 / 4 批 · private body"), "");
});

test("explicit premise-bank runs separate source verification, purpose review and shared freeze", () => {
  const stages = missionStages({ run_id: "r", status: "running", strict_scope_enabled: true, premise_bank_enabled: true });
  assert.deepEqual(stages.map(stage => stage.title), ["检索资料", "原文事实核验", "用途审核", "共享冻结", "双路独立研判", "仲裁与发布"]);
  assert.ok(!stages[1]!.stages.includes("freeze"));
  assert.deepEqual(stages[2]!.stages, ["purpose_review"]);
  assert.deepEqual(stages[3]!.stages, ["freeze"]);
  const refine = missionStages({ run_id: "r", status: "running", mode: "quality-refine", premise_bank_enabled: true });
  assert.equal(refine[0]!.title, "候选工件导入");
  assert.equal(refine[1]!.title, "原文事实核验");
  assert.equal(refine.filter(stage => stage.stages.includes("purpose_review")).length, 1);
});

test("historical runs without purpose execution retain the original stage grouping", () => {
  for (const flag of [undefined, false]) {
    const run: Run = { run_id: "r", status: "succeeded", strict_scope_enabled: true, premise_bank_enabled: flag };
    const groups = missionStages(run, [coverageEvent(1, "purpose_review_planned", { bank_fact_count: 4 }, "q", "another-run")]);
    assert.equal(groups.length, 4);
    assert.equal(groups[1]!.title, "抽取与核验");
    assert.ok(groups[1]!.stages.includes("freeze"));
    assert.ok(!groups.some(group => group.stages.includes("purpose_review")));
  }
  assert.equal(missionStages(null).length, 4);
  assert.equal(missionStages({ run_id: "r", status: "running", question_ids: ["q"] }, [coverageEvent(1, "node_started", { node: "purpose_review" }, "another-question")]).length, 4);
  assert.equal(missionStages({ run_id: "r", status: "running" }, [coverageEvent(-1, "node_started", { node: "purpose_review" })]).length, 4);
});

test("recorded purpose execution reveals its stage even when an older API omits the flag", () => {
  const groups = missionStages({ run_id: "r", status: "running" }, [coverageEvent(1, "node_started", { node: "purpose_review" })]);
  assert.ok(groups.some(group => group.title === "用途审核"));
  assert.equal(stageLabel("purpose_review"), "事实用途审核");
  assert.equal(toolName("review_fact_purpose"), "事实用途审核");
  assert.equal(toolName("purpose_review_repair"), "事实用途审核 · 定向修正");
  assert.equal(toolName("purpose_review_repair_capacity_retry"), "事实用途审核 · 定向修正 · 输出扩容重试");
  assert.equal(eventLabel(coverageEvent(1, "purpose_review_planned", {})), "内部事实库已准备 · 待审核用途");
  assert.equal(eventLabel(coverageEvent(2, "purpose_review_progress", {})), "事实用途审核进度已记录");
});

test("purpose failure remains failed while source review is done and shared freeze is blocked", () => {
  const activity = { run: { run_id: "r", status: "failed", question_ids: ["q"] }, progress: {
    completed_stages: [{ question_id: "q", stage: "scope_review" }], active_stages: [],
    failed_stages: [{ question_id: "q", stage: "purpose_review", code: "model_output_invalid" }],
  } } as unknown as Activity;
  assert.equal(stageState(activity, ["scope_review"]), "done");
  assert.equal(stageState(activity, ["purpose_review"]), "failed");
  assert.equal(stageState(activity, ["freeze"]), "blocked");
  activity.run.status = "running";
  activity.progress.completed_stages.push({ question_id: "q", stage: "purpose_review" });
  assert.equal(stageState(activity, ["purpose_review"]), "done");
  assert.equal(stageState(activity, ["freeze"]), "pending");
});

test("purpose tool counters label local batch permission separately from a formal total", () => {
  assert.equal(extractionBatchSummary({ batch_index: 1, batch_count: 3, bank_fact_count: 20, permitted_fact_count: 7, secret: "private" }), "第 1 / 3 批 · 本批内部事实 20 · 本批用途许可 7");
  assert.equal(toolSummary({ bank_fact_count: 22, formal_fact_count: 0, hidden_reasoning: "private" }), "内部事实库 22 · 正式许可投影 0");
});
