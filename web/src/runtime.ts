import type { Activity, Run, RunEvent, ToolCall } from "./types.ts";
import { capacityDiagnostic } from "./presentation.ts";

export const stageLabels: Record<string, string> = {
  resume: "用量与续跑检查",
  source_bundle_selection: "连续原文检索与选证",
  strict_selection: "完整原段选证", strict_annotation: "入选证据整理",
  paragraph_queries: "问题拆解", paragraph_recall: "候选段落召回", paragraph_rank: "逐段排序",
  paragraph_annotate: "整段事实概括", paragraph_fields: "段落字段核验",
  candidate_import: "候选工件导入", scope_review: "严格事实复核", purpose_review: "事实用途审核", retrieve: "资料检索", extract: "证据抽取", verify: "原文核验", freeze: "快照冻结",
  authority_analyst: "来源研判 A", analyst_a: "来源研判 A", authority: "来源研判 A",
  cross_validation_analyst: "交叉研判 B", analyst_b: "交叉研判 B", cross: "交叉研判 B",
  arbitrate: "综合仲裁", snapshot: "结果校验", publish: "版本发布", baseline: "基线对照", finish: "流程归档",
};
export const runLabels: Record<string, string> = {
  queued: "等待执行", running: "正在研判", pausing: "正在保存并暂停", paused: "已暂停",
  cancelling: "正在停止", cancel_requested: "正在停止", cancelled: "已停止", failed: "执行失败 · 未发布",
  succeeded: "已发布", completed: "已发布", ready: "已发布", superseded: "已有后续版本", awaiting_review: "等待复核",
};
export interface MissionStage { title: string; sub: string; stages: string[] }
/** Show new stages only for explicit run configuration or recorded execution. */
export function missionStages(run: Run | null, events: readonly RunEvent[] = []): MissionStage[] {
  const bundled = run?.source_bundle_enabled === true || !!run && events.some(event =>
    event.run_id === run.run_id && Number.isInteger(event.seq) && event.seq >= 0 &&
    (!run.question_ids?.length || typeof event.data.question_id === "string" && run.question_ids.includes(event.data.question_id)) &&
    ["node_started", "node_completed"].includes(event.type) && event.data.node === "source_bundle_selection");
  if (bundled) return [
    { title: "连续原文选证", sub: run?.source_bundle_front_framework === "langchain" ? "本地检索 · 完整候选 · LangChain 选证" : "本地检索 · 全文读取 · DSH 选证", stages: ["source_bundle_selection"] },
    { title: "来源与范围核验", sub: "逐字原文 · 完整送达 · 共享冻结", stages: ["freeze"] },
    { title: "双路独立研判", sub: "A 来源核查 / B 反例核查", stages: ["authority_analyst", "cross_validation_analyst"] },
    { title: "仲裁与发布", sub: "明确主张 · 证据角色 · 导出校验", stages: ["arbitrate", "publish"] },
  ];
  const strict = run?.strict_evidence_enabled === true || !!run && events.some(event =>
    event.run_id === run.run_id && Number.isInteger(event.seq) && event.seq >= 0 &&
    (!run.question_ids?.length || typeof event.data.question_id === "string" && run.question_ids.includes(event.data.question_id)) &&
    ["node_started", "node_completed"].includes(event.type) &&
    ["strict_selection", "strict_annotation"].includes(String(event.data.node)));
  if (strict) return [
    { title: "完整原段选证", sub: "标题与正文召回 · 双方主张与限定", stages: ["strict_selection"] },
    { title: "证据整理与核验", sub: "严格结构 · 逐字原文 · 冻结", stages: ["strict_annotation", "freeze"] },
    { title: "双路独立研判", sub: "A 来源核查 / B 反例核查", stages: ["authority_analyst", "cross_validation_analyst"] },
    { title: "仲裁与发布", sub: "冲突消解 · 图与导出校验", stages: ["arbitrate", "publish"] },
  ];
  const ranked = run?.paragraph_ranked_enabled === true || !!run && events.some(event => event.run_id === run.run_id && Number.isInteger(event.seq) && event.seq >= 0 &&
    (!run.question_ids?.length || typeof event.data.question_id === "string" && run.question_ids.includes(event.data.question_id)) &&
    (["paragraph_recall_planned", "paragraph_rank_progress"].includes(event.type) ||
      ["node_started", "node_completed"].includes(event.type) && typeof event.data.node === "string" && event.data.node.startsWith("paragraph_")));
  if (ranked) return [
    { title: "问题与召回", sub: "双方假设 · 官方原文 · 全局条件", stages: ["paragraph_queries", "paragraph_recall"] },
    { title: "逐段排序", sub: "回答 / 反证 / 必要限定", stages: ["paragraph_rank"] },
    { title: "整段概括与核验", sub: "一段一条 · 原文与字段 · 冻结", stages: ["paragraph_annotate", "paragraph_fields", "freeze"] },
    { title: "双路独立研判", sub: "A 来源核查 / B 反例核查", stages: ["authority_analyst", "cross_validation_analyst"] },
    { title: "仲裁与发布", sub: "冲突消解 · 图与导出校验", stages: ["arbitrate", "publish"] },
  ];
  const purpose = run?.premise_bank_enabled === true || !!run && events.some(event => event.run_id === run.run_id && Number.isInteger(event.seq) && event.seq >= 0 &&
    (!run.question_ids?.length || typeof event.data.question_id === "string" && run.question_ids.includes(event.data.question_id)) &&
    (["purpose_review_planned", "purpose_review_progress"].includes(event.type) ||
      ["node_started", "node_completed"].includes(event.type) && event.data.node === "purpose_review"));
  const stages: MissionStage[] = run?.mode === "quality-refine" ? [
    { title: "候选工件导入", sub: "固定版本 · 重新校验原文", stages: ["candidate_import"] },
    { title: purpose ? "原文事实核验" : "严格事实复核", sub: "全候选逐项审 · 修正后二验", stages: ["scope_review", ...(purpose ? [] : ["freeze"])] },
  ] : [
    { title: "检索资料", sub: "相关度排序 · 全文扫描", stages: ["retrieve"] },
    { title: purpose ? "原文事实核验" : "抽取与核验", sub: run?.strict_scope_enabled ? purpose ? "逐字核验 · 解释独立复核" : "逐字核验 · 严格事实复核" : "逐字核验 · 证据精筛",
      stages: ["extract", "verify", ...(run?.strict_scope_enabled || purpose ? ["scope_review"] : []), ...(purpose ? [] : ["freeze"])] },
  ];
  if (purpose) stages.push(
    { title: "用途审核", sub: "同题用途 · 反证与限定", stages: ["purpose_review"] },
    { title: "共享冻结", sub: "仅冻结获准用途的正式事实", stages: ["freeze"] },
  );
  stages.push(
    { title: "双路独立研判", sub: "A 来源核查 / B 反例核查", stages: ["authority_analyst", "cross_validation_analyst"] },
    { title: "仲裁与发布", sub: "冲突消解 · 图与导出校验", stages: ["arbitrate", "publish"] },
  );
  return stages;
}
export function viewStatus(run: Run): string { return run.view_status || run.status; }
export function isRunning(run: Run | null): boolean {
  return !!run && ["queued", "running", "pausing", "cancelling", "cancel_requested", "awaiting_review"].includes(viewStatus(run));
}
export function attachRun(runs: Run[], selectedId?: string): Run | null {
  return runs.find(isRunning) || runs.find(run => run.run_id === selectedId) || runs[0] || null;
}
export function stageLabel(stage: string | null | undefined): string { return stage ? stageLabels[stage] || stage : "等待开始"; }
export function toolName(name: string): string {
  if (name === "parent_token_pack") return "候选原段预算装箱";
  if (name === "sealed_front_recovery") return "已付选证封存恢复";
  const bundles: Record<string, string> = { source_bundle_selection: "连续原文检索与选证", source_bundle_select: "DSH 选证模型请求", langchain_source_bundle_select: "LangChain 完整候选选证", local_complete_parent_rank: "本地全文排序（无付费模型调用）", search_source_bundles: "全专题原文检索", read_source_bundles: "连续原文全文读取", submit_bundle_evidence: "选证结果提交与来源核验" };
  if (bundles[name]) return bundles[name];
  if (name.startsWith("strict_") && stageLabels[name]) return stageLabels[name]!;
  if (name.startsWith("paragraph_") && stageLabels[name]) return stageLabels[name]!;
  if (name === "snapshot_integrity_validation") return "结果完整性校验";
  if (name.endsWith("_capacity_retry")) return `${toolName(name.slice(0, -15))} · 输出扩容重试`;
  if (name.endsWith("_repair")) return `${toolName(name.slice(0, -7))} · 定向修正`;
  if (name === "review_fact_purpose" || name === "purpose_review") return "事实用途审核";
  if (name === "repair_fact_purpose") return "事实用途审核 · 定向修正";
  const names: Record<string, string> = { candidate_import: "候选工件校验导入", verify_evidence_scope: "原句蕴含与目标复核", reverify_evidence_scope: "修正事实独立二验", narrow_evidence_annotations: "同引文定向事实修正", retrieve: "原始段落检索", verify: "逐字引用核验", freeze: "证据快照冻结", bm25_retrieve: "BM25 段落检索", source_span_verifier: "逐字引用核验", freeze_evidence_snapshot: "证据快照冻结", extract: "证据抽取", evidence_batch_extraction: "原文分批抽证", evidence_precision_review: "分批证据精筛", evidence_precision: "证据精筛", authority_analyst: "来源研判 A", cross_validation_analyst: "交叉研判 B", arbitrate: "综合仲裁", baseline: "基线对照" };
  return names[name] || name.replace(/_/g, " ");
}
export function duration(seconds: number): string {
  const safe = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  return safe < 60 ? `${safe} 秒` : `${Math.floor(safe / 60)} 分 ${String(safe % 60).padStart(2, "0")} 秒`;
}
export function clockTime(value?: string | null): string {
  if (!value || Number.isNaN(Date.parse(value))) return "—";
  return new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(new Date(value));
}
export function stageState(activity: Activity | null, stages: string[]): string {
  if (!activity) return "pending";
  const state = activity.run ? viewStatus(activity.run) : "running";
  const includes = (stage: string) => stages.includes(stage) || stages.includes("publish") && ["snapshot", "build_snapshot"].includes(stage);
  const questions = activity.run?.question_ids || [];
  if (stages.every(stage => stage === "publish" || !questions.length
    ? activity.progress.completed_stages.some(item => item.stage === stage)
    : questions.every(questionId => activity.progress.completed_stages.some(item => item.stage === stage && item.question_id === questionId)))) return "done";
  if (activity.progress.failed_stages?.some(item => includes(item.stage))) return "failed";
  if (activity.progress.paused_stages?.some(item => includes(item.stage))) return "paused";
  if (activity.progress.active_stages.some(item => includes(item.stage))) {
    return ["paused", "pausing"].includes(state) ? "paused" : state === "running" ? "active" : "blocked";
  }
  if (["failed", "cancelled", "superseded"].includes(state)) return "blocked";
  if (["paused", "pausing", "cancelling", "cancel_requested"].includes(state)) return "blocked";
  if (activity.progress.completed_stages.some(item => stages.includes(item.stage))) return "partial";
  if (state === "queued") return "queued";
  return "pending";
}
export const stageStateLabels: Record<string, string> = { failed: "执行失败", paused: "已暂停", queued: "排队中", active: "执行中", blocked: "未执行 · 已阻断", done: "已完成", partial: "部分完成", pending: "待执行" };
export function analystStatus(activity: Activity | null, role: string): string {
  const status = stageState(activity, [role]);
  if (status === "failed") {
    const failure = activity?.progress.failed_stages?.find(item => item.stage === role);
    if (failure?.code === "model_request_failed") return "请求失败";
    if (failure?.code && ["validation_failed", "model_output_invalid", "model_output_truncated", "model_output_filtered", "snapshot_validation_failed", "publish_validation_failed"].includes(failure.code) || typeof failure?.details?.check === "string") return "校验失败";
    return "执行失败";
  }
  if (status !== "pending") return stageStateLabels[status] || status;
  return stageState(activity, ["freeze"]) === "done" ? "等待执行" : "等待证据快照";
}
export function toolSummary(value: unknown): string {
  if (typeof value === "string") return value;
  if (!value || typeof value !== "object" || Array.isArray(value)) return "";
  if (validCount((value as Record<string, unknown>).batch_index) !== null && validCount((value as Record<string, unknown>).batch_count) !== null) return extractionBatchSummary(value);
  const fields: Record<string, string> = { fact_view_count: "事实版本", reviewed_count: "已审版本", verified_fact_count: "通过事实", narrowing_count: "待修正", unresolved_count: "待处理", paragraph_count: "检索段落", candidate_count: "候选证据", candidates_count: "候选证据", selected_count: "精筛保留", rejected_count: "精筛排除", manual_preserved_count: "人工保留", verified_count: "核验证据", excluded_count: "排除证据", evidence_count: "证据", judgment_count: "中间判断", conflict_count: "冲突", reused_count: "复用阶段", input_tokens: "输入 tokens", output_tokens: "输出 tokens", total_tokens: "总 tokens" };
  Object.assign(fields, { bank_fact_count: "内部事实库", permitted_fact_count: "用途许可", formal_fact_count: "正式许可投影" });
  const counts = Object.entries(value).filter(([name,count]) => Object.prototype.hasOwnProperty.call(fields,name) && typeof count === "number" && Number.isFinite(count) && count >= 0).map(([name,count]) => `${fields[name]} ${count}`).join(" · ");
  return [capacityDiagnostic(value), counts].filter(Boolean).join(" · ");
}
export function toolCapacity(call: Pick<ToolCall, "call_id" | "summary">, events: readonly RunEvent[]): string {
  const summary = capacityDiagnostic(call.summary);
  if (summary) return summary;
  // Only an explicit call identity can attach event metadata to a call card.
  // Matching by role or question would mislabel concurrent or successive attempts.
  const event = [...events].reverse().find(item => item.type.startsWith("model_capacity_retry_") && item.data.call_id === call.call_id);
  return event ? capacityDiagnostic(event.data) : "";
}
export function eventLabel(event: RunEvent): string {
  const stage = typeof event.data.node === "string" ? event.data.node : typeof event.data.stage === "string" ? event.data.stage : null;
  const attempt = typeof event.data.attempt === "number" && Number.isInteger(event.data.attempt) && event.data.attempt > 0 && event.data.attempt <= 100 ? `（第 ${event.data.attempt} 次）` : "";
  const names: Record<string, string> = {
    paragraph_recall_planned: "段落召回完成 · 待逐段排序", paragraph_rank_progress: "候选段落评分进度已记录",
    candidate_artifact_attached: "新复核任务已绑定固定候选工件",
    candidate_artifact_imported: "工件已对原文校验 · 本任务未重新扫描",
    scope_review_planned: "完整候选事实复核已规划", scope_review_progress: "逐事实复核进度已记录", scope_review_finished: "事实复核已完成 · 待处理项单列",
    scope_narrowing_unresolved: "部分解释仍待处理 · 原候选已保留",
    purpose_review_planned: "内部事实库已准备 · 待审核用途", purpose_review_progress: "事实用途审核进度已记录",
    run_queued: "任务进入队列", run_started: "执行器已接管", run_succeeded: "校验通过 · 版本已发布",
    run_failed: "任务停在校验边界", run_cancelled: "执行已停止", cancel_requested: "已请求停止",
    pause_requested: "已请求暂停", checkpoint_reused: "已复用完成的检查点", publication_started: "正在校验并发布版本",
    run_paused: "任务已暂停 · 检查点保留", resume_queued: "检查点续接已入队",
    model_repair_started: `${stageLabel(stage)}定向修正开始（第 1 次）`,
    model_repair_completed: `${stageLabel(stage)}定向修正通过`,
    model_repair_failed: `${stageLabel(stage)}定向修正失败 · 已停止`,
    model_capacity_retry_started: `${stageLabel(stage)}输出扩容重试开始${attempt}`,
    model_capacity_retry_completed: `${stageLabel(stage)}输出扩容重试响应完成${attempt}`,
    model_capacity_retry_failed: `${stageLabel(stage)}输出扩容重试未完成${attempt}`,
    extract_coverage_planned: "全量原文扫描已规划", extract_coverage_progress: "原文扫描进度已记录",
    source_coverage_planned: "全量原文扫描已规划", source_coverage_progress: "原文扫描进度已记录",
    evidence_precision_progress: "分批证据精筛进度已记录",
    tool_started: "调用开始", tool_completed: "调用完成", tool_failed: "调用失败", tool_deferred: "调用未执行 · 额度保护", tool_cancelled: "调用已停止",
    model_response_reused: "复用已付模型响应 · 本次未新增调用", cost_lineage_seeded: "已承接本题历史用量",
    model_syntax_recovered: event.data.scope === "single_unmatched_closing_delimiter_deletion_only" ? "本地移除多余闭合符 · 未新增模型调用" : "本地恢复引号转义 · 未新增模型调用",
    sealed_front_recovery_completed: "已复用封存选证 · 未新增模型调用",
    evidence_balanced: "证据按双方主张与来源平衡 · 原评分保留",
    annotation_skipped_no_selection: "未选中合格证据 · 跳过模型整理",
    node_started: `${stageLabel(stage)}开始`, node_completed: `${stageLabel(stage)}完成`,
  };
  return names[event.type] || "执行记录";
}

function validCount(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 10_000_000 ? value : null;
}

export interface ParagraphRankCoverage { sourceParagraphs: number; candidateCount: number; rankedParagraphs: number; completedBatches: number; batchCount: number }
/** This measures scoring of retrieved candidates, never full-source model scanning or evidence recall. */
export function paragraphRankCoverage(events: readonly RunEvent[], runId: string | undefined, questionId: string | undefined): ParagraphRankCoverage | null {
  if (!runId || !questionId) return null;
  let coverage: ParagraphRankCoverage | null = null;
  const recorded = events.filter(e => e.run_id === runId && e.data.question_id === questionId && Number.isInteger(e.seq) && e.seq >= 0).sort((a,b) => a.seq-b.seq);
  for (const event of recorded) {
    const d = event.data;
    if (event.type === "paragraph_recall_planned") {
      const sourceParagraphs = validCount(d.source_paragraph_count), candidateCount = validCount(d.candidate_count), batchCount = validCount(d.batch_count);
      if (d.complete_source_scan !== false || sourceParagraphs === null || candidateCount === null || batchCount === null || candidateCount > sourceParagraphs) continue;
      coverage = { sourceParagraphs, candidateCount, batchCount, rankedParagraphs: 0, completedBatches: 0 };
    } else if (coverage && event.type === "paragraph_rank_progress") {
      const rankedParagraphs = validCount(d.ranked_paragraph_count), completedBatches = validCount(d.completed_batches);
      if (d.candidate_count !== coverage.candidateCount || d.batch_count !== coverage.batchCount || rankedParagraphs === null || completedBatches === null ||
          rankedParagraphs < coverage.rankedParagraphs || rankedParagraphs > coverage.candidateCount || completedBatches < coverage.completedBatches || completedBatches > coverage.batchCount) continue;
      coverage = { ...(coverage as ParagraphRankCoverage), rankedParagraphs, completedBatches };
    }
  }
  return coverage;
}
export function paragraphRankCoverageText(value: ParagraphRankCoverage): string {
  return `从 ${value.sourceParagraphs.toLocaleString("en-US")} 段原文召回 ${value.candidateCount} 段 · 已评分 ${value.rankedParagraphs} / ${value.candidateCount} 段 · ${value.completedBatches} / ${value.batchCount} 批`;
}

export interface ExtractionCoverage {
  scannedParagraphs: number;
  totalParagraphs: number;
  completedBatches: number;
  batchCount: number;
  candidatesCount: number | null;
}

/** Recorded source coverage is distinct from evidence recall and stage completion. */
export function extractionCoverage(events: readonly RunEvent[], runId: string | undefined, questionId: string | undefined): ExtractionCoverage | null {
  if (!runId || !questionId) return null;
  let coverage: ExtractionCoverage | null = null;
  const recorded = events.filter(event => event.run_id === runId && event.data.question_id === questionId && Number.isInteger(event.seq) && event.seq >= 0).sort((a, b) => a.seq - b.seq);
  for (const event of recorded) {
    if (!["extract_coverage_planned", "extract_coverage_progress", "source_coverage_planned", "source_coverage_progress"].includes(event.type)) continue;
    const data = event.data;
    const totalParagraphs = validCount(data.total_paragraphs), batchCount = validCount(data.batch_count);
    const isPlan = event.type.endsWith("_planned");
    if (isPlan && data.strategy !== "full_source_scan") continue;
    const scannedParagraphs = isPlan ? 0 : validCount(data.scanned_paragraphs);
    const completedBatches = isPlan ? 0 : validCount(data.completed_batches);
    if (totalParagraphs === null || batchCount === null || scannedParagraphs === null || completedBatches === null || scannedParagraphs > totalParagraphs || completedBatches > batchCount) continue;
    coverage = { totalParagraphs, batchCount, scannedParagraphs, completedBatches, candidatesCount: isPlan ? null : validCount(data.candidates_count) };
  }
  return coverage;
}

export function extractionCoverageText(coverage: ExtractionCoverage): string {
  return `已扫描 ${coverage.scannedParagraphs.toLocaleString("en-US")} / ${coverage.totalParagraphs.toLocaleString("en-US")} 段 · 已完成 ${coverage.completedBatches} / ${coverage.batchCount} 批`;
}

export function extractionBatchSummary(value: unknown): string {
  // The activity API projects summaries to fixed Chinese strings. Accept only
  // its bounded batch/count grammar here, never arbitrary model prose.
  if (typeof value === "string") {
    const match = /^第 (\d{1,6}) \/ (\d{1,6}) 批(?: · \d{1,7} (?:段原文|条候选|条精筛保留|条精筛排除|条人工保留|个事实版本|个已审版本|个通过事实|个待缩窄|个待处理))*$/.exec(value);
    return match && Number(match[1]) >= 1 && Number(match[1]) <= Number(match[2]) ? value : "";
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return "";
  const data = value as Record<string, unknown>;
  const index = validCount(data.batch_index), count = validCount(data.batch_count);
  if (index === null || count === null || index < 1 || index > count) return "";
  const parts = [`第 ${index} / ${count} 批`];
  const paragraphs = validCount(data.paragraph_count), codepoints = validCount(data.source_codepoints ?? data.codepoint_count), candidates = validCount(data.candidates_count ?? data.candidate_count);
  if (paragraphs !== null) parts.push(`本批 ${paragraphs} 段`);
  if (codepoints !== null) parts.push(`原文 ${codepoints.toLocaleString("en-US")} 字符`);
  if (candidates !== null) parts.push(`候选证据 ${candidates}`);
  const selected = validCount(data.selected_count), rejected = validCount(data.rejected_count);
  if (selected !== null) parts.push(`精筛保留 ${selected}`);
  if (rejected !== null) parts.push(`精筛排除 ${rejected}`);
  const factCounts: Record<string, string> = { fact_view_count: "事实版本", reviewed_count: "累计复核次数", verified_fact_count: "通过事实", narrowing_count: "待修正", unresolved_count: "待处理", bank_fact_count: "本批内部事实", permitted_fact_count: "本批用途许可" };
  for (const [key, label] of Object.entries(factCounts)) {
    const value = validCount(data[key]);
    if (value !== null) parts.push(`${label} ${value}`);
  }
  return parts.join(" · ");
}
export function tokenTotal(usage: unknown): number | null {
  if (!usage || typeof usage !== "object") return null;
  let found = false, total = 0;
  const walk = (value: unknown) => {
    if (Array.isArray(value)) { value.forEach(walk); return; }
    if (!value || typeof value !== "object") return;
    const row = value as Record<string, unknown>;
    if (typeof row.total_tokens === "number" && Number.isFinite(row.total_tokens)) { total += row.total_tokens; found = true; return; }
    Object.values(row).forEach(walk);
  };
  walk(usage);
  return found ? total : null;
}
