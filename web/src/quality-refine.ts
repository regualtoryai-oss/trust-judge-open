import type { Activity, Run, RunEvent } from "./types.ts";

export function refineRequest(questionId: string, baseRevisionId: string | null, sourceRevisionId: string) {
  if (!questionId || !sourceRevisionId) throw new Error("请选择已发布的完整候选版本。");
  return { question_ids: [questionId], base_revision_id: baseRevisionId, source_revision_id: sourceRevisionId,
    include_baseline: false, mode: "quality-refine" as const };
}
const count = (value: unknown): number | null => typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 1_000_000 ? value : null;

export interface PurposeReviewCoverage {
  bankFactCount: number;
  batchCount: number | null;
  completedBatches: number | null;
  currentBatchFactCount: number | null;
  currentBatchPermittedCount: number | null;
  formalFactCount: number | null;
  complete: boolean;
  reused?: boolean;
}
function reusedPurposeCoverage(event: RunEvent): PurposeReviewCoverage | null {
  const data = event.data;
  if (event.type !== "checkpoint_reused" || data.kind !== "checkpoint_continuation" ||
    !Array.isArray(data.reused_nodes) || !data.reused_nodes.includes("purpose_review") ||
    ![data.source_thread_id, data.checkpoint_id].every(value => typeof value === "string" && value.length > 0 && value.length <= 256)) return null;
  const raw = data.reused_purpose_coverage;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const stored = raw as Record<string, unknown>;
  const bank = count(stored.bank_fact_count), formal = count(stored.formal_fact_count);
  if (stored.complete !== true || stored.reused !== true || typeof stored.bank_id !== "string" || !stored.bank_id || stored.bank_id.length > 256 ||
    bank === null || formal === null || formal > bank) return null;
  return { bankFactCount: bank, batchCount: null, completedBatches: null, currentBatchFactCount: null,
    currentBatchPermittedCount: null, formalFactCount: formal, complete: true, reused: true };
}
function purposeEventSummary(event: RunEvent): string {
  const reused = reusedPurposeCoverage(event);
  if (reused) return `复用已完成用途审核（检查点） · 内部事实库 ${reused.bankFactCount} 个 · 正式许可 ${reused.formalFactCount} 个${reused.formalFactCount === 0 ? " · 无获准事实，不能作确定性判断" : ""}`;
  const data = event.data, bank = count(data.bank_fact_count), batches = count(data.batch_count);
  if (event.type === "purpose_review_planned") return bank === null || batches === null || batches > bank || (bank === 0) !== (batches === 0) || data.formal_evidence !== false
    ? "" : `内部事实库 ${bank} 个 · 用途审核 ${batches} 批 · 尚非正式证据`;
  if (event.type === "purpose_review_progress") {
    const done = count(data.completed_batches), index = count(data.batch_index), permitted = count(data.permitted_fact_count);
    return bank === null || bank < 1 || batches === null || done === null || done < 1 || done > batches || index !== done || permitted === null || permitted > bank
      ? "" : `用途审核 ${done}/${batches} 批 · 本批内部事实 ${bank} 个 · 本批用途许可 ${permitted} 个（非正式总数）`;
  }
  if (event.type === "node_completed" && data.node === "purpose_review") {
    const formal = count(data.formal_fact_count);
    return bank === null || formal === null || formal > bank ? ""
      : `用途审核完成 · 内部事实库 ${bank} 个 · 正式许可 ${formal} 个${formal === 0 ? " · 无获准事实，不能作确定性判断" : ""}`;
  }
  return "";
}

/** Batch permission counts are local; only the completed node records a formal total. */
export function purposeReviewCoverage(events: readonly RunEvent[], runId: string | undefined, questionId: string | undefined): PurposeReviewCoverage | null {
  if (!runId || !questionId) return null;
  let result: PurposeReviewCoverage | null = null;
  const ordered = events.filter(event => event.run_id === runId && event.data.question_id === questionId && Number.isInteger(event.seq) && event.seq >= 0).sort((a, b) => a.seq - b.seq);
  for (const event of ordered) {
    if (!purposeEventSummary(event)) continue;
    const reused = reusedPurposeCoverage(event);
    if (reused) { result = reused; continue; }
    const data = event.data;
    const previous = result as PurposeReviewCoverage | null;
    if (event.type === "purpose_review_planned") result = { bankFactCount: count(data.bank_fact_count)!, batchCount: count(data.batch_count)!, completedBatches: 0,
      currentBatchFactCount: null, currentBatchPermittedCount: null, formalFactCount: null, complete: false };
    else if (event.type === "purpose_review_progress" && previous && !previous.complete && data.batch_count === previous.batchCount && count(data.bank_fact_count)! <= previous.bankFactCount) {
      if (count(data.completed_batches)! < (previous.completedBatches || 0)) continue;
      result = { ...previous, completedBatches: count(data.completed_batches), currentBatchFactCount: count(data.bank_fact_count), currentBatchPermittedCount: count(data.permitted_fact_count) };
    } else if (event.type === "node_completed" && data.node === "purpose_review" && (!previous || data.bank_fact_count === previous.bankFactCount)) {
      result = { bankFactCount: count(data.bank_fact_count)!, batchCount: previous?.batchCount ?? null, completedBatches: previous?.completedBatches ?? null,
        currentBatchFactCount: null, currentBatchPermittedCount: null, formalFactCount: count(data.formal_fact_count), complete: true };
    }
  }
  return result;
}
export function purposeCoverageText(coverage: PurposeReviewCoverage): string {
  if (coverage.complete) return `${coverage.reused ? "复用已完成用途审核 · " : ""}内部事实库 ${coverage.bankFactCount} 个 · 正式许可 ${coverage.formalFactCount} 个${coverage.formalFactCount === 0 ? " · 无获准事实，不能作确定性判断" : ""}`;
  const parts = [`内部事实库 ${coverage.bankFactCount} 个`, coverage.completedBatches === null || coverage.batchCount === null ? "用途审核进度未记录" : `用途审核 ${coverage.completedBatches}/${coverage.batchCount} 批`];
  if (coverage.currentBatchFactCount !== null && coverage.currentBatchPermittedCount !== null) parts.push(`本批用途许可 ${coverage.currentBatchPermittedCount}/${coverage.currentBatchFactCount} 个`);
  parts.push("正式许可总数尚未形成");
  return parts.join(" · ");
}

/** Program event counts describe work performed; none is an accuracy estimate. */
export function scopeEventSummary(event: RunEvent, runId: string | undefined, questionId?: string): string {
  if (!runId || event.run_id !== runId || !Number.isInteger(event.seq) || event.seq < 0 ||
    typeof event.data.question_id !== "string" || !event.data.question_id ||
    (questionId !== undefined && event.data.question_id !== questionId)) return "";
  const data = event.data;
  const purpose = purposeEventSummary(event);
  if (purpose) return purpose;
  if (event.type === "scope_review_planned") {
    const batches = count(data.batch_count), facts = count(data.fact_view_count), candidates = count(data.candidate_count);
    return batches === null || facts === null || candidates === null ? "" : `完整候选 ${candidates} 条 · ${facts} 个事实版本 · 初审 ${batches} 批`;
  }
  if (event.type === "scope_review_progress") {
    const done = count(data.completed_batches), total = count(data.batch_count), reviewed = count(data.reviewed_count);
    const accepted = count(data.verified_fact_count), narrowing = count(data.narrowing_count);
    if (done === null || total === null || total === 0 || done > total || reviewed === null || accepted === null || narrowing === null ||
      !["original", "reverify"].includes(String(data.phase))) return "";
    return `${data.phase === "reverify" ? "修正后二验" : "严格事实初审"} ${done}/${total} 批 · 累计复核 ${reviewed} 次 · 当前通过 ${accepted} 个 · 初审标记需修正 ${narrowing} 个`;
  }
  if (event.type === "scope_review_finished") {
    const accepted = count(data.verified_fact_count), unresolved = count(data.unresolved_count), manual = count(data.manual_preserved_count);
    return accepted === null || unresolved === null || manual === null ? "" : `事实复核完成 · 通过 ${accepted} 个 · 待处理 ${unresolved} 个 · 人工保留 ${manual} 个`;
  }
  return "";
}
export function scopeProgressText(activity: Activity | null, questionId: string, selectedRunId?: string): string {
  if (selectedRunId && activity?.run.run_id !== selectedRunId) return "";
  if (!activity || !questionId || !activity.run.question_ids?.includes(questionId)) return "";
  const ordered = [...activity.events].sort((a, b) => b.seq - a.seq);
  for (const event of ordered) {
    const summary = scopeEventSummary(event, activity.run.run_id, questionId);
    if (summary) return summary;
  }
  return "";
}
export function historicalCoverageText(run: Run | null, questionId: string | undefined): string {
  if (run?.mode !== "quality-refine" || !questionId || !run.question_ids?.includes(questionId)) return "";
  const inputs = run.derived_inputs?.filter(item => item.question_id === questionId) || [];
  if (inputs.length !== 1) return "";
  const input = inputs[0], coverage = input.historical_source_coverage;
  if (!coverage || typeof coverage !== "object" || Array.isArray(coverage)) return "";
  const data = coverage as Record<string, unknown>;
  const scanned = count(data.scanned_paragraphs), total = count(data.total_paragraphs);
  const done = count(data.completed_batches), batches = count(data.batch_count), candidates = count(input.candidate_count);
  if (data.strategy !== "full_source_scan" || data.complete !== true || scanned === null || total === null ||
    done === null || batches === null || candidates === null || scanned !== total || done !== batches) return "";
  return `历史扫描 ${scanned}/${total} 段 · ${done}/${batches} 批 · 本次复用 ${candidates} 条完整候选（本次未重新扫描）`;
}
