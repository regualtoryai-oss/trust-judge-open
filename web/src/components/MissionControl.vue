<script setup lang="ts">
import { computed } from "vue";
import type { Activity, Run } from "../types";
import { analystStatus, duration, extractionCoverage, extractionCoverageText, paragraphRankCoverage, paragraphRankCoverageText, missionStages, runLabels, stageLabel, stageState, stageStateLabels, tokenTotal, viewStatus } from "../runtime";
import { historicalCoverageText, purposeReviewCoverage, purposeCoverageText } from "../quality-refine";
import { diagnosticText, formatRunError, providerFailureAction } from "../presentation";
const props = defineProps<{ run: Run | null; activity: Activity | null; busy: boolean; connected: boolean; hasPublished: boolean; resultIsSelectedRun: boolean; questionLabel: string; questionId?: string; canRetry: boolean }>();
defineEmits<{ pause: []; resume: []; retry: [] }>();
const stages = computed(() => missionStages(props.run, props.activity?.events || []));
const state = computed(() => props.run ? viewStatus(props.run) : "idle");
const providerAction = computed(() => providerFailureAction(props.run?.error));
const current = computed(() => {
  if (state.value === "failed") {
    const error = props.run?.error;
    if (error && typeof error === "object" && error.code === "experiment_usage_stop") return "用量保护已触发 · 本次未发布";
    return error && typeof error === "object" && error.stage ? `${stageLabel(error.stage)}失败 · 本次未发布` : "执行失败 · 本次未发布";
  }
  if (state.value === "running") {
    const active = props.activity?.progress.active_stages.map(item => stageLabel(item.stage)).join(" + ");
    if (active) return active;
  }
  return props.run ? runLabels[state.value] || state.value : "准备就绪";
});
const completedCount = computed(() => Math.max(0, props.activity?.progress.completed_count || 0));
const totalCount = computed(() => Math.max(0, props.activity?.progress.total_count || 0));
const progressFraction = computed(() => totalCount.value ? Math.min(1, completedCount.value / totalCount.value) : 0);
const tokens = computed(() => tokenTotal(props.activity?.usage));
const priorTokens = computed(() => {
  const event = props.activity?.events.find(event => event.run_id === props.run?.run_id && event.type === "cost_lineage_seeded");
  const value = event?.data.prior_known_tokens;
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
});
const cachedStages = computed(() => new Set((props.activity?.events || []).filter(event =>
  event.run_id === props.run?.run_id && event.type === "model_response_reused" &&
  event.data.new_provider_calls === 0 && event.data.new_tokens === 0).map(event => event.data.operation)).size);
const syntaxRecoveredStages = computed(() => new Set((props.activity?.events || []).filter(event =>
  event.run_id === props.run?.run_id && event.type === "model_syntax_recovered" &&
  event.data.new_provider_calls === 0 && ["quote_escape_insertions_only", "single_unmatched_closing_delimiter_deletion_only"].includes(String(event.data.scope)))
  .map(event => event.data.operation)).size);
const errorText = computed(() => props.run?.status === "failed" && state.value !== "paused" ? formatRunError(props.run.error) : "");
const errorDetail = computed(() => props.run?.error && typeof props.run.error === "object" ? diagnosticText(props.run.error.details) : "");
const sourceCoverage = computed(() => {
  if (!props.run || props.activity?.run.run_id !== props.run.run_id) return null;
  const ids = props.run.question_ids || [];
  const question = props.questionId && ids.includes(props.questionId) ? props.questionId : ids.length === 1 ? ids[0] : undefined;
  return extractionCoverage(props.activity.events, props.run.run_id, question);
});
const historicalCoverage = computed(() => historicalCoverageText(props.run, props.questionId));
const rankedCoverage = computed(() => {
  if (!props.run || props.activity?.run.run_id !== props.run.run_id) return null;
  const ids = props.run.question_ids || [];
  const question = props.questionId && ids.includes(props.questionId) ? props.questionId : ids.length === 1 ? ids[0] : undefined;
  return paragraphRankCoverage(props.activity.events, props.run.run_id, question);
});
const purposeCoverage = computed(() => {
  if (!props.run || props.activity?.run.run_id !== props.run.run_id) return null;
  const ids = props.run.question_ids || [];
  const question = props.questionId && ids.includes(props.questionId) ? props.questionId : ids.length === 1 ? ids[0] : undefined;
  return purposeReviewCoverage(props.activity.events, props.run.run_id, question);
});
const sourceCoverageFraction = computed(() => sourceCoverage.value?.totalParagraphs ? sourceCoverage.value.scannedParagraphs / sourceCoverage.value.totalParagraphs : 0);
const lanes = [{ role: "authority_analyst", letter: "A", title: "来源与可核验性", sub: "核对原始来源 · 事实边界" }, { role: "cross_validation_analyst", letter: "B", title: "独立印证与反例", sub: "寻找冲突证据 · 保留异议" }];
function visualStageState(stages: string[]): string {
  const status = stageState(props.activity, stages);
  if (status !== "active" || state.value === "running") return status;
  return ["paused", "pausing"].includes(state.value) ? "paused" : "partial";
}
function laneLabel(role: string): string {
  return analystStatus(props.activity, role);
}
</script>

<template>
  <section class="mission-control" :class="{ 'mission-active': state === 'running', 'mission-paused': state === 'paused' }" aria-label="任务控制台">
    <div class="mission-heading">
      <div class="mission-state"><span class="live-orb" :class="state"></span><div><span class="eyebrow">MISSION CONTROL / 任务控制</span><h2 aria-live="polite">{{ current }}</h2></div></div>
      <div class="mission-buttons">
        <button v-if="run && ['queued','running'].includes(state)" class="button pause-button" :disabled="busy" @click="$emit('pause')"><span>Ⅱ</span> 暂停任务</button>
        <button v-if="run?.resumable" class="button primary" :disabled="busy" @click="$emit('resume')"><span>▶</span> 继续任务</button>
        <span v-if="state === 'pausing'" class="paused-hint">正在确认暂停…</span>
        <span v-if="!connected && run" class="connection-warning">连接中断 · 自动重连中</span>
      </div>
    </div>
    <div v-if="activity && totalCount" class="mission-progress-track" role="progressbar" aria-label="任务阶段完成情况" :aria-valuemin="0" :aria-valuemax="totalCount" :aria-valuenow="Math.min(completedCount,totalCount)" :aria-valuetext="`${completedCount} / ${totalCount} 个阶段完成`"><span class="mission-progress-fill" :style="{ transform: `scaleX(${progressFraction})` }"></span></div>
    <div class="mission-pipeline" :class="{ 'purpose-pipeline': stages.length > 4 }">
      <div v-for="(item,index) in stages" :key="item.title" class="pipeline-step" :class="visualStageState(item.stages)">
        <span class="pipeline-icon"><span class="pipeline-mark-content">{{ stageState(activity,item.stages) === 'done' ? '✓' : String(index+1).padStart(2,'0') }}</span></span>
        <div><strong>{{ item.title }}</strong><small>{{ item.sub }}</small><span class="pipeline-state">{{ stageStateLabels[visualStageState(item.stages)] || '待执行' }}</span></div><span v-if="index < stages.length - 1" class="pipeline-connector"></span>
      </div>
    </div>
    <div v-if="run?.mode === 'quality-refine'" class="pause-explanation">新质量复核任务 · 复用已完成来源扫描的候选工件；本次仅记录新增调用用量，历史扫描用量留在来源任务。</div>
    <div v-if="run?.source_bundle_enabled && run?.resumable" class="pause-explanation">已完成的流程检查点可复用；连续原文选证若尚未完成，续跑会重新执行该阶段，已知用量继续累计。</div>
    <div v-if="cachedStages" class="pause-explanation">已复用 {{ cachedStages }} 个已付模型响应，原请求和原响应均保留。这些复用阶段未再次调用模型；其他阶段的新增用量单列，原费用计入历史用量。</div>
    <div v-if="syntaxRecoveredStages" class="pause-explanation">{{ syntaxRecoveredStages }} 个阶段在本地修复了结构语法，未改写字段内容，未新增模型调用；原始响应与逐字符修复记录均已保留。</div>
    <div v-if="historicalCoverage" class="source-coverage"><div class="coverage-heading"><strong>复用来源工件</strong><span>{{ historicalCoverage }}</span></div></div>
    <div v-if="rankedCoverage" class="source-coverage" aria-live="polite"><div class="coverage-heading"><strong>候选段落排序</strong><span>{{ paragraphRankCoverageText(rankedCoverage) }}</span><small>候选评分进度 ≠ 全文扫描或证据召回率</small></div></div>
    <div v-if="purposeCoverage" class="source-coverage" aria-live="polite">
      <div class="coverage-heading"><strong>事实用途审核</strong><span>{{ purposeCoverageText(purposeCoverage) }}</span><small>内部事实库 ≠ 正式证据</small></div>
    </div>
    <div v-if="sourceCoverage && run?.mode !== 'quality-refine'" class="source-coverage" aria-live="polite">
      <div class="coverage-heading"><strong>{{ (run?.question_ids?.length || 0) > 1 ? '当前选题原文覆盖' : '原文扫描覆盖' }}</strong><span>{{ extractionCoverageText(sourceCoverage) }}</span><small>扫描覆盖 ≠ 证据召回率</small></div>
      <div class="coverage-track" role="progressbar" aria-label="原文段落扫描覆盖，不代表证据召回率" :aria-valuemin="0" :aria-valuemax="sourceCoverage.totalParagraphs" :aria-valuenow="sourceCoverage.scannedParagraphs" :aria-valuetext="extractionCoverageText(sourceCoverage)"><span :style="{ transform: `scaleX(${sourceCoverageFraction})` }"></span></div>
    </div>
    <div class="analyst-lanes"><div v-for="lane in lanes" :key="lane.role" class="analyst-lane" :class="visualStageState([lane.role])"><span class="analyst-avatar">{{ lane.letter }}<span v-if="visualStageState([lane.role]) === 'active'" class="analyst-signal" aria-hidden="true"></span></span><div><strong>{{ lane.title }}</strong><small>{{ lane.sub }}</small></div><span class="lane-status">{{ laneLabel(lane.role) }}</span></div><span class="independent-snapshot">同一快照 · 独立输入</span></div>
    <div class="mission-metrics">
      <span v-if="run"><i class="metric-dot"></i>任务 <code>{{ run.run_id.slice(0,8) }}</code></span><span v-else>选择一个问题，启动证据研判</span>
      <span v-if="questionLabel" :title="questionLabel">问题 {{ questionLabel }}</span>
      <span v-if="activity"><b>{{ activity.progress.completed_count }}</b> / {{ activity.progress.total_count }} 个阶段完成</span>
      <span v-if="activity">已用 {{ duration(activity.progress.elapsed_seconds) }}</span>
      <span v-if="tokens !== null">{{ priorTokens === null ? '已记录' : '本次新增' }} {{ tokens.toLocaleString() }} tokens</span>
      <span v-if="priorTokens !== null">历史已用 {{ priorTokens.toLocaleString() }} · 本题累计 {{ (priorTokens + (tokens || 0)).toLocaleString() }} tokens</span>
      <span v-if="hasPublished" class="published-note">{{ resultIsSelectedRun ? '结果区为本次已发布产物' : '结果区为其他任务已发布版本 · 非本次产物' }}</span>
    </div>
    <div v-if="errorText" class="mission-failure" role="status"><span class="failure-icon">!</span><div><strong>{{ errorText }}</strong><p v-if="errorDetail" class="failure-diagnostic">{{ errorDetail }}</p><p v-if="providerAction" class="failure-diagnostic">{{ providerAction }}</p><p v-if="!run?.resumable">不可直接继续：{{ run?.resume_blocked_reason || '请重新研判当前问题，历史任务记录保留。' }}</p><p v-else>可继续：复用已完成的检查点，重新执行未完成阶段；新增模型调用和用量会单独记录。</p><p>本次任务未发布结果；失败记录和已产生的用量保留。</p></div><button v-if="!run?.resumable" class="text-button" :disabled="busy || !canRetry" @click="$emit('retry')">重新研判 ↗</button></div>
    <div v-if="run?.status === 'failed'" class="pause-explanation"><a class="text-button" :href="`/api/runs/${encodeURIComponent(run.run_id)}/failure-appendix`" download>下载失败附录 ↓</a><span>固定未决说明 + 检索候选原文；未核验，不作为比赛证据，不新增模型调用。</span></div>
    <div v-if="state === 'paused' && !run?.resumable" class="mission-failure"><span class="failure-icon">Ⅱ</span><div><strong>当前任务暂无法直接继续</strong><p>{{ run?.resume_blocked_reason || '请基于当前资料重新研判，历史任务保留。' }}</p></div><button class="text-button" :disabled="busy || !canRetry" @click="$emit('retry')">重新研判 ↗</button></div>
    <div v-if="state === 'paused'" class="pause-explanation">任务已停止，已完成的检查点保留。继续会新建关联任务；暂停前发出的模型请求可能已产生用量。</div>
    <div v-if="run?.resumed_from_run_id" class="resume-lineage">↳ 续接自任务 <code>{{ run.resumed_from_run_id.slice(0,8) }}</code> · {{ run.resume_mode === 'start_unstarted' ? '从未开始处执行' : '已完成阶段按检查点复用' }}</div>
  </section>
</template>

<style scoped>
.mission-pipeline.purpose-pipeline{grid-template-columns:repeat(3,minmax(0,1fr))}@media(max-width:760px){.mission-pipeline.purpose-pipeline{grid-template-columns:repeat(2,minmax(0,1fr))}}
.pipeline-state{display:block;font-size:8px;line-height:1.5;color:#6d94a2;margin-top:5px}.pipeline-step.failed .pipeline-icon,.analyst-lane.failed .analyst-avatar{color:#ffc49b;border-color:#b17a52;background:#53382c55}.pipeline-step.failed strong,.pipeline-step.failed .pipeline-state,.analyst-lane.failed .lane-status{color:#ffc49b}.pipeline-step.blocked,.analyst-lane.blocked{opacity:.66}.pipeline-step.paused .pipeline-state,.analyst-lane.paused .lane-status{color:#dec799}.failure-diagnostic{color:#e8c391!important;overflow-wrap:anywhere}.published-note{overflow-wrap:anywhere}
.source-coverage{padding:0 20px 16px}.coverage-heading{display:flex;align-items:center;gap:12px;flex-wrap:wrap;font-size:10px;color:#92b3bb}.coverage-heading strong{font-size:10px;color:#bde3df}.coverage-heading small{margin-left:auto;color:#6d94a2;font-size:9px}.coverage-track{height:3px;background:#23434d;margin-top:9px;overflow:hidden;border-radius:2px}.coverage-track span{display:block;width:100%;height:100%;background:#61d9c8;transform-origin:left;transition:transform .35s ease}.mission-paused .coverage-track span{background:#dec799}@media(prefers-reduced-motion:reduce){.coverage-track span{transition:none}}
</style>
