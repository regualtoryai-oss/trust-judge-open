<script setup lang="ts">
import { computed, ref, watch, nextTick, onMounted, onBeforeUnmount } from "vue";
import type { Activity, Run } from "../types";
import { clockTime, duration, eventLabel, runLabels, viewStatus, toolSummary, toolName, toolCapacity, extractionBatchSummary, extractionCoverage, extractionCoverageText } from "../runtime";
import { scopeEventSummary } from "../quality-refine";
import { diagnosticText, eventDiagnostic, formatRunError } from "../presentation";
const props = defineProps<{ run: Run | null; activity: Activity | null; runs: Run[]; connected: boolean; motion: boolean }>();
defineEmits<{ select: [id: string]; reconnect: [] }>();
const replay = ref(false);
const cursor = ref(0);
const now = ref(Date.now());
const listMotionEnabled = ref(false);
const executionIdentity = computed(() => props.activity?.run.run_id || props.run?.run_id || "empty");
let clock: ReturnType<typeof setInterval> | undefined;
onMounted(() => { clock = setInterval(() => { now.value = Date.now(); },1000); });
onBeforeUnmount(() => { clearInterval(clock); });
watch(() => props.run?.run_id, () => { replay.value = false; cursor.value = 0; });
watch([() => props.run?.run_id, () => props.activity?.run.run_id], async ([selectedId, loadedId], _previous, onCleanup) => {
  // Loading a historical task must not make all its recorded calls look new.
  listMotionEnabled.value = false;
  if (!loadedId || loadedId !== selectedId) return;
  let current = true;
  onCleanup(() => { current = false; });
  await nextTick();
  if (current) listMotionEnabled.value = true;
}, { immediate: true });
const events = computed(() => {
  const actual = props.activity?.events || [];
  return (replay.value ? actual.slice(0,cursor.value+1) : actual).slice(-35).reverse();
});
const activeTools = computed(() => props.activity?.tool_calls.slice().reverse() || []);
function startReplay() { replay.value = !replay.value; cursor.value = 0; }
function runningElapsed(startedAt: string): string { return duration(Math.max(0,(now.value-Date.parse(startedAt))/1000)); }
function failureText(value: { code?: string; message?: string; stage?: string | null }): string {
  return value.code && value.message ? formatRunError(value) : value.code && /^[a-z][a-z0-9_]{0,63}$/.test(value.code) ? `调用未完成（错误代码：${value.code}）` : "未记录详细原因";
}
function sourceProgressText(event: import("../types").RunEvent): string {
  const questionId = typeof event.data.question_id === "string" ? event.data.question_id : undefined;
  const coverage = extractionCoverage([event], props.run?.run_id, questionId);
  return coverage ? extractionCoverageText(coverage) : "";
}
</script>

<template>
  <aside class="activity-sidebar" :class="{ 'is-executing': run && viewStatus(run) === 'running' && !replay }" aria-label="真实执行记录">
    <div class="activity-header"><div><span class="eyebrow">EXECUTION OBSERVATORY</span><h2>执行观测台 <span class="live-label" :class="{ stale: !connected }">{{ connected ? 'LIVE' : '离线' }}</span></h2></div><button class="icon-button" aria-label="重新连接执行记录" @click="$emit('reconnect')">↻</button></div>
    <div class="history-selector"><label for="run-history">任务记录 <span>{{ runs.length }}</span></label><select id="run-history" :value="run?.run_id || ''" @change="$emit('select',($event.target as HTMLSelectElement).value)"><option v-if="!runs.length" value="">尚无执行任务</option><option v-for="item in runs" :key="item.run_id" :value="item.run_id">{{ clockTime(item.created_at) }} · {{ runLabels[viewStatus(item)] || viewStatus(item) }} · {{ item.run_id.slice(0,6) }}</option></select></div>
    <section class="tools-section">
      <div class="rail-heading"><h3>实际工具与模型调用</h3><span>{{ activeTools.length }}</span></div>
      <p v-if="!activeTools.length" class="rail-empty">本任务尚无工具调用记录。<br>旧任务未采集的调用不会补写。</p>
      <TransitionGroup :key="executionIdentity" name="execution-list" tag="div" class="tool-call-list" :css="motion && listMotionEnabled">
        <details v-for="(call,index) in activeTools" :key="call.call_id" class="tool-call" :class="call.status" :style="{ '--arrival-index': Math.min(index,4) }">
          <summary>
            <span class="tool-status"><span>{{ call.status === 'completed' ? '✓' : call.status === 'running' ? '◉' : call.status === 'deferred' ? '–' : '!' }}</span></span>
            <div>
              <strong>{{ toolName(call.name) }}</strong><span v-if="call.name.endsWith('_repair')" class="repair-badge">额外模型调用 · 最多 1 次</span><span v-else-if="call.name.endsWith('_capacity_retry')" class="repair-badge">额外模型调用 · 输出扩容重试</span>
              <small v-if="call.name.endsWith('_capacity_retry') && toolCapacity(call, activity?.events || [])" class="capacity-note">{{ toolCapacity(call, activity?.events || []) }}</small>
              <small v-if="extractionBatchSummary(call.summary)" class="batch-note">{{ extractionBatchSummary(call.summary) }}</small>
              <span v-if="call.kind || call.engine" class="tool-engine">{{ call.kind === 'model' ? '模型' : '本地工具' }}<template v-if="call.engine"> · {{ call.engine }}</template></span>
              <small>{{ clockTime(call.started_at) }} <span>· {{ call.status === 'running' ? '执行中' : call.status === 'completed' ? '已完成' : call.status === 'cancelled' ? '已停止' : call.status === 'deferred' ? '未执行 · 额度保护' : '未完成' }}</span><span v-if="call.status === 'running'"> · 已用 {{ runningElapsed(call.started_at) }}</span><span v-else-if="call.elapsed_ms != null"> · {{ duration(call.elapsed_ms/1000) }}</span></small>
            </div>
            <span class="detail-chevron">⌄</span>
          </summary>
          <div v-if="call.status === 'failed'" class="call-failure" role="status"><strong>{{ failureText(call) }}</strong><p v-if="diagnosticText(call.details)">{{ diagnosticText(call.details) }}</p></div>
          <p v-if="call.summary">{{ toolSummary(call.summary) }}</p>
          <code>{{ call.name }}</code>
          <small v-if="call.question_id" class="tool-question">问题 {{ call.question_id.slice(0,8) }}</small>
        </details>
      </TransitionGroup>
    </section>
    <section class="event-section">
      <div class="rail-heading"><h3>{{ replay ? '执行记录回放' : '持久化事件流' }}</h3><button class="text-button" :disabled="!activity?.events.length" @click="startReplay">{{ replay ? '返回实时' : '过程回放' }}</button></div>
      <div v-if="replay && activity?.events.length" class="replay-controls"><input v-model.number="cursor" type="range" min="0" :max="activity.events.length-1" aria-label="执行记录回放位置"><small>{{ cursor+1 }} / {{ activity.events.length }} 条 · 仅回放事件，结果图保持已发布版本</small></div>
      <p v-if="!events.length" class="rail-empty">从开始研判到发布，每一步均有据可查。</p>
      <TransitionGroup :key="`${executionIdentity}:${replay ? 'replay' : 'live'}`" name="event-list" tag="ol" class="event-timeline" :css="motion && listMotionEnabled && !replay">
        <li v-for="(event,index) in events" :key="`${event.run_id}:${event.seq}`" :class="{ failed: event.type.includes('failed'), complete: event.type.includes('completed') || event.type === 'run_succeeded' }" :style="{ '--arrival-index': Math.min(index,4) }">
          <span class="event-marker"></span><time>{{ clockTime(event.created_at) }}</time><strong>{{ eventLabel(event) }}</strong><small v-if="sourceProgressText(event)" class="source-progress-note">{{ sourceProgressText(event) }}</small><small v-if="scopeEventSummary(event, run?.run_id)" class="source-progress-note">{{ scopeEventSummary(event, run?.run_id) }}</small><small v-if="eventDiagnostic(event.data)" class="event-diagnostic">{{ eventDiagnostic(event.data) }}</small><small v-if="typeof event.data.summary === 'string'">{{ event.data.summary }}</small><span class="event-seq">#{{ String(event.seq).padStart(3,'0') }}</span>
        </li>
      </TransitionGroup>
    </section>
    <footer class="rail-footer"><span class="shield-icon">◇</span> 仅显示实际执行记录<br><small>不展示模型隐藏思考；未采集的数据留空。</small></footer>
  </aside>
</template>

<style scoped>
.call-failure{margin-top:9px;padding:8px;border-radius:4px;border:1px solid #775b3f55;background:#70432220;overflow-wrap:anywhere}.call-failure strong{font-size:9px;color:#e6bd93;line-height:1.9}.call-failure p{margin-top:4px;color:#be9d7d;font-size:9px}.repair-badge{font-size:7px;line-height:1.8;color:#d1b580}.event-diagnostic{color:#c5a987!important}
.tool-call .capacity-note{display:block;color:#ddc397;line-height:1.7;margin-top:5px;overflow-wrap:anywhere;font-size:9px}
.tool-call .batch-note{display:block;color:#9ccbc5;line-height:1.7;margin-top:5px;overflow-wrap:anywhere;font-size:9px}.source-progress-note{color:#9ccbc5!important}
</style>
