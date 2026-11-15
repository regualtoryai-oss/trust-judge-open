<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { api, messageOf } from "../api";
import { exportFailure, exportMatches, exportProgress, exportReady, readExportDownload } from "../export";
import type { ExportReadiness } from "../export";
import type { ExportJob } from "../types";

const props = defineProps<{ topicId: string; revisionId: string; questionCount: number; selectedQuestion?: string }>();
const readiness = ref<ExportReadiness | null>(null);
const job = ref<ExportJob | null>(null);
const checking = ref(false);
const creating = ref(false);
const downloading = ref<"preview" | "zip" | null>(null);
const error = ref("");
const notice = ref("");
let generation = 0;
let alive = true;
let pollTimer: ReturnType<typeof setTimeout> | undefined;
const processing = computed(() => creating.value || (job.value && ["queued", "running"].includes(job.value.status)));
const completed = computed(() => readiness.value?.completed_questions ?? 0);
const total = computed(() => readiness.value?.total_questions ?? props.questionCount);
const basePath = computed(() => `/topics/${encodeURIComponent(props.topicId)}/revisions/${encodeURIComponent(props.revisionId)}`);

function validContext(token: number, topic: string, revision: string): boolean {
  return alive && token === generation && topic === props.topicId && revision === props.revisionId;
}

async function check() {
  if (!props.revisionId || !props.topicId) return;
  const token = generation, topic = props.topicId, revision = props.revisionId;
  checking.value = true;
  error.value = "";
  try {
    const result = await api<ExportReadiness>(`${basePath.value}/export-readiness`);
    if (!validContext(token, topic, revision)) return;
    if (result.topic_id !== topic || result.revision_id !== revision) throw new Error("导出版本不一致，请刷新后重试。");
    readiness.value = result;
    if (result.existing_export && exportMatches(result.existing_export, topic, revision)) {
      job.value = result.existing_export;
      if (["queued", "running"].includes(job.value.status)) schedulePoll(token, topic, revision);
    } else job.value = null;
  } catch (reason) {
    if (validContext(token, topic, revision)) { readiness.value = null; error.value = `无法检查导出资格：${messageOf(reason)}`; }
  } finally {
    if (validContext(token, topic, revision)) checking.value = false;
  }
}

function schedulePoll(token: number, topic: string, revision: string) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => void poll(token, topic, revision), 1200);
}

async function poll(token: number, topic: string, revision: string) {
  if (!validContext(token, topic, revision) || !job.value) return;
  const exportId = job.value.export_id;
  try {
    const result = await api<ExportJob>(`/exports/${encodeURIComponent(exportId)}`);
    if (!validContext(token, topic, revision) || exportId !== job.value?.export_id) return;
    if (!exportMatches(result, topic, revision)) throw new Error("导出任务与当前版本不一致，请刷新后重试。");
    job.value = result;
    if (exportReady(result)) notice.value = "全专题提交包已准备好，点击“下载正式 ZIP”。";
    else if (result.status === "failed") error.value = exportFailure(result.error);
    else schedulePoll(token, topic, revision);
  } catch (reason) {
    if (validContext(token, topic, revision)) error.value = `读取打包状态失败：${messageOf(reason)} 点击“检查导出状态”可重新连接。`;
  }
}

async function create() {
  if (!readiness.value?.can_export || processing.value || checking.value || !props.revisionId) return;
  const token = generation, topic = props.topicId, revision = props.revisionId;
  creating.value = true;
  error.value = "";
  notice.value = "";
  try {
    const result = await api<ExportJob>(`/topics/${encodeURIComponent(topic)}/exports`, { revision_id: revision }, true);
    if (!validContext(token, topic, revision)) return;
    if (!exportMatches(result, topic, revision)) throw new Error("导出任务与当前版本不一致，请刷新后重试。");
    job.value = result;
    await poll(token, topic, revision);
  } catch (reason) {
    if (validContext(token, topic, revision)) error.value = `提交包未生成：${messageOf(reason)}`;
  } finally {
    if (validContext(token, topic, revision)) creating.value = false;
  }
}

async function download(kind: "preview" | "zip") {
  if (downloading.value || !props.revisionId) return;
  if (kind === "preview" && (!readiness.value || completed.value < 1)) return;
  if (kind === "zip" && (!job.value || !exportReady(job.value) || !exportMatches(job.value, props.topicId, props.revisionId))) return;
  const token = generation, topic = props.topicId, revision = props.revisionId;
  const path = kind === "preview" ? `/api${basePath.value}/submission-preview` : `/api/exports/${encodeURIComponent(job.value!.export_id)}/download`;
  downloading.value = kind;
  error.value = "";
  notice.value = "";
  try {
    const response = await fetch(path, { headers: { Accept: kind === "zip" ? "application/zip" : "application/json" } });
    const blob = await readExportDownload(response, kind === "zip" ? "application/zip" : "application/json");
    if (!validContext(token, topic, revision)) return;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = kind === "zip" ? `正式提交包-${revision.slice(0, 8)}.zip` : `非正式结果预览-${revision.slice(0, 8)}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    notice.value = `${kind === "zip" ? "正式 ZIP" : "非正式 JSON"}已准备下载，请在浏览器下载列表查看。`;
  } catch (reason) {
    if (validContext(token, topic, revision)) error.value = `下载未完成：${messageOf(reason)}`;
  } finally {
    if (validContext(token, topic, revision)) downloading.value = null;
  }
}

watch(() => [props.topicId, props.revisionId], () => {
  generation++;
  clearTimeout(pollTimer);
  readiness.value = null; job.value = null;
  checking.value = false; creating.value = false; downloading.value = null;
  error.value = ""; notice.value = "";
  void check();
}, { immediate: true });
onBeforeUnmount(() => { alive = false; generation++; clearTimeout(pollTimer); });
</script>

<template>
  <section class="export-panel" aria-label="导出结果">
    <div class="export-panel-heading">
      <div><span class="eyebrow">RESULT DELIVERY / 结果交付</span><h2>导出结果</h2></div>
      <div class="export-panel-heading-actions"><span class="export-count" :class="{ complete: readiness?.can_export }">已完成 {{ completed }} / {{ total }} 题</span><button class="text-button" :disabled="!revisionId || checking || creating" @click="check">{{ checking ? '检查中…' : '检查导出状态' }}</button></div>
    </div>
    <p class="export-scope">导出范围：当前发布版本的已完成结果<span v-if="revisionId" class="export-version" :title="revisionId"> · {{ revisionId.slice(0, 8) }}</span>。{{ selectedQuestion ? `左侧选中的 ${selectedQuestion} 及图视图筛选不改变文件范围。` : '图视图筛选不改变文件范围。' }}导出不调用模型。</p>
    <div class="export-options">
      <article class="export-option preview-option">
        <div class="export-option-copy"><h3>已完成结果 <span class="export-tag">非正式 JSON</span></h3><p>查看 {{ completed }} 题的结构化证据、结论、冲突和质量数据；供核对与评测。</p></div>
        <button class="button" :disabled="!readiness || completed < 1 || !!downloading" @click="download('preview')">{{ downloading === 'preview' ? '准备下载…' : '↓ 下载结果 JSON' }}</button>
      </article>
      <article class="export-option formal-option">
        <div class="export-option-copy"><h3>全专题提交包 <span class="export-tag formal">正式 ZIP</span></h3><p>全部 {{ total }} 题完成并通过原文、引用和结构校验后，生成竞赛格式文件与清单。</p></div>
        <div class="export-formal-actions"><button v-if="exportReady(job)" class="button primary" :disabled="!!downloading" @click="download('zip')">{{ downloading === 'zip' ? '准备下载…' : '↓ 下载正式 ZIP' }}</button><button v-else class="button" :disabled="!readiness?.can_export || !!processing || checking" @click="create">{{ processing ? exportProgress(job?.status || 'queued') : job?.status === 'failed' ? '重新生成正式 ZIP' : '生成正式 ZIP' }}</button></div>
      </article>
    </div>
    <p v-if="!revisionId" class="export-blocker">尚无已发布结果。完成问题研判后，可先下载结果 JSON；全专题全部完成后可生成正式 ZIP。</p>
    <div v-else-if="readiness && !readiness.can_export" class="export-blockers"><p v-for="blocker in readiness.blockers" :key="blocker.code">{{ blocker.message }}</p><details v-if="readiness.missing_questions.length"><summary>待完成 {{ readiness.missing_questions.length }} 题 · 查看题目</summary><ul><li v-for="question in readiness.missing_questions" :key="question.id"><b>{{ question.raw_question_id }}</b> {{ question.question }}</li></ul></details></div>
    <p v-if="job && processing" class="export-job-status" role="status">{{ exportProgress(job.status) }} · 任务 {{ job.export_id.slice(0, 8) }}。刷新页面后可恢复状态；生成导出不会调用模型。</p>
    <p v-if="error" class="export-error" role="alert">{{ error }}</p>
    <p v-if="notice" class="export-notice" role="status">{{ notice }}</p>
  </section>
</template>

<style scoped>
.export-panel{margin:0 0 22px;padding:20px;border:1px solid #31505d;border-radius:10px;background:linear-gradient(110deg,#14303a99,#0d202b);color:#bdd3dc}.export-panel-heading,.export-panel-heading-actions{display:flex;align-items:center;justify-content:space-between;gap:14px}.export-panel-heading h2{margin:5px 0 0;font-size:17px;color:#e2f3f4}.export-panel .eyebrow{font-size:8px;letter-spacing:1.4px;color:#6e9eac}.export-count{padding:5px 9px;border:1px solid #6a6045;border-radius:5px;background:#5f4c2726;font-size:11px;color:#e8c58d;white-space:nowrap}.export-count.complete{border-color:#378b78;color:#82e1c9;background:#26765d20}.export-panel .text-button{font-size:10px;color:#8acaba;white-space:nowrap}.export-scope{margin:10px 0 15px;font-size:10px;line-height:1.8;color:#7d9caa}.export-version{font-family:Consolas,monospace}.export-options{display:grid;grid-template-columns:1fr 1fr;gap:12px}.export-option{display:flex;align-items:center;justify-content:space-between;gap:18px;border:1px solid #2b4653;border-radius:7px;padding:15px;background:#0b1e2866}.formal-option{border-color:#326359;background:#17373136}.export-option h3{margin:0;font-size:12px;font-weight:600;color:#d5e9ed}.export-tag{display:inline-block;margin-left:7px;padding:3px 6px;font-size:8px;color:#a1b8c3;background:#3754614a;border-radius:3px;vertical-align:1px}.export-tag.formal{color:#91dec7;background:#316c5733}.export-option p{margin:7px 0 0;max-width:360px;font-size:10px;line-height:1.7;color:#7f9dab}.export-option .button{font-size:10px;white-space:nowrap;min-height:35px}.export-option .button:disabled{opacity:.45;cursor:not-allowed}.export-blocker,.export-blockers,.export-job-status,.export-error,.export-notice{font-size:10px;line-height:1.8;margin:13px 0 0}.export-blocker,.export-blockers{color:#d9b987}.export-blockers p{margin:0}.export-blockers details{margin-top:5px}.export-blockers summary{cursor:pointer;color:#afbdc4}.export-blockers ul{padding-left:20px;margin:7px 0 0}.export-blockers li{margin:5px 0}.export-blockers b{color:#e4cda7;font-family:Consolas,monospace}.export-error{color:#f1ab92;border-left:2px solid #d79b6b;padding-left:10px}.export-notice{color:#89ddc9}.export-job-status{color:#8cb6c0}@media(max-width:1120px){.export-options{grid-template-columns:1fr}.export-option-copy{max-width:75%}.export-option p{max-width:none}}@media(max-width:760px){.export-panel{padding:14px}.export-panel-heading{align-items:flex-start}.export-panel-heading-actions{flex-direction:column;align-items:flex-end;gap:7px}.export-panel-heading h2{font-size:15px}.export-option{padding:12px;gap:10px}.export-option-copy{max-width:none}.export-option h3{font-size:11px}.export-tag{display:block;width:max-content;margin:6px 0 0}.export-option p{font-size:9px}.export-option .button{font-size:9px;padding:8px}.export-scope{font-size:9px}.export-count{font-size:10px}}
</style>
