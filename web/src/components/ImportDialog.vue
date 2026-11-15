<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import { api, messageOf } from "../api";
import { decodeImportEntry, downloadImportTemplate, validateImportFileSelection } from "../import";
import type { ImportPreview, ImportResult } from "../import";

const emit = defineEmits<{ close: []; imported: [result: ImportResult] }>();
const dialog = ref<HTMLDialogElement | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const files = ref<File[]>([]);
const title = ref("");
const preview = ref<ImportPreview | null>(null);
const acknowledged = ref(false);
const pending = ref<"preview" | "commit" | "">("");
const error = ref("");
const dragging = ref(false);
const canCommit = computed(() => !!preview.value && !pending.value && (!preview.value.excluded_fields.length || acknowledged.value));
let returnFocus: HTMLElement | null = null;

function close() { if (!pending.value) emit("close"); }
function resetPreview() { preview.value = null; acknowledged.value = false; error.value = ""; }
function setFiles(selected: FileList | File[]) {
  if (pending.value) return;
  resetPreview();
  files.value = Array.from(selected);
  try { validateImportFileSelection(files.value); }
  catch (reason) { error.value = messageOf(reason); }
}
function selectFiles(event: Event) {
  const input = event.target as HTMLInputElement;
  if (input.files) setFiles(input.files);
  input.value = "";
}
function dropFiles(event: DragEvent) {
  dragging.value = false;
  if (event.dataTransfer?.files) setFiles(event.dataTransfer.files);
}
async function preflight() {
  if (pending.value) return;
  resetPreview();
  pending.value = "preview";
  try {
    validateImportFileSelection(files.value);
    const entries = await Promise.all(files.value.map(async file => decodeImportEntry(file.name, await file.arrayBuffer())));
    preview.value = await api<ImportPreview>("/imports/preview", { title: title.value.trim() || undefined, entries });
    await nextTick();
    dialog.value?.querySelector<HTMLElement>(".import-preview-heading")?.focus();
  } catch (reason) { error.value = messageOf(reason); }
  finally { pending.value = ""; }
}
async function commit() {
  if (!canCommit.value || !preview.value) return;
  pending.value = "commit";
  error.value = "";
  try {
    const result = await api<ImportResult>("/imports/commit", {
      preview_id: preview.value.preview_id,
      fingerprint: preview.value.fingerprint,
      acknowledge_excluded_fields: acknowledged.value,
    });
    emit("imported", result);
  } catch (reason) { error.value = messageOf(reason); }
  finally { pending.value = ""; }
}
onMounted(() => {
  returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  dialog.value?.showModal();
  fileInput.value?.focus();
});
onBeforeUnmount(() => { dialog.value?.close(); returnFocus?.focus(); });
</script>

<template>
  <dialog ref="dialog" class="import-dialog" aria-labelledby="import-title" @cancel.prevent="close">
    <div class="import-heading">
      <div><span class="eyebrow">COMPETITION DATA / 新赛题入口</span><h2 id="import-title">导入原始数据</h2><p>先校验，再创建专题。确认导入不会调用模型。</p></div>
      <button class="import-close" aria-label="关闭导入窗口" :disabled="!!pending" @click="close">×</button>
    </div>
    <div class="import-body">
      <ol class="import-steps" aria-label="导入步骤"><li :class="{ active: !preview }"><b>01</b>选择资料</li><li :class="{ active: preview }"><b>02</b>预检确认</li></ol>
      <template v-if="!preview">
        <div class="import-format"><strong>支持两种 JSON 格式</strong><p>单个 <code>input.json</code>（含 documents、questions）；或同时选择 <code>documents.json / raw_documents.json</code> 与 <code>questions.json / conclusions.json</code>。</p><small>仅保留原始文档与问题。答案、标准证据不参与运行；问题文件中的答案字段会在预检时列出并排除。</small></div>
        <label class="import-dropzone" :class="{ dragging }" @dragover.prevent="dragging = !pending" @dragleave.prevent="dragging = false" @drop.prevent="dropFiles">
          <span class="import-upload-icon" aria-hidden="true">↥</span><strong>{{ files.length ? '重新选择文件' : '点击选择，或将文件拖到这里' }}</strong><span>UTF-8 JSON · 最多 2 个文件 · 单个 6 MiB / 共 8 MiB</span>
          <input ref="fileInput" type="file" accept=".json,application/json" multiple :disabled="!!pending" aria-label="选择原始数据 JSON 文件" @change="selectFiles" />
        </label>
        <ul v-if="files.length" class="import-file-list"><li v-for="file in files" :key="file.name"><span>{{ file.name }}</span><small>{{ (file.size / 1024).toFixed(1) }} KiB</small></li></ul>
        <label class="import-title-label" for="import-topic-title">专题名称 <small>可选，留空使用文件内名称</small></label>
        <input id="import-topic-title" v-model="title" class="import-title-input" maxlength="120" placeholder="例如：赛题 01 · 某事件多源研判" :disabled="!!pending" />
        <p class="import-integrity">原始 doc_id、para_id、question_id 与原文保留。段落位置必须匹配原文的 Unicode 字符区间 [char_start, char_end)。</p>
      </template>
      <section v-else class="import-preview" aria-label="数据预检结果">
        <h3 class="import-preview-heading" tabindex="-1">{{ preview.title }}</h3>
        <div class="import-counts"><div><strong>{{ preview.counts.documents }}</strong><span>原始文档</span></div><div><strong>{{ preview.counts.paragraphs }}</strong><span>来源段落</span></div><div><strong>{{ preview.counts.questions }}</strong><span>研判问题</span></div></div>
        <p v-if="preview.existing_topic_id" class="import-tip">相同内容已存在。确认后将打开已有专题，保留已有资料与结果。</p>
        <p v-else class="import-tip">将创建独立专题，保留已有专题与历史结果。</p>
        <ul v-if="preview.warnings.length" class="import-warning-list"><li v-for="warning in preview.warnings" :key="warning">{{ warning }}</li></ul>
        <div v-if="preview.excluded_fields.length" class="import-exclusions"><strong>以下字段已从运行数据中排除</strong><ul><li v-for="(item, index) in preview.excluded_fields" :key="index"><code>{{ item.path }}</code><span>{{ item.fields.join('、') }} · {{ item.count }} 项</span></li></ul><label><input v-model="acknowledged" type="checkbox" :disabled="!!pending" />我已核对：上述字段不进入资料库、检索或模型输入。</label></div>
        <div class="import-question-preview"><h4>问题预览</h4><ol><li v-for="question in preview.questions" :key="question.question_id"><code>{{ question.question_id }}</code><span>{{ question.question }}</span></li></ol><p v-if="preview.truncated_questions_count">另有 {{ preview.truncated_questions_count }} 个问题，导入后可在左侧查看。</p></div>
      </section>
      <div v-if="error" class="import-error" role="alert"><strong>暂未导入</strong><p>{{ error }}</p><small>请修正对应文件后重新预检；已有专题不受影响。</small></div>
      <p v-if="pending" class="import-pending" role="status">{{ pending === 'preview' ? '正在预检文件与原文完整性…' : '正在提交已校验的数据，请稍候…' }}</p>
    </div>
    <footer class="import-footer">
      <button class="button import-template-button" :disabled="!!pending" @click="downloadImportTemplate">↓ 下载输入模板</button>
      <div><button class="button" :disabled="!!pending" @click="preview ? resetPreview() : close()">{{ preview ? '返回修改' : '取消' }}</button><button v-if="!preview" class="button primary" :disabled="!files.length || !!pending" @click="preflight">{{ pending === 'preview' ? '预检中…' : '预检数据 →' }}</button><button v-else class="button primary" :disabled="!canCommit" @click="commit">{{ pending === 'commit' ? '导入中…' : preview.existing_topic_id ? '打开已有专题' : '确认导入' }}</button></div>
    </footer>
  </dialog>
</template>

<style>
.import-entry{width:100%;display:flex;align-items:center;justify-content:space-between;margin:-10px 0 23px;border:1px solid #35625f;background:#133b3b;color:#a5eadb;padding:10px 12px;border-radius:6px;font-size:12px}.import-entry:hover{background:#1b4a46}.import-entry span:last-child{font-size:16px}
.batch-run-actions{display:flex;flex-direction:column;align-items:flex-end;gap:8px;flex-shrink:0}.batch-run-actions .button{white-space:nowrap}.batch-limit-note{font-size:10px;line-height:1.7;color:#cfb386;max-width:190px;text-align:right}
.import-dialog{padding:0;border:1px solid #37525f;border-radius:13px;background:#0c202b;color:#dce8ec;width:min(690px,calc(100vw - 32px));max-height:calc(100dvh - 42px);overflow:auto;box-shadow:0 25px 100px #0009}.import-dialog::backdrop{background:#020b14bd;backdrop-filter:blur(5px)}.import-heading{padding:24px 26px 19px;display:flex;align-items:flex-start;justify-content:space-between;gap:12px;border-bottom:1px solid #263d48}.import-heading h2{font-weight:500;font-size:23px;color:#edf5f2;margin:4px 0}.import-heading p{font-size:12px;color:#8ba6b4}.import-close{border:1px solid #2e4855;background:#142f3a;border-radius:5px;color:#a9c3ce;min-width:29px;height:29px;font-size:20px;line-height:1}.import-body{padding:19px 26px 21px}.import-steps{list-style:none;margin:0 0 20px;padding:0;display:flex;gap:24px;color:#607f8e;font-size:11px}.import-steps li{display:flex;align-items:center;gap:8px}.import-steps b{font:10px Consolas,monospace;border:1px solid #2b4653;border-radius:4px;padding:4px 5px;font-weight:400}.import-steps .active{color:#8de1d1}.import-steps .active b{border-color:#40827a;background:#183a3b}.import-format strong{font-size:12px;color:#c1d9de;font-weight:500}.import-format p{font-size:11px;line-height:1.9;color:#8ca7b4;margin:5px 0}.import-format code{color:#b9d1d6}.import-format small{display:block;color:#708f9e;font-size:10px;line-height:1.8}.import-dropzone{position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;margin:17px 0 12px;padding:23px 16px;border:1px dashed #3a6170;background:#102b36;border-radius:8px;cursor:pointer;text-align:center}.import-dropzone:hover,.import-dropzone.dragging,.import-dropzone:focus-within{background:#153a42;border-color:#69cfc1}.import-dropzone input{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer}.import-dropzone strong{font-size:13px;font-weight:500;color:#c6e7e2}.import-dropzone>span:last-of-type{font-size:10px;color:#7396a4}.import-upload-icon{font-size:24px;color:#68d8c7;line-height:1.2}.import-file-list{list-style:none;padding:0;margin:0 0 17px}.import-file-list li{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:7px 0;color:#b4cbd3;font-size:11px;border-bottom:1px solid #253e4a}.import-file-list li>span{overflow-wrap:anywhere;min-width:0}.import-file-list small{color:#7698a7;white-space:nowrap}.import-title-label{display:block;font-size:11px;color:#bbd2da;margin-bottom:7px}.import-title-label small{font-size:10px;color:#648596;margin-left:9px}.import-title-input{width:100%;border:1px solid #34515e;background:#091a24;border-radius:5px;padding:10px 11px;color:#c8dfe4;font-size:12px;outline-offset:3px}.import-title-input::placeholder{color:#4d6f80}.import-integrity{font-size:10px;line-height:1.9;color:#638797;margin-top:12px}.import-preview-heading{font-size:18px;font-weight:500;line-height:1.7;overflow-wrap:anywhere}.import-counts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:#29424e;border:1px solid #29424e;border-radius:7px;overflow:hidden;margin:15px 0 12px}.import-counts>div{padding:12px 15px;background:#112b35}.import-counts strong{display:block;font:25px/1.4 Consolas,monospace;color:#a6e4d7}.import-counts span{color:#7a9ca9;font-size:10px}.import-tip{font-size:11px;color:#80a8ad;margin:9px 0}.import-warning-list{color:#d9c39b;font-size:11px;line-height:1.9;padding-left:18px}.import-exclusions{padding:13px 14px;border:1px solid #68563d;background:#342e23;border-radius:6px;margin:14px 0;color:#e3c597}.import-exclusions>strong{font-size:12px;font-weight:500}.import-exclusions ul{list-style:none;padding:0;margin:8px 0}.import-exclusions li{display:flex;flex-direction:column;gap:2px;overflow-wrap:anywhere;font-size:10px;padding:5px 0;border-bottom:1px solid #68563d66}.import-exclusions code{color:#b5a689;font-size:9px}.import-exclusions label{display:flex;align-items:flex-start;gap:6px;font-size:11px;line-height:1.8;cursor:pointer}.import-exclusions input{accent-color:#d7b77e;margin-top:4px}.import-question-preview{margin-top:19px}.import-question-preview h4{font-size:11px;font-weight:500;color:#b3d1d6;margin-bottom:7px}.import-question-preview ol{list-style:none;padding:0;margin:0;max-height:230px;overflow:auto}.import-question-preview li{display:flex;gap:12px;align-items:flex-start;padding:10px 0;border-bottom:1px solid #25404d;font-size:11px}.import-question-preview code{font-size:9px;color:#5e91a5;min-width:50px;overflow-wrap:anywhere;max-width:100px}.import-question-preview li>span{min-width:0;overflow-wrap:anywhere;line-height:1.9;color:#a8c4ce}.import-question-preview>p{font-size:10px;color:#688b9c;margin-top:10px}.import-error{background:#38272a;border:1px solid #694349;border-radius:6px;padding:12px 14px;margin-top:15px;color:#edbbad;overflow-wrap:anywhere}.import-error strong{font-size:12px;font-weight:500}.import-error p{font-size:11px;line-height:1.9;margin:4px 0}.import-error small{font-size:10px;color:#b58f8e}.import-pending{font-size:11px;color:#80cdbf;margin-top:13px}.import-footer{padding:16px 26px;border-top:1px solid #29414d;display:flex;justify-content:space-between;gap:12px;background:#0b1d27;position:sticky;bottom:0}.import-footer>div{display:flex;gap:9px}.import-template-button{background:transparent!important;color:#7ba8b7!important;border-color:transparent!important;padding-left:0!important}.import-dialog button:focus-visible,.import-dialog input:focus-visible,.import-preview-heading:focus-visible{outline:2px solid #62ddc7;outline-offset:3px}.empty-library .import-empty-actions{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:20px}.empty-library .import-empty-actions .button{margin-top:0}
@media(max-width:760px){.import-entry{margin:0 0 15px;max-width:none}.import-heading{padding:19px 17px 15px}.import-heading h2{font-size:20px}.import-body{padding:16px 17px}.import-footer{padding:13px 17px;flex-wrap:wrap}.import-footer>div{margin-left:auto}.import-footer .button{font-size:10px;padding:7px 9px}.import-counts>div{padding:10px 12px}.import-title-label small{display:block;margin:3px 0 0}.import-dialog{max-height:calc(100dvh - 24px);width:calc(100vw - 20px)}}
</style>
