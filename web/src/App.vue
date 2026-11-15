<script setup lang="ts">
import { computed, markRaw, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { VueFlow } from "@vue-flow/core";
import type { Edge, Node, VueFlowStore } from "@vue-flow/core";
import { Background } from "@vue-flow/background";
import { Controls } from "@vue-flow/controls";
import JudgmentNode from "./components/JudgmentNode.vue";
import MissionControl from "./components/MissionControl.vue";
import ActivityLog from "./components/ActivityLog.vue";
import ImportDialog from "./components/ImportDialog.vue";
import ExportPanel from "./components/ExportPanel.vue";
import type { ImportResult } from "./import";
import { selectRunQuestions } from "./run-selection";
import { refineRequest, scopeProgressText } from "./quality-refine";
import { attachRun, isRunning } from "./runtime";
import { evidencePositions, focusEvidence } from "./graph-layout";
import { api, items, messageOf } from "./api";
import { codepointExcerpt } from "./source";
import { sourceContextExcerpt, sourceContextWindow } from "./source-context";
import {
  filterQuestionArtifacts,
  isConflictRelation,
  nodeText,
  relationLabels,
} from "./presentation";
import type {
  Artifacts,
  Activity,
  BusinessNode,
  ChangePreview,
  Evidence,
  Graph,
  Health,
  Paragraph,
  Question,
  Run,
  SourceContextBundle,
  Topic,
} from "./types";

const health = ref<Health | null>(null);
const topics = ref<Topic[]>([]);
const topicId = ref("");
const questions = ref<Question[]>([]);
const questionId = ref("");
const graph = ref<Graph | null>(null);
const artifacts = ref<Artifacts>({});
const revisionId = ref("");
const run = ref<Run | null>(null);
const runHistory = ref<Run[]>([]);
const activity = ref<Activity | null>(null);
const activityConnected = ref(true);
const loading = ref(false);
const importOpen = ref(false);
const busy = ref(false);
const motionEnabled = ref(true);
const systemReducedMotion = ref(false);
const pageVisible = ref(!document.hidden);
const motionAllowed = computed(() => motionEnabled.value && !systemReducedMotion.value);
let motionPreference: MediaQueryList | undefined;
function updateMotionPreference() { systemReducedMotion.value = motionPreference?.matches ?? false; }
function updatePageVisibility() { pageVisible.value = !document.hidden; }
function toggleMotion() {
  if (systemReducedMotion.value) return;
  motionEnabled.value = !motionEnabled.value;
  try { localStorage.setItem("trust-judge:motion", motionEnabled.value ? "on" : "off"); } catch { /* Optional presentation preference. */ }
}
const error = ref("");
const notice = ref("");
const tab = ref("graph");
const showAll = ref(false);
const compactGraph = ref(true);
let fitCurrentView: VueFlowStore["fitView"] | null = null;
const selectedNode = ref<BusinessNode | null>(null);
const evidence = ref<Evidence | null>(null);
const paragraph = ref<Paragraph | null>(null);
const inspectorLoading = ref(false);
const inspectorError = ref("");
const sourceContext = ref<SourceContextBundle | null>(null);
const sourceContextLoading = ref(false);
const sourceContextError = ref("");
const preview = ref<ChangePreview | null>(null);
const changeReason = ref("");
const previewBaseRevision = ref("");
let pollTimer: ReturnType<typeof setTimeout> | undefined;
let alive = true;
let contextGeneration = 0;
let inspectorGeneration = 0;
let sourceContextGeneration = 0;
function resetSourceContext() {
  sourceContextGeneration++;
  sourceContext.value = null;
  sourceContextLoading.value = false;
  sourceContextError.value = "";
}
watch([selectedNode, topicId, questionId, revisionId], () => {
  inspectorGeneration++;
  resetSourceContext();
  evidence.value = null;
  paragraph.value = null;
  inspectorLoading.value = false;
  inspectorError.value = "";
}, { flush: "sync" });
const resumeKeys = new Map<string, string>();
const nodeTypes = { judgment: markRaw(JudgmentNode) };
const activeTopic = computed(() =>
  topics.value.find((t) => t.id === topicId.value),
);
const activeQuestion = computed(() =>
  questions.value.find((q) => q.id === questionId.value),
);
const activeRun = computed(() => isRunning(run.value));
const topicHasActiveRun = computed(() => runHistory.value.some(isRunning) || activeRun.value);
const runQuestionLabel = computed(() => run.value?.question_ids?.map(id => questions.value.find(item => item.id === id)?.raw_question_id || id.slice(0,8)).join("、") || "");
const canRun = computed(
  () =>
    !!health.value?.model_configured &&
    !!health.value?.workflow_available &&
    !!questionId.value &&
    !topicHasActiveRun.value &&
    !busy.value,
);
const questionLimit = computed(() => Number.isInteger(health.value?.max_questions_per_run) && (health.value?.max_questions_per_run || 0) > 0 ? health.value!.max_questions_per_run! : 20);
const allQuestionError = computed(() => {
  try { selectRunQuestions(questions.value.map(item => item.id), "all", questionLimit.value); return ""; }
  catch (reason) { return messageOf(reason); }
});
const canRunAll = computed(() => canRun.value && !allQuestionError.value);
const publishedView = computed(
  () =>
    !!revisionId.value &&
    revisionId.value === activeTopic.value?.published_revision_id,
);
const tabs = [
  { id: "graph", label: "证据星图" },
  { id: "conclusions", label: "结论" },
  { id: "agents", label: "独立研判" },
  { id: "conflicts", label: "冲突记录" },
  { id: "confidence", label: "质量依据" },
];
const visibleNodes = computed(() => {
  const nodes = graph.value?.nodes ?? [];
  if (showAll.value || !questionId.value) return nodes;
  const keep = new Set(
    nodes
      .filter((n) => n.question_id === questionId.value)
      .map((n) => n.id ?? n.node_id ?? ""),
  );
  const edges = graph.value?.edges ?? [];
  let changed = true;
  while (changed) {
    changed = false;
    for (const edge of edges) {
      if (keep.has(edge.to_node_id) && !keep.has(edge.from_node_id)) {
        keep.add(edge.from_node_id);
        changed = true;
      }
    }
  }
  return nodes.filter((n) => keep.has(n.id ?? n.node_id ?? ""));
});
const graphEvidenceCount = computed(() => visibleNodes.value.filter(node => node.kind === "evidence").length);
const displayedNodes = computed(() => compactGraph.value ? focusEvidence(visibleNodes.value, graph.value?.edges || []) : visibleNodes.value);
const displayedEvidenceCount = computed(() => displayedNodes.value.filter(node => node.kind === "evidence").length);
const flowNodes = computed<Node[]>(() => {
  const nodes = displayedNodes.value;
  const positions = evidencePositions(nodes, graph.value?.edges || []);
  return nodes.map((n, index) => {
    const id = n.id ?? n.node_id ?? "";
    const selectedId = selectedNode.value?.id ?? selectedNode.value?.node_id;
    return {
      id,
      type: "judgment",
      position: positions.get(id) || {x:20,y:35},
      data: n,
      class: [preview.value?.affected_node_ids.includes(id) ? "impact-highlight" : "", selectedId === id ? "is-focused" : ""].filter(Boolean).join(" "),
      style: { "--node-enter-delay": `${Math.min(index, 8) * 32}ms` },
    };
  });
});
const flowEdges = computed<Edge[]>(() => {
  const ids = new Set(flowNodes.value.map((n) => n.id));
  return (graph.value?.edges ?? [])
    .filter((e) => ids.has(e.from_node_id) && ids.has(e.to_node_id))
    .map((e, i) => ({
      id: `edge-${i}`,
      source: e.from_node_id,
      target: e.to_node_id,
      class: [
        "edge-live",
        selectedNode.value ? ((selectedNode.value.id ?? selectedNode.value.node_id) === e.from_node_id || (selectedNode.value.id ?? selectedNode.value.node_id) === e.to_node_id ? "edge-focus" : "edge-muted") : "",
      ].filter(Boolean).join(" "),
      label: isConflictRelation(e.relation) || e.relation === "prerequisite" || selectedNode.value?.id === e.from_node_id || selectedNode.value?.id === e.to_node_id ? relationLabels[e.relation] ?? e.relation : undefined,
      sourceHandle: e.relation === "prerequisite" ? "bottom" : "right",
      targetHandle: e.relation === "prerequisite" ? "top" : "left",
      style: {
        stroke: isConflictRelation(e.relation) ? "#e8aa67" : "#458c9f",
        strokeWidth: 1.6,
      },
      labelStyle: { fill: "#b4d2df", fontSize: 10 },
      labelBgStyle: { fill: "#102536", fillOpacity: 0.9 },
      animated: motionAllowed.value && !!selectedNode.value && ((selectedNode.value.id ?? selectedNode.value.node_id) === e.from_node_id || (selectedNode.value.id ?? selectedNode.value.node_id) === e.to_node_id),
      markerEnd: "arrowclosed",
    }));
});
async function fitGraph() {
  await nextTick();
  if (tab.value !== "graph" || !fitCurrentView) return;
  await fitCurrentView({padding:0.06,minZoom:compactGraph.value ? 0.6 : 0.25,maxZoom:1.1,duration:0});
}
function initializeGraph(store: VueFlowStore) { fitCurrentView = store.fitView; void fitGraph(); }
watch([revisionId, questionId, showAll, compactGraph], () => { void fitGraph(); }, {flush:"post"});
const excerpt = computed(() => {
  if (!paragraph.value || !evidence.value) return null;
  const { start_cp: start, end_cp: end, quote } = evidence.value;
  if (evidence.value.span_scope === "document") {
    if (!Number.isSafeInteger(paragraph.value.start_cp)) return null;
    return codepointExcerpt(paragraph.value.text, quote, start - paragraph.value.start_cp!, end - paragraph.value.start_cp!);
  }
  return codepointExcerpt(paragraph.value.text, quote, start, end);
});
const contextExcerpt = computed(() => sourceContext.value && evidence.value
  ? sourceContextExcerpt(sourceContext.value, evidence.value) : null);
const artifactRows = computed(() => {
  const key = tab.value === "agents" ? "agent_outputs" : tab.value;
  const value = artifacts.value[key];
  const rows = Array.isArray(value)
    ? value
    : value && typeof value === "object"
      ? Object.entries(value).map(([id, detail]) => ({ id, detail }))
      : [];
  if (showAll.value || !activeQuestion.value) return rows;
  return filterQuestionArtifacts(
    rows,
    activeQuestion.value,
    visibleNodes.value,
  );
});
function titleOf(value: unknown, index: number): string {
  if (!value || typeof value !== "object") return `记录 ${index + 1}`;
  const row = value as Record<string, unknown>;
  if (row.object_type === "conclusion") return "结论的可核查质量";
  if (row.object_type === "evidence") {
    const ref = row.object_ref as Record<string, unknown> | undefined;
    return typeof ref?.quote === "string"
      ? `证据质量 · ${ref.quote}`
      : "证据的可核查质量";
  }
  for (const key of [
    "statement",
    "conclusion",
    "fact_des",
    "summary",
    "content",
  ]) {
    if (typeof row[key] === "string" && row[key].trim()) return row[key];
  }
  const roles: Record<string, string> = {
    authority_analyst: "研判 A · 来源与可核验性",
    cross_validation_analyst: "研判 B · 独立印证与反例",
    arbitrate: "仲裁",
    baseline: "单 Agent 基线",
  };
  if (typeof row.role === "string") return roles[row.role] ?? row.role;
  if (row.side_a && typeof row.side_a === "object") {
    const side = row.side_a as Record<string, unknown>;
    if (typeof side.position === "string") return side.position;
  }
  return `记录 ${index + 1}`;
}
function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

async function refresh() {
  loading.value = true;
  error.value = "";
  try {
    const results = await Promise.allSettled([
      api<Health>("/health"),
      api<Topic[] | { items: Topic[] }>("/topics"),
    ]);
    const [h, t] = results;
    if (h.status === "fulfilled") health.value = h.value;
    else {
      health.value = null;
      error.value = messageOf(h.reason);
    }
    if (t.status === "fulfilled") {
      topics.value = items(t.value);
      if (!topicId.value && topics.value[0]) {
        const preferred = localStorage.getItem("trust-judge:selected-topic");
        await chooseTopic(topics.value.find(item => item.id === preferred)?.id || topics.value[0].id);
      } else {
        const published = activeTopic.value?.published_revision_id;
        if (published && published !== revisionId.value) {
          const generation = contextGeneration;
          try {
            await loadRevision(published, generation);
            if (generation !== contextGeneration) return;
            selectedNode.value = null;
            evidence.value = null;
            paragraph.value = null;
            preview.value = null;
            notice.value = "已加载最新完成版本。";
          } catch (e) {
            if (generation === contextGeneration) error.value = messageOf(e);
          }
        }
      }
      if (topicId.value) await loadRuns(contextGeneration, true);
    } else error.value = messageOf(t.reason);
  } finally {
    loading.value = false;
  }
}
async function chooseTopic(id: string) {
  if (id === topicId.value && questions.value.length) return;
  contextGeneration++;
  const generation = contextGeneration;
  clearTimeout(pollTimer);
  topicId.value = id;
  localStorage.setItem("trust-judge:selected-topic", id);
  questionId.value = "";
  questions.value = [];
  graph.value = null;
  artifacts.value = {};
  run.value = null;
  runHistory.value = [];
  activity.value = null;
  revisionId.value = "";
  selectedNode.value = null;
  evidence.value = null;
  paragraph.value = null;
  preview.value = null;
  error.value = "";
  notice.value = "";
  busy.value = true;
  try {
    const result = await api<Question[] | { items: Question[] }>(
      `/topics/${id}/questions`,
    );
    if (generation !== contextGeneration) return;
    questions.value = items(result);
    questionId.value = questions.value[0]?.id ?? "";
    const revision = activeTopic.value?.published_revision_id;
    if (revision) await loadRevision(revision, generation);
    await loadRuns(generation);
  } catch (e) {
    if (generation === contextGeneration) error.value = messageOf(e);
  } finally {
    if (generation === contextGeneration) busy.value = false;
  }
}
async function importSucceeded(result: ImportResult) {
  importOpen.value = false;
  try {
    topics.value = items(await api<Topic[] | { items: Topic[] }>("/topics"));
    tab.value = "graph";
    showAll.value = false;
    await chooseTopic(result.topic_id);
    notice.value = `${result.idempotent ? "已打开已有专题" : "新专题导入完成"}：${result.counts.documents} 篇文档、${result.counts.paragraphs} 段原文、${result.counts.questions} 个问题。选择问题后点击“开始研判”；尚未调用模型。`;
  } catch (reason) {
    error.value = `资料已导入，但读取专题失败：${messageOf(reason)}。点击“刷新”后可选择该专题。`;
  }
}
async function loadRevision(id: string, generation = contextGeneration) {
  const [g, a] = await Promise.all([
    api<Graph>(`/topics/${topicId.value}/revisions/${id}/graph`),
    api<Artifacts>(`/topics/${topicId.value}/revisions/${id}/artifacts`),
  ]);
  if (generation !== contextGeneration) return;
  graph.value = g;
  artifacts.value = a;
  revisionId.value = id;
}
const canRefine = computed(() => canRun.value && !!revisionId.value &&
  Array.isArray(artifacts.value.candidate_artifacts) && artifacts.value.candidate_artifacts.some(
    item => item && typeof item === "object" && (item as Record<string,unknown>).question_id === questionId.value && (item as Record<string,unknown>).can_refine === true));
const qualityReview = computed(() => Array.isArray(artifacts.value.quality_reviews) ? artifacts.value.quality_reviews.find(
  item => item && typeof item === "object" && (item as Record<string,unknown>).question_id === questionId.value) as Record<string,unknown> | undefined : undefined);
const scopeProgress = computed(() => scopeProgressText(activity.value, questionId.value, run.value?.run_id));
async function startQualityRefine() {
  if (!canRefine.value || !activeTopic.value) return;
  busy.value = true; error.value = ""; notice.value = "";
  try {
    run.value = await api<Run>(`/topics/${topicId.value}/runs`,
      refineRequest(questionId.value, activeTopic.value.head_revision_id, revisionId.value), true);
    activity.value = null; runHistory.value = [run.value, ...runHistory.value];
    notice.value = "已新建质量复核任务；导入固定候选，不重新扫描，接下来的模型调用另计用量。";
    await pollRun(contextGeneration);
  } catch (reason) { error.value = messageOf(reason); }
  finally { busy.value = false; }
}
async function startRun() {
  await submitRun([questionId.value]);
}
async function startAllQuestions() {
  if (!canRunAll.value) return;
  await submitRun("all");
}
async function submitRun(selection: "all" | readonly string[]) {
  if (!canRun.value || !activeTopic.value) return;
  let ids: string[];
  try { ids = selectRunQuestions(questions.value.map(item => item.id), selection, questionLimit.value); }
  catch (reason) { error.value = messageOf(reason); return; }
  busy.value = true;
  error.value = "";
  notice.value = "";
  preview.value = null;
  try {
    run.value = await api<Run>(
      `/topics/${topicId.value}/runs`,
      {
        question_ids: ids,
        include_baseline: false,
        base_revision_id: activeTopic.value.head_revision_id,
        mode: "initial",
      },
      true,
    );
    activity.value = null;
    runHistory.value = [run.value, ...runHistory.value];
    await pollRun(contextGeneration);
  } catch (e) {
    error.value = messageOf(e);
  } finally {
    busy.value = false;
  }
}
function schedulePoll(generation: number) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => void pollRun(generation), activeRun.value ? 1200 : 6500);
}
async function loadRuns(generation: number, keepSelection = false) {
  const history = await api<Run[]>(`/topics/${topicId.value}/runs`);
  if (generation !== contextGeneration) return;
  runHistory.value = history;
  const attached = attachRun(history, keepSelection ? run.value?.run_id : undefined);
  if (!attached) return;
  if (attached.run_id !== run.value?.run_id) activity.value = null;
  run.value = attached;
  await pollRun(generation);
}
async function selectRun(id: string) {
  const live = runHistory.value.find(isRunning);
  if (live && live.run_id !== id) {
    notice.value = "当前专题仍有任务执行中，控制台保持连接；暂停或结束后可查看历史任务。";
    return;
  }
  clearTimeout(pollTimer);
  run.value = runHistory.value.find(item => item.run_id === id) || null;
  activity.value = null;
  await pollRun(contextGeneration);
}
async function pollRun(generation: number) {
  if (!alive || generation !== contextGeneration || !run.value) return;
  try {
    const selected = run.value.run_id;
    const observed = await api<Activity>(`/runs/${selected}/activity`);
    if (generation !== contextGeneration || selected !== run.value?.run_id) return;
    const previousStatus = run.value.status;
    const result = observed.run;
    activity.value = observed;
    activityConnected.value = true;
    run.value = result;
    runHistory.value = runHistory.value.map(item => item.run_id === result.run_id ? result : item);
    if (!isRunning(result) && previousStatus !== result.status) {
      const refreshedTopics = items(
        await api<Topic[] | { items: Topic[] }>("/topics"),
      );
      if (generation !== contextGeneration) return;
      topics.value = refreshedTopics;
      const published = activeTopic.value?.published_revision_id;
      if (published && published !== revisionId.value) await loadRevision(published, generation);
      if (["completed", "succeeded", "ready"].includes(result.status))
        notice.value =
          "研判完成，已加载通过校验的发布版本。";
    }
  } catch (e) {
    if (generation === contextGeneration) activityConnected.value = false;
  } finally {
    if (alive && generation === contextGeneration) schedulePoll(generation);
  }
}
async function pauseRun() {
  if (!run.value || busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    run.value = await api<Run>(`/runs/${run.value.run_id}/pause`, {});
    notice.value = "已请求暂停。完成的阶段将保留，执行器确认后可继续。";
    await pollRun(contextGeneration);
  } catch (e) {
    error.value = messageOf(e);
  } finally {
    busy.value = false;
  }
}
async function resumeRun() {
  if (!run.value?.resumable || busy.value || activeRun.value) return;
  busy.value = true;
  error.value = "";
  try {
    const sourceId = run.value.run_id;
    const requestKey = resumeKeys.get(sourceId) || crypto.randomUUID();
    resumeKeys.set(sourceId, requestKey);
    run.value = await api<Run>(`/runs/${sourceId}/resume`, {}, requestKey);
    activity.value = null;
    runHistory.value = [run.value, ...runHistory.value];
    notice.value = run.value.resume_mode === "start_unstarted" ? "已创建续接任务，从尚未开始的阶段执行。" : "已创建续接任务，完成的检查点将被复用。";
    await pollRun(contextGeneration);
  } catch (e) { error.value = messageOf(e); }
  finally { busy.value = false; }
}
async function inspectNode(node: Node) {
  selectedNode.value = node.data as BusinessNode;
  const request = ++inspectorGeneration;
  resetSourceContext();
  evidence.value = null;
  paragraph.value = null;
  inspectorLoading.value = false;
  inspectorError.value = "";
  preview.value = null;
  changeReason.value = "";
  if (!selectedNode.value.evidence_id) return;
  inspectorLoading.value = true;
  const selectedId = selectedNode.value.id ?? selectedNode.value.node_id;
  const evidenceId = selectedNode.value.evidence_id;
  const generation = contextGeneration;
  const isCurrent = () => alive && generation === contextGeneration && request === inspectorGeneration &&
    (selectedNode.value?.id ?? selectedNode.value?.node_id) === selectedId && selectedNode.value?.evidence_id === evidenceId;
  try {
    const value = await api<Evidence>(
      `/evidence/${encodeURIComponent(evidenceId)}?revision_id=${encodeURIComponent(revisionId.value)}`,
    );
    if (!isCurrent()) return;
    const source = await api<Paragraph>(`/paragraphs/${encodeURIComponent(value.paragraph_id)}`);
    if (!isCurrent()) return;
    evidence.value = value;
    paragraph.value = source;
    if (value.span_scope === "document") await openSourceContext();
  } catch (e) {
    if (isCurrent()) inspectorError.value = messageOf(e);
  } finally {
    if (isCurrent()) inspectorLoading.value = false;
  }
}
async function openSourceContext() {
  const selectedEvidence = evidence.value;
  if (!selectedEvidence || sourceContextLoading.value) return;
  const paragraphId = selectedEvidence.paragraph_id;
  const generation = contextGeneration;
  const inspectorRequest = inspectorGeneration;
  const request = ++sourceContextGeneration;
  const isCurrent = () => alive && generation === contextGeneration && inspectorRequest === inspectorGeneration &&
    request === sourceContextGeneration && evidence.value === selectedEvidence && evidence.value?.paragraph_id === paragraphId;
  sourceContextLoading.value = true;
  sourceContextError.value = "";
  sourceContext.value = null;
  try {
    const window = sourceContextWindow(selectedEvidence);
    const value = await api<SourceContextBundle>(`/paragraphs/${encodeURIComponent(paragraphId)}/context?before=${window.before}&after=${window.after}`);
    if (!isCurrent()) return;
    if (!value || value.anchor_paragraph_id !== paragraphId) throw new Error("相邻原文的锚点与当前来源不一致，未显示其他来源的上下文。");
    if (typeof value.quote !== "string" || !Array.isArray(value.members) || !value.members.length ||
        value.members.some(member => !member || typeof member.paragraph_id !== "string" || typeof member.raw_para_id !== "string")) {
      throw new Error("相邻原文缺少完整正文或范围成员，暂不能显示。请重新读取。");
    }
    sourceContext.value = value;
  } catch (reason) {
    if (isCurrent()) sourceContextError.value = messageOf(reason);
  } finally {
    if (isCurrent()) sourceContextLoading.value = false;
  }
}
async function previewExclusion() {
  if (!evidence.value || !changeReason.value.trim() || !activeTopic.value)
    return;
  busy.value = true;
  error.value = "";
  try {
    previewBaseRevision.value = activeTopic.value.head_revision_id ?? "";
    preview.value = await api<ChangePreview>(
      `/topics/${topicId.value}/changes/preview`,
      {
        base_revision_id: previewBaseRevision.value,
        operations: [
          {
            op: "exclude_evidence",
            evidence_id: selectedNode.value?.evidence_id,
          },
        ],
        reason: changeReason.value.trim(),
      },
    );
  } catch (e) {
    error.value = messageOf(e);
  } finally {
    busy.value = false;
  }
}
async function commitChange() {
  if (!preview.value) return;
  busy.value = true;
  error.value = "";
  try {
    run.value = await api<Run>(
      `/topics/${topicId.value}/changes`,
      {
        preview_id: preview.value.preview_id,
        preview_content_hash: preview.value.content_hash,
        base_revision_id: previewBaseRevision.value,
      },
      true,
    );
    preview.value = null;
    notice.value = "修订已提交，独立研判与下游判断将按影响范围重算。";
    schedulePoll(contextGeneration);
  } catch (e) {
    error.value = messageOf(e);
  } finally {
    busy.value = false;
  }
}
function chooseQuestion(id: string) {
  questionId.value = id;
  selectedNode.value = null;
  evidence.value = null;
  paragraph.value = null;
  preview.value = null;
}
async function reconnect() {
  await refresh();
}
onMounted(() => {
  try { motionEnabled.value = localStorage.getItem("trust-judge:motion") !== "off"; } catch { /* Storage can be disabled. */ }
  motionPreference = window.matchMedia("(prefers-reduced-motion: reduce)");
  updateMotionPreference();
  motionPreference.addEventListener("change", updateMotionPreference);
  document.addEventListener("visibilitychange", updatePageVisibility);
  void refresh();
});
onBeforeUnmount(() => {
  alive = false;
  clearTimeout(pollTimer);
  motionPreference?.removeEventListener("change", updateMotionPreference);
  document.removeEventListener("visibilitychange", updatePageVisibility);
});
</script>

<template>
  <div class="workbench shell-enter" :class="{ 'motion-still': !motionAllowed, 'page-inactive': !pageVisible }">
    <header class="app-header">
      <div class="brand">
        <span class="brand-mark"><svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 3 28 10v13L16 30 4 23V10Z M4 10l12 7 12-7 M16 17v13 M16 3v14" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="16" cy="17" r="3" fill="currentColor"/></svg></span>
        <div><strong>证据研判<span class="brand-separator"> / </span>指挥台</strong><span>EVIDENCE INTELLIGENCE WORKSPACE</span></div>
      </div>
      <div class="header-right">
        <span class="foundation-label">{{ health?.source_bundle_enabled ? (health?.source_bundle_front_framework === 'langchain' ? 'LANGCHAIN + LANGGRAPH' : 'DSH + LANGGRAPH') : 'LANGGRAPH ENGINE' }}</span
        ><span v-if="health?.model_configured && health.model_name" class="foundation-label">{{ health.model_name }}</span
        ><span class="service-state" :class="{ connected: health }"
          ><i></i>{{ health ? "本地服务已连接" : "服务未连接" }}</span
        ><button class="motion-toggle" :aria-pressed="motionAllowed" :disabled="systemReducedMotion" :aria-label="systemReducedMotion ? '系统已开启减少动态效果' : motionAllowed ? '关闭界面动效' : '开启界面动效'" :title="systemReducedMotion ? '跟随系统减少动态效果设置' : '仅切换界面动效，不影响研判任务'" @click="toggleMotion"><span aria-hidden="true">{{ motionAllowed ? '∿' : '—' }}</span>{{ motionAllowed ? '动效已开' : '静态模式' }}</button
        ><button class="header-refresh" :disabled="loading" @click="reconnect">
          {{ loading ? "连接中…" : "刷新" }}
        </button>
      </div>
    </header>
    <Transition name="notice-reveal"><div v-if="error" class="message-bar error" role="alert">
      <span>{{ error }}</span
      ><button aria-label="关闭错误" @click="error = ''">×</button>
    </div></Transition>
    <Transition name="notice-reveal"><div v-if="notice" class="message-bar notice" role="status">
      <span>{{ notice }}</span
      ><button aria-label="关闭提示" @click="notice = ''">×</button>
    </div></Transition>
    <div class="body-grid">
      <aside class="sidebar">
        <div class="side-heading">
          <span class="eyebrow">案件资料库</span
          ><small>{{ topics.length }} 个专题</small>
        </div>
        <button class="import-entry" :disabled="busy" @click="importOpen = true"><span>导入新赛题</span><span aria-hidden="true">＋</span></button>
        <label class="field-label" for="topic">当前专题</label>
        <select
          id="topic"
          :value="topicId"
          :disabled="!topics.length || busy"
          @change="chooseTopic(($event.target as HTMLSelectElement).value)"
        >
          <option v-if="!topics.length" value="">暂无已导入专题</option>
          <option v-for="topic in topics" :key="topic.id" :value="topic.id">
            {{ topic.title }}
          </option>
        </select>
        <div v-if="activeTopic" class="source-counts">
          <div>
            <strong>{{ activeTopic.document_count }}</strong
            ><span>文档</span>
          </div>
          <div>
            <strong>{{ activeTopic.paragraph_count }}</strong
            ><span>段落</span>
          </div>
          <div>
            <strong>{{ activeTopic.question_count }}</strong
            ><span>问题</span>
          </div>
        </div>
        <div class="question-heading">
          <h2>研判问题</h2>
          <span>{{ questions.length.toString().padStart(2, "0") }}</span>
        </div>
        <nav class="question-list" aria-label="研判问题">
          <button
            v-for="(question, i) in questions"
            :key="question.id"
            :class="{ selected: questionId === question.id }"
            @click="chooseQuestion(question.id)"
          >
            <span class="question-index">{{
              (i + 1).toString().padStart(2, "0")
            }}</span>
            <div>
              <small>{{ question.raw_question_id }}</small
              ><span>{{ question.question }}</span>
            </div>
          </button>
        </nav>
        <p v-if="!questions.length" class="sidebar-empty">
          导入专题后，问题会显示在这里。
        </p>
        <div class="sidebar-foot">
          <span class="tiny-label">SOURCE INTEGRITY</span>
          <p>原始资料只读<br />每一条判断，都能回到原文</p>
          <small v-if="health">API {{ health.version }}</small>
        </div>
      </aside>
      <main class="main-area">
        <section class="workspace-heading">
          <div>
            <span class="eyebrow">INVESTIGATION / 当前研判</span>
            <h1>{{ activeTopic?.title || "先导入一组真实资料" }}</h1>
            <p>
              {{
                activeQuestion?.question ||
                "从专题问题出发，查看证据、独立判断与来源。"
              }}
            </p>
          </div>
          <div class="run-actions batch-run-actions">
            <button
              class="button primary"
              :disabled="!canRun"
              @click="startRun"
            >
              {{ busy ? "处理中…" : topicHasActiveRun ? "任务执行中" : "↗ 开始研判" }}
            </button>
            <button v-if="revisionId" class="button" :disabled="!canRefine" title="新建付费复核任务：复用此已发布版本的完整候选，逐项严格复核；本任务不重新扫描原文" @click="startQualityRefine">严格复核已有候选</button>
            <button v-if="questions.length > 1" class="button" :disabled="!canRunAll" :title="allQuestionError || '为当前专题的全部问题创建一个任务；点击后开始调用模型'" @click="startAllQuestions">研判全部问题（{{ questions.length }}）</button>
            <small v-if="questions.length > questionLimit" class="batch-limit-note">单任务最多 {{ questionLimit }} 题，请逐题研判或拆分专题分批执行。</small>
          </div>
        </section>
        <p v-if="run?.question_ids?.length && !run.question_ids.includes(questionId)" class="scope-review-progress" role="status">任务控制台属于 {{ runQuestionLabel }}；当前查看 {{ activeQuestion?.raw_question_id }}。选择任务所含的问题，可查看对应事实复核进度。</p>
        <p v-if="scopeProgress" class="scope-review-progress" role="status">{{ scopeProgress }} · 审核覆盖不代表精确率或召回率</p>
        <details v-if="qualityReview && !run?.strict_evidence_enabled" class="technical-detail"><summary>事实质量复核：通过 {{ qualityReview.verified_fact_view_count }} 个 · 待处理 {{ qualityReview.unresolved_count }} 个</summary><p>原候选完整保留；修正后事实需独立二验。人工材料保留，未通过项单列待确认。</p><pre>{{ pretty(qualityReview) }}</pre></details>
        <section
          v-if="!health?.model_configured || !health?.workflow_available"
          class="configuration-note"
        >
          <span class="note-icon">!</span>
          <div>
            <strong>{{
              !health
                ? "等待后端服务连接"
                : !health.model_configured
                  ? "尚未配置真实模型"
                  : "编排组件尚未就绪"
            }}</strong>
            <p>
              {{
                !health
                  ? "启动 FastAPI 服务后点击右上角“刷新”。"
                  : !health.model_configured
                    ? "可浏览已导入资料。完成后端模型配置并启动 worker 后，才能开始研判。"
                    : "请检查 LangGraph 运行依赖与后端启动日志。"
              }}
            </p>
          </div>
          <span class="pill neutral">研判未启用</span>
        </section>
        <MissionControl :run="run" :activity="activity" :busy="busy" :connected="activityConnected" :has-published="!!revisionId" :result-is-selected-run="!!run && revisionId === run.target_revision_id" :question-label="runQuestionLabel" :question-id="questionId" :can-retry="canRun" @pause="pauseRun" @resume="resumeRun" @retry="startRun" />
        <ExportPanel v-if="activeTopic" :topic-id="topicId" :revision-id="revisionId" :question-count="activeTopic.question_count" :selected-question="activeQuestion?.raw_question_id" />
        <section v-if="!topics.length" class="empty-library">
          <div class="empty-symbol">⌁</div>
          <span class="eyebrow">尚无运行数据</span>
          <h2>从专题原文开始</h2>
          <p>
            上传原始文档和问题，预检确认后创建专题。<br />工作台不会使用样例答案生成展示结果。
          </p>
          <div class="empty-steps">
            <span><b>01</b> 导入原始文档</span
            ><span><b>02</b> 校验问题与编号</span
            ><span><b>03</b> 配置模型并研判</span>
          </div>
          <div class="import-empty-actions"><button class="button primary" @click="importOpen = true">导入新赛题</button><button class="button" :disabled="loading" @click="reconnect">重新读取专题</button></div>
        </section>
        <section v-else class="results-shell">
          <div class="results-toolbar">
            <div class="tab-list" role="tablist" aria-label="研判结果">
              <button
                v-for="item in tabs"
                :key="item.id"
                role="tab"
                :aria-selected="tab === item.id"
                :class="{ active: tab === item.id }"
                @click="tab = item.id"
              >
                {{ item.label }}
              </button>
            </div>
            <span v-if="revisionId" class="revision-label" :title="revisionId">版本 {{ revisionId.slice(0, 8) }}</span>
          </div>
          <template v-if="graph">
            <div
              v-if="tab === 'graph'"
              class="graph-workspace"
              :class="{ 'with-inspector': selectedNode }"
            >
              <div class="graph-canvas" :class="{ 'is-tracing': !!selectedNode, 'is-executing': run?.status === 'running' && revisionId === run.target_revision_id, 'is-paused': ['paused', 'pausing', 'cancelled'].includes(run?.view_status || run?.status || '') }">
                <div class="graph-legend">
                  <span class="canvas-label">EVIDENCE MAP</span>
                  <span><i class="legend-evidence"></i>证据</span
                  ><span><i class="legend-judgment"></i>中间判断</span
                  ><span><i class="legend-conclusion"></i>结论</span
                  ><label
                    ><input v-model="showAll" type="checkbox" />全部问题</label
                  >
                </div>
                <div v-if="flowNodes.length" class="graph-viewbar"><span>{{ compactGraph ? '聚焦视图' : '完整视图' }} <b>{{ displayedEvidenceCount }} / {{ graphEvidenceCount }}</b> 条证据<span class="graph-filter-explanation"> · 判断与结论完整保留</span></span><div><button v-if="graphEvidenceCount > 6" class="graph-view-button" @click="compactGraph = !compactGraph">{{ compactGraph ? `展开全部 ${graphEvidenceCount} 条` : '聚焦关键连接' }}</button><button class="graph-view-button" @click="fitGraph">适应画布 ↗</button></div></div>
                <VueFlow
                  v-if="flowNodes.length"
                  id="evidence-map"
                  :nodes="flowNodes"
                  :edges="flowEdges"
                  :node-types="nodeTypes"
                  :nodes-draggable="false"
                  :nodes-connectable="false"
                  :elements-selectable="true"
                  :min-zoom="compactGraph ? 0.6 : 0.25"
                  :max-zoom="1.5"
                  @init="initializeGraph"
                  @nodes-initialized="fitGraph"
                  @node-click="({ node }) => inspectNode(node)"
                  ><Background :gap="28" pattern-color="#25404e" /><Controls
                    :show-interactive="false"
                /></VueFlow>
                <div v-else class="panel-empty graph-no-result"><svg class="constellation" viewBox="0 0 500 270" aria-hidden="true"><g class="constellation-links"><path d="M65 130 190 70 330 112 425 50 M190 70l40 120 100-78 90 100 M65 130l165 60-60 55 M230 190l190 22"/></g><g class="constellation-points"><circle cx="65" cy="130" r="5"/><circle cx="190" cy="70" r="8"/><circle cx="330" cy="112" r="12"/><circle cx="425" cy="50" r="4"/><circle cx="230" cy="190" r="7"/><circle cx="420" cy="212" r="5"/><circle cx="170" cy="245" r="3"/></g><circle class="constellation-ring" cx="330" cy="112" r="29"/></svg><h3>等待证据连接成判断</h3><p>当前问题尚无发布图。执行进度会在上方持续更新。</p></div>
                <div class="canvas-caption">
                  {{ compactGraph && graphEvidenceCount > 6 ? '当前聚焦连接最多的 6 条证据，完整引用可一键展开。' : '依据 → 中间判断 → 结论' }} · 拖拽平移 / 滚轮缩放 / 点击核查
                </div>
              </div>
              <Transition name="panel-reveal"><aside v-if="selectedNode" class="inspector">
                <div class="inspector-heading">
                  <span class="eyebrow">节点核查</span
                  ><button
                    aria-label="关闭节点详情"
                    @click="selectedNode = null"
                  >
                    ×
                  </button>
                </div>
                <span class="pill neutral"
                  >{{ selectedNode.kind }} ·
                  {{ selectedNode.status || "未标记" }}</span
                >
                <h3>
                  {{ nodeText(selectedNode) }}
                </h3>
                <p v-if="selectedNode.locked" class="small-note">
                  此判断已由人工锁定；依据变化后需要复核。
                </p>
                <p v-if="inspectorLoading" class="small-note">
                  正在读取证据与原文…
                </p>
                <p v-if="inspectorError" class="inline-error">
                  {{ inspectorError }}
                </p>
                <template v-if="evidence"
                  ><div class="inspector-section">
                    <h4>证据引用</h4>
                    <blockquote>{{ evidence.quote }}</blockquote>
                    <p v-if="evidence.fact_des" class="small-note">
                      {{ evidence.fact_des }}
                    </p>
                  </div>
                  <div v-if="paragraph" class="inspector-section">
                    <h4>
                      来源段落
                      <span
                        >{{ paragraph.raw_doc_id }} /
                        {{ paragraph.raw_para_id }}</span
                      >
                    </h4>
                    <template v-if="evidence.span_scope === 'document'">
                      <p class="small-note">这是跨段文档引用。此处为起始原段，完整范围与逐字高亮见下方原文核对。</p>
                      <div class="source-paragraph">{{ paragraph.text }}</div>
                    </template>
                    <div v-else-if="excerpt" class="source-paragraph">
                      {{ excerpt.before }}<mark>{{ excerpt.quote }}</mark
                      >{{ excerpt.after }}
                    </div>
                    <template v-else
                      ><p class="inline-error">
                        引用与字符位置未通过本地校验，原文暂不高亮。
                      </p>
                      <div class="source-paragraph">
                        {{ paragraph.text }}
                      </div></template
                    ><small class="span-label"
                      >Unicode code point · {{ evidence.span_scope === 'document' ? '文档' : '段内' }} [{{ evidence.start_cp }},
                      {{ evidence.end_cp }})</small
                    >
                    <button class="button full-width" :disabled="sourceContextLoading" @click="openSourceContext">
                      {{ sourceContextLoading ? '正在读取相邻原文…' : '查看相邻原文（不调用模型）' }}
                    </button>
                    <p v-if="sourceContextError" class="inline-error" role="alert">{{ sourceContextError }}</p>
                    <div v-if="sourceContext" class="inspector-section">
                      <h4>相邻原文核对<span>{{ sourceContext.raw_doc_id }} · {{ sourceContext.members[0]?.raw_para_id }} — {{ sourceContext.members[sourceContext.members.length - 1]?.raw_para_id }}</span></h4>
                      <p class="small-note">上下文仅供核对，未增加到已发布结论或导出；本次读取不调用模型。</p>
                      <p v-if="!contextExcerpt" class="inline-error" role="alert">原证据引用、来源范围或字符位置未通过逐字校验，暂不高亮；以下保留返回的完整原文，未修正引用。</p>
                      <div v-if="contextExcerpt" class="source-paragraph">{{ contextExcerpt.before }}<mark>{{ contextExcerpt.quote }}</mark>{{ contextExcerpt.after }}</div>
                      <div v-else class="source-paragraph">{{ sourceContext.quote }}</div>
                      <small class="span-label">文档 Unicode code point · [{{ sourceContext.start_cp }}, {{ sourceContext.end_cp }}) · {{ sourceContext.paragraph_count }} 个原段</small>
                      <details class="technical-detail">
                        <summary>查看范围成员ID</summary>
                        <p v-for="member in sourceContext.members" :key="member.paragraph_id" class="small-note">{{ member.raw_para_id }} · [{{ member.start_cp }}, {{ member.end_cp }})<br />{{ member.paragraph_id }}</p>
                      </details>
                    </div>
                  </div>
                  <div class="inspector-section">
                    <h4>如果拿掉这条证据？</h4>
                    <p class="small-note">
                      预览哪些判断依赖它，再决定是否排除并重算。
                    </p>
                    <label class="sr-only" for="change-reason"
                      >排除证据的理由</label
                    ><textarea
                      id="change-reason"
                      v-model="changeReason"
                      :disabled="!!preview"
                      placeholder="填写排除理由"
                      rows="3"
                    ></textarea
                    ><button
                      class="button full-width"
                      :disabled="
                        !changeReason.trim() ||
                        busy ||
                        !!activeRun ||
                        !publishedView ||
                        !!preview ||
                        !health?.model_configured
                      "
                      @click="previewExclusion"
                    >
                      查看下游影响
                    </button>
                    <div v-if="preview" class="change-preview">
                      <strong>本次修改的影响</strong>
                      <p>
                        {{ preview.affected_question_ids.length }} 个问题 ·
                        {{ preview.affected_node_ids.length }} 个节点
                      </p>
                      <p v-if="preview.must_rerun_agents?.length">
                        重跑：{{ preview.must_rerun_agents.join("、") }}
                      </p>
                      <small
                        >基于
                        {{
                          previewBaseRevision.slice(0, 8)
                        }}，提交时重新检查版本。</small
                      >
                      <div class="preview-actions">
                        <button
                          class="button compact"
                          :disabled="busy"
                          @click="preview = null"
                        >
                          返回修改</button
                        ><button
                          class="button compact primary"
                          :disabled="busy || !!activeRun"
                          @click="commitChange"
                        >
                          确认并重算
                        </button>
                      </div>
                    </div>
                  </div></template
                >
                <details class="technical-detail">
                  <summary>节点结构</summary>
                  <pre>{{ pretty(selectedNode) }}</pre>
                </details>
              </aside></Transition>
            </div>
            <Transition name="artifact-switch" mode="out-in"><div v-if="tab !== 'graph'" :key="tab" class="artifact-panel">
              <div class="artifact-context">
                <h2>{{ tabs.find((t) => t.id === tab)?.label }}</h2>
                <span class="pill neutral">{{
                  showAll
                    ? "全部问题"
                    : `当前问题 · ${activeQuestion?.raw_question_id ?? ""}`
                }}</span>
                <p v-if="tab === 'agents'">
                  同一证据快照下的独立研判与仲裁产物。
                </p>
                <p v-else-if="tab === 'confidence'">
                  质量分表示可核查程度，不能当作事实正确概率。
                </p>
                <p v-else-if="tab === 'conflicts'">
                  保留双方依据、取舍原因与未被采纳的意见。
                </p>
                <p v-else>已发布版本的结论与支持依据。</p>
              </div>
              <div v-if="!artifactRows.length" class="panel-empty">
                <h3>
                  当前版本暂无{{ tabs.find((t) => t.id === tab)?.label }}记录
                </h3>
                <p>这里仅展示服务实际返回的产物。</p>
              </div>
              <article
                v-for="(row, index) in artifactRows"
                :key="index"
                class="artifact-card"
              >
                <span class="record-number">{{
                  String(index + 1).padStart(2, "0")
                }}</span>
                <div>
                  <h3>{{ titleOf(row, index) }}</h3>
                  <div v-if="tab === 'conclusions' && row.evidence_relation_scope" class="inspector-section">
                    <h4>支持与冲突针对的主张</h4>
                    <p>{{ (row.evidence_relation_scope as Record<string, unknown>).principal_claim }}</p>
                    <p class="small-note">这是模型本次研判的关系目标，须结合结论中的归属、时点与未决限制阅读。</p>
                  </div>
                  <details>
                    <summary>查看完整依据与字段</summary>
                    <pre>{{ pretty(row) }}</pre>
                  </details>
                </div>
              </article>
            </div></Transition>
          </template>
          <div v-else class="empty-results">
            <svg class="constellation" viewBox="0 0 500 270" aria-hidden="true"><defs><radialGradient id="orbit-glow"><stop stop-color="#39d4cd" stop-opacity=".18"/><stop offset="1" stop-color="#39d4cd" stop-opacity="0"/></radialGradient></defs><circle cx="250" cy="135" r="125" fill="url(#orbit-glow)"/><g class="constellation-links"><path d="M65 140 180 65 315 118 425 60 M180 65l55 140 80-87 105 95 M65 140l170 65-70 40 M235 205l185 8"/></g><g class="constellation-points"><circle cx="65" cy="140" r="5"/><circle cx="180" cy="65" r="8"/><circle cx="315" cy="118" r="12"/><circle cx="425" cy="60" r="4"/><circle cx="235" cy="205" r="7"/><circle cx="420" cy="213" r="5"/><circle cx="165" cy="245" r="3"/></g><circle class="constellation-ring" cx="315" cy="118" r="28"/></svg>
            <span class="eyebrow">FROM SOURCES TO CONCLUSIONS</span>
            <h2>让每一个判断，都有迹可循</h2>
            <p>
              选择左侧问题后开始研判。完成的结果将在这里形成证据图，<br />可继续查看独立判断、冲突记录和原文来源。
            </p>
            <span v-if="!health?.model_configured" class="pill neutral"
              >等待真实模型配置</span
            ><span v-else-if="activeRun" class="pill is-running"
              >任务执行中，完成后自动刷新</span
            >
          </div>
        </section>
        <footer class="workspace-footer">
          <span>原文可定位 · 判断可追溯 · 修订有版本</span
          ><span>本机开发环境</span>
        </footer>
      </main>
      <ActivityLog :run="run" :activity="activity" :runs="runHistory" :connected="activityConnected" :motion="motionAllowed && pageVisible" @select="selectRun" @reconnect="reconnect" />
    </div>
    <ImportDialog v-if="importOpen" @close="importOpen = false" @imported="importSucceeded" />
  </div>
</template>
