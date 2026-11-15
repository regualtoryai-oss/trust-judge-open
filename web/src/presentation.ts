import type { BusinessNode, Question } from "./types";

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

const failureStages: Record<string, string> = {
  source_bundle_selection: "原文选证",
  paragraph_queries: "问题拆解", paragraph_recall: "候选段落召回", paragraph_rank: "逐段排序",
  paragraph_annotate: "整段事实概括", paragraph_fields: "段落字段核验", verify_evidence_scope: "证据范围核验",
  candidate_import: "候选工件导入", scope_review: "严格事实复核", purpose_review: "事实用途审核", retrieve: "资料检索",
  extract: "证据抽取",
  verify: "原文核验",
  freeze: "证据快照保存",
  authority_analyst: "独立研判 A",
  analyst_a: "独立研判 A",
  cross_validation_analyst: "独立研判 B",
  analyst_b: "独立研判 B",
  arbitrate: "综合仲裁",
  baseline: "对照研判",
  snapshot: "结果组装校验",
  build_snapshot: "结果组装校验",
  publish: "结果发布校验",
  export: "提交包校验",
};

export function formatRunError(error: unknown): string {
  const fallback = "研判失败：未获得可展示的错误说明，请查看任务记录。";
  const value = record(error);
  const code = value.code;
  if (typeof code !== "string" || !/^[a-z][a-z0-9_]{0,63}$/.test(code))
    return fallback;
  const message = typeof value.message === "string" ? value.message.trim() : "";
  // Only the backend's bounded, public message is displayable. Extra details,
  // stacks and provider response objects must never become the banner text.
  if (
    !message ||
    message.length > 400 ||
    /[\r\n]/.test(message) ||
    /^[{\[]|```|\bTraceback\b|\bBearer\s+|\bsk-[a-z0-9]{12,}/i.test(message)
  )
    return fallback;
  const stage =
    typeof value.stage === "string" &&
    Object.prototype.hasOwnProperty.call(failureStages, value.stage)
      ? failureStages[value.stage]
      : undefined;
  return `${stage ? `${stage}失败` : "研判失败"}：${message}（错误代码：${code}）`;
}

const checkLabels: Record<string, string> = {
  lc_bundle_source_compile_failed: "选证来源或注释未通过业务校验",
  lc_bundle_unknown_or_duplicate_window: "选证包含未知或重复的原文窗口编号",
  lc_bundle_duplicate_original_start_anchor: "多条选证共用同一原始起始段",
  lc_bundle_time_not_literal: "选证时间不是所选原文中的逐字文本",
  lc_bundle_entity_not_literal_or_duplicate: "选证实体未在所选原文中逐字出现或存在重复",
  lc_bundle_annotation_number_not_in_source: "选证注释包含所选原文不支持的数值或单位",
  lc_bundle_empty_annotation: "选证缺少事实概括或内容说明",
  lc_bundle_output_schema: "选证编号或注释字段不符合规定结构",
  lc_bundle_output_object_required: "选证结果不是规定的结构化对象",
  lc_bundle_prepared_contract_changed: "选证目录与绑定的原始资料不一致",
  premise_binding_material: "内部事实绑定内容无效", premise_source_identity: "内部事实来源身份不一致", premise_source_span: "内部事实引用未通过原文校验",
  premise_scope_exclusions: "内部事实排除记录不一致", premise_scope_corrections: "内部事实修订记录不一致", premise_scope_assessments: "原文事实核验记录不完整",
  premise_scope_coverage: "原文事实核验覆盖不完整", premise_detailed_quote_review_required: "内部事实缺少完整原文核验", premise_scope_assessment_replay: "原文核验记录重放失败",
  premise_original_scope_selection_replay: "原有事实选择记录重放失败", premise_missing_quote_review: "内部事实缺少引文核验", premise_bank_replay: "内部事实库重放校验失败",
  premise_bank_binding: "内部事实库绑定不一致", premise_question_binding: "用途审核题目绑定不一致", premise_purpose_batch_identity: "用途审核批次身份不一致",
  premise_purpose_contract: "用途审核结构或逐项决定不完整", premise_purpose_duplicate_json_key: "用途审核 JSON 包含重复字段", premise_purpose_equivalence: "用途审核同义替代关系不符合要求",
  premise_purpose_coverage: "用途审核未完整覆盖内部事实", premise_purpose_ledger_replay: "用途审核记录重放失败", premise_purpose_full_coverage: "用途审核未覆盖完整事实库",
  premise_formal_authorization_replay: "正式事实用途许可重放失败", premise_manual_purpose_unverified: "人工观点尚未取得用途许可", premise_formal_projection_replay: "正式事实投影校验失败",
  scope_question_invalid: "复核题目绑定无效", scope_catalog_invalid: "原子事实目录无效",
  scope_decision_contract: "逐事实复核映射不完整", scope_target_contract: "目标范围绑定不符合要求",
  scope_verification_duplicate_json_key: "事实复核 JSON 包含重复字段",
  annotation_refocus_catalog_invalid: "定向修正目录无效", annotation_refocus_decision_contract: "定向修正映射不完整",
  annotation_refocus_duplicate_json_key: "定向修正 JSON 包含重复字段", annotation_refocus_ledger_invalid: "定向修正身份校验失败",
  scope_ledger_invalid: "事实复核身份校验失败", annotation_refocus_manual_protected: "人工解释保留，禁止自动改写", annotation_correction_contract: "定向修正结构不完整", annotation_correction_empty: "定向修正未返回事实", annotation_correction_no_supported_nonempty: "无可支持事实时不应附带改写",
  scope_annotation_narrowing: "原解释不能重复进行内容修正",
  refine_artifact_hash: "候选工件哈希不一致", refine_question_binding: "候选工件题目不一致",
  refine_source_manifest: "候选工件来源已变化", refine_overlay_binding: "人工标注已变化",
  refine_incomplete_coverage: "来源扫描未完整完成", refine_candidate_source: "工件原文定位校验失败",
  schema: "结构不符合要求", candidate_schema: "证据结构不符合要求", json: "未返回有效 JSON",
  ungrounded_judgment: "判断没有证据或已声明的前提", unknown_prerequisite: "判断引用了未声明的前提",
  duplicate_judgment_id: "判断编号重复", prerequisite_cycle: "判断前提存在循环", assertion_without_support: "确定性结论没有支持证据",
  invalid_evidence_reference: "证据引用重复或超出冻结快照", duplicate_conflict_id: "冲突编号重复",
  unknown_paragraph: "引用段落不属于本轮资料", unseen_paragraph: "引用了未提供的段落", quote_occurrence: "引用不匹配或有多处相同原文", quote_span: "引用与原文位置不一致",
  unknown_interpretation_target: "解释修订缺少可定位证据", structured_output: "结构化输出无效",
  unknown_source_selection: "所选原文片段不属于本轮资料", source_selection: "原文片段选择不符合要求",
  source_catalog: "原文片段目录完整性校验失败", source_catalog_id: "原文目录绑定编号不匹配", source_catalog_limit: "原文片段目录超过单批容量", evidence_selection: "证据精筛结果未通过完整性校验",
  source_interval_selection: "原文区间选择不符合要求",
  source_window_catalog: "原文窗口目录完整性校验失败", source_window_selection: "原文窗口选择不符合要求",
  precision_decisions: "精筛决定映射未通过完整性校验", precision_duplicate_json_key: "精筛 JSON 包含重复字段", precision_candidate_catalog: "精筛候选目录不完整或编号重复",
  duplicate_prerequisite: "判断的前提引用重复", overlapping_evidence_relations: "同一证据同时被标为支持和冲突",
  overlapping_conflict_sides: "冲突双方引用了相同证据", ambiguous_conflict_resolution: "冲突消解类型与依据不明确", uncited_numeric_literal: "结论数值没有对应引用证据",
};

const schemaFields = new Set(["bank_id", "purpose_catalog_id", "purpose_binding_id", "introduces_unstated_source_fact", "fact_ids", "evidence", "paragraph_id", "quote", "fact_des", "argument", "entities", "start_cp", "end_cp", "source_span_ids", "source_start_id", "source_end_id", "source_window_id", "source_catalog_id", "candidate_catalog_id", "decisions", "selected_evidence_ids", "assessments", "evidence_id", "relation", "reason", "equivalent_to", "fact_entailment", "target_relation", "target_question_text", "qualifier_dimension", "requires_unstated_bridge", "target_fact_present_but_annotation_incomplete", "corrections", "annotations", "annotation", "时间", "主体", "地点", "内容", "影响", "数值口径", "decision", "conclusion", "judgments", "supporting_evidence_ids", "conflicting_evidence_ids", "id", "statement", "evidence_ids", "prerequisite_judgment_ids", "conflicts", "status", "resolution_type", "basis", "accepted_reason", "rejected_reason", "minority_opinion", "reopen_conditions"]);
const schemaTypeLabels: Record<string, string> = {
  missing: "缺少必填字段", extra_forbidden: "不允许额外字段", string_type: "应为文本", string_too_short: "文本过短", string_too_long: "文本过长", list_type: "应为数组", dict_type: "应为对象", model_type: "应为结构化对象", model_attributes_type: "应为结构化对象", int_type: "应为整数", int_parsing: "整数格式无效", float_type: "应为数字", bool_type: "应为布尔值", literal_error: "不属于允许的选项", enum: "不属于允许的选项", greater_than: "数值低于范围", greater_than_equal: "数值低于范围", less_than: "数值超过范围", less_than_equal: "数值超过范围", too_short: "内容数量不足", too_long: "内容数量超限", finite_number: "应为有限数值", value_error: "字段值不符合规则", json_invalid: "JSON 格式无效", json_type: "应为 JSON 数据", none_required: "应为空值", union_tag_invalid: "类型选项无效", union_tag_not_found: "缺少类型选项", schema: "结构不符合要求",
};

function schemaDiagnostic(value: unknown): string {
  if (!Array.isArray(value)) return "";
  return value.slice(0, 3).map(row => {
    const item = record(row);
    if (!Array.isArray(item.loc)) return "";
    const path = item.loc.slice(0, 8).map(part => typeof part === "number" && Number.isInteger(part) && part >= 0 && part <= 1_000_000 ? `[${part}]` : typeof part === "string" && schemaFields.has(part) ? part : "<field>").reduce<string>((text, part) => `${text}${text && !part.startsWith("[") ? "." : ""}${part}`, "");
    const label = typeof item.type === "string" && Object.prototype.hasOwnProperty.call(schemaTypeLabels, item.type) ? schemaTypeLabels[item.type] : schemaTypeLabels.schema;
    return `${path ? `字段 ${path}` : "根对象"}：${label}`;
  }).filter(Boolean).join("；");
}

function boundedCount(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 1_000_000 ? value : null;
}

const sourceIntervalIssueLabels: Record<string, string> = {
  unknown_source_start: "起点不属于本批原文目录", unknown_source_end: "终点不属于本批原文目录",
  cross_paragraph_interval: "起终点跨越段落", reversed_source_interval: "原文区间顺序颠倒",
  invalid_source_interval: "原文区间结构无效", source_catalog_mismatch: "原文目录绑定不匹配",
  unknown_source_window: "所选窗口不属于本批原文目录", invalid_source_window_catalog: "原文窗口目录无效", invalid_source_window: "原文窗口定位校验失败",
  precision_duplicate_self: "重复证据指向自身", precision_duplicate_target_not_retained: "重复证据未指向保留项",
  precision_unexpected_equivalent_to: "非重复项填写了替代引用", precision_decision_coverage: "精筛决定未完整覆盖本批候选",
};

function intervalIssueDiagnostic(value: unknown): string {
  if (!Array.isArray(value)) return "";
  return value.slice(0, 3).map(row => {
    const issue = record(row);
    const label = typeof issue.check === "string" && Object.prototype.hasOwnProperty.call(sourceIntervalIssueLabels, issue.check) ? sourceIntervalIssueLabels[issue.check] : "";
    if (!label) return "";
    const index = boundedCount(issue.candidate_index);
    return `${index === null ? "" : `第 ${index + 1} 条候选：`}${label}`;
  }).filter(Boolean).join("；");
}

/** Output capacity is a request ceiling, never an observed token-usage estimate. */
export function capacityDiagnostic(data: unknown): string {
  const value = record(data), parts: string[] = [];
  const previous = boundedCount(value.previous_max_output_tokens);
  const current = boundedCount(value.max_output_tokens);
  const attempt = boundedCount(value.attempt);
  // Content repairs also carry `attempt`; it alone never identifies a capacity retry.
  if (!(previous !== null && previous > 0) && !(current !== null && current > 0)) return "";
  if (attempt !== null && attempt > 0) parts.push(`第 ${attempt} 次输出扩容重试`);
  if (previous !== null && previous > 0 && current !== null && current > 0) parts.push(`输出额度 ${previous.toLocaleString("en-US")} → ${current.toLocaleString("en-US")} tokens（上限，非已用）`);
  else if (current !== null && current > 0) parts.push(`输出额度 ${current.toLocaleString("en-US")} tokens（上限，非已用）`);
  else if (previous !== null && previous > 0) parts.push(`上次输出额度 ${previous.toLocaleString("en-US")} tokens（上限，非已用）`);
  return parts.join(" · ");
}

/** Only fixed diagnostic labels and bounded counters are displayable. */
export function diagnosticText(details: unknown): string {
  const value = record(details), parts: string[] = [];
  const knownTokens = boundedCount(value.known_tokens), estimatedInput = boundedCount(value.input_estimate);
  if (knownTokens !== null && estimatedInput !== null) {
    const reserved = boundedCount(value.last_max_output_tokens);
    return [`本题累计已用 ${knownTokens.toLocaleString("en-US")} tokens`,
      `下一次输入预估 ${estimatedInput.toLocaleString("en-US")} tokens`,
      ...(reserved !== null ? [`另需预留最多 ${reserved.toLocaleString("en-US")} tokens 输出`] : []),
      "触发用量保护，下一次模型请求未发出"].join(" · ");
  }
  const httpStatus = boundedCount(value.provider_http_status);
  if (httpStatus !== null && httpStatus >= 400 && httpStatus <= 599) parts.push(`供应商 HTTP ${httpStatus}`);
  const retryAfter = boundedCount(value.retry_after_seconds);
  if (retryAfter !== null && retryAfter <= 86400) parts.push(`供应商建议至少等待 ${retryAfter} 秒`);
  if (value.usage_known === false) parts.push("该次请求未返回用量，费用未知");
  const check = typeof value.check === "string" && Object.prototype.hasOwnProperty.call(checkLabels, value.check) ? checkLabels[value.check] : "";
  if (check) parts.push(check);
  const count = (name: string): number | null => boundedCount(value[name]);
  const judgment = count("judgment_index"), candidate = count("candidate_index"), conflict = count("conflict_index");
  if (judgment !== null) parts.push(`第 ${judgment + 1} 条判断`);
  else if (candidate !== null) parts.push(`第 ${candidate + 1} 条证据候选`);
  else if (conflict !== null) parts.push(`第 ${conflict + 1} 条冲突`);
  const evidence = count("evidence_count"), prerequisites = count("prerequisite_count");
  if (evidence !== null) parts.push(`证据引用 ${evidence}`);
  if (prerequisites !== null) parts.push(`前提引用 ${prerequisites}`);
  const invalidCandidates = count("invalid_candidate_count");
  if (invalidCandidates !== null) parts.push(`${value.check === "source_window_selection" ? "窗口" : value.check === "precision_decisions" || value.check === "precision_candidate_catalog" ? "精筛" : "区间"}无效候选 ${invalidCandidates}`);
  const duplicateKeys = count("duplicate_key_count");
  if (duplicateKeys !== null) parts.push(`重复 JSON 字段 ${duplicateKeys}`);
  const path = value.field_path;
  if (typeof path === "string" && path.length <= 160 && /^(?:judgments|conflicts|candidates|evidence|assessments)(?:(?:\[\d{1,6}\]|\.\d{1,6}))?(?:\.(?:id|statement|evidence_ids|prerequisite_judgment_ids|supporting_evidence_ids|conflicting_evidence_ids|paragraph_id|quote|start_cp|end_cp|source_span_ids|source_start_id|source_end_id|source_window_id|evidence_id|relation|reason|equivalent_to)(?:\[\d{1,6}\]|\.\d{1,6})?)?$|^decisions(?:\[\d{1,6}\]|\.\d{1,6})?(?:\.(?:relation|reason|equivalent_to))?$|^(?:decision|conclusion|supporting_evidence_ids|conflicting_evidence_ids|source_span_ids|source_start_id|source_end_id|source_window_id|source_catalog_id|candidate_catalog_id|selected_evidence_ids)(?:\[\d{1,6}\]|\.\d{1,6})?$/.test(path)) parts.push(`字段 ${path}`);
  const schema = schemaDiagnostic(value.schema_errors);
  if (schema) parts.push(schema);
  const intervalIssues = intervalIssueDiagnostic(value.selection_issues);
  if (intervalIssues) parts.push(intervalIssues);
  if (count("repair_attempt") === 1 && value.repair_exhausted === true) parts.push("已达 1 次定向修正上限");
  const attempts = count("capacity_attempts"), lastBudget = count("last_max_output_tokens"), capacityLimit = count("max_output_tokens_limit");
  const retried = attempts !== null && attempts > 0;
  const truncated = value.finish_reason === "length";
  if (retried) parts.push(`已执行 ${attempts} 次输出扩容重试`);
  if (truncated) parts.push("供应商确认输出长度截断");
  if (lastBudget !== null && lastBudget > 0) parts.push(retried || truncated
    ? `最后输出额度 ${lastBudget.toLocaleString("en-US")} tokens`
    : `本次请求配置输出额度 ${lastBudget.toLocaleString("en-US")} tokens（上限，非已用）`);
  if (capacityLimit !== null && capacityLimit > 0) parts.push(`本次配置输出上限 ${capacityLimit.toLocaleString("en-US")} tokens`);
  if ((retried || truncated) && lastBudget !== null && capacityLimit !== null && capacityLimit > 0 && lastBudget >= capacityLimit) parts.push("已达本次可配置输出上限");
  return parts.join(" · ");
}

export function providerFailureAction(error: unknown): string {
  const value = record(error);
  const actions: Record<string, string> = {
    model_insufficient_balance: "账户余额不足：请补充 DeepSeek 账户余额后再继续。当前任务不会自动重试。",
    model_authentication_failed: "认证未通过：请检查本机模型密钥配置，修复后再继续。",
    model_access_denied: "模型访问被拒绝：请检查账户权限后再继续。",
    model_rate_limited: "请求被限流：请按供应商等待提示稍后续跑，已完成检查点保留。",
    model_service_unavailable: "供应商服务暂时不可用：请稍后续跑，避免连续重复提交。",
  };
  return typeof value.code === "string" && Object.prototype.hasOwnProperty.call(actions, value.code) ? actions[value.code] : "";
}

export function eventDiagnostic(data: unknown): string {
  const value = record(data);
  const details = record(value.details);
  return [capacityDiagnostic(value), diagnosticText({ ...details, ...(typeof value.check === "string" ? { check: value.check } : {}) })].filter(Boolean).join(" · ");
}

function firstText(value: Record<string, unknown>): string {
  for (const key of [
    "statement",
    "conclusion",
    "fact_des",
    "quote",
    "summary",
  ]) {
    if (typeof value[key] === "string" && value[key].trim()) return value[key];
  }
  return "";
}

export function nodeText(node: BusinessNode): string {
  const content = node.content;
  if (typeof content === "string" && content.trim()) return content;
  return firstText(record(content)) || firstText(node) || "点击查看节点详情";
}

export const relationLabels: Record<string, string> = {
  supports: "支持",
  support: "支持",
  contradicts: "冲突",
  conflicts: "冲突",
  conflict: "冲突",
  prerequisite: "前提",
  premise: "前提",
  depends_on: "前提",
};

export function isConflictRelation(relation: string): boolean {
  return ["contradicts", "conflicts", "conflict"].includes(relation);
}

export function filterQuestionArtifacts(
  rows: unknown[],
  question: Pick<Question, "id" | "raw_question_id">,
  evidenceNodes: BusinessNode[],
): unknown[] {
  const ids = new Set([question.id, question.raw_question_id]);
  const evidenceRefs = new Set(
    evidenceNodes
      .filter((node) => node.kind === "evidence")
      .map((node) => {
        const value = record(node.content);
        return JSON.stringify([
          value.raw_doc_id,
          value.raw_para_id,
          value.quote,
        ]);
      }),
  );
  return rows.filter((value) => {
    const row = record(value);
    const ref = record(row.object_ref);
    for (const id of [row.question_id, row.raw_question_id, ref.question_id]) {
      if (typeof id === "string") return ids.has(id);
    }
    if (ref.doc_id && ref.para_id && typeof ref.quote === "string") {
      return evidenceRefs.has(
        JSON.stringify([ref.doc_id, ref.para_id, ref.quote]),
      );
    }
    return false;
  });
}
