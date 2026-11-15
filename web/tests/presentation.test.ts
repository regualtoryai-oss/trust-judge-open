import test from "node:test";
import assert from "node:assert/strict";
import {
  nodeText,
  formatRunError,
  filterQuestionArtifacts,
  isConflictRelation,
  relationLabels,
  diagnosticText,
  eventDiagnostic,
  capacityDiagnostic,
  providerFailureAction,
} from "../src/presentation.ts";

test("a budget stop distinguishes consumed tokens and input estimate from output truncation", () => {
  const text = diagnosticText({known_tokens:46251,input_estimate:12130,last_max_output_tokens:8192,
    max_output_tokens_limit:8192,capacity_attempts:0});
  assert.match(text,/累计已用 46,251/);
  assert.match(text,/输入预估 12,130/);
  assert.match(text,/请求未发出/);
  assert.doesNotMatch(text,/已达本次可配置输出上限|扩容重试/);
});

test("source compile failures show the fixed rule and configured capacity without inventing exhaustion", () => {
  const text = diagnosticText({check:"lc_bundle_entity_not_literal_or_duplicate",capacity_attempts:0,
    last_max_output_tokens:4096,max_output_tokens_limit:4096,response:"private model body"});
  assert.match(text,/选证实体未在所选原文中逐字出现或存在重复/);
  assert.match(text,/本次请求配置输出额度 4,096 tokens（上限，非已用）/);
  assert.match(text,/本次配置输出上限 4,096 tokens/);
  assert.doesNotMatch(text,/已达本次可配置输出上限|扩容重试|长度截断|private/);
  assert.match(formatRunError({stage:"source_bundle_selection",code:"lc_bundle_source_compile_failed",
    message:"选证实体未在所选原文中逐字出现或存在重复，未发布。"}),/^原文选证失败：选证实体/);
  for (const check of ["lc_bundle_unknown_or_duplicate_window", "lc_bundle_duplicate_original_start_anchor",
    "lc_bundle_time_not_literal", "lc_bundle_annotation_number_not_in_source", "lc_bundle_empty_annotation"]) {
    assert.notEqual(diagnosticText({check}),"");
  }
  assert.equal(diagnosticText({check:"private arbitrary exception body"}),"");
});

test("only a confirmed length finish or recorded capacity retry can mark the configured cap reached", () => {
  const limit = {last_max_output_tokens:4096,max_output_tokens_limit:4096,capacity_attempts:0};
  assert.match(diagnosticText({...limit,finish_reason:"length"}),/供应商确认输出长度截断.*已达本次可配置输出上限/);
  for (const finish_reason of [undefined,"stop","tool_calls","length guessed",true]) {
    assert.doesNotMatch(diagnosticText({...limit,finish_reason}),/长度截断|已达本次可配置输出上限|扩容重试/);
  }
  assert.doesNotMatch(diagnosticText({...limit,capacity_attempts:"1"}),/已达本次可配置输出上限|扩容重试/);
  assert.match(diagnosticText({...limit,capacity_attempts:1}),/已执行 1 次输出扩容重试.*已达本次可配置输出上限/);
  assert.doesNotMatch(diagnosticText({...limit,last_max_output_tokens:2048,finish_reason:"length"}),/已达本次可配置输出上限/);
});

test("provider diagnostics show bounded status and fixed recovery guidance", () => {
  assert.equal(diagnosticText({provider_http_status:402,usage_known:false,body:"private",retry_after_seconds:"key"}), "供应商 HTTP 402 · 该次请求未返回用量，费用未知");
  assert.equal(diagnosticText({provider_http_status:429,retry_after_seconds:30}), "供应商 HTTP 429 · 供应商建议至少等待 30 秒");
  assert.equal(diagnosticText({provider_http_status:"private",retry_after_seconds:Infinity}), "");
  assert.match(providerFailureAction({code:"model_insufficient_balance",message:"private"}), /补充 DeepSeek 账户余额/);
  assert.match(providerFailureAction({code:"model_rate_limited"}), /已完成检查点保留/);
  assert.equal(providerFailureAction({code:"constructor",message:"private"}), "");
  assert.match(formatRunError({code:"model_rate_limited",stage:"paragraph_rank",message:"模型服务限流。"}), /^逐段排序失败/);
});

test("run failures identify the failed workflow stage with a public explanation", () => {
  assert.equal(
    formatRunError({
      code: "model_output_truncated",
      stage: "extract",
      message: "模型输出达到长度上限，本次结果未发布。",
    }),
    "证据抽取失败：模型输出达到长度上限，本次结果未发布。（错误代码：model_output_truncated）",
  );
  assert.match(
    formatRunError({ code: "invalid_reference", stage: "cross_validation_analyst", message: "引用不属于本轮证据。" }),
    /^独立研判 B失败：引用不属于本轮证据。/,
  );
  assert.match(
    formatRunError({ code: "invalid_snapshot", stage: "snapshot", message: "结论引用未通过校验。" }),
    /^结果组装校验失败：/,
  );
});

test("validation diagnostics show safe location and counts without model prose", () => {
  assert.equal(diagnosticText({check:"ungrounded_judgment",judgment_index:0,evidence_count:0,prerequisite_count:0,field_path:"judgments[0].evidence_ids",repair_attempt:1,repair_exhausted:true,rawbody:"secret"}),"判断没有证据或已声明的前提 · 第 1 条判断 · 证据引用 0 · 前提引用 0 · 字段 judgments[0].evidence_ids · 已达 1 次定向修正上限");
  assert.equal(diagnosticText({check:"private-model-text",judgment_index:-2,evidence_count:Infinity,field_path:"private.provider-body"}),"");
  assert.equal(diagnosticText({check:"constructor",judgment_index:true}),"");
  assert.equal(eventDiagnostic({check:"schema",details:{response:"secret"}}),"结构不符合要求");
});

test("schema errors show at most three safe paths and controlled type descriptions", () => {
  const formatted = diagnosticText({schema_errors:[
    {loc:["judgments",0,"statement"],type:"string_type",msg:"private-body"},
    {loc:["evidence",1,"argument","时间"],type:"missing"},
    {loc:["private-field",true,-1,{response:"private"}],type:"private-type"},
    {loc:["conclusion"],type:"missing"},
  ]});
  assert.equal(formatted,"字段 judgments[0].statement：应为文本；字段 evidence[1].argument.时间：缺少必填字段；字段 <field>.<field>.<field>.<field>：结构不符合要求");
  assert.doesNotMatch(formatted,/private|conclusion|response/);
  assert.equal(diagnosticText({schema_errors:[{loc:[],type:"json_invalid"}]}),"根对象：JSON 格式无效");
});

test("source selection diagnostics expose only whitelisted evidence paths, never source content", () => {
  assert.equal(diagnosticText({check:"unknown_source_selection",candidate_index:1,field_path:"evidence[1].source_span_ids[0]",source_text:"private original",source_span_ids:["private-id"]}), "所选原文片段不属于本轮资料 · 第 2 条证据候选 · 字段 evidence[1].source_span_ids[0]");
  assert.equal(diagnosticText({check:"source_selection",schema_errors:[{loc:["evidence",0,"source_span_ids"],type:"missing",input:"private source"}]}), "原文片段选择不符合要求 · 字段 evidence[0].source_span_ids：缺少必填字段");
  assert.doesNotMatch(diagnosticText({check:"source_selection",field_path:"evidence[0].private_response",response:"private"}), /private|response/);
});

test("precision and catalog failures expose fixed Chinese checks and safe schema locations", () => {
  assert.equal(diagnosticText({check:"source_catalog_limit",source_text:"private"}), "原文片段目录超过单批容量");
  assert.equal(diagnosticText({check:"source_catalog",response:"private"}), "原文片段目录完整性校验失败");
  const text = diagnosticText({check:"evidence_selection",field_path:"selected_evidence_ids[2]",schema_errors:[
    {loc:["assessments",0,"reason"],type:"string_type",input:"private reason"},
    {loc:["selected_evidence_ids",2],type:"string_type",input:"private identity"},
    {loc:["assessments",0,"equivalent_to"],type:"missing",msg:"private body"},
  ],reason:"private reason",selected_evidence_ids:["private identity"]});
  assert.equal(text, "证据精筛结果未通过完整性校验 · 字段 selected_evidence_ids[2] · 字段 assessments[0].reason：应为文本；字段 selected_evidence_ids[2]：应为文本；字段 assessments[0].equivalent_to：缺少必填字段");
  assert.doesNotMatch(text,/private|identity|body/);
  assert.equal(diagnosticText({check:"evidence_selection",field_path:"assessments[0].private_response"}), "证据精筛结果未通过完整性校验");
});

test("explicit source interval diagnostics show fixed reasons without source identifiers", () => {
  assert.equal(diagnosticText({check:"source_catalog_id",field_path:"source_catalog_id",source_catalog_id:"private-binding",schema_errors:[{loc:["source_catalog_id"],type:"literal_error",input:"private-binding"}]}), "原文目录绑定编号不匹配 · 字段 source_catalog_id · 字段 source_catalog_id：不属于允许的选项");
  const text = diagnosticText({check:"source_interval_selection",candidate_index:1,invalid_candidate_count:5,field_path:"candidates[1].source_end_id",selection_issues:[
    {check:"cross_paragraph_interval",candidate_index:1,field_path:"candidates[1].source_end_id",source_end_id:"private-end"},
    {check:"reversed_source_interval",candidate_index:2},
    {check:"unknown_source_start",candidate_index:3},
    {check:"invalid_source_interval",candidate_index:4},
  ],source_intervals:[{first_id:"private-start"}],response:"private-body"});
  assert.equal(text,"原文区间选择不符合要求 · 第 2 条证据候选 · 区间无效候选 5 · 字段 candidates[1].source_end_id · 第 2 条候选：起终点跨越段落；第 3 条候选：原文区间顺序颠倒；第 4 条候选：起点不属于本批原文目录");
  assert.doesNotMatch(text,/private|first_id|source_intervals|第 5 条/);
  assert.equal(diagnosticText({schema_errors:[{loc:["evidence",0,"source_start_id"],type:"missing"},{loc:["evidence",0,"source_end_id"],type:"string_type"}]}),"字段 evidence[0].source_start_id：缺少必填字段；字段 evidence[0].source_end_id：应为文本");
  assert.equal(diagnosticText({check:"source_interval_selection",selection_issues:[{check:{private:"body"}},{check:"private-body"}],invalid_candidate_count:true}),"原文区间选择不符合要求");
});

test("predefined source window diagnostics are distinct and preserve interval history", () => {
  assert.equal(diagnosticText({check:"source_window_catalog",source_window_catalog:[{quote:"private-source"}]}), "原文窗口目录完整性校验失败");
  const text = diagnosticText({check:"source_window_selection",candidate_index:2,invalid_candidate_count:3,field_path:"candidates[2].source_window_id",selection_issues:[
    {check:"unknown_source_window",candidate_index:2,source_window_id:"private-window"},
    {check:"invalid_source_window_catalog",candidate_index:3},
    {check:"invalid_source_window",candidate_index:4},
  ],source_window_catalog:[{id:"private-window",quote:"private-source"}],reason:"private-body"});
  assert.equal(text,"原文窗口选择不符合要求 · 第 3 条证据候选 · 窗口无效候选 3 · 字段 candidates[2].source_window_id · 第 3 条候选：所选窗口不属于本批原文目录；第 4 条候选：原文窗口目录无效；第 5 条候选：原文窗口定位校验失败");
  assert.doesNotMatch(text,/private|source_window_catalog/);
  assert.equal(diagnosticText({check:"schema",schema_errors:[{loc:["evidence",2,"source_window_id"],type:"literal_error",input:"private-window"}]}),"结构不符合要求 · 字段 evidence[2].source_window_id：不属于允许的选项");
  assert.equal(diagnosticText({check:"source_interval_selection",selection_issues:[{check:"cross_paragraph_interval",candidate_index:0}]}),"原文区间选择不符合要求 · 第 1 条候选：起终点跨越段落");
});

test("bound precision decision-map diagnostics show fixed candidate positions without records", () => {
  assert.equal(diagnosticText({check:"precision_duplicate_json_key",duplicate_key_count:2,duplicate_keys:["private-key"]}), "精筛 JSON 包含重复字段 · 重复 JSON 字段 2");
  assert.equal(diagnosticText({check:"precision_candidate_catalog",invalid_candidate_count:2,candidate_catalog:["private-id"]}), "精筛候选目录不完整或编号重复 · 精筛无效候选 2");
  const text = diagnosticText({check:"precision_decisions",candidate_index:2,invalid_candidate_count:2,field_path:"decisions[2].equivalent_to",selection_issues:[
    {check:"precision_duplicate_self",candidate_index:2,record:{reason:"private-body"}},
    {check:"precision_duplicate_target_not_retained",candidate_index:3,equivalent_to:"private-id"},
    {check:"precision_unexpected_equivalent_to",candidate_index:4},
  ],decisions:{"private-id":{reason:"private-body"}}});
  assert.equal(text,"精筛决定映射未通过完整性校验 · 第 3 条证据候选 · 精筛无效候选 2 · 字段 decisions[2].equivalent_to · 第 3 条候选：重复证据指向自身；第 4 条候选：重复证据未指向保留项；第 5 条候选：非重复项填写了替代引用");
  assert.doesNotMatch(text,/private|record/);
  assert.equal(diagnosticText({check:"precision_decisions",selection_issues:[{check:"precision_decision_coverage"}]}), "精筛决定映射未通过完整性校验 · 精筛决定未完整覆盖本批候选");
  assert.equal(diagnosticText({check:"schema",schema_errors:[{loc:["decisions",2,"equivalent_to"],type:"literal_error",input:"private-ref"},{loc:["candidate_catalog_id"],type:"literal_error",input:"private-binding"}]}), "结构不符合要求 · 字段 decisions[2].equivalent_to：不属于允许的选项；字段 candidate_catalog_id：不属于允许的选项");
  assert.equal(diagnosticText({check:"precision_decisions",field_path:"decisions.private-record.reason"}), "精筛决定映射未通过完整性校验");
});

test("capacity exhaustion remains distinct from the content repair bound and claims no unrecorded cap", () => {
  const text = diagnosticText({capacity_attempts:2,last_max_output_tokens:131072,max_output_tokens_limit:131072,repair_attempt:1,repair_exhausted:true,rawbody:"private"});
  assert.match(text, /已达 1 次定向修正上限/);
  assert.match(text, /已执行 2 次输出扩容重试.*最后输出额度 131,072 tokens.*本次配置输出上限 131,072 tokens.*已达本次可配置输出上限/);
  assert.doesNotMatch(text, /已用|选边|private/);
  assert.doesNotMatch(diagnosticText({capacity_attempts:1,last_max_output_tokens:65536}), /可配置输出上限/);
  assert.equal(capacityDiagnostic({max_output_tokens:32768}), "输出额度 32,768 tokens（上限，非已用）");
  assert.equal(capacityDiagnostic({attempt:"2",max_output_tokens:"65536",previous_max_output_tokens:-1}), "");
  assert.equal(eventDiagnostic({attempt:1,check:"schema"}), "结构不符合要求");
  assert.equal(eventDiagnostic({attempt:1,previous_max_output_tokens:32768,max_output_tokens:65536,details:{check:"source_selection"},response:"private"}), "第 1 次输出扩容重试 · 输出额度 32,768 → 65,536 tokens（上限，非已用） · 原文片段选择不符合要求");
});

test("unknown failures never stringify arbitrary objects, stacks or provider bodies", () => {
  const fallback = "研判失败：未获得可展示的错误说明，请查看任务记录。";
  for (const error of [
    undefined,
    null,
    "untrusted response body",
    new Error("private stack"),
    { rawbody: "private provider body", stack: "private stack" },
    { code: "unsafe", message: '{"rawbody":"private provider body"}' },
    { code: "unsafe", message: "Traceback (most recent call last): private stack" },
    { code: "unsafe", message: "Bearer private-credential-value" },
  ])
    assert.equal(formatRunError(error), fallback);
  assert.equal(
    formatRunError({ code: "validation_failed", stage: "unknown_private_stage", message: "结果未通过校验。" }),
    "研判失败：结果未通过校验。（错误代码：validation_failed）",
  );
  assert.equal(
    formatRunError({ code: "validation_failed", stage: "constructor", message: "结果未通过校验。" }),
    "研判失败：结果未通过校验。（错误代码：validation_failed）",
  );
});

test("diagnostic details never replace the safe primary message", () => {
  const formatted = formatRunError({
    code: "quote_span_mismatch",
    stage: "verify",
    message: "候选引用与原文位置不一致。",
    details: { rawbody: "private provider body", stack: "private stack", question_id: "internal-id" },
    rawbody: "private top-level body",
  });
  assert.equal(formatted, "原文核验失败：候选引用与原文位置不一致。（错误代码：quote_span_mismatch）");
  assert.doesNotMatch(formatted, /private|internal-id|rawbody|stack/);
});

test("snapshot dictionary content displays its business statement rather than JSON", () => {
  assert.equal(
    nodeText({
      kind: "judgment",
      content: { id: "internal", statement: "范围尚不能直接合并" },
    }),
    "范围尚不能直接合并",
  );
  assert.equal(
    nodeText({ kind: "conclusion", content: { conclusion: "当前证据不足" } }),
    "当前证据不足",
  );
  assert.equal(
    nodeText({
      kind: "evidence",
      content: { fact_des: "事实描述", quote: "逐字引用" },
    }),
    "事实描述",
  );
  assert.equal(
    nodeText({ kind: "evidence", content: { quote: "逐字引用" } }),
    "逐字引用",
  );
  assert.equal(
    nodeText({ kind: "judgment", content: { id: "internal" } }),
    "点击查看节点详情",
  );
});

test("actual snapshot edge relations have Chinese labels and conflict styling", () => {
  assert.equal(relationLabels.contradicts, "冲突");
  assert.equal(relationLabels.prerequisite, "前提");
  assert.equal(isConflictRelation("contradicts"), true);
  assert.equal(isConflictRelation("prerequisite"), false);
});

test("artifacts match internal or raw question IDs and only connected evidence quality", () => {
  const selected = { id: "question-a-uuid", raw_question_id: "q_001" };
  const rows = [
    { question_id: "question-a-uuid", conclusion: "当前题" },
    { question_id: "question-b-uuid", conclusion: "另一题" },
    {
      object_type: "conclusion",
      object_ref: { question_id: "q_001" },
      score: 0.5,
    },
    {
      object_type: "conclusion",
      object_ref: { question_id: "q_002" },
      score: 0.5,
    },
    {
      object_type: "evidence",
      object_ref: { doc_id: "doc_001", para_id: "p1", quote: "当前依据" },
    },
    {
      object_type: "evidence",
      object_ref: { doc_id: "doc_002", para_id: "p2", quote: "其他依据" },
    },
  ];
  const evidence = [
    {
      kind: "evidence",
      content: { raw_doc_id: "doc_001", raw_para_id: "p1", quote: "当前依据" },
    },
  ];
  assert.deepEqual(filterQuestionArtifacts(rows, selected, evidence), [
    rows[0],
    rows[2],
    rows[4],
  ]);
});

test("purpose-stage errors and fixed contract diagnostics are readable without provider content", () => {
  assert.equal(formatRunError({ code: "model_output_invalid", stage: "purpose_review", message: "用途审核未完整覆盖当前固定批次。" }),
    "事实用途审核失败：用途审核未完整覆盖当前固定批次。（错误代码：model_output_invalid）");
  assert.equal(diagnosticText({ check: "premise_purpose_batch_identity", response: "private", bank_id: "private-bank" }), "用途审核批次身份不一致");
  assert.equal(diagnosticText({ check: "premise_manual_purpose_unverified", reason: "private" }), "人工观点尚未取得用途许可");
  const text = diagnosticText({ check: "premise_purpose_contract", schema_errors: [
    { loc: ["decisions", "private-fact-id", "purpose_binding_id"], type: "literal_error", input: "private-response" },
    { loc: ["bank_id"], type: "missing" },
    { loc: ["decisions", "private-fact-id", "introduces_unstated_source_fact"], type: "bool_type" },
  ] });
  assert.match(text, /用途审核结构或逐项决定不完整/);
  assert.match(text, /decisions\.<field>\.purpose_binding_id/);
  assert.match(text, /字段 bank_id：缺少必填字段/);
  assert.match(text, /introduces_unstated_source_fact：应为布尔值/);
  assert.doesNotMatch(text, /private|response/);
});
