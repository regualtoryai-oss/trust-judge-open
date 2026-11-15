# 隔离 DSH 证据实验入口

复用官方 `@deepseek-ai/dsh` / `dsh-sdk-client` **0.2.1-rc.2**。模型—工具循环、会话、事件和结束任务均由 DSH 执行；本目录只实现比赛业务工具、预算策略和 SDK 启动适配。尚未替换主工作台。

## 安装与启动

运行库留在 `.runtime/dsh-official-r1`，不使用其他项目的医药 fork 或 Companion 配置。`runtime/package-lock.json` 保存此次依赖锁：

```powershell
New-Item -ItemType Directory -Force .runtime/dsh-official-r1
Copy-Item integrations/dsh/runtime/package*.json .runtime/dsh-official-r1/
npm ci --prefix .runtime/dsh-official-r1 --ignore-scripts --no-audit --no-fund
node integrations/dsh/run_comparison.mjs <已冻结manifest.json>
```

真实启动只读取项目 `.env` 的 `TRUST_MODEL_API_KEY`，凭据只进入子进程环境，不写 profile、报告或产物。固定官方 Messages endpoint、V4 Pro、thinking off、provider retry 0。独立 home、原文输入和输出文件隔离；不暴露 shell、文件、网络、子智能体或自动摘要工具。

`--fixture` 使用预先声明的本地 SSE 假供应商，无真实网络；模拟 usage 不算真实用量。`--initialize-only` 仅握手，不发送问题。

## 预算与证据边界

整题使用同一 ledger：已用 + 实际序列化请求的官方 tokenizer 估算 + 输出预留 + 安全余量，不超过 20,000。输入估算不是服务端账单保证；最终以供应商 usage 核账。工具定义、历史和输出均计入。未知费用、截断、非预期网络或源文件变动均停止，不自动重试。

SDK启动成功不保证可选插件已激活。启动器必须先验证预算插件写出的独占readiness凭据及本轮source bindings，再提交问题；预算插件未就绪时不得向模型发送请求。

程序掌握原始 ID、完整段落、引用和 codepoint 坐标；模型只选择段落和提出注释、支撑/冲突关系。完整原文已送达并不等于语义正确。提交后使用 DSH 官方 `concludeTurn()`，避免额外确认调用。所有试验产物保持 `publishable:false`，未实施比赛所需独立 A/B 仲裁，不能冒充完整比赛交付。

受控对照目前优先使用 manifest 的 `contextMode:"full-candidate-context"`：完整候选池数据放入官方 `systemPrompt.section`，关闭插值，只注册终止提交工具。原生 fetch 前验证完整数据串和唯一提交工具；任何缺段、改字或增加搜索工具均不发送。必须先通过整题预算预检，不为凑预算删原文。该模式仅适合能装入预算的候选池，不能把 24 段开发样例等同全专题召回。

保留的 `tools` 模式先尝试动态预算下的完整候选包，过大才逐次搜索/读全文；精确重复预览仅回引用。前两题已证明让模型自由搜索仍会浪费历史输入，不能称该分支已解决成本或质量问题。没有默认恢复其付费测试。

历史参照与新输出须先 CLOSED、核费用及 SHA，独立离线进程再读参考。不得把参考、评分协议或参考 ID 混入工具目录和原始输入。

## 段落选择实验

manifest 可显式设置 `selectionPolicy:"source-local-v2"`，追加固定的段落内摘要、归属、限定和关系对象说明；默认 `baseline` 保持原提示。新增政策不增加工具或模型调用，文本与实现均进入来源 SHA 绑定，receipt 记录政策名和文本 SHA。它是待验证的选证政策，不能单凭提示修改宣称 P/R 已提高。

`native_fit_suite.mjs` 只提供 `prepare` 与 `run-fixtures`，用于封印输入的完整容纳性对照。执行强制本地假供应商，固定一请求、6,144输出预留、512安全余量和整题20,000预算；读官方 SDK 实际序列化请求的现有预算 guard 结果。假输出、假 usage 和 fit 均不能当比赛产物、真实用量或质量成绩。原历史18题协议及原文不分发，公开验证口径见[评测协议](../../docs/evaluation.md)。

可选 `contextMode:"staged-short-window"` 已接业务插件，要求 `selectionPolicy:"source-local-v2"`、`maxCalls:2`。首轮完整原题、完整标题和程序逐字160码点短窗，只开放选段；读取后撤下候选区，完整所选原段只由工具结果提供，次轮只开放提交。代码复用 DSH 原生循环、动态 section、工具 disposer 和结束接口。

阶段守卫在实际请求发送前核验完整数据串、唯一阶段工具、读取调用与结果的关联、全文结果仅一份、旧候选区不残留历史。读取前仅做内容成本预估负筛；后续实际原生请求仍须通过完整预算预检，不保证未来必然可容纳。预算服务通过 Cordis required service 完成初始化后，业务插件才能激活。阶段校验失败记录明确代码，不能包装成无结果的正常结束。

`run_staged_raw_fixture.mjs <新.runtime目录>` 只对固定04/q_002的40个原段做无网接口验收；预先模拟读取原序前6段及注释，**不代表模型选择或准确率**，没有付费开关。正式比赛流程、A/B仲裁和工作台暂停续跑尚未迁移。

## 验证

```powershell
node --test integrations/dsh/*.test.mjs
```

公开版不分发历史有限对照的原始资料与报告，也不据此宣称框架性能优劣。见[公开范围清单](../../docs/public-edition.md)。在任何新对照中保留首轮失败，不通过换目录把同题预算重置。
