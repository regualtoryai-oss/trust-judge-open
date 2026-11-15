# 原文束工具与显式工作台接线

这一分支复用官方 DSH `0.2.1-rc.2` 的工具循环、工具注册/撤销和 `concludeTurn()`。旧 `profile.mjs`、`run_comparison.mjs` 和主工作台默认路径保持不变。早期 `buildBundleFixturePatch`/`bundle_guard.mjs` 仍强制无网；新增显式工作台启动接口见下节，不能把两种模式混为一次真实测试。

## 显式工作台接口（2026-10-20）

`app/agents/bundle_transport.py::BundleWorkbenchProvider` 只在 `model_source_bundle_enabled=True` 且官方模型配置一致时构造。`select_source_bundles(question,documents,paragraphs,excluded_spans)` 固定唯一原题、完整原文、可信 manifest、源码归档、费用父链和事前预算，返回 `{artifact,proof,usage_report}`；原生循环由官方 SDK 执行，后续沿原有独立 A/B 和 scoped 仲裁三调用。

`bundle_workbench_launcher.mjs` 只接受绝对 launch manifest 路径与 SHA。新 `bundle_workbench_guard.mjs` 独立校验 workbench intent、唯一题目、source SHA 和 readiness；生产密钥仅通过受控环境传入，不读任意文件或写入报告。默认配置不启用这条路径。所有 native request 原始字节、原始响应流、工具 transcript、usage/presence journal 都单独保存；没有 submit 不补一个结果。

前半实际原生请求的保护线是 `min(front_token_stop,whole_token_stop-tail_reserved_tokens)`，已知父费用也从同一线扣减。每次按真正 native wire 和当前允许工具做输出预留：只有导航时用 navigation capacity；允许提前 submit 后必须预留 submission capacity，即使该次实际选择再读一次。尾部只有在所有 native 费用完整结算后才一次 seed `父全部记录+本次每个实际 native 请求`；失败和未知费用不重置，父费用不作为本次新用量重复写入。已完成 source checkpoint 直接续进尾部时同样 lazy seed 父记录。

原生 source 阶段中途取消会保留官方会话与 journal；尚未实现该半程的原生会话恢复，下一次重新启动并累计父费用。已经完成的 LangGraph source 节点可以复用经过重放校验的结果。工具与每个实际 native 请求通过标准 `tool_started/completed/failed` 流式展示，另外保留审计事件。

### 工作台导航更新

生产 bundle profile 现在先程序化调用一次完整原题的本地多 facet / word / bigram 排名融合检索，固定展示40个去重候选（不足40则全部）。模型不能扩展每个 query 的条数，也没有付费 search 规划回合。`program-navigation.json` 保存真实完成时间、完整原始结果、初始40条预览及其后撤回视图；本地检索完成单独显示，明确模型调用为0，不伪造开始时间或模型工具调用。

初始导航使用官方动态 `systemPrompt.section`。一次完整 read 后，下一次真实 wire 撤回80码点预览，只保留候选地址、文档标题与未知状态收据，便于第二批读取。原题绑定和所有完整 read tool result 始终保留。工具定义在同一响应执行期间保持注册，官方 assembly 按阶段过滤模型可见工具；官方 `tools.guard` 每模型 step 至多执行一次 read，多余调用明确 `DEFERRED`，未读取且不消耗 Python read 限额。仍允许两个不同 step 各读一批。

新增完整验收实际运行官方 SDK → 正常 StateGraph → 真 `Worker`；尾部仅 HTTP Mock，分别验证完整结果发布与未读地址拒绝后不发布。前半一批 read 的模拟请求由 search/read/submit 三次减为 read/submit 两次；两批仍可 read/read/submit。所有这些验收都无网络、无付费，不提供 P/R 提升结论。即使源码正确，真实题目的预冻结预算仍须包含全部父费用、全文输入和真实每阶段输出预留。

新增 provider 的全入口零费测试实际运行官方 SDK fixture → `run_analysis` StateGraph → 来源重放 → HTTP Mock A/B/仲裁；也覆盖已完成 source 的费用承接、无 submit、未知 usage、前半预算阻断和运行中取消。与相邻模块合计 **151 项 Python、18 项 Node** 通过；模拟 usage 不是账单。本模块开发阶段没有调用付费模型，没有新的 P/R 数据。整题实际预算与准确率仍须通过用户要求的真实工作台实验确认。

## 业务接口

- `scripts/source_bundle_tool.py <absolute_case_path> <case_sha256>`：stdin 仅接收 `{operation, arguments, history}`。原始 case 包含 `format: frozen-source-bundle-case-v1`、原题、完整 documents/paragraphs、可信原段 manifest、limits。调用端固定文件与脚本 SHA；模型参数没有路径、shell 或文件读取能力。
- `bundle_bridge.mjs`：受控 `execFile` 调用本机 Python，子进程环境排除凭据与 Python 注入配置。每次重放带结果哈希的全部历史，失败读取也消耗原有限额；历史不重复保存全文，只保留工具参数、状态与 canonical JSON 哈希。正文仍按真实 read tool result 传给框架。
- 当前工作台模型仅能调用 `read_source_bundles`、`submit_bundle_evidence`，首次只有 read；读取后开放 submit，两批后只显示 submit。早期隔离 fixture profile 保留 search/read/submit 原协议，用于旧协议回归，不能冒充当前生产策略。
- submit 只接受原始起止段 ID、`fact_des`、六项 `argument`、`entities`、待核验的 `proposed_roles`。传入 quote、offset、hash 或未知字段会失败。同一原始起点只能一条，重复起点而终点不同也明确拒绝，不合并或扩大范围。程序从原始文档连续切片，引用只能属于一次成功全文读取的完整范围，不能跨两批拼造读取证明。
- 所有产物保持 `proposed_unverified`、`publishable:false`。来源绑定与实际送达验证不表示注释、支持/冲突角色或结论正确。
- 桥复用现有数字字面检查，并沿严格注释验证的基础规则拒绝：`argument.内容` 为空、时间不是本条完整原文束中的字面文本、注释/argument/entities 新增原文未出现的数字。只拒绝，不自动补值或付费修复；这不检验语义真值和数字的主体归属。

## 原生送达与预算

`bundle_presence.mjs` 对实际 HTTP 发送前的 native Messages body 校验：完整原题、当前工具集合、每次 read 的 `tool_use_id` 与参数、唯一且未裁剪的全文 `tool_result`。已有预览、后台已读标记或单独离线重放都不能替代它。submit 必须来自已经通过当前请求校验的状态；同一响应中先 read 再直接 submit 会被拒绝。

新 guard 复用原有 `EpisodeBudgetGuard`、实际原生请求序列化和固定官方 tokenizer。原文、脚本和实现 SHA 在请求前复核。送达失败记录具体原因并在发送前拒绝；读取超码点/字节上限整批失败，不删中间段或缩短引用。

**完整历史仍有成本。** 两批全文都保留在 DSH 历史中，后续请求会再次计入输入；动态 system section 并不会删除历史 tool body。上节的新工作台适配已经实现预留、费用继承与 checkpoint 阶段边界；早期 fixture 不能证明真实赛题的预算可容纳性，不能简单去掉旧 guard 的 fixture 限制来替代新入口。

## 已完成的零付费验证

```powershell
.venv\Scripts\python.exe -m pytest tests/test_source_bundle_tool_bridge.py -q
node --test integrations/dsh/bundle_presence.test.mjs integrations/dsh/bundle_bridge.test.mjs integrations/dsh/bundle_sdk.test.mjs
```

Python 22 项与 Node 18 项通过。Node 包含实际已安装官方 SDK/loop/serializer 的无网络合成验收：search → read → read → submit，4 次模拟供应商请求，实际 native 请求的完整 read 数依次为 `[0, 0, 1, 2]`，最后仅保留 submit 工具；原段间 CRLF、非换行分隔符与 emoji 的文档码点原样保存。

首份闭合记录：`.runtime/dsh-bundle-two-burst-20261020-r1/closed-receipt.json`。真实 API 请求 0、付费 tokens 0；模拟 usage 600 不是账单。此前冻结的四次 native 输入估算为 759、1840、2887、3419 tokens，仅用于说明历史输入会递增，不能当实际赛题成本。这是工具协议验证，**不是用户要求的真实工作台质量测试，没有 P/R 成绩**。worker 接线及整题预算完成后，仍须按用户要求通过实际工作台运行、发布、导出，再独立离线评分。
