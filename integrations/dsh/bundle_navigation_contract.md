# Official DSH navigation and tool scheduling contract

2026-10-20; local `@deepseek-ai/*` 0.2.1-rc.2 source inspection. No network, model generation, reference-answer reads, or quality measurement.

## Chosen workbench change

Root selected a smaller route than a model-generated query batch: before the first model request, the program calls the existing whole-corpus search once with the **complete original question and limit 40**. That backend already performs multiple facets and word/bigram RRF. A dynamic official `systemPrompt.section` presents its deduplicated result; the first model can read directly. After a complete read, the next actual native wire replaces the preview section with address/title/hash receipts. The original `program-navigation.json` keeps all initial previews and their source binding, and every full read stays in the actual conversation. No assistant or tool event is fabricated for this program search.

The installed public `SystemPrompt.section` interface returns an exact disposer (`dsh-system-prompt/lib/types/index.d.ts:250`), and its `PromptSection.text` accepts a function evaluated at each assembly (`:56`). The existing official-SDK dynamic-context fixture proves that this route removes old preview text from the actual subsequent native request, including its history, while the full read remains once. The new production provider tests also check this on actual native wires.

The workbench now uses that dynamic section, not the surface replacement primitive below. It retains both read bursts. Definitions remain registered throughout the returned call list; official `system-prompt/assemble` filters visible tools, and `agent/pre-step` updates the execution view. Official `tools.guard` denies additional same-step reads as `DEFERRED` before the codec; their source range is explicitly **not read** and their read allowance is not consumed.

## Earlier alternatives considered

1. One search invocation accepts a small array of query strings. The program owns per-query and whole-question candidate limits and produces one deduplicated navigation table. Model-supplied limits cannot expand the payload. Only visible candidate anchors are allowed for reading; omitted candidates remain unknown.
2. After a complete read, replace old search navigation results with small hash receipts through the official **logged surface replacement** API. Keep the original immutable search events and every full read message. This is context selection, never evidence deletion or quote truncation.
3. Permit one business `read` dispatch per model step through the official tool guard. Two separate read steps remain possible. Additional same-step requests receive explicit denials before the Python codec, so they do not consume its read allowance. Refresh tool registrations at the following pre-step, avoiding tools disappearing midway through the already-produced call list.

The candidate cap is a cost policy requiring explicit freeze. It does not establish recall; subsequent real-workbench comparisons must measure that trade-off.

## Exact local API evidence

All paths below are relative to `.runtime/dsh-official-r1/node_modules/@deepseek-ai/`.

| Requirement | Public interface and implementation |
|---|---|
| Immutable original log plus model-only replacement | `dsh-session/lib/types/types.d.ts:449–465`: positional `SurfaceOp` replace, including “any surface-replacing producer”; `sourceEventSeqs` lists the shadowed events. |
| Append and reconstruct | `dsh-session/lib/types/index.d.ts:277`: `Session.append(type, data, intent)`; `:347`: `deriveMessages()`; `:61`: `session/event(session,event)` feed. |
| Preserve tool identity | `dsh-session/lib/index.js:394–410`: a `tool/result` replacement must target exactly one current result and may change only message content. Call ID, result state, turn/step, and other identity fields are protected. |
| Apply before actual request assembly | `dsh-agent/lib/types/runtime-types.d.ts:304`: `agent/pre-step`; `dsh-agent-loop/lib/index.js:1055–1078` subsequently commits logged inputs and builds the request; `:1274` reads `session.deriveMessages()`. |
| Avoid unsupported wire mutation | `dsh-agent/lib/types/runtime-types.d.ts:319`: `agent/request` cannot mutate messages. `dsh-llm/lib/types/index.d.ts:39`: loop-built requests are frozen; stream middleware reads them rather than rewriting them. |
| Official per-tool denial | `dsh-tools/lib/types/index.d.ts:47`: `tools/pre-execute`; `:656`: monotonic `ctx.tools.guard(guard)`, which returns a denial reason before tool execution. |
| Parallelism is not call count | `dsh-agent-loop/lib/types/index.d.ts:85`: `maxParallelToolCalls`; `dsh-agent-loop/lib/index.js:506–537`: iterates the entire assistant call list even when execution is exclusive/serial. |
| No native single-call option in this installed adapter | `dsh-llm-deepseek/lib/index.js:1715–1731` serializes Messages request fields, but has no `parallel_tool_calls`, `disable_parallel_tool_use`, or `tool_choice`. A business guard controls actual execution, not what the model originally emits. |

`startsRequestSeries` (`runtime-types.d.ts:98`) changes the request/tool declaration series. `buildRequest` still derives the existing session surface; it is **not** a history reset.

## Verified official replacement primitive

```js
session.append('tool/result', {
  ...original.data,
  message: { ...original.data.message, content: receiptBlocks },
}, {
  surfaceOp: { op: 'replace', startSeq: original.seq, endSeq: original.seq },
  sourceEventSeqs: [original.seq],
});
```

`node --test integrations/dsh/bundle_context_contract.test.mjs`: three passing tests using the installed official `Session` class. They verify that old navigation is absent from the derived model view while its original event remains unchanged, the full read message is byte-identical, changed call identity is rejected, and replacing a range spanning the read is rejected. This is an API-contract test, not a real-model or P/R result.
