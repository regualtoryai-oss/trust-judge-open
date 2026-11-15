# Technical foundation

- Public design and scope: `docs/architecture.md`, `docs/evaluation.md`, and `docs/public-edition.md`. Original private competition design materials are not distributed.
- Runtime input may contain only raw source documents and projected questions. Never import `evidence_samples.json` or conclusion answers into the runtime database, retrieval index, prompts, or tests pretending to be blind evaluation.
- Preserve source doc/paragraph/question IDs. Use separate internal UUIDs.
- Source text and quote identities are immutable. Validate codepoint half-open spans before accepting evidence.
- Automatic model coordinates are hints: derive canonical coordinates only from a unique literal match in the given paragraph. Repeated quotes require a valid explicit location; never silently change human-selected spans or use fuzzy quote matching.
- `undetermined` is a valid result with empty supporting refs and quality gate 0. Asserted conclusions must still have qualified support; never fabricate support to make an unresolved case pass validation.
- Preserve safe error stage/code and known token counters on failures. Do not expose provider response bodies, hidden reasoning, authentication data or arbitrary exception text in user-facing errors.
- No fake success or silent provider fallback. Default model configuration is disabled. Deterministic providers belong to tests and must be explicitly injected.
- Persist tasks, events, complete graph revisions and provenance. Check cancellation/current revision before publication.
- Resume by forking a matching checkpoint and invoking it without fresh initial input. Preserve successful parallel writes and original usage; distinguish reused outputs from new calls. Normal retries must never inherit recovery or continuation control flags.
- Display only real execution stages and recorded tool calls. Event replay and focused graph projections must not invent progress, historical graph states or source evidence.
- Keep original competition documents and source samples unchanged.
- Tests must cover meaningful data isolation, span/graph integrity, idempotency, revision and failure behavior. Do not commit secrets or runtime databases.

# Evidence quality development

- 2026-10-19 user clarification: framework choice is open. Prefer existing agent frameworks (DSH / DeepSeek Harness is an explicit candidate); do not build a replacement agent loop or orchestration framework from scratch. Application-specific retrieval, evidence validation/export and framework plugins/adapters remain necessary integration work. Select by measured final precision/recall, failures and whole-episode token use, not allegiance to LangGraph or an assumption that DSH is inherently cheaper.
- Abstract rules from multiple topics and apply one runtime strategy. Do not add sample-specific selection or prompt branches keyed by a topic, person, publisher, year, source ID or reference label. Reading such values dynamically from the original question is allowed; it does not establish a fact or justify a special case.
- Single-question validation controls cost and diagnoses failures. It does not demonstrate overall quality or justify promotion. Freeze code and policy before scoring; report cross-topic gains and losses with complete denominators. Previously inspected public samples are retrospective regression data, never an unseen blind test.
- Keep candidate source-ID coverage, actual model reading scope, selected paragraphs, and final exported evidence/relations as separate measurements. Do not call candidate coverage final precision or recall, or non-reference IDs semantic false positives.
- 2026-10-20 latest user clarification supersedes the former hard 20,000-token rule: 20,000 per question is now a cost-saving soft target; modest excess is allowed for complete results, but avoid large paid runs. Freeze an explicit experimental stopping threshold and call limit before launch, count all failures/retries in the episode, and never reset use per stage. Do not automatically resume full-dataset paid runs; reuse raw-only caches and offline comparisons.
- 2026-10-20 user clarification: real validation must run through the actual workbench task/worker/publication path and verify its UI. Standalone probes, mocked tests and offline replay are preparation/diagnostics, not substitutes for end-to-end workbench results.
