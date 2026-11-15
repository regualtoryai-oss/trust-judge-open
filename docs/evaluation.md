# Evaluation protocol

## Published evidence and current limits

This edition publishes code, synthetic source-only import fixtures, engineering
regressions and a toy offline scoring artifact. It does **not** publish a real
model benchmark, an official competition score, a blind holdout, or a calibrated
forecasting result. Historical exploratory quality reports used inspected
development samples and are not redistributed. No consistent frozen public
baseline result is currently claimed.

### 1. Engineering verification (no model)

Run `uv run python scripts/prepare_tokenizer_cache.py` once during dependency
preparation. It may fetch the public `cl100k_base` table and verifies its pinned
SHA-256; it makes no model call. Set `TIKTOKEN_CACHE_DIR` when a dedicated cache
is desired. All following pytest processes continue to reject external network
attempts; real encoding counts are not replaced by synthetic counters in the
token-proxy invariance tests.

```sh
uv run python scripts/demo_offline.py
uv run pytest -q tests/test_import_scenarios.py tests/test_data.py tests/test_domain.py
uv run pytest -q tests/test_repository.py tests/test_workflow_resume.py tests/test_export_readiness.py
```

The original synthetic inputs are under `fixtures/import_samples/`, containing
only raw documents and questions. Test providers are explicitly injected and
must not be enabled implicitly in the normal API. Cases cover duplicate raw
IDs, immutable text, answer-field isolation, Unicode coordinates, missing
evidence, repeated quotes and invalid references. Passing these checks is a
statement about those engineering properties, not a model accuracy score.

### 2. Reproduce the offline scorer (no model)

```sh
uv run python scripts/evaluate_paragraphs.py --bundle evaluation-fixtures/toy-bundle.json --output .runtime/public-toy-score
```

Use a new output directory on each run. The bundle is an explicitly hand-built
scoring example, **not model output**. It has two reference records and two
submitted records with one matching `(doc_id, para_id)` pair. The expected
**ID-only** precision, recall and F1 are each `0.5`. This intentionally shows
both a false-positive location and a missed reference location. Semantic labels
are absent, so semantic point metrics must remain unknown rather than taking
the ID-only score as semantic accuracy.

`evaluation-fixtures/` is offline evaluator input. It is separate from runtime
source-only `fixtures/import_samples/`; never import reference annotations or
scorer outputs into the application, prompts or retrieval index.

### 3. Protocol for a future real-model comparison

Before starting a paid run, freeze code commit, source bytes, dataset split,
questions, model/provider configuration, prompt version, request/cost limits and
evaluation method. Keep raw-only runtime inputs and gold annotations in separate
processes/stores. Previously inspected development cases cannot be relabelled
as unseen holdout data. Record SHA-256 guards for sources, outputs and scorer.

Compare the same full question set and denominators across a declared baseline
and candidate. Preserve failed, cancelled and unfinished questions in completion
and cost reporting. A subset improvement is not a whole-system improvement.

For deterministic ID-only diagnostics, use one-to-one matching:

- Precision = confirmed matches / all scoped submitted records.
- Recall = confirmed matches / all scoped reference records.
- F1 = `2 * matches / (submitted + reference)`.
- Preserve duplicate reference records in the raw denominator; show unique
  paragraph metrics separately. Empty denominators are `null`, not fabricated
  success. Reference mapping issues must remain visible.

Same paragraph identity is not semantic equivalence. Semantic evaluation must
also check subject, time, value and statistical scope against the original
quotes. Existing semantic-judgment cache entries must bind to frozen source and
prompt hashes. Unknown/unjudged pairs stay unknown; independent model judges
remain fallible development proxies. Evidence selection, supporting/conflicting
relations, conclusion correctness and completion rate are separate measures.

## Failure, cost and latency reporting

Keep each attempted request, failure stage/code, retry, recovery and known
provider usage in the whole episode. Do not reset token costs per stage or omit
failed calls. Reused checkpoint work must be distinguished from new calls.
If usage or price is unavailable, report it as unknown. Report wall-clock
latency with dataset size and configuration; do not turn synthetic usage or
capacity reservations into real costs or performance results.

Probability calibration, Brier score, forecast locking and event-resolution
backtesting are not implemented or claimed by this project. They would require
a separate forecasting evaluation protocol and dated real outcomes.
