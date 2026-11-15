# Architecture and review guide

## System boundary

Trust Judge is a local, single-operator evidence-research workbench. The browser
uses the FastAPI `/api` boundary; the API persists requests, and a separate
worker executes the LangGraph pipeline. SQLite business records and LangGraph
checkpoints are separate stores under the ignored `.runtime/` directory.

| Boundary | Responsibility | Failure behavior |
| --- | --- | --- |
| Import preview/commit | Validate source IDs, raw paragraphs, questions, size and excluded fields | Reject invalid input before partial import; no model call |
| Task creation | Validate scope, idempotency and base revision | Return explicit conflict/validation error |
| Extraction/verification | Map model choices to immutable original text and codepoint spans | Invalid or ambiguous citations cannot silently become evidence |
| Independent A/B analysis | Consume the same frozen evidence snapshot | Record stage and known usage on failure |
| Arbitration/publication | Check relations and support in the current task/revision | Do not publish an invalid or cancelled result |
| Export | Bind JSON/ZIP to a fixed published revision | Incomplete work cannot masquerade as a complete submission |

Original document IDs remain provenance identifiers. Internal UUIDs prevent two
topics containing `doc_001` from colliding. Text is immutable once imported;
derived interpretations and publication versions are separate records.

## Reviewer walkthrough

1. Run `scripts/demo_offline.py`: inspect input schema, import idempotency and
   Unicode codepoint verification without any model or credential.
2. Read `tests/test_import_scenarios.py`: canned providers are explicitly
   injected, while the API/worker/graph/export paths are real application code.
3. Inspect `app/workflows/analysis.py` and
   `app/workflows/checkpoint_resume.py`: distinguish retry, checkpoint reuse,
   cancellation and publication revision checks.
4. Inspect `app/domain/spans.py`, `app/data/ingest.py` and `app/repository.py`:
   source identity and relation integrity must survive model-output failures.
5. Run frontend tests and build; with a local API, import one original synthetic
   sample through the UI and inspect the source drawer.

## Security and operating limits

The service defaults to loopback and has no production multi-tenant login. Do
not expose it directly to the Internet. A production deployment would require
identity, authorization, resource isolation, request quotas, deployment secrets,
backup/recovery and operational review. Those capabilities are not claimed by
this public edition.

Provider text, source documents and source-embedded instructions are untrusted
inputs. The project validates structural and provenance constraints; this is not
a proof of complete prompt-injection resistance or semantic correctness.

Optional DSH adapters and local-rerank experiments are not the default runtime.
Their SDK/model/tokenizer assets are separately installed, and historical
archive-bound experiments are excluded from this release.

## Contribution attribution

Application integration, Vue interface, business schema/provenance rules,
workflow adapters and engineering regressions are the project author's work,
implemented with AI coding assistance. LangGraph owns its orchestration
primitive and checkpoint APIs; provider models own text generation. The project
does not claim to have authored those frameworks or trained a foundation model.

See [ADR 001](adr/001-immutable-sources-and-versioned-publication.md) for the main
storage/publication tradeoff and [evaluation protocol](evaluation.md) for the
separate quality-verification boundary.
