# ADR 001: Immutable sources and explicit publication revisions

Status: **implemented in the existing workbench; public documentation added
2026-11-14**.

## Problem

A research answer can look correct while its evidence points at the wrong
paragraph, a rewritten quote, another topic with the same raw ID, or the result
of a cancelled task. Browser refresh and retry make these errors harder to see.

## Decision

Preserve imported documents, paragraphs and questions as immutable source
records. Keep external source IDs separate from internal UUIDs. Validate literal
quotes with Unicode codepoint half-open spans. Keep checkpoint state separate
from business publication revisions; publish a complete graph only after
validating its task state, evidence references and current revision.

Retries create their own execution history. Checkpoint continuation explicitly
records which completed work is reused and which calls are new. The UI replays
recorded events, but does not invent historic graph states.

## Alternatives considered

- Mutate the current answer/evidence rows in place: fewer tables and simpler UI,
  but weak reproducibility and harder race/conflict diagnosis.
- Treat the LangGraph checkpoint as the user-facing result: easier integration,
  but incomplete/intermediate state could appear published.
- Repair model quotes using fuzzy text matching: more apparent completion, but
  may silently relocate a citation and weaken auditability.

## Consequences

The design adds version-aware queries and explicit conflict handling. It allows
stable exports and source inspection after retries, and prevents an apparently
successful response from bypassing provenance checks. It does not ensure the
chosen passage semantically supports the conclusion; that requires independent
evaluation and human review.

Relevant implementation: `app/data/ingest.py`, `app/repository.py`,
`app/domain/spans.py`, `app/workflows/checkpoint_resume.py`, and
`app/workflows/analysis.py`.
