# Public edition manifest

Prepared on 2026-11-14 from private development revision
`dd02aee0b819f91eee06f612161482ebe00d698d`. The public Git repository starts with
a clean history; this identifier documents code provenance without publishing
the private repository's objects or its archived source materials.

## Included

- FastAPI application, Vue workbench, schema contracts and dependency locks.
- Original synthetic source-only import examples and engineering regressions.
- Optional application-specific DSH adapters, without an installed SDK tree.
- Generic deterministic evaluator code and an explicit hand-built scoring demo.
- Architecture, an implemented-design ADR, evaluation protocol and CI.

## Excluded

- The private Git history and private repository configuration.
- Original competition documents, organizer sample sources and answer files.
- Historical research/evaluation reports, raw model responses and paid caches.
- `.env`, databases, checkpoints, exports, real account settings and model assets.
- `scripts/run_obligation_selection_probe.py` and its private-boundary runner
  test: the historical paid-experiment launcher imports a frozen local research
  harness that is not distributed. The generic workflow and source-only tools
  remain available; no replacement paid launcher is advertised.

The historical offline reference mapper's four generic functions are retained
in `scripts/reference_mapping.py`, with a pinned SHA guard in
`scripts/offline_reference_tools.py`. This removes a dependency on an excluded
private report without changing its mapping behavior.

Explicit official-SDK integration tests are marked
`optional_dsh_runtime`. They require the original pinned tokenizer/Python runtime
assets that are not redistributed, so the public default suite skips them with
an explicit reason. `--run-dsh-integration` opts in only after those assets have
been independently provisioned. Pure contract and mocked transport tests remain
in the default suite; the skip does not claim the native integration passed.

Local HTTP fixture integrations explicitly inject a synthetic planning counter
through `tests/synthetic_planning.py`, consistent with their mocked wire-token
counter. Those cases verify budget plumbing and accounting states, not real
tokenization, provider capacity or billing. Production tokenizer guards are
unchanged and fail closed when the required asset is unavailable.

The archived-code replay test for a fixed private historical Git commit skips
when that commit is absent. The clean public repository deliberately contains
no original private Git objects; current-source replay regressions still run.

## Verification interpretation

Local tests use synthetic providers and source fixtures. GitHub Actions is
configured to repeat those checks; its badge/result is authoritative only after
the public repository's workflow actually completes. A workflow definition is
not evidence that CI has passed. No new real model request is needed for the
public-edition checks.
