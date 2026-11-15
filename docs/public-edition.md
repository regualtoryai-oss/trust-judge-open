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

## CI partition

`archival` marks exactly two modules: `tests/test_bundle_archived_replay.py`
and `tests/test_lc_bundle_workbench.py`. They deliberately run isolated frozen
source/dependency replay processes, including missing-dependency and cleanup
checks. CI runs these in an **Archived proof integrity** job; all other backend
cases run in **Core engineering**. Both jobs run on every push/PR and must pass.

```sh
uv run pytest -q --maxfail=1 -m 'not archival'
uv run pytest -q --maxfail=1 -m archival
```

These are disjoint selections whose union is the default collected test set.
Unqualified `uv run pytest -q` continues to execute both groups. External-asset
optional skips and the absent-private-history skip retain their explicit
reasons; CI partitioning does not turn those into successful integrations.

The 2026-11-14 collection-only audit found **3,749** default cases: **3,704**
core and **45** archival, with an empty intersection and no missing/extra case.
These are collection counts, not a claim that all tests have passed.

## Verification interpretation

Local tests use synthetic providers and source fixtures. GitHub Actions is
configured to repeat those checks; its badge/result is authoritative only after
the public repository's workflow actually completes. A workflow definition is
not evidence that CI has passed. No new real model request is needed for the
public-edition checks.
