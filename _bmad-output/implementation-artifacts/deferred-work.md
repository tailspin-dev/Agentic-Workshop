## Deferred from: code review of 1-triage-decision-schema.md (2026-09-26)

- Epic 1 SPEC.md still lists the route-match and sentence-count questions under Open Questions, and CAP-1's success line says any in-vocabulary route "passes". The story's human decisions (2026-09-26) reject mismatched routes and check only that the rationale is non-blank. Sync through `/bmad-spec` so Epic 2/3 builders reading SPEC.md see the same contract the code enforces.

## Deferred from: code review of 2-seed-loader.md (2026-09-26)

- `_bmad-output/implementation-artifacts/epic-1-context.md` still lists the route-match and sentence-count questions under "Still open in the spec". Story 1.1 settled both: `route_matches_category` rejects mismatched routes, and sentence count is not enforced. Regenerate the epic context after the SPEC.md sync above, so that Epic 2 and 3 builders don't read stale guidance.
