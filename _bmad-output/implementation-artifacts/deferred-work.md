## Deferred from: code review of 1-triage-decision-schema.md (2026-09-26)

- Epic 1 SPEC.md still lists the route-match and sentence-count questions under Open Questions, and CAP-1's success line says any in-vocabulary route "passes". The story's human decisions (2026-09-26) reject mismatched routes and check only that the rationale is non-blank. Sync through `/bmad-spec` so Epic 2/3 builders reading SPEC.md see the same contract the code enforces.

## Deferred from: code review of 2-seed-loader.md (2026-09-26)

- `_bmad-output/implementation-artifacts/epic-1-context.md` still lists the route-match and sentence-count questions under "Still open in the spec". Story 1.1 settled both: `route_matches_category` rejects mismatched routes, and sentence count is not enforced. Regenerate the epic context after the SPEC.md sync above, so that Epic 2 and 3 builders don't read stale guidance.

## Deferred from: code review of 2-human-gated-escalation.md (2026-09-26)

- source_spec: `_bmad-output/specs/spec-epic-2/stories/2-human-gated-escalation.md`
  summary: Confirm that a resumed escalation run (invoke plus Command resume) stays in one MLflow trace under run_agent.py's triage span.
  evidence: Unverified, medium if false; Epic 3's tool_order scorer depends on it. Settle it by running `run_agent.py T-1044`, answering yes, and checking that the trace has a single root with both ainvoke calls nested.
- source_spec: `_bmad-output/specs/spec-epic-2/stories/2-human-gated-escalation.md`
  summary: Add a concurrency test that the default terminal approver doesn't block the event loop while waiting for input.
  evidence: Removing the thread offload passes all tests, but nothing else runs on the loop in the CLI path today, so it becomes relevant only if triage is run concurrently.
- source_spec: `_bmad-output/specs/spec-epic-2/stories/2-human-gated-escalation.md`
  summary: Silence or fix the `MlflowLangchainTracer.on_interrupt/on_resume` AttributeError noise printed on every escalation pause and resume.
  evidence: Seen live on T-1044 (2026-09-26). The installed MLflow tracer doesn't implement LangGraph's interrupt/resume callbacks. Runs and traces are unaffected, but the error text sits next to the `Escalate? [y/N]` prompt. Fix by upgrading MLflow once it supports these callbacks, or by filtering this log in run_agent.py.
- source_spec: `_bmad-output/specs/spec-epic-2/stories/2-human-gated-escalation.md`
  summary: RESOLVED: the one-trace check for resumed escalation runs (earlier entry).
  evidence: Live T-1044 runs on Groq each produced one trace with a single `triage` root and both LangGraph invokes nested under it.
