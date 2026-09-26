---
title: 'The eval run and the four code scorers'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '18052b228fac0cbf2f36a1fbbfa8d123aab9ce82'
context: ['{project-root}/_bmad-output/specs/spec-epic-3/SPEC.md', '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Nothing measures the Epic 2 agent. There is no eval run, no scores against the 20 labelled tickets, and escalating tickets would block a run waiting for a person (SPEC CAP-1 to CAP-5, CAP-8).

**Approach:** `eval/run_eval.py` builds a dataset from all 20 CSV rows. It runs `agent.triage` through `mlflow.genai.evaluate` with an auto-approving approver and a traced predict function. It scores each ticket with `valid_schema`, `category_match`, `priority_match` and `tool_order`, logging one run to the `triage-agent` experiment.

## Boundaries & Constraints

**Always:**
- Set tracking URI `sqlite:///mlflow.db` and experiment `triage-agent`, and call `mlflow.langchain.autolog()` and `load_dotenv()`.
- Wrap `predict_fn` with `@mlflow.trace`, so each ticket's agent run, including an escalation resume, is one trace.
- `predict_fn(ticket_id)` runs `asyncio.run(agent.triage(ticket_id, approve=auto_approve))`.
- `auto_approve` returns `True` and increments a thread-safe escalation counter that Story 3.2 will report.
- Set `MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION=True` before evaluating. Without it, MLflow re-runs the first ticket as a check, which costs an extra agent run.
- A ticket whose agent run raises stays in the run with `outputs=None` and scores 0 on the three output scorers. The eval continues.
- Scorers are `@mlflow.genai.scorer` functions returning 0/1:
  - `valid_schema`: `TriageDecision.model_validate(outputs)` succeeds;
  - `category_match`: `outputs["category"] == expectations["expected_category"]`;
  - `priority_match`: the same, with `expected_priority`;
  - `tool_order`: the earliest `get_ticket` span in the trace starts before the earliest `get_customer_history` span. 0 if either is missing.
- Resolve paths from the script's location, so the command works from any directory.
- Keep the scorer list and the escalation counter easy for Story 3.2 to extend and read.

**Never:**
- Edit `agent.py`, `run_agent.py`, `eval/labelled_tickets.csv`, `TRIAGE_POLICY.md`, `triage_schema.py`, `mcp/`, `seed/`.
- Hand-roll the scoring loop.
- Add the rationale judge, the printed means report or `eval/latest_report.json` (Story 3.2).
- Make network calls in scorers.
- Read terminal input.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Full run | 20 CSV rows, working model key | One MLflow run in `triage-agent`, 20 traces, four `<scorer>/mean` metrics | N/A |
| Escalating ticket | T-1044, T-1048, T-1057 | Auto-approved, no terminal read, counter +1 each, one trace per ticket | N/A |
| Correct output | `access`/`P1`/`access-team` for T-1044 | All four scorers 1 | N/A |
| Wrong priority | P2 where P1 expected | `priority_match` 0; the others unaffected | N/A |
| Agent raises | `GroundingError`/`EscalationError`/`TriageOutputError` | `outputs=None`; the output scorers are 0; `tool_order` still reads the trace | Eval continues |
| Order reversed or missing | Trace has `get_customer_history` first, or no `get_ticket` | `tool_order` 0 | N/A |
| Rate limited | provider raises a rate-limit error (Groq `RateLimitError`, Gemini `GoogleRateLimitError` or a 429) | Ticket retried after a wait; the scores reflect the eventual run | After 5 retries the error propagates (the ticket scores 0) |

**Decision (human, 2026-09-26): run one ticket at a time, with retries.**
- Set `MLFLOW_GENAI_EVAL_MAX_WORKERS=1` before evaluating.
- `predict_fn` retries `agent.triage` for that ticket on a provider rate-limit error, and on no other error:
  - wait for the provider's retry hint when one is parsable, otherwise 5s, 10s, 20s and so on;
  - give up after 5 retries.
- The retries stay inside the ticket's single trace.
- Run the eval with `PROVIDER=groq`, because Gemini's free daily cap can't cover 20 tickets.

**Decision (human, 2026-09-26):** keep the full spec as one story.

</frozen-after-approval>

## Code Map

- `agent.py` -- read-only. `async triage(ticket_id, approve=None) -> dict`:
  - it returns the four `TriageDecision` fields;
  - it raises `TriageError` subclasses;
  - the approver gets `{"name", "args"}` and must return exactly `True`;
  - it prints `Escalated to a person: ...` to stderr;
  - `build_model()` picks the provider from `PROVIDER`.
- `triage_schema.py` -- `TriageDecision` (strict, `extra="forbid"`, route must match category).
- `eval/labelled_tickets.csv` -- read-only, 20 rows. Columns `ticket_id,expected_category,expected_priority,expected_tools,judge_notes`. Put `ticket_id` in `inputs` and the other four columns in `expectations`, so Story 3.2's judge can read `judge_notes`.
- `mlflow.genai.evaluate(data=[{"inputs": {...}, "expectations": {...}}], scorers=[...], predict_fn=...)` (MLflow 3.16). Verified offline on 2026-09-26:
  - `predict_fn` is called with `**inputs`;
  - a raising `predict_fn` yields `outputs=None` and the run continues;
  - result metrics are keyed `<scorer>/mean`;
  - exactly one run is logged.
  - Without `@mlflow.trace` on `predict_fn`, each top-level span becomes its own trace and the scorer sees only one.
  - Trace-validation re-runs the first sample unless `MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION` is set.
- Scorer signature: `@scorer def name(inputs, outputs, expectations, trace) -> int`, taking only the arguments it needs. For `tool_order`, use `trace.data.spans`, with `.name` and `.start_time_ns`.
- Rate-limit exception classes (installed versions): `groq.RateLimitError`, and `langchain_google_genai.chat_models.GoogleRateLimitError`. The Gemini message carries a `Please retry in 51.7s` hint.
- `run_agent.py` -- the reference for MLflow setup (same URI and experiment, `autolog`).
- `tests/` -- plain pytest. `pyproject.toml` puts the repo root on `pythonpath`, but `eval/` isn't a package: import `run_eval` in tests via `importlib.util.spec_from_file_location` from its path. `run_eval.py` puts the repo root on `sys.path` so `import agent` works when it's run as a script.

## Tasks & Acceptance

**Execution:**
- [x] `eval/run_eval.py` -- the script as described above, with `main()` under `__main__`. The dataset builder and the scorers are module-level, so tests can import them.
- [x] `tests/test_run_eval.py` -- offline tests with no network or keys:
  - the dataset has 20 rows with the right `inputs`/`expectations`;
  - each scorer's 1 and 0 cases, including `None` outputs;
  - `tool_order` on real traces built in a temporary MLflow store: right order, reversed, missing;
  - `auto_approve` returns `True` and counts;
  - `predict_fn` with a stubbed `agent.triage` passes `auto_approve`, returns the dict, and lets errors propagate;
  - an end-to-end `mlflow.genai.evaluate` with a stubbed `triage` against a temporary tracking URI logs one run with the four metrics;
  - a stubbed `triage` that raises a rate-limit error twice then succeeds is retried, with sleep monkeypatched, and returns the decision;
  - a non-rate-limit error is not retried;
  - retries stop after 5;
  - the worker and skip-validation env vars are set.

**Acceptance Criteria:**
- Given `app.db`, a working model key and `PROVIDER=groq`, when `uv run python eval/run_eval.py` runs, then it completes with no terminal input and logs exactly one new run in `triage-agent` with `valid_schema/mean`, `category_match/mean`, `priority_match/mean` and `tool_order/mean`. The escalating tickets are auto-approved.
- Given no network, when `uv run pytest` runs, then all tests pass.

## Implementation Notes

- **Files:** `eval/run_eval.py` (new) and `tests/test_run_eval.py` (new). No existing file changed.
- **Tracking URI:** the absolute path to `<repo>/mlflow.db`, so the command works from any directory.
- **Retries and escalations:**
  - Each retry attempt runs in a `triage_attempt` child span, and `tool_order` scores only the last attempt.
  - Escalation approvals count only for the successful attempt.
  - The `main()` pre-flight check exits before `evaluate` when `app.db` or the provider key is missing.
- **Review pass 1:** 9 patch entries were applied. One fix was corrected by the orchestrator: `_error_chain` now follows only explicit `__cause__`, because following an unsuppressed implicit `__context__` still retried unrelated errors raised while handling a 429 (triage row 3). The suite is at 183 passed, offline.
- **Live acceptance on Groq** (`PROVIDER=groq`, 2026-09-26, run `ba6b914878cf4c8b91d6acd0e176cac5` in `triage-agent`) passes:
  - exit 0 and 20 traces, each with a `triage_attempt` span;
  - 3 escalations auto-approved with no terminal input;
  - no rate limits hit;
  - means: `valid_schema` 1.0, `tool_order` 1.0, `category_match` 0.95 (T-1045 got `how-to`, expected `billing`), `priority_match` 1.0.

  A pre-review run, `bcace56f58c94732b8bf7676cb8656ee`, scored 1.0 / 1.0 / 0.95 / 0.95.
- **Known noise:** the `MlflowLangchainTracer.on_interrupt/on_resume` AttributeError is printed on each escalation; it's already recorded in `deferred-work.md`.

## Spec Change Log

## Review Triage Log

Review pass 1 (2026-09-26): blind-hunter (BH), edge-case-hunter (ECH), verification-gap (VG).

| # | Source | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|---|
| 1 | BH1, ECH2 | A rate-limited attempt's spans stay in the trace, and `tool_order` takes the earliest spans across all attempts. | medium | `tool_order` scans all of `trace.data.spans`. The frozen rule is "scores reflect the eventual run". The fix is to wrap each attempt in its own child span and score only the last one. | patch |
| 2 | BH2, ECH1 | A retry after an auto-approved escalation counts it twice. | medium | `auto_approve` increments the global counter on every attempt, and `triage` restarts from scratch. Story 3.2 reports this number. The fix is to count only the successful attempt's approvals. | patch |
| 3 | BH3 | The error chain follows `__context__`, so a non-rate-limit error raised while handling a 429 is retried. | low | `_error_chain` walks `__context__` without checking `__suppress_context__`, while the frozen rule says "and on no other error". The fix is a direct correction: follow `__cause__`, and `__context__` only when it isn't suppressed. | patch |
| 4 | BH4, ECH3, ECH4, ECH5 | Hours- or minutes-only hints aren't parsed, waits are unbounded, and `retry-after` `nan`/`inf` breaks `sleep`. | medium | `_HINT` lacks `h` and minutes-only forms. Groq daily-token 429s ("try again in 7m") fall back to 155s of retries that can't succeed, while an unbounded hint could sleep for hours. The fix is to parse `h`/`m`, add `MAX_WAIT_SECONDS` = 120 (fail fast above it), and reject non-finite values. | patch |
| 5 | BH5 | A broken setup (no `app.db` or no provider key) logs a real-looking run of zeros to `triage-agent`. | medium | Every ticket raises and gets `outputs=None`, so an all-zero run pollutes the experiment. The fix is a pre-flight check in `main()`: `app.db` exists and `agent.build_model()` succeeds, otherwise exit with a clear message before `evaluate`. | patch |
| 6 | VG1, BH7 | `tool_order` is untested on real autolog spans from a real `agent.triage` run, including an escalation resume. | medium | Pre-verified: the tests hand-build the span names. The live run scored 1.0, but nothing guards against regressions. | patch |
| 7 | VG2, BH6 | The `main()` wiring (URI, experiment, autolog, dotenv) is never executed in tests. | low | Pre-verified and cheap to close alongside #6. | patch |
| 8 | VG3, BH10 | The `retry-after` header branch, and the `EscalationError`/`TriageOutputError` failure rows, are untested. | medium | Pre-verified for the header. The matrix lists all three error types. | patch |
| 9 | BH9 | The `tracking` fixture doesn't restore the active experiment. | low | The fix is a direct correction: restore it on teardown. | patch |
| 10 | ECH6 | The hint may be read from a non-rate-limit link in the chain. | low | Needs an unrelated "retry in" text in a wrapped error. Unlikely, and the fix adds a branch. | reject |
| 11 | ECH7 | A string `'429'` code isn't detected. | low | The installed providers raise typed classes (`groq.RateLimitError`, `GoogleRateLimitError`) or an integer code. | reject |
| 12 | ECH8 | An import inside the except handler could raise `ImportError`. | false | `groq` and `langchain_google_genai` are project dependencies, installed by `uv sync`. | reject |
| 13 | ECH9 | `asyncio.run` inside a running loop. | false | `evaluate` calls `predict_fn` from worker threads with no loop. The live run on 2026-09-26 completed all 20 tickets. | reject |
| 14 | ECH10-12 | CSV BOM, whitespace, empty file or duplicate IDs. | false | `eval/labelled_tickets.csv` is read-only and has 20 clean, unique rows with no BOM. | reject |
| 15 | ECH13 | `expectations` is None. | false | `build_dataset` always supplies all four expectation keys. | reject |
| 16 | ECH14 | Equal `start_time_ns` values. | low | The tool calls are sequential model turns seconds apart, and a tie-break adds complexity. | reject |
| 17 | ECH15 | The eval env vars aren't restored after `run_eval`. | low | It's a one-shot script process, and the tests use fixtures. | reject |
| 18 | ECH16 | The escalation count is never printed or logged. | false | Reporting belongs to Story 3.2 by the frozen intent. The counter is exposed for it. | reject |
| 19 | BH8 | The diff doesn't record the live acceptance run. | n/a | Not a code defect. Recorded in Implementation Notes: run `bcace56f58c94732b8bf7676cb8656ee`. | - |

## Verification

**Commands:**
- `uv run pytest` -- expected: all pass, offline.
- `PROVIDER=groq uv run python eval/run_eval.py` -- expected: completes unattended; one new MLflow run with the four means.
