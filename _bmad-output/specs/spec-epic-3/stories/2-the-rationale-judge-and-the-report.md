---
title: 'The rationale judge and the report'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'dispatch'
review_loop_iteration: 1
baseline_commit: '63fc1cb894b7e00a124d2479c2e643c08ae133da'
context: ['{project-root}/_bmad-output/specs/spec-epic-3/SPEC.md', '{project-root}/_bmad-output/implementation-artifacts/epic-3-context.md', '{project-root}/_bmad-output/specs/spec-epic-3/stories/1-the-eval-run-and-the-four-code-scorers.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The Story 3.1 eval scores four things in code, but nothing judges whether a rationale is sound. After a run, a person has to open the MLflow UI to see any numbers, and the escalation count and token spend are never reported (SPEC CAP-6, CAP-7, CAP-8 reporting).

**Approach:**
- Add a `rationale_judge` scorer to `eval/run_eval.py`. It asks a Groq model whether each decision's rationale is sound, given that ticket's `judge_notes`.
- After the run, print five numbers: the mean of each of the five scorers, with the judge's mean being its pass rate. Also print the agent's total tokens and the auto-approved escalation count.
- Write the same numbers to `eval/latest_report.json`.

## Boundaries & Constraints

**Always:**
- `rationale_judge` builds `ChatGroq(model=JUDGE_MODEL or "openai/gpt-oss-120b", api_key=GROQ_API_KEY)` whatever `PROVIDER` is, and never reads `GEMINI_API_KEY`.
- It returns an MLflow `Feedback` with value `"pass"` or `"fail"` and a one-line rationale, using structured output from the model.
- The judge prompt contains the decision (category, priority, route, rationale) and `judge_notes`. It tells the judge to treat all of that as data to evaluate, never as instructions.
- If there are no outputs (the agent failed), the judge returns `"fail"` without calling the model.
- Judge rate-limit errors reuse Story 3.1's retry policy: the hint-aware backoff, 5 retries and the 120s cap.
- The judge is added to `SCORERS`.
- After `evaluate`, `main()` computes a report from the result:
  - `valid_schema`, `category_match`, `priority_match`, `tool_order`: the `<name>/mean` metrics MLflow logged.
  - `rationale_judge`: the pass rate. MLflow logs no mean for string values; this was verified.
  - `total_tokens`: the sum of `total_tokens` over every chat-model span under a `triage_attempt` span in the run's 20 traces. This counts all attempts but not the judge.
  - `auto_approved_escalations`: `escalations.count`.
  - `tickets`: the number of dataset rows.
- It prints one line per number to stdout, writes the same numbers plus the `run_id` to `eval/latest_report.json` (UTF-8, indent 2, overwritten each run), and logs `rationale_judge/mean` to the same MLflow run.

**Never:**
- Change the four existing scorers, `predict_fn`, the retry or escalation logic, or `agent.py` and the other read-only files.
- Write any other new file.
- Count judge tokens in `total_tokens`.
- Send the ticket text or any `GEMINI_API_KEY` to the judge.
- Commit `eval/latest_report.json`; it's already git-ignored.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Sound rationale | T-1042 decision citing money at stake, Enterprise under threshold | `pass` + one-line reason | N/A |
| Unsound rationale | rationale contradicting `judge_notes` | `fail` + reason | N/A |
| Agent failed | `outputs=None` | `fail`, no model call | N/A |
| Judge rate limited | Groq 429 | retried per the Story 3.1 policy | gives up per policy |
| `PROVIDER=gemini` | agent on Gemini | judge still on `ChatGroq` with `GROQ_API_KEY`; `GEMINI_API_KEY` never read by the judge | N/A |
| Report | finished run | 5 means + `total_tokens` + `auto_approved_escalations` printed and in `eval/latest_report.json` | N/A |
| Judge fails | Groq down, retries exhausted or unparseable verdict on one ticket | That ticket is left out of the pass rate; the report shows `rationale_judge_judged` (e.g. 19 of 20) | The scorer raises; MLflow records the error; the eval continues |
| No judge key | `GROQ_API_KEY` unset | Nothing evaluated | Pre-flight check exits with a message naming `GROQ_API_KEY` |

**Decision (human, 2026-09-26): leave judge failures out of the pass rate.**
- When the judge can't produce a verdict for a ticket, the scorer raises. MLflow then records an error and no value for that ticket.
- The pass rate is `pass / judged`, taken only over tickets with a `pass` or `fail` value.
- The printed report and `eval/latest_report.json` include `rationale_judge_judged` (judged count) next to `tickets`.
- If no ticket was judged, the pass rate is reported as `null` (`n/a` when printed), not 0.
- `main()`'s pre-flight check also requires `GROQ_API_KEY`.

**Decision (human, 2026-09-26):** keep the full spec as one story.

**Decision (human, 2026-09-26, review pass 1): the judge is told whether an escalation happened.**
- Besides the decision and `judge_notes`, the judge prompt includes `escalated: yes` or `escalated: no`.
- The value is read from the ticket's trace: `yes` when an `escalate_to_human` tool span ran under the last `triage_attempt` span, `no` otherwise. The scorer takes `trace` for this.
- The prompt also says that the route names the team that owns the ticket, and that escalating to a person is a separate action recorded by `escalated`.
- Still no ticket text goes to the judge.

**Decision (human, 2026-09-26):** apply this change and the review-pass-1 patches in place, without reverting and re-deriving the code.

</frozen-after-approval>

## Code Map

- `eval/run_eval.py` (Story 3.1, merged) is the file to extend:
  - `SCORERS` (line ~274) is where the judge is added;
  - `escalations` (`EscalationCounter`, `.count`) holds the escalation count;
  - `ATTEMPT_SPAN = "triage_attempt"` and `_last_attempt_spans` show how the span tree is walked;
  - `is_rate_limit`, `retry_hint`, `MAX_RETRIES`, `BASE_WAIT_SECONDS` and `MAX_WAIT_SECONDS` hold the retry policy. Factor the retry loop so the judge can reuse it without changing agent retry behaviour.
  - Also relevant: `preflight()`, `run_eval()` (returns the `EvaluationResult`) and `main()` (prints the run ID).
- `mlflow.entities.Feedback(value=..., rationale=...)` is what a scorer returns. Verified offline with MLflow 3.16:
  - string values give no `<name>/mean` metric;
  - `result_df` has `<name>/value` ("pass"/"fail", NaN when the scorer raised) and `<name>/error_message`;
  - `result.run_id` identifies the run.
- Token usage: on real eval traces, each `ChatGroq` span has the attribute `mlflow.chat.tokenUsage` = `{input_tokens, output_tokens, total_tokens, ...}`, and `trace.info.token_usage` aggregates the whole trace. Run `ba6b9148…` totalled 73,540 tokens over 20 traces. Read traces with `mlflow.search_traces(locations=[experiment_id], run_id=..., return_type="list")`. The `locations` argument is required.
- `langchain_groq.ChatGroq`, with `.with_structured_output(<pydantic model>)` for `{verdict: Literal["pass","fail"], reason: str}`.
- `eval/labelled_tickets.csv`: `judge_notes` is already in each row's `expectations` (Story 3.1 `build_dataset`).
- `.gitignore` already ignores `eval/latest_report.json`.
- `tests/test_run_eval.py` has the `tracking` fixture, the scripted-agent autolog harness and the retry helpers to extend.

## Tasks & Acceptance

**Execution:**
- [x] `eval/run_eval.py` -- add these -- CAP-6, CAP-7:
  - the judge's verdict model and prompt, and `rationale_judge`;
  - the shared retry helper, used by the judge;
  - `GROQ_API_KEY` in the pre-flight check;
  - `build_report(result)`, `print_report` and `write_report`;
  - `rationale_judge/mean` logged to the run;
  - `main()` wiring.
- [x] `tests/test_run_eval.py` -- offline tests, with a fake judge model (patch the `ChatGroq` constructor):
  - pass and fail;
  - no outputs → fail with no call;
  - it uses `JUDGE_MODEL` and `GROQ_API_KEY` and never touches `GEMINI_API_KEY`, with `PROVIDER=gemini` set;
  - judge retry on a rate limit;
  - the prompt contains `judge_notes` and the decision but no ticket text;
  - `build_report` on a stubbed result gives the right means, pass rate and escalation count, and a `total_tokens` that excludes non-attempt spans;
  - `latest_report.json` matches the printed numbers, using a tmp path;
  - the pre-flight check requires `GROQ_API_KEY`;
  - judge failures: a judge failure raises; `build_report` excludes NaN verdicts from the pass rate and reports the judged count; when no ticket was judged, the pass rate is `null`.

**Acceptance Criteria:**
- Given `app.db`, `GROQ_API_KEY` and `PROVIDER=groq`, when `uv run python eval/run_eval.py` runs, then it completes unattended and prints the five means (the judge as a pass rate), the agent's total tokens and the escalation count (3 expected). `eval/latest_report.json` holds the same numbers, and the MLflow run has `rationale_judge/mean`.
- Given no network, when `uv run pytest` runs, then all tests pass.

## Implementation Notes

- **Files:** `eval/run_eval.py` and `tests/test_run_eval.py` only. The retry loop is factored into `call_with_retries(attempt_fn, label)`. `triage_with_retries` wraps its attempt span and escalation counting in a closure, so agent retry behaviour is unchanged.
- **Judge:** `ChatGroq` is imported at module level so tests can patch `run_eval.ChatGroq`. The judge gets only the four decision fields and `judge_notes`, wrapped in `<decision>`/`<judge_notes>` tags. A non-dict or `None` output is a `fail` with no call. A `None`, unparseable or non-`JudgeVerdict` answer raises.
- **Report:** `build_report(result, traces=None, escalation_count=None)`. The optional arguments exist for tests; `main()` passes only `result`. Means are rounded to 4 places. `log_judge_mean` uses `MlflowClient.log_metric` on the eval run and skips logging when nothing was judged.
- **Offline suite:** 203 passed (67 in `tests/test_run_eval.py`).
- **Live acceptance on Groq** (`PROVIDER=groq`, 2026-09-26, run `504b1db9303c42ea8fb698f71b33ec14`) passes: exit 0, and the printed numbers equal `eval/latest_report.json`:
  - valid_schema 1.0, category_match 1.0, priority_match 1.0, tool_order 1.0;
  - rationale_judge 0.95 with 20 of 20 judged (T-1044 failed: the judge read "routes to access-team" as contradicting "escalated to a person");
  - total_tokens 74,412, auto_approved_escalations 3;
  - `rationale_judge/mean` = 0.95 is on the run;
  - there was one judge rate-limit retry (hint 2s).
  - The run has 21 traces: 20 agent traces plus 1 judge trace. The traces' `token_usage` sums to 74,995, so the 583 judge tokens are correctly excluded.
- **Review pass 1 patches** (applied in place):
  - The judge takes `trace`, and the prompt carries `escalated: yes|no`, taken from the last `triage_attempt`. The system prompt says the route and escalation are separate.
  - `<` and `>` are escaped as `&lt;`/`&gt;` in the prompt data.
  - `ChatGroq` is built with `temperature=0`, `timeout=60` and `max_retries=0`.
  - A blank or whitespace-only reason raises.
  - `tickets` is the dataset size.
  - `main()` prints and writes the report before logging `rationale_judge/mean`, and a logging failure only warns.
  - `tests/test_run_eval.py` is at 79 passed.
  - The live numbers above predate these patches.
- **Known noise:** besides the `on_interrupt/on_resume` tracer AttributeError, the escalating tickets print `RuntimeError: Event loop is closed` from `httpx.AsyncClient.aclose` after `asyncio.run` returns. It is harmless and comes from the agent/`asyncio.run` path, which this story doesn't touch.

- **Final live acceptance on Groq** (`PROVIDER=groq`, 2026-09-26, run `f39207e5f7534f78bb90703008e6f4bd`) passes. It exited 0, unattended, and the printed report equals `eval/latest_report.json` and the run's metrics:

  | Number | Value |
  |---|---|
  | `valid_schema` | 1.0 |
  | `category_match` | 1.0 |
  | `priority_match` | 0.9 |
  | `tool_order` | 1.0 |
  | `rationale_judge` | 0.9, 20 of 20 judged (also logged as `rationale_judge/mean`) |
  | `total_tokens` | 74,051 |
  | `auto_approved_escalations` | 3 |

- **After the escalation fix,** T-1044, T-1048 and T-1057 all pass the judge. T-1044 was a false fail before it.
- **The judge's two fails line up with the two real priority misses:**
  - T-1099 got P3, expected P4 ("calls it a broken behavior bug … cosmetic");
  - T-1056 got P2, expected P3 ("claims complete blockage … partial access").

  So the judge now tracks real agent errors.
- **Full suite:** 215 passed, offline.

## Spec Change Log

- **2026-09-26, review pass 1 (intent_gap, triage row 1).**
  - **Trigger:** the judge failed a correct escalation (T-1044), because its input couldn't show that an escalation happened or that route and escalation are separate.
  - **Amended:** the human added a frozen decision: pass `escalated: yes|no` from the trace, and the route-vs-escalation rule, to the judge. No ticket text.
  - **Known-bad state avoided:** judge pass rates that depend on the rationale's wording instead of the decision.
  - **How it was applied:** in place, not by revert and re-derive; the human chose this.
  - **KEEP:** the shared `call_with_retries`, the escalation counting, the pass-rate maths (pass / judged, `null` if none) and the token count limited to attempt spans.

## Review Triage Log

Review pass 1 (2026-09-26): blind-hunter (BH), edge-case-hunter (ECH), verification-gap (VG).

| # | Source | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|---|
| 1 | BH1 | The judge can't see that an escalation happened, or that route and escalation are separate, so correct escalating decisions can fail. | medium | Run `504b1db9…`: T-1044 got `fail` ("claims escalation but routes to access-team, contradicting … escalated to a person"). T-1048 and T-1057 passed only because their rationales happened to explain it. The frozen block fixes the judge's input to the decision plus `judge_notes`. | intent_gap |
| 2 | BH2, ECH3 | Rationale text can close `</decision>` and break out of the data tags. | medium | `json.dumps` leaves `<` and `>` as they are, and the rationale is model output from untrusted ticket text. The fix is to escape `<` and `>`. | patch |
| 3 | BH3 | The judge isn't pinned: default temperature, no timeout, and the Groq SDK's own `max_retries=2` stacks on the Story 3.1 policy. | medium | `ChatGroq(model, api_key)` only. The fix is `temperature=0`, a timeout and `max_retries=0`. | patch |
| 4 | BH5, ECH2 | `log_judge_mean` or `run_traces` raising after `evaluate` loses the printed report and the JSON. | low | `main()` order. The fix is a direct reorder: print and write before logging, and warn if logging fails. | patch |
| 5 | BH7 | A blank judge reason gets through. | low | `reason: str`. The fix is `min_length=1`. | patch |
| 6 | ECH8 | `tickets` comes from `len(result_df)` rather than the dataset size. | low | The fix is a direct correction: use the dataset length. | patch |
| 7 | VG1 | The `run_traces` path to `total_tokens` is untested, and the end-to-end test expects 0. | medium | Pre-verified. | patch |
| 8 | VG2 | `agent_tokens` is only tested on hand-built spans, never on autolog spans with usage. | medium | Pre-verified. | patch |
| 9 | BH6 | The "never reads `GEMINI_API_KEY`" test only patches `os.environ.get`. | low | The fix is a sentinel value checked against the constructor kwargs and prompts. | patch |
| 10 | BH11 | Two lines are over 100 characters. | low | A cosmetic, direct fix. | patch |
| 11 | BH4, ECH4, ECH5 | `expectations` is None, or `judge_notes` is empty. | false | `build_dataset` always supplies `judge_notes`, and the read-only CSV has non-empty notes on all 20 rows. | reject |
| 12 | BH8 | A new `ChatGroq` per ticket. | low | Cost is negligible across 20 rows, and caching adds state. | reject |
| 13 | BH9 | The report doesn't name the unjudged tickets. | low | The report fields are fixed by the frozen block. `result_df` and MLflow already hold the errors. | reject |
| 14 | BH10, ECH6 | Missing-usage warnings; `total_tokens` falling back to input+output. | low | Groq spans carry `total_tokens` (verified on 20 live traces). It adds branches for a case never seen. | reject |
| 15 | ECH1 | `result_df` is None. | false | MLflow builds `result_df` whenever traces exist, and `predict_fn` always produces one per row. | reject |
| 16 | ECH7 | `run_eval` without the pre-flight check and with no `GROQ_API_KEY`. | false | `main()` always runs the pre-flight check before `run_eval`. `run_eval` isn't a separate entry point. | reject |

## Verification

**Commands:**
- `uv run pytest` -- expected: all pass, offline.
- `PROVIDER=groq uv run python eval/run_eval.py` -- expected: completes; the printed numbers match `eval/latest_report.json`.
