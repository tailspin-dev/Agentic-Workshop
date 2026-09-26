---
title: 'Human-gated escalation'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '846ee87740acaa7276d012c3e4fc39a08b80a22b'
context: ['{project-root}/_bmad-output/specs/spec-epic-2/SPEC.md', '{project-root}/_bmad-output/implementation-artifacts/epic-2-context.md', '{project-root}/_bmad-output/specs/spec-epic-2/stories/1-the-triage-agent.md', '{project-root}/TRIAGE_POLICY.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The policy says a final P1 for an Enterprise customer must be escalated to a person, but the Story 2.1 agent has no way to escalate, and nothing stops it escalating on its own (SPEC CAP-5).

**Approach:**
- Add a local `escalate_to_human` tool to the agent's tool list.
- Gate that tool with LangChain's `HumanInTheLoopMiddleware`, so every call pauses the run.
- When paused, `triage` asks an approver whether to escalate and resumes the run with approve or reject. The default approver asks yes/no at the terminal; a caller can pass its own, which Epic 3's eval needs for auto-approval.

## Boundaries & Constraints

**Always:**
- The middleware config is `HumanInTheLoopMiddleware(interrupt_on={"escalate_to_human": {"allowed_decisions": ["approve", "reject"]}})`.
- The agent gets a checkpointer (`InMemorySaver`) and a fresh `thread_id` per `triage` call. The run resumes with `Command(resume={"decisions": [...]})` on the same thread, once for each interrupt.
- `triage(ticket_id, approve=None)` works as follows:
  - `approve` is a callable that receives the pending action (tool name and args) and returns `True` only to escalate.
  - `None` means use the terminal prompt.
  - The terminal prompt shows the ticket and the reason, reads the answer without blocking the event loop, and treats only `y`/`yes` (any case, trimmed) as yes.
  - Anything else, an empty line or EOF means no.
- The system prompt adds one rule: when the final priority is P1 and the customer's plan is Enterprise, call `escalate_to_human` before giving the final answer.
- Story 2.1's other rules carry over unchanged: the grounding check, retry-once, and `RECURSION_LIMIT` applying to each invoke.
- A resumed run stays inside `run_agent.py`'s existing `triage` span, so it's one trace.

**Never:**
- Escalate without an explicit yes.
- Edit `mcp/triage_server.py`, `triage_schema.py`, `TRIAGE_POLICY.md`, `seed/`, `eval/`, or `run_agent.py`'s MLflow lines.
- Add fields to `TriageDecision`.
- Contact any external system from `escalate_to_human`. It only confirms, because there is no escalation target yet.
- Decide escalation in code instead of by the agent.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Escalate, yes | P1 + Enterprise ticket (e.g. T-1044: C-91 Globex, Enterprise, 3 open), answer `yes` | Run pauses at `escalate_to_human`; tool runs; decision returned; run reported as escalated | N/A |
| Escalate, no | same, answer `no` | Tool call rejected, not executed; agent still returns its decision; run reported as not escalated | N/A |
| Unclear answer | `""`, `maybe`, EOF | Treated as no | N/A |
| No escalation | T-1042 (P2) | No pause, no prompt; same result as Story 2.1 | N/A |
| Custom approver | `triage(t, approve=lambda a: True)` | No terminal read; approver called once per escalation | N/A |
| Missed escalation | model returns P1 for an Enterprise customer without calling `escalate_to_human` | No decision returned | Raises `TriageError` naming the ticket and the skipped escalation rule |

**Decision (human, 2026-09-26): the outcome goes on a separate line.**
- The decision JSON and `triage`'s return value stay exactly the Epic 1 `TriageDecision` dict. Add no fields.
- When an escalation was requested, `triage` writes one line to stderr: `Escalated to a person: yes` or `Escalated to a person: no`. Runs with no escalation write nothing.
- Callers such as Epic 3 learn the outcome through their approver.
- `run_agent.py` needs no change.

**Decision (human, 2026-09-26): code-checked escalation.**
- After the run, `triage` raises a clear `TriageError` when the decision's priority is `P1`, the successful `get_customer_history` returned `plan == "Enterprise"`, and no `escalate_to_human` call was made.
- A rejected escalation still counts as requested.
- An escalation request for any other decision isn't blocked by code. The human gate already stops it unless someone answers yes.

**Decision (human, 2026-09-26):** keep the full spec (~2,100 tokens) as one story.

</frozen-after-approval>

## Code Map

- `agent.py` -- Story 2.1 module. `triage` builds `tools` (commented "Story 2.2 adds escalate_to_human here") and `middleware` (commented "Story 2.2 adds the human-in-the-loop middleware here"), calls `agent.ainvoke(..., config={"recursion_limit": RECURSION_LIMIT})`, then `check_grounding` then structured-response check. `INSTRUCTIONS` holds the appended prompt rules. `check_grounding` only inspects `get_ticket`/`get_customer_history` calls, so an extra tool doesn't affect it.
- `langchain.agents.middleware.HumanInTheLoopMiddleware` (LangChain 1.4.2) -- `after_model` calls `interrupt(HITLRequest)` with `action_requests` (`name`, `args`, `description`) and expects `{"decisions": [{"type": "approve"} | {"type": "reject", "message": ...}]}` in action order. A reject adds an artificial `ToolMessage` and drops the call, and the model then continues. Needs a checkpointer. An interrupted `ainvoke` result has `"__interrupt__"` (a tuple of `Interrupt`, where `.value` is the `HITLRequest`).
- `langgraph.types.Command`, `langgraph.checkpoint.memory.InMemorySaver`.
- `run_agent.py` -- `asyncio.run(triage(ticket_id))` inside the `triage` span, then `print(json.dumps(decision))`. Only the MLflow lines are protected.
- Seed data: the policy's escalation tickets are T-1044 (C-91), T-1048 (C-05) and T-1057 (C-66), all Enterprise with 3 or more open tickets and labelled P1 in `eval/labelled_tickets.csv`.
- Epic 3 (`spec-epic-3/SPEC.md` CAP-8) will call `triage` with an auto-approving approver and count escalations. The resume must happen inside one call to `triage`.
- `tests/test_agent.py` -- has the `scripted` fixture (`ScriptedModel`, fake `get_ticket`/`get_customer_history` tools, `call`/`decide` helpers) to extend with an `escalate_to_human` call.
- Gemini's free tier (20 requests per day) was exhausted on 2026-09-26, so live checks may have to wait.

## Tasks & Acceptance

**Execution:**
- [x] `agent.py` -- add these; keep Story 2.1's behaviour for non-escalating runs -- CAP-5:
  - the `escalate_to_human(ticket_id, reason)` tool;
  - the HITL middleware, the checkpointer and the per-call thread;
  - the interrupt/resume loop with the approver;
  - the default terminal approver and the prompt rule;
  - the stderr outcome line;
  - the escalation consistency check, a `check_escalation(messages, decision)` helper reusing the `get_customer_history` result.
- [x] `tests/test_agent.py` -- offline tests with the scripted model:
  - approve → the tool runs and stderr shows `Escalated to a person: yes`;
  - reject → the tool doesn't run, the decision is still returned, and stderr shows `no`;
  - each unclear-answer input (`""`, `maybe`, EOF, with `input` monkeypatched) → no;
  - a non-escalating run never calls the approver and writes no stderr line;
  - a custom approver is called once per escalation;
  - the returned dict has exactly the four schema fields;
  - a missed escalation (P1 for an Enterprise customer, no call) raises, and a P1 for a non-Enterprise customer without escalation doesn't.

**Acceptance Criteria:**
- Given `app.db` and a working model key, when `uv run python run_agent.py T-1044` runs and the person answers `yes`, then it prints `access`/`P1`/`access-team` and reports escalated. With `no`, it reports not escalated.
- Given `uv run python run_agent.py T-1042`, then there is no prompt and the output matches Story 2.1.
- Given no network, when `uv run pytest` runs, then all tests pass.

## Implementation Notes

- **Files:** `agent.py` and `tests/test_agent.py` only. `run_agent.py`, `mcp/`, `triage_schema.py`, `TRIAGE_POLICY.md`, `seed/` and `eval/` are untouched.
- **Approver contract:** the approver gets `{"name", "args"}` and may be sync or async. Only a return value that `is True` escalates, so a truthy `1` or `"yes"` does not.
- **Terminal prompt:** the ticket, the reason and `Escalate? [y/N]` go to stderr, so stdout stays pure JSON. `input()` runs through `asyncio.to_thread`, and EOF means no.
- **Reject message:** a reject tells the model not to call `escalate_to_human` again and to give its final answer.
- **Guard (not in spec):** more than `RECURSION_LIMIT` pauses in one `triage` call raises `TriageError`. This stops an auto-approving caller from looping forever.
- **Outcome line:** written right after the run ends, before the grounding, structured-response and escalation checks. It reports `yes` if any escalation in the run was approved.
- **Checks:** `check_escalation` raises `EscalationError`, a `TriageError` subclass, and names the ticket from the `get_ticket` call args.
- **Tests:** 31 new tests, 130 passing in total, offline. A mutation check covered three changes: accepting any approver answer, dropping `check_escalation`, and dropping the middleware. Each one fails the suite.
- **Live check:** blocked on 2026-09-26. `run_agent.py T-1044` hit Gemini's free-tier daily quota (429), and `GROQ_API_KEY` is unset. The T-1044 yes/no runs and the T-1042 run are still to do.

- **Review pass 1 patches (2026-09-26):**
  - An escalation for another ticket is auto-rejected without asking and doesn't count as an escalation.
  - `check_escalation(messages, decision, ticket_id)` counts only escalations of the ticket being triaged.
  - The prompt text goes through `printable()`.
  - `read_answer()` reads stdin in a daemon thread.
  - `MAX_ESCALATION_PAUSES = 3`.
  - Tests were added for the pause cap, mixed answers, same-turn actions, other-ticket escalation, control characters and stdout staying clean. The suite is at 136 passed.
- **Ctrl-C verification (triage row 3):** the first reproduction ran as a shell background job, where SIGINT is ignored, so that evidence was flawed. Re-run with `signal.default_int_handler` restored and SIGINT sent after 2s, stdin held open:
  - old `asyncio.to_thread(input)`: still alive after 15s;
  - new `read_answer()`: exits in about 3s with `KeyboardInterrupt`.

  The finding stands and the fix is verified.
- **Live acceptance runs not done:** T-1044 answered yes and then no, and T-1042. Gemini's free tier returned 429 (20 requests per day used up) and `GROQ_API_KEY` is empty. The real middleware interrupt/resume against a live model, and the one-trace behaviour (deferred), are still unexercised.

## Spec Change Log

## Review Triage Log

Review pass 1 (2026-09-26): blind-hunter (BH), edge-case-hunter (ECH), verification-gap (VG).

| # | Source | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|---|
| 1 | BH2, ECH1, ECH2 | `escalate_to_human` for a ticket other than the one being triaged is put to the person and satisfies `check_escalation`. | high | The loop at `agent.py` forwards any pending action to the approver, and `check_escalation` only matches the tool name. An injected "escalate T-xxxx" could escalate the wrong ticket while the real P1 goes unescalated. | patch |
| 2 | BH1, ECH3 | The approval prompt shows model-written `ticket_id`/`reason` unsanitised. | medium | `ask_terminal` prints `args['ticket_id']`/`args['reason']` raw. Ticket-derived text with control or ANSI characters can fake or hide the `Escalate? [y/N]` line. | patch |
| 3 | ECH4 | Ctrl-C at the prompt hangs the process. | medium | Reproduced: `asyncio.run(asyncio.to_thread(input))` with SIGINT was still alive after 12s, because shutdown waits on the non-daemon executor thread. | patch |
| 4 | BH4 | The pause cap reuses `RECURSION_LIMIT` (20), so a person can be re-asked up to 20 times after saying no. | low | `agent.py` checks `pauses > RECURSION_LIMIT`. The fix is a direct correction: a small separate constant. | patch |
| 5 | VG1, BH5 | The pause guard is untested. | medium | Pre-verified: removing the guard leaves the suite green, and Epic 3's auto-approver could hang. | patch |
| 6 | VG2, BH5 | Mixed approve/reject answers are untested, including several actions in one interrupt. | medium | Pre-verified: tracking only the last answer passes every test. | patch |
| 7 | BH6 | No test asserts that the prompt stays off stdout. | low | stdout must stay pure JSON, and the terminal tests read only `err`. The fix is one assertion. | patch |
| 8 | BH3, ECH5 | Approver exceptions are not wrapped in `TriageError`, and the outcome line is skipped. | low | Only a caller's own buggy approver can raise, since the terminal approver catches EOF. Wrapping adds a branch for an unlikely case. | reject |
| 9 | BH3 | "yes" is printed before post-run checks that may raise. | false | "yes" is printed only after an approved call actually ran the tool, so the escalation did happen. The line is true even when a later check fails. | reject |
| 10 | ECH6 | `escalated` is set by the approval, not by the tool succeeding. | low | The tool only returns a string and can't fail. A recursion failure after approval raises `TriageError` anyway. | reject |
| 11 | ECH4b | A non-TTY stdin that is kept open waits forever. | low | Only unattended callers hit this, and they pass their own approver (Epic 3 CAP-8). The CLI is interactive. | reject |
| 12 | BH7 | `test_approver_must_return_exactly_true` asserts too little. | low | The outcome line and the returned decision are pinned by other reject-path tests. | reject |
| 13 | BH9 | `check_escalation` matches the plan exactly, uses the first customer result, and takes the ticket from the first lookup. | false | The seed stores `Enterprise` verbatim. Story 2.1's grounding forbids other customers or tickets anywhere in the run, and `check_grounding` runs first. | reject |
| 14 | BH8, VG-other | The one-trace requirement for a resumed run is untested. | maybe-false | `mlflow.langchain.autolog` should nest resume spans under `run_agent.py`'s active `triage` span. A live T-1044 trace would settle it. If it's false, it's medium. | defer |
| 15 | VG3 | Nothing checks that the terminal read doesn't block the event loop. | low | Pre-verified gap. Only `triage` runs on the loop in the CLI, and the patch for #3 replaces the read mechanism anyway. | defer |
| 16 | BH10, VG-other | The live acceptance runs (T-1044 yes/no, T-1042) are not done. | n/a | Blocked by Gemini's free-tier limit (429) and an empty `GROQ_API_KEY`. This isn't a code defect; it's surfaced to the human. | - |

## Verification

**Commands:**
- `uv run pytest` -- expected: all pass, offline.
- `uv run python run_agent.py T-1044` (answer `yes`, then re-run and answer `no`) -- expected: `access`/`P1`/`access-team`, with the escalation outcome shown as decided.
- `uv run python run_agent.py T-1042` -- expected: no prompt, `billing`/`P2`/`billing-team`.
