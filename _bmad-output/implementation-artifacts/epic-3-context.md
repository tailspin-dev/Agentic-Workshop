# Epic 3 Context: Measure the agent

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Close the loop on the triage agent with an MLflow evaluation. It runs the Epic 2 agent over 20 hand-labelled tickets, scores every run with four code scorers and one independent LLM judge, and reports the scores, token spend and how often a human was needed. Attendees leave with a measured baseline, not just a working agent. There are no separate PRD, architecture or UX documents; the epic spec, the labelled tickets and the Epic 2 agent are the planning source.

## Stories

- Story 3.1: The eval run and the four code scorers
- Story 3.2: The rationale judge and the report

## Requirements & Constraints

- **Entry point:** `uv run python eval/run_eval.py` evaluates all 20 rows of `eval/labelled_tickets.csv` in one pass. It logs exactly one MLflow run to `sqlite:///mlflow.db` under the `triage-agent` experiment.
- **Code scorers,** each 0/1 per ticket:
  - `valid_schema`: the output validates against the Epic 1 `TriageDecision`.
  - `category_match`: the output's category equals `expected_category`.
  - `priority_match`: the output's priority equals `expected_priority`.
  - `tool_order`: the ticket's trace has a `get_ticket` span starting before the `get_customer_history` span.
- **Rationale judge:** `rationale_judge` calls `ChatGroq` with `JUDGE_MODEL` (default `openai/gpt-oss-120b`) and `GROQ_API_KEY`. It returns pass/fail plus a one-line reason given the ticket's `judge_notes`. It never reads `GEMINI_API_KEY`, whatever `PROVIDER` is. Its reported mean is the pass rate.
- **Report:** print the mean of all five scorers, the agent's total tokens (read from the MLflow traces), and the count of auto-approved escalations. Write the same numbers to `eval/latest_report.json`, which is git-ignored and the only new file outside MLflow.
- **Unattended escalation:** every escalation during the eval is auto-approved (the labels expect T-1044, T-1048 and T-1057), and the run never blocks on terminal input. `run_agent.py` still asks a person.
- **Network:** no calls beyond the agent's model and the judge's Groq call. The code scorers run locally.
- **Out of scope:** dashboards, CI, hosting, and tuning the agent to raise its score.

## Technical Decisions

- Built on `mlflow.genai.evaluate(data=..., scorers=[...], predict_fn=...)`, not a hand-rolled loop. MLflow 3.16. Scorers use `@mlflow.genai.scorer` and receive `inputs`, `outputs`, `expectations` and `trace`.
- Each ticket's prediction is one MLflow trace. The Epic 2 agent resumes after an approved escalation inside a single `triage()` call, so the whole run, resume included, nests under the predict function's trace.
- **The agent is read-only for this epic.** Call `agent.triage(ticket_id, approve=...)` as it is:
  - It returns the four-field `TriageDecision` dict.
  - It raises `TriageError` subclasses (`GroundingError`, `EscalationError`, `TriageOutputError`) when a run can't be trusted.
  - The approver receives `{"name", "args"}` and must return exactly `True` to approve.
  - The agent prints `Escalated to a person: yes/no` to stderr.
- `mlflow.langchain.autolog()` produces the tool spans (`span_type` `TOOL`, named `get_ticket`, `get_customer_history`, `escalate_to_human`).
- Provider limits seen on 2026-09-26: Gemini free tier allows 20 requests per day (about 6 tickets). Groq `openai/gpt-oss-120b` on-demand allows 8,000 tokens per minute, and one triage run uses several thousand tokens.
- Read-only: `eval/labelled_tickets.csv`, `TRIAGE_POLICY.md`, `agent.py`'s decision logic, prompts and policy handling, `seed/`, `triage_schema.py` and `mcp/triage_server.py`.
- Python 3.12+ with uv. Add packages only with `uv add`. Never commit `.env`, `app.db` or `mlflow.db`.

## Cross-Story Dependencies

- Story 3.2 adds `rationale_judge` and the printed/JSON report to the `run_eval.py` built in 3.1. So 3.1 should leave the scorer list and the escalation count easy to extend and read.
- Depends on Epic 1 (the schema, `app.db`) and Epic 2 (`agent.triage` with an approver hook).
