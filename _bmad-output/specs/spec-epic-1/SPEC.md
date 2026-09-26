---
id: SPEC-epic-1
companions: [../../../TRIAGE_POLICY.md, ../../../mcp/triage_server.py]
sources: [../../../INTENT.md]
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Epic 1: triage data and schema

## Why

The workshop's triage agent (Epic 2) and its eval (Epic 3) both need two things that don't exist yet: a single definition of what a valid triage decision is, and the seed tickets and customers in the SQLite database that `mcp/triage_server.py` already reads. Epic 1 is the foundation: it has no model, no network and no keys, so attendees can build and test it before any LLM is involved.

## Capabilities

- **CAP-1**
  - **intent:** Every triage decision is checked against one schema: a JSON object with a category, a priority, a route and a one-sentence rationale.
  - **success:** An object with category in {billing, bug, access, performance, how-to}, priority in {P1, P2, P3, P4}, route in {billing-team, bug-team, access-team, performance-team, how-to-team} and a rationale passes. Any other value, a missing field, or a non-object is rejected with an error that names the offending field.

- **CAP-2**
  - **intent:** One command loads the seed data into a local SQLite database.
  - **success:** `uv run python load_seed.py` creates `app.db` with tables `tickets` (24 rows from `seed/tickets.csv`) and `customers` (20 rows from `seed/customers.csv`), each with exactly the CSV's columns. Running it a second time leaves both tables identical, with no duplicate rows.

## Constraints

- Python 3.12 or newer, managed with uv. New packages only via `uv add`.
- The files in `seed/` are read-only.
- No network calls and no API keys anywhere in this epic.
- `mcp/triage_server.py` is unchanged and must keep working against `app.db`: `tickets(ticket_id, customer_id, created_at, text)` and `customers(customer_id, name, plan, open_tickets)`.
- The schema is importable Python. Epic 2 uses it as the agent's structured output, and Epic 3's `valid_schema` scorer validates against it.
- The category, priority and route values match `TRIAGE_POLICY.md` exactly.

## Non-goals

- The agent, the MCP tools, evals and any user interface (Epics 2 and 3).

## Success signal

- On a fresh clone, after `uv run python load_seed.py`, calling `mcp/triage_server.py`'s `get_ticket("T-1042")` and `get_customer_history("C-77")` returns that ticket and customer. `uv run pytest` shows the schema accepting a valid decision (e.g. `billing` / `P2` / `billing-team` with a rationale) and rejecting a bad one (e.g. priority `P5`) with a field-named error.

## Assumptions

- Keys other than category, priority, route and rationale are rejected ("anything else is rejected" covers extra fields).
- `load_seed.py` rebuilds both tables from scratch on every run (replace, not upsert).
- `open_tickets` is stored as an integer so the policy's "3 or more open tickets" check works. Other columns stay text.
- The schema uses pydantic, which is already a dependency, so no new packages are needed.

## Open Questions

- Must the schema reject a route that doesn't match its category in `TRIAGE_POLICY.md`'s table (e.g. `billing` + `bug-team`), or only check each field on its own?
- How strictly is "one-sentence rationale" enforced: non-empty only, or also reject text with more than one sentence?
