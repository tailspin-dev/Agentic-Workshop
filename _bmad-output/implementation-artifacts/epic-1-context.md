# Epic 1 Context: Triage data and schema

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Lay the offline foundation for the support-ticket triage agent: one importable definition of a valid triage decision, and a single command that loads the seed tickets and customers into the local SQLite database the existing MCP server reads. The agent (Epic 2) uses the schema as its structured output and the eval (Epic 3) validates against it, so both depend on this epic. It involves no model, no network and no API keys, so it can be built and tested before any LLM is involved. There are no separate PRD, architecture or UX documents; the epic spec, the triage policy and the MCP server are the planning source.

## Stories

- Story 1.1: Triage decision schema
- Story 1.2: Seed loader

## Requirements & Constraints

- **Triage decision schema:** a JSON object with exactly four fields: category, priority, route, and a one-sentence rationale.
  - category: `billing`, `bug`, `access`, `performance`, `how-to`
  - priority: `P1`, `P2`, `P3`, `P4`
  - route: `billing-team`, `bug-team`, `access-team`, `performance-team`, `how-to-team`
  - The schema rejects any other value, a missing field, an extra field or a non-object, and the error names the field at fault.
  - These values must match the triage policy exactly.
- **Seed loading:** `uv run python load_seed.py` creates `app.db` with two tables:
  - `tickets`: 24 rows from `seed/tickets.csv`
  - `customers`: 20 rows from `seed/customers.csv`
  - Each table has exactly the columns of its CSV.
  - A second run leaves both tables identical, with no duplicate rows. Each run rebuilds the tables from scratch rather than upserting.
- **Success signal:** on a fresh clone, after loading the seed data:
  - the MCP server's `get_ticket("T-1042")` and `get_customer_history("C-77")` return that ticket and that customer;
  - `uv run pytest` shows the schema accepting a valid decision (for example `billing` / `P2` / `billing-team` with a rationale) and rejecting a bad one (for example priority `P5`) with an error that names the field.
- **Out of scope:** the agent, the MCP tools, evals and any user interface.
- **Still open in the spec:**
  - Must the schema reject a route that does not match its category, or only check each field on its own?
  - How strictly is "one sentence" enforced: non-empty only, or also reject text with more than one sentence?

  Check what Story 1.1 decided before depending on either behaviour.

## Technical Decisions

- **Environment:** Python 3.12 or newer, managed with uv. Add packages only with `uv add`. The schema uses pydantic, which is already installed, so this epic needs no new packages.
- **Schema module:** plain importable Python, because Epic 2 uses it as the agent's structured output and Epic 3's `valid_schema` scorer validates against it.
- **Database contract:** `mcp/triage_server.py` must stay unchanged and keep working against `app.db` at the repo root. It expects:
  - `tickets(ticket_id, customer_id, created_at, text)`
  - `customers(customer_id, name, plan, open_tickets)`
- **Column types:** `open_tickets` is stored as an integer so the policy's "Enterprise with 3 or more open tickets" check works. Every other column stays text.
- **Category and route pairs in the triage policy:**
  - billing goes to billing-team
  - bug goes to bug-team
  - access goes to access-team
  - performance goes to performance-team
  - how-to goes to how-to-team
- **Read-only and never committed:**
  - `seed/`, `eval/labelled_tickets.csv` and `TRIAGE_POLICY.md` are read-only.
  - Never commit `app.db`, `mlflow.db` or `.env`.

## Cross-Story Dependencies

- The two stories are independent of each other. Both are prerequisites for Epic 2 (the agent reads `app.db` through the MCP server and returns the schema) and Epic 3 (the eval validates against the schema).
