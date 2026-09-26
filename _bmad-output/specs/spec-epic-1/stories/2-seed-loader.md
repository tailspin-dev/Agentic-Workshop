---
title: 'Seed loader'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '4cb999a9f12a2b622a8268dbc1b301a7454d839c'
context: ['{project-root}/_bmad-output/specs/spec-epic-1/SPEC.md', '{project-root}/_bmad-output/implementation-artifacts/epic-1-context.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `mcp/triage_server.py` reads `tickets` and `customers` from `app.db`, but nothing creates that database, so Epic 2's agent has no data to read (SPEC CAP-2).

**Approach:** One script, `load_seed.py`, that drops and recreates both tables from `seed/*.csv` inside a single transaction, so every run yields the same database and the MCP server works unchanged.

## Boundaries & Constraints

**Always:** Tables `tickets(ticket_id, customer_id, created_at, text)` and `customers(customer_id, name, plan, open_tickets)`, exactly the CSV columns in CSV order. `open_tickets` stored as INTEGER; every other column TEXT, values copied verbatim (no trimming or reformatting). Rebuild from scratch on every run (replace, not upsert). Paths resolve from the script's own location, so the command works from any working directory. Python 3.12+, stdlib only (`csv`, `sqlite3`), no new packages, no network, no API keys.

**Never:** Edit `seed/`, `TRIAGE_POLICY.md`, `mcp/triage_server.py` or `triage_schema.py`. Touch tables in `app.db` other than `tickets` and `customers`. Commit `app.db`. Build the agent, MCP tools or evals.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Fresh load | no `app.db` | `app.db` created; 24 tickets, 20 customers; prints both counts | N/A |
| Re-run | `app.db` from a previous run | Both tables identical to the first run, no duplicates | N/A |
| Stale data | `app.db` with extra/edited rows in either table | Tables match the CSVs again | N/A |
| Other tables | `app.db` holds an unrelated table | That table is left untouched | N/A |
| Text with commas/quotes | e.g. a quoted `text` field | Stored exactly as the CSV value | N/A |
| Bad seed | header differs from the expected columns, or `open_tickets` not an integer | Nothing written; existing tables unchanged | Exits non-zero with a message naming the file and column |

</frozen-after-approval>

## Code Map

- `load_seed.py` (new, repo root) -- the loader. Expose `load_seed(db_path, seed_dir)` so tests can point it at a temp directory; `__main__` calls it with `ROOT / "app.db"` and `ROOT / "seed"`, where `ROOT = Path(__file__).resolve().parent`.
- `mcp/triage_server.py` -- read-only. Queries `tickets` by `ticket_id` and by `customer_id`, `customers` by `customer_id`; `DB_PATH` is a module-level `Path` (repo-root `app.db`) and `_query` reads it at call time, so tests can monkeypatch it. `get_ticket`/`get_customer_history` are plain functions after `@server.tool()`.
- `seed/tickets.csv`, `seed/customers.csv` -- read-only. UTF-8, LF, no BOM; 24 and 20 rows; unique IDs; no blank values; every ticket's `customer_id` exists; `open_tickets` values 0–4.
- `tests/test_triage_schema.py` -- existing; style reference (plain pytest, parametrize).
- `pyproject.toml` -- already has `pythonpath = ["."]`, so `from load_seed import load_seed` works; no change.
- `.gitignore` -- already ignores `app.db`; no change.

## Tasks & Acceptance

**Execution:**
- [x] `load_seed.py` -- read each CSV with `csv.DictReader`, check its header equals the expected columns, convert `open_tickets` with `int()`; then in one `sqlite3` transaction `DROP TABLE IF EXISTS` and `CREATE TABLE` both tables (ID column `TEXT PRIMARY KEY`) and `executemany` the rows; print the row counts; on a bad seed exit non-zero with a clear message -- implements CAP-2
- [x] `tests/test_load_seed.py` -- one test per I/O matrix row, using `tmp_path` (copying `seed/` there for the bad-seed cases, never touching the real `app.db` or `seed/`); plus the success signal: load into a temp db, monkeypatch the server's `DB_PATH` (import `mcp/triage_server.py` via `importlib.util.spec_from_file_location`, since `mcp` is also the installed SDK), and check `get_ticket("T-1042")` and `get_customer_history("C-77")` -- locks the contract

**Acceptance Criteria:**
- Given a fresh clone after `uv sync`, when `uv run python load_seed.py` runs from the repo root or another directory, then the repo-root `app.db` has `tickets` (24 rows) and `customers` (20 rows) whose `PRAGMA table_info` column names equal the CSV headers in order.
- Given the loaded `app.db`, when `get_ticket("T-1042")` and `get_customer_history("C-77")` are called, then they return that ticket (customer `C-77`) and that customer with `T-1042` in `ticket_ids`, and `open_tickets` is a Python `int`.
- Given no API keys and no network, when `uv run pytest` runs, then all tests pass and the real `app.db` is not created or changed by the tests.

### Review Findings

Code review pass 2 (2026-09-26; blind-hunter, edge-case-hunter, verification-gap, acceptance-auditor). Acceptance auditor: all three ACs met. Verification gap: none.

- [x] [Review][Patch] Test helpers never close their SQLite connections: `with sqlite3.connect(db)` commits but does not close, so on Python 3.14 every `uv run pytest` prints about 35 `ResourceWarning: unclosed database` from line 23 alone. Wrap the helpers and the inline blocks in `contextlib.closing` [tests/test_load_seed.py:21]
- [x] [Review][Patch] `test_duplicate_id_rolls_back` accepts any `SeedError`, so it would still pass if a header or row-length error fired instead of the `IntegrityError` path. Add `match="UNIQUE constraint failed"` [tests/test_load_seed.py:166]
- [x] [Review][Patch] The header-mismatch cases assert only the file name, never a column, though the matrix's Bad seed row asks for "file and column". Set `named` to a column (`created_at`, `open_tickets`) for `tickets-header` and `customers-header` [tests/test_load_seed.py:122]
- [x] [Review][Defer] `epic-1-context.md` still lists route-match and sentence-count as open, but Story 1.1 settled both (`route_matches_category`; sentence count not enforced) [_bmad-output/implementation-artifacts/epic-1-context.md] — deferred: its fix edits a planning artifact. Regenerate it after the SPEC.md sync already in deferred-work.md.

Rejected:
- low (repeats pass 1 #5): a duplicate ID on a fresh checkout leaves a 0-byte `app.db`, which contradicts "nothing written" in `SeedError`, `main()` and the Implementation Notes. Reproduced. It needs a duplicate in the read-only seed, and the fix adds a pre-check or cleanup branch. The Implementation Notes wording is a spec edit.
- false (repeats pass 1 #6): "the duplicate-ID message names no file". It ends with sqlite's `UNIQUE constraint failed: tickets.ticket_id`, and duplicates are not a matrix row.
- low (repeats pass 1 #8): `int()` accepts `" 4 "`, `"4_0"` and `"-3"`. "Verbatim" governs the TEXT columns, and the spec asks for `int()` conversion of `open_tickets`. The read-only seed holds only 0–4.
- low (repeats pass 1 #7): a missing CSV, non-UTF-8 input, `csv.Error` or `sqlite3.Error` escape `main()` as tracebacks. They still exit non-zero and name the cause, and fixing them adds except branches.
- low (repeats pass 1 #9): blank IDs and orphan tickets are not rejected. They are not in the matrix, the read-only seed has none, and the fix adds cross-file validation.
- low: no subprocess test runs `python load_seed.py` from another directory. `test_root_is_repo_root` pins `ROOT` (pass 1 #3), and the `__main__` guard is two lines.
- low: the rollback when the customers insert fails midway is untested. `test_duplicate_id_rolls_back` already fails after both DROPs and CREATEs and proves they roll back, and every statement shares one transaction.
- reject (spec edit): `epic-1-context.md` is not listed in the Implementation Notes' file list, and the frontmatter says `review_loop_iteration: 0` despite pass 1. Both fixes edit the spec.
- false: "stale-data test only covers extra `tickets` rows". The whole table is dropped, and the edited `C-77` row already proves `customers` is rebuilt.
- false: "test oracle reads with `utf-8`, loader with `utf-8-sig`". The read-only seed has no BOM, and the plain read pins that. `test_seed_with_utf8_bom_loads` covers the BOM path.
- false: "appended test rows assume a trailing newline". Both seed files end in `\n` (checked with `xxd`) and are read-only.

## Implementation Notes

- Implemented by a fresh subagent from this spec. Files: `load_seed.py`, `tests/test_load_seed.py` (both new); nothing else edited.
- `load_seed()` returns `(ticket_count, customer_count)` and raises `SeedError` on bad seed; `main()` prints the counts, or prints the error to stderr and returns 1.
- Beyond the matrix: a row with too many or too few fields is a bad seed, and a duplicate ID (`IntegrityError`) becomes `SeedError` after a full rollback. Both have tests.
- Validation runs before `sqlite3.connect`, so a bad seed on a fresh checkout never creates `app.db`.
- Verification: `uv run pytest` with API keys unset → 54 passed (14 loader + 40 schema); the real `app.db` is not touched by tests. `load_seed.py` run from a directory outside the repo wrote the repo-root `app.db` with 24/20 rows; `.schema` shows the CSV columns and `open_tickets INTEGER`.
- Review pass 1 patches: CSVs opened with `utf-8-sig` (a BOM-prefixed seed loads); tests added for short/long rows, `ROOT` = repo root and a BOM seed; the fresh-db bad-seed test now asserts `app.db` is not created. Re-verified: `uv run pytest` → 58 passed (18 loader + 40 schema); deleting the row-length check fails the two new row tests.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-09-26; blind-hunter, edge-case-hunter, verification-gap).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | verification+blind | Row-length check in `_read_csv` untested; Implementation Notes claim a test | low | Pre-verified by verification-gap; reproduced: a short `customers.csv` row raises `SeedError`, but no test sends one | patch: two parametrize cases |
| 2 | blind+edge+verification | `test_bad_seed_on_fresh_db_creates_no_tables` guarded by `if db.exists()` | low | The guard lets the test pass if `app.db` is created | patch: `assert not db.exists()` |
| 3 | verification+blind | `ROOT` never checked; both `main()` tests monkeypatch it | low | Pre-verified: `ROOT = Path.cwd()` keeps all tests green; the AC "works from another directory" rests on a manual run | patch: assert `ROOT` is the repo root |
| 4 | edge | UTF-8 BOM seed rejected with header `'\ufeffticket_id'` | low | Reproduced. Seed has no BOM today, but the fix is one token (`utf-8-sig`) | patch + test |
| 5 | blind+edge | Duplicate ID on a fresh checkout leaves an empty `app.db` | low | Reproduced (0-byte file). Needs a duplicate in the read-only seed, which has none; fix adds a pre-check or cleanup branch | rejected (unlikely, adds complexity) |
| 6 | blind | Duplicate-ID message names no file or column | false | Message ends with sqlite's `UNIQUE constraint failed: tickets.ticket_id`, which names table and column | rejected |
| 7 | blind+edge | Missing CSV, non-UTF-8, `csv.Error` or locked db escape `main()` as tracebacks | low | Reproduced `FileNotFoundError`. The traceback names the file and exits non-zero; seed is read-only and present; fix adds except branches | rejected (unlikely, adds complexity) |
| 8 | blind+edge+verification | `int()` accepts `"4_0"`, `" 4 "`, `"-3"` | low | Reproduced (40, 4, -3). Read-only seed holds only 0-4; a strict check adds a branch | rejected (unlikely, adds complexity) |
| 9 | blind+edge | Blank IDs and orphan tickets not rejected | low | Real, but neither is in the matrix; read-only seed has none (checked at planning); fix adds cross-file validation | rejected (unlikely, adds complexity) |
| 10 | edge | Header-only CSV empties the tables without warning | false | That is the CSV's content; the tables then match the CSV, as the intent requires | rejected |

## Design Notes

Reading and validating both CSVs before opening the transaction means a bad seed never leaves a half-built database. The drop, create and insert all run in one transaction, and SQLite rolls DDL back as well, so an error mid-load leaves the previous tables in place. Python's default `sqlite3` mode does not open a transaction before DDL, so connect with `autocommit=False` (3.12+) and use `with conn:`; a plain `with conn:` alone would commit the `DROP` immediately. The primary keys are constraints, not extra columns: the column list still matches the CSV, and a duplicate ID in the seed would fail loudly instead of doubling rows.

## Verification

**Commands:**
- `uv run pytest` -- expected: all tests pass (schema and loader)
- `uv run python load_seed.py && uv run python load_seed.py` -- expected: both runs print 24 tickets and 20 customers
- `sqlite3 app.db ".schema"` -- expected: the two tables with the CSV columns, `open_tickets INTEGER`
