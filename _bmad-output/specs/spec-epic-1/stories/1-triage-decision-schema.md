---
title: 'Triage decision schema'
type: 'feature'
created: '2026-09-26'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '13f4764dfe0274a7da2eb782beca8feb1b92b985'
context: ['{project-root}/_bmad-output/specs/spec-epic-1/SPEC.md', '{project-root}/TRIAGE_POLICY.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Nothing in the repo defines a valid triage decision, yet Epic 2 needs one as the agent's structured output and Epic 3's `valid_schema` scorer needs one to validate against (SPEC CAP-1).

**Approach:** One importable pydantic model, `TriageDecision`, with the four fields and the exact vocabularies from `TRIAGE_POLICY.md`; anything else fails validation with an error that names the offending field.

## Boundaries & Constraints

**Always:** Category, priority and route values match `TRIAGE_POLICY.md` exactly and are case-sensitive. All four fields are required. Keys other than the four are rejected. Rejection raises pydantic's `ValidationError`, whose message names each offending field. Python 3.12+, no new packages (pydantic is already a dependency). No network calls, no API keys.

**Decisions (human, 2026-09-26):** The route must match its category per `TRIAGE_POLICY.md` (billing→billing-team, bug→bug-team, access→access-team, performance→performance-team, how-to→how-to-team); a mismatch is rejected with an error naming `route`. The rationale only has to be non-blank; sentence count is not checked (Epic 3's `rationale_judge` covers that).

**Never:** Edit `seed/`, `TRIAGE_POLICY.md`, `mcp/triage_server.py` or `run_agent.py`. Build the loader (story 2), the agent, MCP tools or evals. Coerce or normalize values (e.g. `"p2"` → `"P2"`, `"Billing"` → `"billing"`).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Valid | `{"category":"billing","priority":"P2","route":"billing-team","rationale":"Double charge puts money at stake (P2)."}` | Returns a `TriageDecision`; `model_dump()` equals the input | N/A |
| Bad enum value | priority `P5`, or category `Billing` | Rejected | `ValidationError` naming `priority` / `category` |
| Missing field | no `route` key | Rejected | `ValidationError` naming `route` |
| Extra field | adds `"confidence": 0.9` | Rejected | `ValidationError` naming `confidence` |
| Route mismatch | category `billing`, route `bug-team` | Rejected | `ValidationError` naming `route` |
| Multi-sentence rationale | `"Charged twice. Money at stake, so P2."` | Accepted | N/A |
| Empty rationale | `""` or whitespace only | Rejected | `ValidationError` naming `rationale` |
| Wrong type | `priority: 2`, or `rationale: null` | Rejected, no coercion | `ValidationError` naming the field |
| Not an object | a list, a string, `None` | Rejected | `ValidationError` |

</frozen-after-approval>

## Code Map

- `triage_schema.py` (new, repo root) -- holds `TriageDecision`. It lives at the root because the repo is flat: `run_agent.py` imports `agent` from the root, and Epic 2/3 will import `from triage_schema import TriageDecision`.
- `tests/test_triage_schema.py` (new) -- the first test file; `pyproject.toml` already sets `testpaths = ["tests"]`.
- `pyproject.toml` -- add `pythonpath = ["."]` under `[tool.pytest.ini_options]`. The project isn't installed as a package, so without it the tests can't import root modules.
- `TRIAGE_POLICY.md` -- read-only source of the vocabularies (Categories and routes table, Priority list).
- `run_agent.py` -- read-only; it prints `json.dumps(decision)`, so the decision must be JSON-serializable via `model_dump()`.

## Tasks & Acceptance

**Execution:**
- [x] `triage_schema.py` -- define `Category`, `Priority`, `Route` as `Literal` types and `TriageDecision(BaseModel)` with `model_config = ConfigDict(extra="forbid", strict=True)` and `rationale` required non-blank; a category→route map and a model validator that rejects a mismatched route with an error naming `route`; short docstring pointing to `TRIAGE_POLICY.md` -- the single contract for Epics 2 and 3
- [x] `pyproject.toml` -- add `pythonpath = ["."]` to pytest options -- lets tests import root modules
- [x] `tests/test_triage_schema.py` -- one test per I/O matrix row, asserting the offending field name appears in the error; plus a check that the three `Literal`s list exactly the policy's values -- locks the contract

**Acceptance Criteria:**
- Given a fresh clone after `uv sync`, when `uv run pytest` runs, then all schema tests pass with no network access and no API keys set.
- Given `TriageDecision.model_json_schema()`, when inspected, then `category`, `priority` and `route` appear as enums with exactly the policy's values, all four fields are required, and `additionalProperties` is false.
- Given a valid decision, when `json.dumps(decision.model_dump())` runs, then it produces the same four keys and values.

### Review Findings

Pass 2 (2026-09-26; blind-hunter, edge-case-hunter, verification-gap, acceptance-auditor).

- [x] [Review][Patch] Assigning `category` after validation skips the route check — move the pairing check to a `model_validator(mode="after")` that raises with `loc` `route`, and add a test that assigns `category` (decision: Kris, 2026-09-26) [triage_schema.py:38]
- [x] [Review][Patch] No test pins that a padded rationale is kept verbatim; `return rationale.strip()` or `str_strip_whitespace=True` would pass all 35 tests [tests/test_triage_schema.py:91]
- [x] [Review][Patch] Whitespace-only rationale cases cover spaces only; add `"\n\t"` [tests/test_triage_schema.py:91]
- [x] [Review][Patch] Non-object test asserts only that `ValidationError` is raised; assert the error type and add a bare string and an int [tests/test_triage_schema.py:104]
- [x] [Review][Defer] SPEC.md still lists the route-match and sentence-count questions as open, and CAP-1 says any in-vocabulary route "passes" [_bmad-output/specs/spec-epic-1/SPEC.md:49] — deferred: needs `/bmad-spec` on the epic spec, outside this story

**Rejected**
- `false` — `validate_assignment=True` not in the task's ConfigDict: it only tightens validation and is recorded in the triage log.
- `false` — Gemini may not accept the schema: `langchain_google_genai` converts it offline; only `additionalProperties` is dropped (with a warning), and `extra="forbid"` still enforces it on validation.
- `low` — Story metadata inconsistent (row #12 says story file excluded from diff; `status: done` / `review_loop_iteration: 0` before review): fix edits the spec under review.
- `low` — JSON schema doesn't express category→route pairing: already rejected in pass 1 (row #10); encoding it needs `oneOf`.
- `low` — Tests copy vocabularies by hand instead of parsing `TRIAGE_POLICY.md`: the policy is read-only; parsing Markdown adds fragile code.
- `low` — No `Field(description=...)` for structured output: prompt guidance belongs to Epic 2; adds surface with no demonstrated harm.
- `low` — `ROUTE_FOR_CATEGORY` is a mutable dict: no caller mutates it; guards undemonstrated state.

## Implementation Notes

- Implemented directly in the session (no subagent). Files: `triage_schema.py`, `tests/test_triage_schema.py`, `pyproject.toml`.
- Route/category check is a `model_validator(mode="after")` that raises a `route_mismatch` error with `loc` `route`. It runs on creation and on every assignment (review pass 2 moved it from a `field_validator`, which let `d.category = ...` bypass it). It does not run when a field is invalid, so a bad category is reported on its own. A failed assignment raises but leaves the instance holding the assigned value (pydantic behaviour). `ROUTE_FOR_CATEGORY` is exported for Epic 2/3 reuse.
- Blank-rationale check does not strip the stored value, so `model_dump()` returns input unchanged.
- A JSON *string* of a valid object is rejected by `model_validate` (not an object); callers holding JSON text use `model_validate_json`.
- Review pass 1 patches (see triage log): `validate_assignment=True`, map typed `dict[Category, Route]`, docstring reworded, tests tightened and extended.
- Verification: `uv run pytest` with `GEMINI_API_KEY`/`GROQ_API_KEY` unset → 40 passed after review pass 2 (35 after pass 1, 26 before review); `model_json_schema()` shows three enums, four required fields, `additionalProperties: false`.

## Spec Change Log

## Review Triage Log

Pass 1 (blind-hunter, edge-case-hunter, verification-gap; verification-gap: no findings).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|-------|---------|---------|----------|-------|
| 1 | edge | Assigning a field after validation skips checks | low | Reproduced: `d.route = "bug-team"` accepted on a billing decision | patch: `validate_assignment=True` + test |
| 2 | edge | `model_validate` passes a `model_construct` instance unchecked | low | Real, but `model_construct` deliberately skips validation and no caller uses it; fix changes semantics | rejected (unlikely, not a direct fix) |
| 3 | edge+blind | Route map not locked to the `Literal`s/policy; drift raises raw `KeyError` | low | No test tied `ROUTE_FOR_CATEGORY` to `Category`/`Route` or the policy table | patch: typed `dict[Category, Route]`, test vs. explicit policy table |
| 4 | edge+blind | Pairing test zips `Literal`s by position | low | `zip(get_args(Category), get_args(Route))` breaks on reorder, never reads the map | patch (same root as #3): parametrize from policy table |
| 5 | edge | Zero-width-only rationale (`"​"`) accepted | low | Reproduced; `str.strip()` keeps U+200B. Unlikely model output; fix adds a guard | rejected (unlikely, adds complexity) |
| 6 | blind | Error-field assertions use `in`, so extra errors go unnoticed; message never checked | low | `rejected_fields` checked membership only | patch: exact-set asserts, route message asserted |
| 7 | blind | Invalid category + mismatched route path untested; relies on field order | low | Skip branch had no test | patch: test + comment on declaration order |
| 8 | blind | `model_validate_json` / `model_dump_json` untested | low | No JSON-text test existed | patch: tests added |
| 9 | blind | Padded enum values and more wrong types untested | low | Only 2 wrong-type cases | patch: cases added |
| 10 | blind | JSON schema doesn't express category→route rule | low | Real: schema lists routes freely; rule enforced after generation. Encoding it needs `oneOf`, not a direct fix; Epic 2's retry-once covers it | rejected (low, adds complexity) |
| 11 | blind | Docstring says "one-sentence" though not enforced | low | Contradicted the recorded decision | patch: docstring reworded |
| 12 | blind | Story file missing from diff; Verification section empty | false | Story file is excluded from the review diff by design; `## Verification` lists two commands | rejected |
| 13 | blind | ~130-char `raise` line | low | Cosmetic, direct fix | patch: split into `expected` variable |

## Design Notes

`strict=True` stops pydantic from turning `2` into `"2"` or accepting other non-string values. `Literal` gives exact-match enums that come out as JSON-schema `enum`, which is what structured-output providers read.

## Verification

**Commands:**
- `uv run pytest` -- expected: all tests pass
- `uv run python -c "from triage_schema import TriageDecision; print(TriageDecision.model_json_schema())"` -- expected: three enums, four required fields, `additionalProperties: false`
