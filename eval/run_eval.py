"""Evaluate the triage agent over every ticket in eval/labelled_tickets.csv with MLflow.

Usage: PROVIDER=groq uv run python eval/run_eval.py

Each ticket runs through `agent.triage` inside one MLflow trace, escalations are approved
automatically, and four code scorers grade the result. One run is logged to the `triage-agent`
experiment in mlflow.db at the repo root.
"""

import asyncio
import csv
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    # So `import agent` and `import triage_schema` work when this runs as a script.
    sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from mlflow.genai import scorer  # noqa: E402
from pydantic import ValidationError  # noqa: E402

import agent  # noqa: E402
from triage_schema import TriageDecision  # noqa: E402

LABELS_PATH = ROOT / "eval" / "labelled_tickets.csv"
APP_DB = ROOT / "app.db"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"  # the repo's sqlite:///mlflow.db, from any directory
EXPERIMENT = "triage-agent"

# Set before evaluating: one ticket at a time (provider rate limits), and no extra validation run.
EVAL_ENV = {
    "MLFLOW_GENAI_EVAL_MAX_WORKERS": "1",
    "MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION": "True",
}

MAX_RETRIES = 5
BASE_WAIT_SECONDS = 5.0
MAX_WAIT_SECONDS = 120.0  # a provider hint longer than this ends the retries for that ticket

ATTEMPT_SPAN = "triage_attempt"


# --- dataset ----------------------------------------------------------------------------------


def build_dataset(path: Path = LABELS_PATH) -> list[dict[str, Any]]:
    """One eval row per CSV row: ticket_id as the input, every other column as an expectation."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "inputs": {"ticket_id": row["ticket_id"]},
            "expectations": {
                "expected_category": row["expected_category"],
                "expected_priority": row["expected_priority"],
                "expected_tools": row["expected_tools"],
                "judge_notes": row["judge_notes"],
            },
        }
        for row in rows
    ]


# --- unattended escalation --------------------------------------------------------------------


class EscalationCounter:
    """Counts auto-approved escalations; safe to bump from several threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0

    def increment(self, by: int = 1) -> None:
        with self._lock:
            self._count += by

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def reset(self) -> None:
        with self._lock:
            self._count = 0


escalations = EscalationCounter()


def auto_approve(action: dict[str, Any], counter: EscalationCounter = escalations) -> bool:
    """The eval's approver: approve every escalation without asking anyone, and count it."""
    counter.increment()
    return True


# --- rate-limit retries -----------------------------------------------------------------------


def _error_chain(exc: BaseException):
    """The exception, then its explicit causes (`raise ... from`).

    Implicit context is not followed: an unrelated error raised while handling a rate limit is not
    itself a rate limit, so it must not be retried.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__


def is_rate_limit(exc: BaseException) -> bool:
    """True for a provider rate-limit error (Groq, Gemini, or any HTTP 429), anywhere in the chain."""
    from groq import RateLimitError as GroqRateLimitError
    from langchain_google_genai.chat_models import GoogleRateLimitError

    for err in _error_chain(exc):
        if isinstance(err, (GroqRateLimitError, GoogleRateLimitError)):
            return True
        for attr in ("status_code", "code"):
            if getattr(err, attr, None) == 429:
                return True
        response = getattr(err, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
    return False


_UNIT = r"(?:ms|h|m|s)(?![a-z])"
_HINT = re.compile(rf"(?:retry|try again) in\s*((?:\d+(?:\.\d+)?\s*{_UNIT}\s*)+)", re.IGNORECASE)
_PART = re.compile(rf"(\d+(?:\.\d+)?)\s*({_UNIT})", re.IGNORECASE)
_SECONDS_PER = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def retry_hint(exc: BaseException) -> float | None:
    """Seconds the provider asked us to wait, from a retry-after header or the message; else None."""
    for err in _error_chain(exc):
        headers = getattr(getattr(err, "response", None), "headers", None)
        if headers is not None:
            try:
                value = float(headers.get("retry-after"))
            except (TypeError, ValueError, AttributeError):
                value = None
            if value is not None and math.isfinite(value):
                return max(value, 0.0)
        match = _HINT.search(str(err))
        if match:
            seconds = sum(float(n) * _SECONDS_PER[unit.lower()] for n, unit in _PART.findall(match.group(1)))
            if math.isfinite(seconds):
                return seconds
    return None


def triage_with_retries(ticket_id: str) -> dict[str, Any]:
    """Run agent.triage, retrying only on a provider rate-limit error, at most MAX_RETRIES times.

    Each attempt is its own ATTEMPT_SPAN, so tool_order can score the last one. Escalations are
    added to the shared counter only when their attempt succeeds, so a retry never counts twice.
    """
    attempt = 0
    while True:
        approved = EscalationCounter()
        try:
            with mlflow.start_span(name=ATTEMPT_SPAN):
                decision = asyncio.run(
                    agent.triage(ticket_id, approve=lambda action: auto_approve(action, approved))
                )
            escalations.increment(approved.count)
            return decision
        except Exception as exc:
            if not is_rate_limit(exc) or attempt >= MAX_RETRIES:
                raise
            wait = retry_hint(exc)
            if wait is None:
                wait = BASE_WAIT_SECONDS * 2**attempt
            elif wait > MAX_WAIT_SECONDS:
                raise
            attempt += 1
            print(
                f"{ticket_id}: rate limited, retry {attempt}/{MAX_RETRIES} in {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)


@mlflow.trace
def predict_fn(ticket_id: str) -> dict[str, Any]:
    """One ticket, one trace: the agent run, any escalation resume and any retries nest here."""
    return triage_with_retries(ticket_id)


# --- scorers ----------------------------------------------------------------------------------


@scorer
def valid_schema(outputs) -> int:
    """1 when the output validates as a TriageDecision."""
    if outputs is None:
        return 0
    try:
        TriageDecision.model_validate(outputs)
    except (ValidationError, TypeError, ValueError):
        return 0
    return 1


@scorer
def category_match(outputs, expectations) -> int:
    """1 when the output's category equals expected_category."""
    if not isinstance(outputs, dict):
        return 0
    return int(outputs.get("category") == expectations["expected_category"])


@scorer
def priority_match(outputs, expectations) -> int:
    """1 when the output's priority equals expected_priority."""
    if not isinstance(outputs, dict):
        return 0
    return int(outputs.get("priority") == expectations["expected_priority"])


def _last_attempt_spans(spans):
    """The spans under the last ATTEMPT_SPAN, or all spans when there is none."""
    attempts = [span for span in spans if span.name == ATTEMPT_SPAN]
    if not attempts:
        return spans
    last = max(attempts, key=lambda span: span.start_time_ns)
    by_id = {span.span_id: span for span in spans}

    def under_last(span) -> bool:
        parent = span.parent_id
        while parent is not None:
            if parent == last.span_id:
                return True
            parent = getattr(by_id.get(parent), "parent_id", None)
        return False

    return [span for span in spans if under_last(span)]


def _earliest_start(spans, name: str) -> int | None:
    starts = [span.start_time_ns for span in spans if span.name == name]
    return min(starts) if starts else None


@scorer
def tool_order(trace) -> int:
    """1 when the earliest get_ticket span starts before the earliest get_customer_history span.

    Only the last attempt counts when a ticket was retried; without attempt spans, the whole trace.
    """
    if trace is None:
        return 0
    spans = _last_attempt_spans(trace.data.spans)
    ticket = _earliest_start(spans, "get_ticket")
    history = _earliest_start(spans, "get_customer_history")
    if ticket is None or history is None:
        return 0
    return int(ticket < history)


# Story 3.2 appends the rationale judge here.
SCORERS = [valid_schema, category_match, priority_match, tool_order]


# --- run --------------------------------------------------------------------------------------


def set_eval_env() -> None:
    os.environ.update(EVAL_ENV)


def run_eval(data: list[dict[str, Any]] | None = None):
    """Evaluate the agent over the labelled tickets; returns MLflow's EvaluationResult.

    Assumes tracking is already configured (see main).
    """
    set_eval_env()
    escalations.reset()
    return mlflow.genai.evaluate(
        data=build_dataset() if data is None else data,
        scorers=SCORERS,
        predict_fn=predict_fn,
    )


def preflight() -> None:
    """Exit with a clear message instead of logging an all-zero run when nothing could succeed."""
    if not APP_DB.exists():
        raise SystemExit(f"{APP_DB} not found. Load the data first: uv run python load_seed.py")
    try:
        agent.build_model()
    except Exception as exc:  # build_model's messages name the env var, never the key
        raise SystemExit(f"Cannot build the agent's model: {exc}") from None


def main() -> None:
    load_dotenv(ROOT / ".env")
    preflight()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    mlflow.langchain.autolog()

    result = run_eval()
    print(f"MLflow run {result.run_id} logged to experiment '{EXPERIMENT}'.")


if __name__ == "__main__":
    main()
