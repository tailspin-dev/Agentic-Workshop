"""Offline tests for eval/run_eval.py: no network, no API keys."""

import asyncio
import importlib.util
import json
import os
import threading
from pathlib import Path

import httpx
import mlflow
import pytest
from groq import RateLimitError as GroqRateLimitError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_google_genai.chat_models import GoogleRateLimitError
from mlflow.tracking import fluent

import agent
from agent import EscalationError, GroundingError, TriageError, TriageOutputError

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)

GOOD = {"category": "access", "priority": "P1", "route": "access-team", "rationale": "Whole team locked out."}
EXPECT = {"expected_category": "access", "expected_priority": "P1", "expected_tools": "", "judge_notes": ""}
EVAL_VARS = ("MLFLOW_GENAI_EVAL_MAX_WORKERS", "MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION")


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    """A throwaway MLflow store; the real mlflow.db is never touched."""
    for var in EVAL_VARS:
        monkeypatch.setenv(var, "unset-by-test")  # so monkeypatch restores the real value afterwards
    previous = mlflow.get_tracking_uri()
    previous_experiment = fluent._active_experiment_id
    previous_experiment_env = os.environ.get("MLFLOW_EXPERIMENT_ID")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("triage-agent-test")
    yield
    mlflow.set_tracking_uri(previous)
    fluent._active_experiment_id = previous_experiment
    if previous_experiment_env is None:
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    else:
        os.environ["MLFLOW_EXPERIMENT_ID"] = previous_experiment_env


@pytest.fixture(autouse=True)
def fresh_counter():
    run_eval.escalations.reset()
    yield
    run_eval.escalations.reset()


def groq_rate_limit(message="Rate limit reached. Please try again in 1.5s.", headers=None):
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://api.groq.com/test"))
    return GroqRateLimitError(message, response=response, body=None)


def last_trace():
    mlflow.flush_trace_async_logging()
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


def traced(names):
    """Build a real trace whose child spans start in the given order; return it."""
    with mlflow.start_span(name="predict_fn"):
        for name in names:
            with mlflow.start_span(name=name, span_type="TOOL"):
                pass
    return last_trace()


# --- dataset ----------------------------------------------------------------------------------


def test_dataset_has_every_labelled_row():
    data = run_eval.build_dataset()
    assert len(data) == 20
    assert all(set(row) == {"inputs", "expectations"} for row in data)
    assert all(set(row["inputs"]) == {"ticket_id"} for row in data)
    first = next(row for row in data if row["inputs"]["ticket_id"] == "T-1044")
    assert first["expectations"]["expected_category"] == "access"
    assert first["expectations"]["expected_priority"] == "P1"
    assert first["expectations"]["expected_tools"] == "get_ticket,get_customer_history"
    assert "escalated" in first["expectations"]["judge_notes"]
    assert len({row["inputs"]["ticket_id"] for row in data}) == 20


# --- output scorers ---------------------------------------------------------------------------


def test_valid_schema():
    assert run_eval.valid_schema(outputs=GOOD) == 1
    assert run_eval.valid_schema(outputs=None) == 0
    assert run_eval.valid_schema(outputs={**GOOD, "route": "billing-team"}) == 0
    assert run_eval.valid_schema(outputs={**GOOD, "extra": 1}) == 0
    assert run_eval.valid_schema(outputs="not a dict") == 0


def test_category_match():
    assert run_eval.category_match(outputs=GOOD, expectations=EXPECT) == 1
    assert run_eval.category_match(outputs={**GOOD, "category": "bug"}, expectations=EXPECT) == 0
    assert run_eval.category_match(outputs=None, expectations=EXPECT) == 0


def test_priority_match():
    assert run_eval.priority_match(outputs=GOOD, expectations=EXPECT) == 1
    wrong = {**GOOD, "priority": "P2"}
    assert run_eval.priority_match(outputs=wrong, expectations=EXPECT) == 0
    assert run_eval.category_match(outputs=wrong, expectations=EXPECT) == 1
    assert run_eval.valid_schema(outputs=wrong) == 1
    assert run_eval.priority_match(outputs=None, expectations=EXPECT) == 0


# --- tool_order on real traces ----------------------------------------------------------------


def test_tool_order_right_order(tracking):
    trace = traced(["get_ticket", "get_customer_history", "escalate_to_human"])
    assert run_eval.tool_order(trace=trace) == 1


def test_tool_order_uses_earliest_spans(tracking):
    trace = traced(["get_ticket", "get_customer_history", "get_ticket"])
    assert run_eval.tool_order(trace=trace) == 1


def test_tool_order_reversed(tracking):
    trace = traced(["get_customer_history", "get_ticket"])
    assert run_eval.tool_order(trace=trace) == 0


@pytest.mark.parametrize("names", [["get_customer_history"], ["get_ticket"], []])
def test_tool_order_missing(tracking, names):
    assert run_eval.tool_order(trace=traced(names)) == 0


def test_tool_order_without_trace():
    assert run_eval.tool_order(trace=None) == 0


# --- auto approval ----------------------------------------------------------------------------


def test_auto_approve_returns_true_and_counts():
    action = {"name": "escalate_to_human", "args": {"ticket_id": "T-1044", "reason": "x"}}
    assert run_eval.auto_approve(action) is True
    assert run_eval.auto_approve(action) is True
    assert run_eval.escalations.count == 2


def test_escalation_counter_is_thread_safe():
    threads = [threading.Thread(target=lambda: [run_eval.auto_approve({}) for _ in range(500)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert run_eval.escalations.count == 4000


# --- predict_fn and retries -------------------------------------------------------------------


@pytest.fixture
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(run_eval.time, "sleep", waits.append)
    return waits


def test_predict_fn_passes_auto_approve_and_returns_decision(tracking, monkeypatch):
    seen = {}

    async def fake_triage(ticket_id, approve=None):
        seen["ticket_id"] = ticket_id
        seen["approved"] = approve({"name": "escalate_to_human", "args": {"ticket_id": ticket_id}})
        return GOOD

    monkeypatch.setattr(agent, "triage", fake_triage)
    assert run_eval.predict_fn("T-1044") == GOOD
    assert seen == {"ticket_id": "T-1044", "approved": True}
    assert run_eval.escalations.count == 1


@pytest.mark.parametrize("error", [GroundingError, EscalationError, TriageOutputError])
def test_predict_fn_lets_errors_propagate(tracking, monkeypatch, no_sleep, error):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        raise error("the run cannot be trusted")

    monkeypatch.setattr(agent, "triage", fake_triage)
    with pytest.raises(error):
        run_eval.predict_fn("T-1044")
    assert calls == ["T-1044"]  # not retried
    assert no_sleep == []


def test_rate_limit_is_retried_then_succeeds(tracking, monkeypatch, no_sleep):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        if len(calls) <= 2:
            raise groq_rate_limit("Rate limit reached. Please try again in 1.5s.")
        return GOOD

    monkeypatch.setattr(agent, "triage", fake_triage)
    assert run_eval.predict_fn("T-1042") == GOOD
    assert len(calls) == 3
    assert no_sleep == [1.5, 1.5]


def test_retries_stay_in_one_trace(tracking, monkeypatch, no_sleep):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        with mlflow.start_span(name="attempt"):
            if len(calls) == 1:
                raise groq_rate_limit()
        return GOOD

    monkeypatch.setattr(agent, "triage", fake_triage)
    run_eval.predict_fn("T-1042")
    trace = last_trace()
    assert [s.name for s in trace.data.spans].count("attempt") == 2


def test_backoff_without_hint_doubles_from_five(monkeypatch, no_sleep):
    async def fake_triage(ticket_id, approve=None):
        raise groq_rate_limit("Rate limit reached.")

    monkeypatch.setattr(agent, "triage", fake_triage)
    with pytest.raises(GroqRateLimitError):
        run_eval.triage_with_retries("T-1042")
    assert no_sleep == [5, 10, 20, 40, 80]


def test_retries_stop_after_five(monkeypatch, no_sleep):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        raise GoogleRateLimitError("429 Resource exhausted. Please retry in 51.7s.")

    monkeypatch.setattr(agent, "triage", fake_triage)
    with pytest.raises(GoogleRateLimitError):
        run_eval.triage_with_retries("T-1042")
    assert len(calls) == 6  # one attempt plus five retries
    assert no_sleep == [51.7] * 5


def test_generic_429_counts_as_rate_limit():
    class Http429(Exception):
        status_code = 429

    assert run_eval.is_rate_limit(Http429("too many requests"))
    assert not run_eval.is_rate_limit(ValueError("nope"))
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = groq_rate_limit()
    assert run_eval.is_rate_limit(wrapped)


@pytest.mark.parametrize(
    "message, seconds",
    [("Please retry in 51.7s.", 51.7), ("Please try again in 1m2.5s.", 62.5), ("try again in 750ms", 0.75), ("nothing", None)],
)
def test_retry_hint_parsing(message, seconds):
    hint = run_eval.retry_hint(Exception(message))
    if seconds is None:
        assert hint is None
    else:
        assert hint == pytest.approx(seconds)


# --- evaluate end to end ----------------------------------------------------------------------


def test_eval_env_vars_are_set(tracking, monkeypatch):
    run_eval.set_eval_env()
    import os

    assert os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] == "1"
    assert os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] == "True"


def test_evaluate_logs_one_run_with_four_metrics(tracking, monkeypatch):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        with mlflow.start_span(name="get_ticket", span_type="TOOL"):
            pass
        with mlflow.start_span(name="get_customer_history", span_type="TOOL"):
            pass
        if ticket_id == "T-1044":
            approve({"name": "escalate_to_human", "args": {"ticket_id": ticket_id}})
        if ticket_id == "T-1045":
            raise GroundingError("boom")
        return GOOD

    monkeypatch.setattr(agent, "triage", fake_triage)
    data = [row for row in run_eval.build_dataset() if row["inputs"]["ticket_id"] in ("T-1044", "T-1045")]
    experiment = mlflow.get_experiment_by_name("triage-agent-test")

    result = run_eval.run_eval(data)

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert sorted(calls) == ["T-1044", "T-1045"]  # no extra validation run
    assert run_eval.escalations.count == 1
    for name in ("valid_schema", "category_match", "priority_match"):
        assert result.metrics[f"{name}/mean"] == pytest.approx(0.5)
    assert result.metrics["tool_order/mean"] == pytest.approx(1.0)  # the failing ticket's trace is still read


# --- review follow-ups: retries, hints, preflight, last-attempt scoring -----------------------


def test_retry_after_header_wins_over_message_hint():
    exc = groq_rate_limit("Please try again in 30s.", headers={"retry-after": "7"})
    assert run_eval.retry_hint(exc) == 7


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_retry_after_is_ignored(value):
    exc = groq_rate_limit("Please try again in 3s.", headers={"retry-after": value})
    assert run_eval.retry_hint(exc) == 3
    assert run_eval.retry_hint(groq_rate_limit("Rate limit reached.", headers={"retry-after": value})) is None


@pytest.mark.parametrize("message, seconds", [("Please retry in 1h2m3s.", 3723), ("Please try again in 7m.", 420)])
def test_retry_hint_parses_hours_and_minutes(message, seconds):
    assert run_eval.retry_hint(Exception(message)) == pytest.approx(seconds)


def test_hint_above_max_wait_reraises_without_sleeping(monkeypatch, no_sleep):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        raise groq_rate_limit("Please try again in 7m.")

    monkeypatch.setattr(agent, "triage", fake_triage)
    with pytest.raises(GroqRateLimitError):
        run_eval.triage_with_retries("T-1042")
    assert calls == ["T-1042"]
    assert no_sleep == []


def test_rate_limit_only_in_suppressed_context_is_not_retried(monkeypatch, no_sleep):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        try:
            raise groq_rate_limit()
        except GroqRateLimitError:
            raise TriageError("gave up") from None

    monkeypatch.setattr(agent, "triage", fake_triage)
    with pytest.raises(TriageError):
        run_eval.triage_with_retries("T-1042")
    assert calls == ["T-1042"]
    assert no_sleep == []


def test_error_chain_follows_explicit_cause_only():
    try:
        try:
            raise groq_rate_limit()
        except GroqRateLimitError:
            raise ValueError("unrelated")
    except ValueError as exc:
        # An unrelated error raised while handling a rate limit is not a rate limit.
        assert not run_eval.is_rate_limit(exc)
    try:
        try:
            raise groq_rate_limit()
        except GroqRateLimitError as limit:
            raise RuntimeError("wrapped") from limit
    except RuntimeError as exc:
        # A wrapper that names the rate limit as its cause (as langchain_google_genai does) still is one.
        assert run_eval.is_rate_limit(exc)


def test_unrelated_error_raised_while_handling_a_rate_limit_is_not_retried(monkeypatch, no_sleep):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        try:
            raise groq_rate_limit()
        except GroqRateLimitError:
            raise TriageError("gave up")

    monkeypatch.setattr(agent, "triage", fake_triage)
    with pytest.raises(TriageError):
        run_eval.triage_with_retries("T-1042")
    assert calls == ["T-1042"]
    assert no_sleep == []


def test_tool_order_scores_only_the_last_attempt(tracking, monkeypatch, no_sleep):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        if len(calls) == 1:
            with mlflow.start_span(name="get_ticket", span_type="TOOL"):
                pass
            raise groq_rate_limit()
        with mlflow.start_span(name="get_customer_history", span_type="TOOL"):
            pass
        with mlflow.start_span(name="get_ticket", span_type="TOOL"):
            pass
        return GOOD

    monkeypatch.setattr(agent, "triage", fake_triage)
    run_eval.predict_fn("T-1042")
    trace = last_trace()
    assert [s.name for s in trace.data.spans].count(run_eval.ATTEMPT_SPAN) == 2
    assert run_eval.tool_order(trace=trace) == 0


def test_tool_order_last_attempt_in_right_order(tracking, monkeypatch, no_sleep):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        if len(calls) == 1:
            with mlflow.start_span(name="get_customer_history", span_type="TOOL"):
                pass
            raise groq_rate_limit()
        for name in ("get_ticket", "get_customer_history"):
            with mlflow.start_span(name=name, span_type="TOOL"):
                pass
        return GOOD

    monkeypatch.setattr(agent, "triage", fake_triage)
    run_eval.predict_fn("T-1042")
    assert run_eval.tool_order(trace=last_trace()) == 1


def test_escalation_counted_once_when_rate_limit_follows_approval(tracking, monkeypatch, no_sleep):
    calls = []

    async def fake_triage(ticket_id, approve=None):
        calls.append(ticket_id)
        assert approve({"name": "escalate_to_human", "args": {"ticket_id": ticket_id}}) is True
        if len(calls) == 1:
            raise groq_rate_limit()
        return GOOD

    monkeypatch.setattr(agent, "triage", fake_triage)
    run_eval.predict_fn("T-1044")
    assert len(calls) == 2
    assert run_eval.escalations.count == 1


def test_failed_attempt_escalations_are_not_counted(tracking, monkeypatch, no_sleep):
    async def fake_triage(ticket_id, approve=None):
        approve({"name": "escalate_to_human", "args": {"ticket_id": ticket_id}})
        raise EscalationError("x")

    monkeypatch.setattr(agent, "triage", fake_triage)
    with pytest.raises(EscalationError):
        run_eval.predict_fn("T-1044")
    assert run_eval.escalations.count == 0


# --- main -------------------------------------------------------------------------------------


@pytest.fixture
def main_env(monkeypatch, tmp_path):
    """main() with the eval, .env loading and MLflow setup stubbed; returns what main recorded."""
    seen = {}
    app_db = tmp_path / "app.db"
    app_db.write_text("")
    monkeypatch.setattr(run_eval, "APP_DB", app_db)
    monkeypatch.setattr(run_eval, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(run_eval.mlflow, "set_tracking_uri", lambda uri: seen.setdefault("uri", uri))
    monkeypatch.setattr(run_eval.mlflow, "set_experiment", lambda name: seen.setdefault("experiment", name))
    monkeypatch.setattr(run_eval.mlflow.langchain, "autolog", lambda *a, **k: seen.setdefault("autolog", True))

    class Result:
        run_id = "run-123"

    def fake_run_eval(data=None):
        seen["evaluated"] = True
        return Result()

    monkeypatch.setattr(run_eval, "run_eval", fake_run_eval)
    for var in ("PROVIDER", "MODEL", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    seen["app_db"] = app_db
    return seen


def test_main_uses_repo_mlflow_db_and_triage_agent(main_env):
    run_eval.main()
    assert main_env["uri"] == f"sqlite:///{ROOT / 'mlflow.db'}"
    assert main_env["experiment"] == "triage-agent"
    assert main_env["autolog"] is True
    assert main_env["evaluated"] is True


def test_main_exits_early_without_app_db(main_env):
    main_env["app_db"].unlink()
    with pytest.raises(SystemExit, match="load_seed.py"):
        run_eval.main()
    assert "evaluated" not in main_env


def test_main_exits_early_without_key(main_env, monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(SystemExit, match="GROQ_API_KEY is not set") as info:
        run_eval.main()
    assert "evaluated" not in main_env
    assert "fake-groq-key" not in str(info.value)


# --- tool_order on real autolog spans ---------------------------------------------------------

TICKET = {"ticket_id": "T-1042", "customer_id": "C-77", "created_at": "2026-09-01T09:14:00", "text": "Charged twice."}
CUSTOMER = {"customer_id": "C-77", "name": "Northwind", "plan": "Enterprise", "open_tickets": 2, "ticket_ids": ["T-1042"]}
BILLING = {"category": "billing", "priority": "P2", "route": "billing-team", "rationale": "Double charge (P2)."}
P1 = {"category": "access", "priority": "P1", "route": "access-team", "rationale": "Team locked out (P1)."}


class ScriptedModel(GenericFakeChatModel):
    """Replays AIMessages in order; tool binding is a no-op so create_agent accepts it."""

    def bind_tools(self, tools, **kwargs):
        return self


@tool("get_ticket")
def fake_get_ticket(ticket_id: str) -> str:
    """Stand-in for the MCP get_ticket tool."""
    return json.dumps(TICKET)


@tool("get_customer_history")
def fake_get_customer_history(customer_id: str) -> str:
    """Stand-in for the MCP get_customer_history tool."""
    return json.dumps(CUSTOMER)


def tool_call(call_id, name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture
def autologged(tracking, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("PROVIDER", raising=False)

    class FakeClient:
        async def get_tools(self):
            return [fake_get_ticket, fake_get_customer_history]

    monkeypatch.setattr(agent, "mcp_client", FakeClient)
    mlflow.langchain.autolog()
    yield
    mlflow.langchain.autolog(disable=True)


def script(monkeypatch, *answers):
    messages = [
        tool_call("1", "get_ticket", ticket_id="T-1042"),
        tool_call("2", "get_customer_history", customer_id="C-77"),
        *answers,
    ]
    monkeypatch.setattr(agent, "build_model", lambda: ScriptedModel(messages=iter(messages)))


def trace_count():
    mlflow.flush_trace_async_logging()
    experiment = mlflow.get_experiment_by_name("triage-agent-test")
    return len(mlflow.search_traces(locations=[experiment.experiment_id]))


def test_tool_order_on_autologged_run(autologged, monkeypatch):
    script(monkeypatch, tool_call("3", "TriageDecision", **BILLING))
    before = trace_count()
    assert run_eval.predict_fn("T-1042") == BILLING
    assert trace_count() == before + 1
    trace = last_trace()
    names = [s.name for s in trace.data.spans]
    assert "get_ticket" in names and "get_customer_history" in names
    assert run_eval.tool_order(trace=trace) == 1


def test_tool_order_on_autologged_escalating_run(autologged, monkeypatch):
    script(
        monkeypatch,
        tool_call("e1", "escalate_to_human", ticket_id="T-1042", reason="P1 for an Enterprise customer."),
        tool_call("3", "TriageDecision", **P1),
    )
    before = trace_count()
    assert run_eval.predict_fn("T-1042") == P1
    assert trace_count() == before + 1  # the resume after approval stays in the same trace
    assert run_eval.escalations.count == 1
    trace = last_trace()
    assert "escalate_to_human" in [s.name for s in trace.data.spans]
    assert run_eval.tool_order(trace=trace) == 1
