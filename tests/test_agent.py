"""Offline tests for agent.py: no network, no API keys."""

import asyncio
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError

import agent
from agent import EscalationError, GroundingError, TriageError, TriageOutputError, check_escalation, check_grounding

KEY_VARS = ("PROVIDER", "MODEL", "GEMINI_API_KEY", "GROQ_API_KEY")


@pytest.fixture
def env(monkeypatch):
    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# --- provider switch -------------------------------------------------------------------------


def test_default_provider_is_gemini(env):
    env.setenv("GEMINI_API_KEY", "fake-gemini-key")
    model = agent.build_model()
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.endswith("gemini-3.8-flash")


def test_groq_provider_and_default_model(env):
    env.setenv("PROVIDER", "groq")
    env.setenv("GROQ_API_KEY", "fake-groq-key")
    model = agent.build_model()
    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"


@pytest.mark.parametrize(
    "provider, key_var, cls, attr",
    [("gemini", "GEMINI_API_KEY", ChatGoogleGenerativeAI, "model"), ("groq", "GROQ_API_KEY", ChatGroq, "model_name")],
)
def test_model_env_overrides_default(env, provider, key_var, cls, attr):
    env.setenv("PROVIDER", provider)
    env.setenv(key_var, "fake-key")
    env.setenv("MODEL", "some-other-model")
    model = agent.build_model()
    assert isinstance(model, cls)
    assert getattr(model, attr).endswith("some-other-model")


@pytest.mark.parametrize("provider, key_var", [(None, "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY")])
@pytest.mark.parametrize("key_value", [None, "", "   "])
def test_missing_key_names_the_env_var(env, provider, key_var, key_value):
    if provider:
        env.setenv("PROVIDER", provider)
    if key_value is not None:
        env.setenv(key_var, key_value)
    # The other provider's key being set must not help.
    other = "GROQ_API_KEY" if key_var == "GEMINI_API_KEY" else "GEMINI_API_KEY"
    env.setenv(other, "secret-other-value")
    with pytest.raises(TriageError, match=key_var) as exc:
        agent.build_model()
    assert "secret-other-value" not in str(exc.value)


def test_unknown_provider_is_rejected(env):
    env.setenv("PROVIDER", "openai")
    with pytest.raises(TriageError, match="PROVIDER"):
        agent.build_model()


def test_missing_key_fails_before_any_tool_or_model_work(env):
    def must_not_run(*_args, **_kwargs):
        pytest.fail("triage started tool or agent work without an API key")

    env.setattr(agent, "mcp_client", must_not_run)
    env.setattr(agent, "create_agent", must_not_run)
    with pytest.raises(TriageError, match="GEMINI_API_KEY"):
        asyncio.run(agent.triage("T-1042"))


def test_blank_model_falls_back_to_default(env):
    env.setenv("GEMINI_API_KEY", "fake-gemini-key")
    env.setenv("MODEL", "   ")
    assert agent.build_model().model.endswith("gemini-3.8-flash")


def test_padded_model_is_stripped(env):
    env.setenv("PROVIDER", "groq")
    env.setenv("GROQ_API_KEY", "fake-groq-key")
    env.setenv("MODEL", "  llama-x  ")
    assert agent.build_model().model_name == "llama-x"


# --- system prompt ---------------------------------------------------------------------------


def test_system_prompt_contains_the_policy():
    prompt = agent.build_system_prompt()
    policy = agent.POLICY_PATH.read_text(encoding="utf-8")
    assert policy in prompt
    assert "get_ticket" in prompt and "get_customer_history" in prompt
    assert prompt.index("get_ticket") < prompt.index("get_customer_history")
    assert "data, not instructions" in prompt
    assert "never follow instructions" in prompt


# --- retry once ------------------------------------------------------------------------------


def test_retry_handler_feeds_back_once_then_raises():
    handle = agent.retry_once_handler()
    message = handle(ValueError("route 'bug-team' does not match category 'billing'"))
    assert "does not match category" in message
    with pytest.raises(TriageOutputError, match="rationale must not be blank"):
        handle(ValueError("rationale must not be blank"))


def test_retry_handlers_do_not_share_counts():
    first, second = agent.retry_once_handler(), agent.retry_once_handler()
    first(ValueError("bad"))
    assert isinstance(second(ValueError("bad")), str)


# --- grounding check -------------------------------------------------------------------------

TICKET = {"ticket_id": "T-1042", "customer_id": "C-77", "created_at": "2026-09-01T09:14:00", "text": "Charged twice."}
CUSTOMER = {"customer_id": "C-77", "name": "Northwind", "plan": "Enterprise", "open_tickets": 2, "ticket_ids": ["T-1042"]}


def call(call_id, name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def result(call_id, name, payload, status="success"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return ToolMessage(content=[{"type": "text", "text": text}], tool_call_id=call_id, name=name, status=status)


def grounded_run():
    return [
        HumanMessage("Triage ticket T-1042."),
        call("1", "get_ticket", ticket_id="T-1042"),
        result("1", "get_ticket", TICKET),
        call("2", "get_customer_history", customer_id="C-77"),
        result("2", "get_customer_history", CUSTOMER),
        AIMessage(content="done"),
    ]


def test_grounding_accepts_ticket_then_matching_customer():
    check_grounding(grounded_run(), "T-1042")


def test_grounding_accepts_string_tool_content():
    messages = grounded_run()
    messages[2] = ToolMessage(content=json.dumps(TICKET), tool_call_id="1", name="get_ticket")
    check_grounding(messages, "T-1042")


def test_grounding_rejects_missing_customer_lookup():
    messages = grounded_run()[:3]
    with pytest.raises(GroundingError, match="never called get_customer_history"):
        check_grounding(messages, "T-1042")


def test_grounding_rejects_missing_ticket_lookup():
    with pytest.raises(GroundingError, match="never called get_ticket"):
        check_grounding([HumanMessage("Triage ticket T-1042."), AIMessage(content="P1!")], "T-1042")


def test_grounding_rejects_ticket_tool_error_naming_ticket_and_error():
    messages = [
        call("1", "get_ticket", ticket_id="T-9999"),
        result("1", "get_ticket", "Error executing tool get_ticket: No ticket with ID T-9999", status="error"),
    ]
    with pytest.raises(GroundingError, match=r"T-9999.*No ticket with ID T-9999"):
        check_grounding(messages, "T-9999")


def test_grounding_rejects_customer_tool_error():
    messages = grounded_run()
    messages[4] = result("2", "get_customer_history", "Error: No customer with ID C-77", status="error")
    with pytest.raises(GroundingError, match="get_customer_history.*failed"):
        check_grounding(messages, "T-1042")


def test_grounding_rejects_wrong_order():
    messages = [
        call("2", "get_customer_history", customer_id="C-77"),
        result("2", "get_customer_history", CUSTOMER),
        call("1", "get_ticket", ticket_id="T-1042"),
        result("1", "get_ticket", TICKET),
    ]
    with pytest.raises(GroundingError, match="before get_ticket"):
        check_grounding(messages, "T-1042")


def test_grounding_rejects_mismatched_customer_id():
    messages = grounded_run()
    messages[3] = call("2", "get_customer_history", customer_id="C-12")
    messages[4] = result("2", "get_customer_history", {**CUSTOMER, "customer_id": "C-12"})
    with pytest.raises(GroundingError, match=r"'C-12'.*'C-77'"):
        check_grounding(messages, "T-1042")


def test_grounding_rejects_lookup_of_a_different_ticket():
    messages = grounded_run()
    messages[1] = call("1", "get_ticket", ticket_id="T-1099")
    with pytest.raises(GroundingError, match="T-1099"):
        check_grounding(messages, "T-1042")


# --- triage with a stubbed agent -------------------------------------------------------------


class StubAgent:
    def __init__(self, result):
        self.result = result

    async def ainvoke(self, _input, config=None):
        return self.result


class StubClient:
    async def get_tools(self):
        return []


@pytest.fixture
def stubbed(env):
    env.setenv("GEMINI_API_KEY", "fake-gemini-key")
    env.setattr(agent, "mcp_client", StubClient)

    def use(result):
        env.setattr(agent, "create_agent", lambda **_kwargs: StubAgent(result))

    return use


def test_triage_rejects_run_without_structured_response(stubbed):
    stubbed({"messages": grounded_run()})
    with pytest.raises(TriageError, match="without a structured decision"):
        asyncio.run(agent.triage("T-1042"))


def test_triage_returns_grounded_decision_as_dict(stubbed):
    decision = agent.TriageDecision(
        category="billing", priority="P2", route="billing-team", rationale="Money at stake (P2)."
    )
    stubbed({"messages": grounded_run(), "structured_response": decision})
    assert asyncio.run(agent.triage("T-1042")) == decision.model_dump()


def test_triage_rejects_ungrounded_decision(stubbed):
    decision = agent.TriageDecision(category="bug", priority="P1", route="bug-team", rationale="Invented.")
    stubbed({"messages": [HumanMessage("Triage ticket T-1042.")], "structured_response": decision})
    with pytest.raises(GroundingError):
        asyncio.run(agent.triage("T-1042"))


# --- full create_agent run with a scripted model ---------------------------------------------


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


def decide(call_id, **decision):
    return AIMessage(content="", tool_calls=[{"name": "TriageDecision", "args": decision, "id": call_id, "type": "tool_call"}])


GOOD = {"category": "billing", "priority": "P2", "route": "billing-team", "rationale": "Double charge: money at stake (P2)."}
BAD = {**GOOD, "route": "bug-team"}


@pytest.fixture
def scripted(env):
    env.setenv("GEMINI_API_KEY", "fake-gemini-key")
    customer_tools = {"tool": fake_get_customer_history}

    class FakeClient:
        async def get_tools(self):
            return [fake_get_ticket, customer_tools["tool"]]

    env.setattr(agent, "mcp_client", FakeClient)

    def use(*answers, customer=None):
        if customer is not None:

            @tool("get_customer_history")
            def other_customer(customer_id: str) -> str:
                """Stand-in for the MCP get_customer_history tool."""
                return json.dumps(customer)

            customer_tools["tool"] = other_customer
        script = [call("1", "get_ticket", ticket_id="T-1042"), call("2", "get_customer_history", customer_id="C-77"), *answers]
        env.setattr(agent, "build_model", lambda: ScriptedModel(messages=iter(script)))

    return use


def test_agent_retries_invalid_output_once_then_returns_decision(scripted):
    scripted(decide("3", **BAD), decide("4", **GOOD))
    assert asyncio.run(agent.triage("T-1042")) == GOOD


def test_agent_stops_after_second_invalid_output(scripted):
    scripted(decide("3", **BAD), decide("4", **BAD))
    with pytest.raises(TriageOutputError, match="failed validation twice"):
        asyncio.run(agent.triage("T-1042"))


# --- MCP tools -------------------------------------------------------------------------------


def test_mcp_tools_load_from_the_real_server():
    tools = asyncio.run(agent.mcp_client().get_tools())
    assert {"get_ticket", "get_customer_history"} <= {tool.name for tool in tools}


# --- grounding: retries and off-target lookups -----------------------------------------------


def test_grounding_accepts_ticket_error_then_retry():
    messages = [
        call("0", "get_ticket", ticket_id="T-1042"),
        result("0", "get_ticket", "Error executing tool get_ticket: database is locked", status="error"),
        *grounded_run()[1:],
    ]
    check_grounding(messages, "T-1042")


def test_grounding_accepts_customer_error_then_retry():
    messages = grounded_run()
    messages[4:4] = [
        result("2", "get_customer_history", "Error: database is locked", status="error"),
        call("3", "get_customer_history", customer_id="C-77"),
        result("3", "get_customer_history", CUSTOMER),
    ]
    check_grounding(messages, "T-1042")


def test_grounding_rejects_customer_lookup_with_no_result():
    messages = grounded_run()
    del messages[4]
    with pytest.raises(GroundingError, match="no tool result"):
        check_grounding(messages, "T-1042")


def test_grounding_rejects_other_ticket_after_grounded_pair():
    messages = [*grounded_run(), call("9", "get_ticket", ticket_id="T-1099"), result("9", "get_ticket", TICKET)]
    with pytest.raises(GroundingError, match="T-1099"):
        check_grounding(messages, "T-1042")


def test_grounding_rejects_other_customer_after_grounded_pair():
    messages = [
        *grounded_run(),
        call("9", "get_customer_history", customer_id="C-12"),
        result("9", "get_customer_history", {**CUSTOMER, "customer_id": "C-12"}),
    ]
    with pytest.raises(GroundingError, match=r"'C-12'.*'C-77'"):
        check_grounding(messages, "T-1042")


def test_grounding_rejects_real_mcp_tool_error_for_unknown_ticket():
    async def unknown_ticket_run():
        tools = {tool.name: tool for tool in await agent.mcp_client().get_tools()}
        tool_call = {"name": "get_ticket", "args": {"ticket_id": "T-9999"}, "id": "x1", "type": "tool_call"}
        message = await tools["get_ticket"].ainvoke(tool_call)
        return [AIMessage(content="", tool_calls=[tool_call]), message]

    with pytest.raises(GroundingError, match="T-9999"):
        check_grounding(asyncio.run(unknown_ticket_run()), "T-9999")


def test_triage_checks_grounding_before_structured_response(stubbed):
    stubbed({"messages": [HumanMessage("Triage ticket T-1042."), AIMessage(content="I give up.")]})
    with pytest.raises(GroundingError):
        asyncio.run(agent.triage("T-1042"))


# --- per-run retry budget and recursion limit, via the scripted fixture -----------------------


def test_triage_builds_a_fresh_retry_handler_per_run(scripted):
    scripted(decide("3", **BAD), decide("4", **GOOD))
    assert asyncio.run(agent.triage("T-1042")) == GOOD
    # Same process, second run: it gets its own retry, so BAD then GOOD still returns GOOD.
    assert asyncio.run(agent.triage("T-1042")) == GOOD


def test_triage_tool_loop_hits_recursion_limit(scripted):
    scripted(*[call(f"t{i}", "get_ticket", ticket_id="T-1042") for i in range(50)])
    with pytest.raises(TriageError, match=r"T-1042 within 20 steps") as exc:
        asyncio.run(agent.triage("T-1042"))
    assert not isinstance(exc.value, GraphRecursionError)


# --- human-gated escalation (Story 2.2) ------------------------------------------------------

P1 = {"category": "access", "priority": "P1", "route": "access-team", "rationale": "Team locked out (P1); Enterprise rule."}
ESCALATE = {"ticket_id": "T-1042", "reason": "P1 for an Enterprise customer."}
OUTCOME = "Escalated to a person:"


def escalate(call_id="e1"):
    return call(call_id, "escalate_to_human", **ESCALATE)


@pytest.fixture
def tool_runs(env):
    """Replace escalate_to_human with a spy of the same name and schema; returns its call list."""
    runs = []

    @tool("escalate_to_human")
    def spy(ticket_id: str, reason: str) -> str:
        """Spy for escalate_to_human."""
        runs.append({"ticket_id": ticket_id, "reason": reason})
        return f"Ticket {ticket_id} was escalated to a person."

    env.setattr(agent, "escalate_to_human", spy)
    return runs


def recorder(answer):
    seen = []

    def approve(action):
        seen.append(action)
        return answer

    approve.seen = seen
    return approve


def test_approve_runs_the_tool_and_reports_yes(scripted, tool_runs, capsys):
    scripted(escalate(), decide("3", **P1))
    approve = recorder(True)
    assert asyncio.run(agent.triage("T-1042", approve=approve)) == P1
    assert tool_runs == [ESCALATE]
    assert approve.seen == [{"name": "escalate_to_human", "args": ESCALATE}]
    err = capsys.readouterr().err
    assert f"{OUTCOME} yes" in err and f"{OUTCOME} no" not in err


def test_reject_skips_the_tool_still_decides_and_reports_no(scripted, tool_runs, capsys):
    scripted(escalate(), decide("3", **P1))
    assert asyncio.run(agent.triage("T-1042", approve=recorder(False))) == P1
    assert tool_runs == []
    err = capsys.readouterr().err
    assert f"{OUTCOME} no" in err and f"{OUTCOME} yes" not in err


@pytest.mark.parametrize("answer", ["", "maybe", "no", "n", "yess", "  ", EOFError])
def test_terminal_unclear_answer_means_no(scripted, tool_runs, capsys, env, answer):
    def fake_input(*_args):
        if answer is EOFError:
            raise EOFError
        return answer

    env.setattr("builtins.input", fake_input)
    scripted(escalate(), decide("3", **P1))
    assert asyncio.run(agent.triage("T-1042")) == P1
    assert tool_runs == []
    out, err = capsys.readouterr()
    assert out == ""
    assert "T-1042" in err and ESCALATE["reason"] in err
    assert f"{OUTCOME} no" in err


@pytest.mark.parametrize("answer", ["y", "yes", " YES ", "Y\n", "Yes"])
def test_terminal_yes_escalates(scripted, tool_runs, capsys, env, answer):
    env.setattr("builtins.input", lambda *_args: answer)
    scripted(escalate(), decide("3", **P1))
    assert asyncio.run(agent.triage("T-1042")) == P1
    assert tool_runs == [ESCALATE]
    out, err = capsys.readouterr()
    assert out == ""
    assert f"{OUTCOME} yes" in err


@pytest.mark.parametrize("answer", [1, "yes", object(), None])
def test_approver_must_return_exactly_true(scripted, tool_runs, answer):
    scripted(escalate(), decide("3", **P1))
    asyncio.run(agent.triage("T-1042", approve=recorder(answer)))
    assert tool_runs == []


def test_async_approver_is_awaited(scripted, tool_runs):
    async def approve(_action):
        return True

    scripted(escalate(), decide("3", **P1))
    assert asyncio.run(agent.triage("T-1042", approve=approve)) == P1
    assert tool_runs == [ESCALATE]


def test_non_escalating_run_never_asks_and_writes_nothing(scripted, tool_runs, capsys, env):
    env.setattr("builtins.input", lambda *_a: pytest.fail("the terminal was read"))
    approve = recorder(True)
    scripted(decide("3", **GOOD))
    assert asyncio.run(agent.triage("T-1042", approve=approve)) == GOOD
    assert approve.seen == [] and tool_runs == []
    assert capsys.readouterr().err == ""


def test_custom_approver_skips_terminal_and_is_called_once_per_escalation(scripted, tool_runs, env):
    env.setattr("builtins.input", lambda *_a: pytest.fail("the terminal was read"))
    approve = recorder(True)
    scripted(escalate("e1"), escalate("e2"), decide("3", **P1))
    assert asyncio.run(agent.triage("T-1042", approve=approve)) == P1
    assert len(approve.seen) == 2
    assert len(tool_runs) == 2


def test_returned_dict_has_exactly_the_schema_fields(scripted, tool_runs):
    scripted(escalate(), decide("3", **P1))
    decision = asyncio.run(agent.triage("T-1042", approve=recorder(True)))
    assert set(decision) == {"category", "priority", "route", "rationale"}
    assert set(decision) == set(agent.TriageDecision.model_fields)


def test_missed_escalation_for_enterprise_p1_raises(scripted, tool_runs, capsys):
    scripted(decide("3", **P1))
    with pytest.raises(EscalationError, match=r"T-1042.*escalat"):
        asyncio.run(agent.triage("T-1042", approve=recorder(True)))
    assert capsys.readouterr().err == ""


def test_p1_for_non_enterprise_customer_needs_no_escalation(scripted, tool_runs):
    scripted(decide("3", **P1), customer={**CUSTOMER, "plan": "Pro"})
    assert asyncio.run(agent.triage("T-1042", approve=recorder(True))) == P1


def test_check_escalation_counts_a_rejected_call():
    messages = [
        *grounded_run(),
        escalate(),
        ToolMessage(content="User rejected the tool call", tool_call_id="e1", name="escalate_to_human", status="error"),
    ]
    check_escalation(messages, P1, "T-1042")


def test_check_escalation_ignores_non_p1_and_failed_customer_lookup():
    check_escalation(grounded_run(), GOOD, "T-1042")
    messages = grounded_run()
    messages[4] = result("2", "get_customer_history", "Error: locked", status="error")
    check_escalation(messages, P1, "T-1042")


def test_check_escalation_accepts_a_decision_model():
    with pytest.raises(EscalationError):
        check_escalation(grounded_run(), agent.TriageDecision(**P1), "T-1042")


def test_escalate_tool_only_confirms():
    assert "T-1044" in agent.escalate_to_human.invoke({"ticket_id": "T-1044", "reason": "P1 Enterprise"})


def test_middleware_gates_escalation_with_approve_or_reject_only():
    middleware = agent.escalation_middleware()
    assert middleware.interrupt_on == {"escalate_to_human": {"allowed_decisions": ["approve", "reject"]}}


def test_system_prompt_has_the_escalation_rule():
    prompt = agent.build_system_prompt()
    assert "escalate_to_human" in prompt[len(agent.POLICY_PATH.read_text(encoding="utf-8")):]
    assert "P1" in agent.INSTRUCTIONS and "Enterprise" in agent.INSTRUCTIONS


def test_each_triage_call_gets_its_own_thread(scripted, tool_runs, env):
    threads = []
    real_create_agent = agent.create_agent

    def spy_create_agent(**kwargs):
        built = real_create_agent(**kwargs)
        original = built.ainvoke

        async def ainvoke(payload, config=None):
            threads.append(config["configurable"]["thread_id"])
            assert config["recursion_limit"] == agent.RECURSION_LIMIT
            return await original(payload, config=config)

        built.ainvoke = ainvoke
        return built

    env.setattr(agent, "create_agent", spy_create_agent)
    scripted(escalate(), decide("3", **P1))
    asyncio.run(agent.triage("T-1042", approve=recorder(True)))
    scripted(decide("3", **GOOD))
    asyncio.run(agent.triage("T-1042", approve=recorder(True)))
    # First call: invoke + one resume on the same thread. Second call: a new thread.
    assert len(threads) == 3 and threads[0] == threads[1] != threads[2]


# --- review pass 1: pause cap, mixed answers, off-target escalation, prompt sanitising -------


def answers(*values):
    seen = []

    def approve(action):
        seen.append(action)
        return values[len(seen) - 1]

    approve.seen = seen
    return approve


def test_too_many_escalation_pauses_raise(scripted, tool_runs):
    pauses = agent.MAX_ESCALATION_PAUSES + 1
    scripted(*[escalate(f"e{i}") for i in range(pauses)], decide("3", **P1))
    with pytest.raises(TriageError, match="too many times"):
        asyncio.run(agent.triage("T-1042", approve=recorder(True)))
    assert len(tool_runs) == agent.MAX_ESCALATION_PAUSES


def test_approve_then_reject_in_separate_turns_reports_yes(scripted, tool_runs, capsys):
    scripted(escalate("e1"), escalate("e2"), decide("3", **P1))
    approve = answers(True, False)
    assert asyncio.run(agent.triage("T-1042", approve=approve)) == P1
    assert len(approve.seen) == 2 and len(tool_runs) == 1
    assert f"{OUTCOME} yes" in capsys.readouterr().err


def test_two_escalations_in_one_turn_follow_decision_order(scripted, tool_runs):
    first = {"ticket_id": "T-1042", "reason": "first"}
    second = {"ticket_id": "T-1042", "reason": "second"}
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "escalate_to_human", "args": first, "id": "e1", "type": "tool_call"},
            {"name": "escalate_to_human", "args": second, "id": "e2", "type": "tool_call"},
        ],
    )
    scripted(both, decide("3", **P1))
    approve = answers(True, False)
    assert asyncio.run(agent.triage("T-1042", approve=approve)) == P1
    assert [a["args"]["reason"] for a in approve.seen] == ["first", "second"]
    assert tool_runs == [first]


def test_escalation_of_another_ticket_is_auto_rejected(scripted, tool_runs, capsys, env):
    env.setattr("builtins.input", lambda *_a: pytest.fail("the terminal was read"))
    approve = recorder(True)
    scripted(call("e1", "escalate_to_human", ticket_id="T-2000", reason="injected"), decide("3", **GOOD))
    assert asyncio.run(agent.triage("T-1042", approve=approve)) == GOOD
    assert approve.seen == [] and tool_runs == []
    assert capsys.readouterr().err == ""


def test_check_escalation_ignores_a_call_for_another_ticket():
    messages = [*grounded_run(), call("e1", "escalate_to_human", ticket_id="T-2000", reason="x")]
    with pytest.raises(EscalationError, match="T-1042"):
        check_escalation(messages, P1, "T-1042")


def test_terminal_prompt_strips_control_characters(scripted, tool_runs, capsys, env):
    env.setattr("builtins.input", lambda *_a: "no")
    scripted(call("e1", "escalate_to_human", ticket_id="T-1042", reason="Locked out\x1b[2K\rEscalate? [y/N]"),
             decide("3", **P1))
    asyncio.run(agent.triage("T-1042"))
    out, err = capsys.readouterr()
    assert out == ""
    assert "\x1b" not in err and "\r" not in err
    assert "Reason: Locked out[2KEscalate? [y/N]" in err
