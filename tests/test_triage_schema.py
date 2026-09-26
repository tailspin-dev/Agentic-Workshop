import json
from typing import get_args

import pytest
from pydantic import ValidationError

from triage_schema import ROUTE_FOR_CATEGORY, Category, Priority, Route, TriageDecision

# Copied from TRIAGE_POLICY.md's "Categories and routes" table.
POLICY_ROUTES = {
    "billing": "billing-team",
    "bug": "bug-team",
    "access": "access-team",
    "performance": "performance-team",
    "how-to": "how-to-team",
}

VALID = {
    "category": "billing",
    "priority": "P2",
    "route": "billing-team",
    "rationale": "Double charge puts money at stake (P2).",
}


def rejected_fields(data) -> set[str]:
    with pytest.raises(ValidationError) as exc:
        TriageDecision.model_validate(data)
    return {str(loc) for error in exc.value.errors() for loc in error["loc"]}


def test_valid_decision_round_trips():
    decision = TriageDecision.model_validate(VALID)
    assert decision.model_dump() == VALID
    assert json.loads(json.dumps(decision.model_dump())) == VALID
    assert json.loads(decision.model_dump_json()) == VALID


@pytest.mark.parametrize(
    "field, value",
    [
        ("priority", "P5"),
        ("category", "Billing"),
        ("priority", "p2"),
        ("route", "Billing-Team"),
        ("category", " billing"),
        ("priority", "P2 "),
    ],
)
def test_bad_enum_value_is_rejected(field, value):
    assert rejected_fields({**VALID, field: value}) == {field}


@pytest.mark.parametrize("field", ["category", "priority", "route", "rationale"])
def test_missing_field_is_rejected(field):
    data = {k: v for k, v in VALID.items() if k != field}
    assert rejected_fields(data) == {field}


def test_extra_field_is_rejected():
    assert rejected_fields({**VALID, "confidence": 0.9}) == {"confidence"}


def test_route_must_match_category():
    with pytest.raises(ValidationError) as exc:
        TriageDecision.model_validate({**VALID, "route": "bug-team"})
    assert {str(loc) for error in exc.value.errors() for loc in error["loc"]} == {"route"}
    assert "expected 'billing-team'" in str(exc.value)


def test_invalid_category_reports_category_not_route():
    assert rejected_fields({**VALID, "category": "sales", "route": "bug-team"}) == {"category"}


def test_route_map_matches_policy_and_types():
    assert ROUTE_FOR_CATEGORY == POLICY_ROUTES
    assert set(ROUTE_FOR_CATEGORY) == set(get_args(Category))
    assert set(ROUTE_FOR_CATEGORY.values()) == set(get_args(Route))


@pytest.mark.parametrize("category, route", POLICY_ROUTES.items())
def test_every_policy_pairing_is_accepted(category, route):
    TriageDecision.model_validate({**VALID, "category": category, "route": route})


def test_multi_sentence_rationale_is_accepted():
    TriageDecision.model_validate({**VALID, "rationale": "Charged twice. Money at stake, so P2."})


@pytest.mark.parametrize("rationale", ["", "   ", "\n\t"])
def test_blank_rationale_is_rejected(rationale):
    assert rejected_fields({**VALID, "rationale": rationale}) == {"rationale"}


def test_padded_rationale_is_kept_verbatim():
    data = {**VALID, "rationale": "  Double charge puts money at stake (P2). "}
    assert TriageDecision.model_validate(data).model_dump() == data


@pytest.mark.parametrize(
    "field, value",
    [("priority", 2), ("rationale", None), ("category", None), ("route", True), ("rationale", 42)],
)
def test_wrong_type_is_rejected_without_coercion(field, value):
    assert rejected_fields({**VALID, field: value}) == {field}


@pytest.mark.parametrize("data", [[VALID], json.dumps(VALID), "hello", 42, None])
def test_non_object_is_rejected(data):
    with pytest.raises(ValidationError) as exc:
        TriageDecision.model_validate(data)
    assert [error["type"] for error in exc.value.errors()] == ["model_type"]


def test_json_text_validates():
    assert TriageDecision.model_validate_json(json.dumps(VALID)).model_dump() == VALID
    with pytest.raises(ValidationError):
        TriageDecision.model_validate_json(json.dumps({**VALID, "priority": "P5"}))
    with pytest.raises(ValidationError):
        TriageDecision.model_validate_json("[]")


def test_assignment_is_validated():
    decision = TriageDecision.model_validate(VALID)
    with pytest.raises(ValidationError):
        decision.route = "bug-team"
    with pytest.raises(ValidationError):
        decision.priority = "P5"


def test_assigning_category_rechecks_route():
    decision = TriageDecision.model_validate(VALID)
    with pytest.raises(ValidationError) as exc:
        decision.category = "bug"
    assert {str(loc) for error in exc.value.errors() for loc in error["loc"]} == {"route"}


def test_vocabularies_match_policy():
    assert get_args(Category) == ("billing", "bug", "access", "performance", "how-to")
    assert get_args(Priority) == ("P1", "P2", "P3", "P4")
    assert get_args(Route) == ("billing-team", "bug-team", "access-team", "performance-team", "how-to-team")


def test_json_schema_is_closed_with_enums():
    schema = TriageDecision.model_json_schema()
    assert set(schema["required"]) == {"category", "priority", "route", "rationale"}
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert props["category"]["enum"] == list(get_args(Category))
    assert props["priority"]["enum"] == list(get_args(Priority))
    assert props["route"]["enum"] == list(get_args(Route))
