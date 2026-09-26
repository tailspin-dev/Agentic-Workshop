"""The triage decision: the one contract the agent returns and the eval validates.

Values come from TRIAGE_POLICY.md. Validate with TriageDecision.model_validate(data);
anything invalid raises pydantic.ValidationError naming the offending field.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

Category = Literal["billing", "bug", "access", "performance", "how-to"]
Priority = Literal["P1", "P2", "P3", "P4"]
Route = Literal["billing-team", "bug-team", "access-team", "performance-team", "how-to-team"]

ROUTE_FOR_CATEGORY: dict[Category, Route] = {
    "billing": "billing-team",
    "bug": "bug-team",
    "access": "access-team",
    "performance": "performance-team",
    "how-to": "how-to-team",
}


class TriageDecision(BaseModel):
    """A category, a priority, the category's route and a non-blank rationale.

    The policy asks for a one-sentence rationale; sentence count is not enforced here.
    """

    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    category: Category
    priority: Priority
    route: Route
    rationale: str

    @field_validator("route")
    @classmethod
    def route_matches_category(cls, route: str, info: ValidationInfo) -> str:
        # category is declared before route, so it is in info.data unless it failed validation.
        category = info.data.get("category")
        if category is None:
            return route
        expected = ROUTE_FOR_CATEGORY[category]
        if route != expected:
            raise ValueError(f"route {route!r} does not match category {category!r}; expected {expected!r}")
        return route

    @field_validator("rationale")
    @classmethod
    def rationale_not_blank(cls, rationale: str) -> str:
        if not rationale.strip():
            raise ValueError("rationale must not be blank")
        return rationale
