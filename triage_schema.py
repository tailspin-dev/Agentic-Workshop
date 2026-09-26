"""The triage decision: the one contract the agent returns and the eval validates.

Values come from TRIAGE_POLICY.md. Validate with TriageDecision.model_validate(data);
anything invalid raises pydantic.ValidationError naming the offending field.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator
from pydantic_core import InitErrorDetails, PydanticCustomError

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

    @model_validator(mode="after")
    def route_matches_category(self) -> "TriageDecision":
        # A model validator runs on every assignment too, so changing category re-checks the route.
        # It only runs once every field is valid, so a bad category is reported on its own.
        expected = ROUTE_FOR_CATEGORY[self.category]
        if self.route != expected:
            error = PydanticCustomError(
                "route_mismatch",
                "route '{route}' does not match category '{category}'; expected '{expected}'",
                {"route": self.route, "category": self.category, "expected": expected},
            )
            raise ValidationError.from_exception_data(
                type(self).__name__, [InitErrorDetails(type=error, loc=("route",), input=self.route)]
            )
        return self

    @field_validator("rationale")
    @classmethod
    def rationale_not_blank(cls, rationale: str) -> str:
        if not rationale.strip():
            raise ValueError("rationale must not be blank")
        return rationale
