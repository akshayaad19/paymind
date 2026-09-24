"""Validator: decide whether a tool call suggested by the LLM may run.

The LLM suggests; this plain code decides. It never calls an LLM, so the same
input always gives the same decision.

  1. Tool allowed   exists, was offered to the agent this turn, the user's role may use it
  2. Parameters     required fields present, right types, no made-up parameters
                    (obvious type slips such as 50 for "50.00" are tidied first)
  3. Business rules amounts positive and in USD, emails look like emails
  4. Grounding      every ID must have appeared in the conversation (user messages or
                    earlier tool results); an ID seen nowhere is probably invented
  5. Confirmation   write tools need the user's "yes"; large amounts get a stronger warning

Outcome:
  ok                  run it
  needs_confirmation  ask the user first (the call itself is valid)
  invalid             send the errors back to the LLM so it can fix the call
  blocked             not allowed for this user; don't retry
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from jsonschema import Draft202012Validator
from langsmith import traceable

from ..app.database import User

Outcome = Literal["ok", "needs_confirmation", "invalid", "blocked"]

CURRENCY = "USD"
LARGE_AMOUNT = Decimal("500")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class Validation:
    outcome: Outcome
    tool: str
    params: dict[str, Any]                      # tidied parameters to send
    errors: list[str] = field(default_factory=list)
    confirmation: str | None = None             # question to ask the user
    large_amount: bool = False

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"


# ---- 2. parameters --------------------------------------------------------------------

def tidy(value: Any, schema: dict) -> Any:
    """Fix obvious type slips so a correct call isn't rejected on a technicality."""
    expected = schema.get("type")
    if expected == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.2f}" if isinstance(value, float) else str(value)
    if expected == "integer" and isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    if expected == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    if expected == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        return {k: tidy(v, props[k]) if k in props else v for k, v in value.items()}
    if expected == "array" and isinstance(value, list):
        return [tidy(v, schema.get("items") or {}) for v in value]
    return value


def check_parameters(card: dict, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    schema = card["parameters"]
    props = schema.get("properties", {})
    errors = [f"unknown parameter '{name}' (allowed: {', '.join(sorted(props)) or 'none'})"
              for name in params if name not in props]
    cleaned = {k: tidy(v, props[k]) if k in props else v for k, v in params.items()}
    for err in Draft202012Validator(schema).iter_errors(cleaned):
        where = ".".join(str(p) for p in err.absolute_path) or "parameters"
        if err.validator == "required":
            errors.append(f"missing required parameter: {err.message.split(chr(39))[1]}")
        else:
            errors.append(f"{where}: {err.message}")
    return cleaned, errors


# ---- 3. business rules ---------------------------------------------------------------------

def walk(value: Any, path: str = ""):
    """Yield (path, value) for every nested value."""
    yield path, value
    if isinstance(value, dict):
        for k, v in value.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, f"{path}[{i}]")


def money_values(params: dict[str, Any]) -> list[tuple[str, dict]]:
    """Every {"currency_code": ..., "value": ...} object in the parameters."""
    return [(p, v) for p, v in walk(params) if isinstance(v, dict) and "value" in v and "currency_code" in v]


def is_email_field(path: str) -> bool:
    """Fields that hold an address: email, email_address, recipient_email... (not email_subject)."""
    key = re.split(r"[.\[]", path)[-1].lower()
    return key in ("email", "email_address") or key.endswith("_email")


def check_business_rules(params: dict[str, Any]) -> tuple[list[str], Decimal]:
    errors, largest = [], Decimal("0")
    for path, money in money_values(params):
        try:
            amount = Decimal(str(money["value"]))
        except (InvalidOperation, TypeError):
            errors.append(f"{path}: amount '{money['value']}' is not a number")
            continue
        if amount <= 0:
            errors.append(f"{path}: amount must be more than 0")
        if amount.as_tuple().exponent < -2:
            errors.append(f"{path}: amount can have at most 2 decimal places")
        if money["currency_code"] != CURRENCY:
            errors.append(f"{path}: only {CURRENCY} is supported, got {money['currency_code']}")
        largest = max(largest, amount)
    for path, value in walk(params):
        if isinstance(value, str) and is_email_field(path) and not EMAIL.match(value):
            errors.append(f"{path}: '{value}' is not a valid email address")
    return errors, largest


# ---- 4. grounding ---------------------------------------------------------------------------

def id_params(card: dict, params: dict[str, Any]) -> dict[str, str]:
    """Parameters that hold IDs: every path parameter, plus top-level *_id fields."""
    props = card["parameters"].get("properties", {})
    return {
        name: str(value) for name, value in params.items()
        if isinstance(value, (str, int)) and (props.get(name, {}).get("x-in") == "path" or name.endswith("_id"))
    }


def check_grounding(card: dict, params: dict[str, Any], seen_text: str) -> list[str]:
    """seen_text = everything the user said plus every earlier tool result in this conversation."""
    haystack = seen_text.lower()
    return [
        f"{name} '{value}' was not mentioned by the user or returned by an earlier tool call. "
        f"Look it up with a tool first instead of guessing."
        for name, value in id_params(card, params).items()
        if value.lower() not in haystack
    ]


# ---- 5. confirmation ---------------------------------------------------------------------------

def confirmation_text(card: dict, params: dict[str, Any], largest: Decimal) -> str:
    details = ", ".join(f"{k}={v}" for k, v in id_params(card, params).items())
    amounts = ", ".join(f"{m['value']} {m['currency_code']}" for _, m in money_values(params))
    parts = [card.get("title") or card["name"]]
    if amounts:
        parts.append(f"for {amounts}")
    if details:
        parts.append(f"({details})")
    text = " ".join(parts) + "?"
    if largest > LARGE_AMOUNT:
        text = f"⚠️ Large amount. {text}"
    return text


# ---- all checks ---------------------------------------------------------------------------------

@traceable(name="validate", run_type="chain")
def validate(
    card: dict | None,
    tool_name: str,
    params: dict[str, Any],
    user: User,
    offered_tools: set[str],
    seen_text: str,
) -> Validation:
    params = params or {}

    # 1. tool allowed
    if card is None:
        return Validation("invalid", tool_name, params, [f"there is no tool named '{tool_name}'"])
    if user.role not in card.get("allowed_roles", []):
        return Validation("blocked", tool_name, params, [f"a {user.role} is not allowed to use {tool_name}"])
    if tool_name not in offered_tools:
        return Validation("invalid", tool_name, params,
                          [f"{tool_name} was not offered this turn; use find_tools to look for it first"])

    # 2. parameters, 3. business rules, 4. grounding
    cleaned, errors = check_parameters(card, params)
    rule_errors, largest = check_business_rules(cleaned)
    errors += rule_errors
    errors += check_grounding(card, cleaned, seen_text)
    if errors:
        return Validation("invalid", tool_name, cleaned, errors)

    # 5. confirmation
    if card.get("requires_confirmation") or card.get("action_type") == "write":
        return Validation("needs_confirmation", tool_name, cleaned,
                          confirmation=confirmation_text(card, cleaned, largest), large_amount=largest > LARGE_AMOUNT)
    return Validation("ok", tool_name, cleaned)
