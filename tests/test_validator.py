import pytest

from paymind.agent.executor import ToolRegistry
from paymind.agent.validator import tidy, validate
from paymind.app.database import User

REGISTRY = ToolRegistry()
ASHA = User("u_asha", "Asha Iyer", "asha@x.com", "accountant", None)
RAHUL = User("u_rahul", "Rahul Sharma", "rahul@x.com", "customer", "user_123")
SEEN = "user: refund payment CAP123 please. tool list_disputes -> dispute PP-D-88561 capture CAP123"


def usd(value):
    return {"currency_code": "USD", "value": value}


def check(tool, params, user=ASHA, offered=None, seen=SEEN):
    offered = offered if offered is not None else {tool}
    return validate(REGISTRY.get(tool), tool, params, user, offered, seen)


# ---- 1. tool allowed -------------------------------------------------------------------

def test_unknown_tool():
    v = check("book_a_flight", {})
    assert v.outcome == "invalid" and "no tool named" in v.errors[0]


def test_role_not_allowed_is_blocked():
    v = check("refund_captured_payment", {"capture_id": "CAP123"}, user=RAHUL)
    assert v.outcome == "blocked" and "customer is not allowed" in v.errors[0]


def test_tool_not_offered_this_turn():
    v = check("list_disputes", {}, offered={"list_invoices"})
    assert v.outcome == "invalid" and "find_tools" in v.errors[0]


# ---- 2. parameters ----------------------------------------------------------------------------

def test_missing_required_parameter():
    v = check("refund_captured_payment", {"amount": usd("5.00")})
    assert v.outcome == "invalid" and "missing required parameter: capture_id" in v.errors


def test_made_up_parameter():
    v = check("list_disputes", {"buyer": "user_123"})
    assert v.outcome == "invalid" and v.errors[0].startswith("unknown parameter 'buyer'")


def test_wrong_type():
    v = check("send_invoice", {"invoice_id": "INV2-AAAA", "send_to_recipient": "maybe"}, seen="INV2-AAAA")
    assert v.outcome == "invalid" and any("send_to_recipient" in e for e in v.errors)


def test_obvious_type_slips_are_tidied():
    assert tidy(50, {"type": "string"}) == "50"
    assert tidy(49.5, {"type": "string"}) == "49.50"
    assert tidy("5", {"type": "integer"}) == 5
    assert tidy("true", {"type": "boolean"}) is True
    v = check("list_disputes", {"page_size": "5"})
    assert v.ok and v.params["page_size"] == 5


# ---- 3. business rules ----------------------------------------------------------------------------

@pytest.mark.parametrize("amount,message", [
    (usd("-5.00"), "more than 0"),
    (usd("0"), "more than 0"),
    (usd("abc"), "not a number"),
    (usd("5.001"), "2 decimal places"),
    ({"currency_code": "EUR", "value": "5.00"}, "only USD"),
])
def test_bad_amounts(amount, message):
    v = check("refund_captured_payment", {"capture_id": "CAP123", "amount": amount})
    assert v.outcome == "invalid" and any(message in e for e in v.errors)


def test_bad_email():
    v = check("search_for_invoices", {"recipient_email": "john-at-x"})
    assert v.outcome == "invalid" and "not a valid email" in v.errors[0]


# ---- 4. grounding ----------------------------------------------------------------------------------

def test_invented_id_is_rejected():
    v = check("refund_captured_payment", {"capture_id": "CAP-999", "amount": usd("5.00")})
    assert v.outcome == "invalid" and any("not mentioned" in e and "CAP-999" in e for e in v.errors)


def test_id_from_earlier_tool_result_is_fine():
    v = check("show_dispute_details", {"dispute_id": "pp-d-88561"})  # case doesn't matter
    assert v.ok


# ---- 5. confirmation ---------------------------------------------------------------------------------

def test_read_tool_runs_without_confirmation():
    v = check("list_disputes", {"dispute_state": "REQUIRED_ACTION"})
    assert v.outcome == "ok" and v.confirmation is None


def test_write_tool_needs_confirmation():
    v = check("refund_captured_payment", {"capture_id": "CAP123", "amount": usd(40)})
    assert v.outcome == "needs_confirmation"
    assert v.params["amount"]["value"] == "40"  # tidied number -> string
    assert v.confirmation == "Refund captured payment for 40 USD (capture_id=CAP123)?"
    assert not v.large_amount


def test_large_amount_gets_stronger_warning():
    v = check("refund_captured_payment", {"capture_id": "CAP123", "amount": usd("750.00")})
    assert v.outcome == "needs_confirmation" and v.large_amount
    assert v.confirmation.startswith("⚠️ Large amount.")


def test_customer_can_use_their_tools():
    v = check("show_dispute_details", {"dispute_id": "PP-D-88561"}, user=RAHUL)
    assert v.ok


def test_all_errors_reported_together():
    v = check("refund_captured_payment", {"capture_id": "CAP-999", "amount": usd("-1"), "reason": "x"})
    assert v.outcome == "invalid" and len(v.errors) == 3  # unknown param, bad amount, invented id


def test_only_address_fields_are_checked_as_emails():
    from paymind.agent.validator import is_email_field
    assert is_email_field("recipient_email") and is_email_field("primary_recipients[0].billing_info.email_address")
    assert not is_email_field("sender_batch_header.email_subject")
    assert not is_email_field("sender_batch_header.email_message")
