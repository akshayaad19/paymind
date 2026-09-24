import pytest
from qdrant_client import QdrantClient

from paymind.retrieval.tool_index import build_index, point_id, search_text, search_tools


def tool(name, description, questions, roles=("accountant",), category="invoices", eval_only=False, action="write"):
    return {
        "name": name, "service": "paypal", "category": category, "description": description,
        "example_questions": list(questions), "action_type": action,
        "allowed_roles": list(roles), "eval_only": eval_only,
    }


TOOLS = [
    tool("send_invoice", "Send a draft invoice to the customer by email.",
         ["send invoice INV-5 to the client", "email the bill to John"]),
    tool("send_invoice_reminder", "Remind a customer about an unpaid invoice.",
         ["nudge them to pay", "send a payment reminder for INV-9"]),
    tool("refund_captured_payment", "Return money to the buyer for a completed payment.",
         ["refund order 123", "give the customer their money back"], category="payments"),
    tool("show_invoice_details", "View the items and status of one invoice.",
         ["show me invoice INV-2", "what's on my bill"], roles=("customer", "accountant"), action="read"),
    tool("stripe_refund_charge", "Refund a Stripe charge back to the card.",
         ["refund the stripe charge"], category="stripe", eval_only=True),
]


@pytest.fixture(scope="module")
def client():
    c = QdrantClient(":memory:")
    assert build_index(c, TOOLS) == len(TOOLS)
    return c


def names(hits):
    return [h.name for h in hits]


def test_search_text_uses_only_searchable_fields():
    text = search_text(TOOLS[0])
    assert text.startswith("send invoice invoices Send a draft invoice")
    assert "email the bill to John" in text
    assert "accountant" not in text


def test_point_id_is_stable_and_unique():
    assert point_id("send_invoice", "paypal") == point_id("send_invoice", "paypal")
    assert point_id("send_invoice", "paypal") != point_id("send_invoice", "stripe")


def test_meaning_match_without_shared_keyword(client):
    assert names(search_tools(client, "hand the buyer their cash back", k=1)) == ["refund_captured_payment"]


def test_keyword_match_separates_siblings(client):
    assert names(search_tools(client, "send a reminder for invoice INV-9", k=1)) == ["send_invoice_reminder"]


def test_role_filter_hides_tools_from_customers(client):
    hits = names(search_tools(client, "refund my order", role="customer", k=5))
    assert hits == ["show_invoice_details"]


def test_eval_only_tools_can_be_hidden(client):
    assert "stripe_refund_charge" in names(search_tools(client, "refund the stripe charge", k=5))
    assert "stripe_refund_charge" not in names(search_tools(client, "refund the stripe charge", k=5, include_eval_only=False))


def test_rebuild_does_not_duplicate(client):
    assert build_index(client, TOOLS) == len(TOOLS)
