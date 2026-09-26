"""Does tool search still find the right tool when there are 1000+ tools?

Hand-written test questions (not taken from the tool cards, so nothing is copied from the
index) are searched over three catalogue sizes:

  112    the real PayPal tools
  ~500   + 400 synthetic tools from look-alike services
  ~1000  + 900 synthetic tools

The synthetic tools come from other payment, billing, CRM, accounting, shipping and support
APIs, and deliberately reuse PayPal's vocabulary ("refund a charge", "list invoices", "open
dispute") so they compete with the real tools. We report how often the right tool is in the
top 5 (what the agent is shown) and in the top 1, for BM25 only, dense only and our hybrid
(both merged with RRF).

Runs locally (in-memory Qdrant + fastembed); no LLM, no network apart from the model download.

    python -m paymind.retrieval.eval_tools
"""

from __future__ import annotations

import itertools
import json
import time

from qdrant_client import QdrantClient, models

from .tool_index import DENSE_MODEL, DENSE_SIZE, SPARSE_MODEL, load_tools, search_text

# (question, acceptable tools). Written by hand, phrased the way users talk.
QUESTIONS: list[tuple[str, set[str]]] = [
    ("Send an invoice for $50 to john@x.com", {"create_draft_invoice", "send_invoice"}),
    ("bill Maria 3 hours of consulting", {"create_draft_invoice"}),
    ("email the invoice I drafted yesterday to the client", {"send_invoice"}),
    ("nudge the customer who hasn't paid their invoice", {"send_invoice_reminder"}),
    ("which invoices are still unpaid?", {"list_invoices", "search_for_invoices"}),
    ("find invoices sent to rahul.sharma@example.com", {"search_for_invoices"}),
    ("void the invoice we sent by mistake", {"cancel_sent_invoice"}),
    ("mark invoice INV-1004 as paid in cash", {"record_payment_for_invoice"}),
    ("what's the next invoice number?", {"generate_invoice_number"}),
    ("make a QR code so the customer can pay the invoice", {"generate_qr_code"}),
    ("What was my total sales volume last month?", {"list_transactions"}),
    ("show me all payments received this week", {"list_transactions"}),
    ("how much money is in the account right now?", {"list_all_balances"}),
    ("Is there a dispute open from user_123?", {"list_disputes"}),
    ("any customer complaints or chargebacks open?", {"list_disputes"}),
    ("show the details of case PP-D-38106", {"show_dispute_details"}),
    ("reply to the buyer on their dispute", {"send_message_about_dispute_to_other_party"}),
    ("we agree with the customer, give them their money back on the dispute", {"accept_claim"}),
    ("offer the buyer $20 to settle the case", {"make_offer_to_resolve_dispute"}),
    ("the customer accepts the shop's settlement offer", {"accept_offer_to_resolve_dispute"}),
    ("refund $15 of payment 2GG279541U471931P", {"refund_captured_payment"}),
    ("give the customer their money back for that payment", {"refund_captured_payment"}),
    ("what happened to refund 1JU08902781691411?", {"show_refund_details"}),
    ("look up captured payment 2GG279541U471931P", {"show_captured_payment_details"}),
    ("capture the money we authorized for order 5O190127TN364715T", {"capture_authorized_payment", "capture_payment_for_order"}),
    ("release the hold on the authorized payment, we won't ship", {"void_authorized_payment"}),
    ("create a checkout order for $120", {"create_order"}),
    ("what's the status of order 5O190127TN364715T?", {"show_order_details"}),
    ("add the Blue Dart tracking number to the order", {"add_tracking_information_for_an_order", "add_tracking_information_for_multiple_paypal_transactions"}),
    ("where is the parcel for this transaction? show tracking", {"show_tracking_information"}),
    ("pay 5 freelancers by email in one go", {"create_batch_payout"}),
    ("did the payout batch go through?", {"show_payout_batch_details"}),
    ("start a monthly plan for customer at $10", {"create_subscription", "create_plan"}),
    ("pause this customer's subscription", {"suspend_subscription"}),
    ("stop the subscription for good", {"cancel_subscription"}),
    ("which products do we sell on subscription?", {"list_products"}),
    ("notify our server when a payment completes", {"create_webhook"}),
    ("list the webhooks we have set up", {"list_webhooks"}),
    ("how many rupees would 100 dollars be?", {"create_currency_exchange_quote", "get_currency_exchange_quote"}),
    ("create a buy button link for a single t-shirt", {"creates_a_payment_resource_for_a_single_item_purchase"}),
    ("save the customer's card for next time", {"create_payment_token", "create_a_setup_token"}),
    ("onboard a new seller account on the platform", {"create_managed_account"}),
]

# Look-alike services: same verbs and nouns as PayPal, different systems.
SERVICES = ["Stripe-style Charges", "Razorpay-style Payments", "Square-style POS", "Adyen-style Checkout", "Chargebee Billing",
            "QuickBooks-style Accounting", "Xero-style Ledger", "Zoho Invoice", "HubSpot-style CRM", "Salesforce-style CRM",
            "Shopify-style Store", "WooCommerce Orders", "FedEx-style Shipping", "Delhivery Logistics", "Zendesk-style Support",
            "Freshdesk Tickets", "Twilio-style Messaging", "Mailchimp-style Email", "Gusto-style Payroll", "Tally ERP",
            "Snowflake-style Analytics", "Looker-style BI", "Intercom-style Chat", "Plaid-style Banking", "Wise-style FX"]
ACTIONS = [
    ("create", "invoice", "Create a new invoice for a customer with line items and a due date."),
    ("send", "invoice", "Email an invoice to the customer."),
    ("list", "invoices", "List invoices, optionally filtered by status or customer."),
    ("remind", "invoice", "Send a payment reminder for an unpaid invoice."),
    ("void", "invoice", "Void or cancel an invoice that was issued."),
    ("refund", "charge", "Refund all or part of a charge back to the customer's card."),
    ("get", "charge", "Retrieve the details of a charge or payment."),
    ("list", "payments", "List payments received in a date range with totals."),
    ("capture", "payment", "Capture a previously authorized payment."),
    ("cancel", "authorization", "Release an authorization hold."),
    ("list", "disputes", "List open disputes and chargebacks raised by customers."),
    ("respond", "dispute", "Submit a response or message on a customer dispute."),
    ("accept", "dispute", "Accept a dispute and refund the customer."),
    ("get", "balance", "Get the current account balance."),
    ("create", "payout", "Send money to recipients in a batch payout."),
    ("create", "subscription", "Start a recurring subscription for a customer."),
    ("cancel", "subscription", "Cancel a customer's subscription."),
    ("pause", "subscription", "Pause billing on a subscription."),
    ("create", "order", "Create a sales order."),
    ("get", "order", "Get the status of an order."),
    ("create", "shipment", "Create a shipment and tracking number for an order."),
    ("track", "shipment", "Get tracking updates for a shipment."),
    ("create", "webhook", "Register a webhook endpoint for event notifications."),
    ("get", "exchange_rate", "Get a currency exchange rate quote."),
    ("create", "payment_link", "Create a shareable payment link for a product."),
    ("save", "card", "Save a customer's card for future payments."),
    ("create", "customer", "Create a customer record."),
    ("search", "customers", "Search customers by name or email."),
    ("create", "ticket", "Open a support ticket for a customer issue."),
    ("list", "transactions", "List ledger transactions for reporting."),
    ("run", "sales_report", "Run a sales report for a period."),
    ("create", "journal_entry", "Post a journal entry in the ledger."),
    ("send", "sms", "Send a text message to a customer."),
    ("run", "payroll", "Run payroll for employees."),
    ("onboard", "merchant", "Onboard a new merchant or seller account."),
    ("create", "coupon", "Create a discount coupon."),
]


def synthetic_tools(n: int) -> list[dict]:
    """n look-alike tools: every (service, action) pair, in a fixed order."""
    out = []
    for service, (verb, noun, what) in itertools.islice(itertools.product(SERVICES, ACTIONS), n):
        slug = service.split("-")[0].split()[0].lower()
        name = f"{slug}_{verb}_{noun}"
        out.append({
            "name": name, "service": slug, "category": noun, "action_type": "write" if verb not in ("get", "list", "track", "search") else "read",
            "allowed_roles": ["accountant"], "eval_only": True,
            "description": f"{service}: {what}",
            "example_questions": [f"{verb} {noun.replace('_', ' ')} in {service.split('-')[0]}",
                                  f"{what.split(',')[0].rstrip('.').lower()} using {service}"],
        })
    return out


def build(client: QdrantClient, tools: list[dict], tiers: list[int], name: str = "tools_eval") -> None:
    client.create_collection(name, vectors_config={"dense": models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE)},
                             sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    points = [models.PointStruct(id=i, vector={"dense": models.Document(text=search_text(t), model=DENSE_MODEL),
                                               "bm25": models.Document(text=search_text(t), model=SPARSE_MODEL)},
                                 payload={"name": t["name"], "tier": tier}) for i, (t, tier) in enumerate(zip(tools, tiers))]
    for i in range(0, len(points), 64):
        client.upsert(name, points=points[i:i + 64])


def search(client: QdrantClient, query: str, mode: str, max_tier: int, k: int = 5, name: str = "tools_eval") -> list[str]:
    flt = models.Filter(must=[models.FieldCondition(key="tier", range=models.Range(lte=max_tier))])
    sparse = models.Document(text=query, model=SPARSE_MODEL)
    dense = models.Document(text=query, model=DENSE_MODEL)
    if mode == "bm25":
        res = client.query_points(name, query=sparse, using="bm25", query_filter=flt, limit=k)
    elif mode == "dense":
        res = client.query_points(name, query=dense, using="dense", query_filter=flt, limit=k)
    else:  # hybrid: the same as production (top 20 from each, merged with RRF)
        res = client.query_points(name, prefetch=[models.Prefetch(query=sparse, using="bm25", filter=flt, limit=20),
                                                  models.Prefetch(query=dense, using="dense", filter=flt, limit=20)],
                                  query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k)
    return [p.payload["name"] for p in res.points]


def main() -> None:
    real = [t for t in load_tools() if not t.get("eval_only")]
    fake = synthetic_tools(900)
    tools, tiers = real + fake, [0] * len(real) + [1] * 400 + [2] * 500
    client = QdrantClient(":memory:")
    started = time.time()
    build(client, tools, tiers)
    print(f"indexed {len(tools)} tools in {time.time() - started:.0f}s")
    sizes = {0: len(real), 1: len(real) + 400, 2: len(real) + 900}
    results = {}
    for tier, size in sizes.items():
        for mode in ("bm25", "dense", "hybrid"):
            top5 = top1 = 0
            took = []
            for q, ok in QUESTIONS:
                t0 = time.perf_counter()
                hits = search(client, q, mode, tier)
                took.append(time.perf_counter() - t0)
                top5 += bool(ok & set(hits))
                top1 += bool(hits and hits[0] in ok)
            results[f"{size}/{mode}"] = {"tools": size, "mode": mode, "top5": round(top5 / len(QUESTIONS), 3),
                                         "top1": round(top1 / len(QUESTIONS), 3),
                                         "ms_per_query": round(1000 * sorted(took)[len(took) // 2], 1)}
            r = results[f"{size}/{mode}"]
            print(f"{size:>5} tools  {mode:<6}  top-5 {r['top5']:.0%}  top-1 {r['top1']:.0%}  {r['ms_per_query']} ms")
    print(json.dumps({"questions": len(QUESTIONS), "results": list(results.values())}))


if __name__ == "__main__":
    main()
