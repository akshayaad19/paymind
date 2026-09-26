"""Dispute endpoints. Reads and writes SQLite.

Deviation from PayPal, on purpose: list items include `disputed_transactions`
with the buyer, so "is there a dispute from user_123?" can be answered from the
list. Real PayPal only returns a summary and needs a details call per dispute.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Request

from .errors import PayPalError, not_found
from .store import Store, amount_of, iso, money

router = APIRouter(prefix="/v1/customer/disputes")

CLOSED = {"RESOLVED"}


def store_of(request: Request) -> Store:
    return request.app.state.store


def actor(request: Request) -> str:
    """Who is acting: BUYER or SELLER. Real PayPal knows this from the access token (buyer and
    seller log in separately); the mock is told through the X-PayPal-Actor header."""
    return "BUYER" if request.headers.get("x-paypal-actor", "").upper() == "BUYER" else "SELLER"


def get_dispute(store: Store, dispute_id: str) -> dict:
    dispute = store.db.get("disputes", dispute_id)
    if not dispute:
        raise not_found("dispute", dispute_id)
    return dispute


def get_open_dispute(store: Store, dispute_id: str) -> dict:
    dispute = get_dispute(store, dispute_id)
    if dispute["status"] in CLOSED:
        raise PayPalError(422, "DISPUTE_ALREADY_RESOLVED", f"Dispute {dispute_id} is already resolved.")
    return dispute


def save(store: Store, dispute: dict) -> dict:
    dispute["update_time"] = iso(store.now())
    return store.db.put("disputes", dispute)


def update(store: Store, dispute: dict, status: str, state: str) -> dict:
    dispute["status"], dispute["dispute_state"] = status, state
    save(store, dispute)
    return {"links": [{"href": f"/v1/customer/disputes/{dispute['dispute_id']}", "rel": "self", "method": "GET"}],
            "dispute_id": dispute["dispute_id"], "status": status}


def refund_buyer(store: Store, dispute: dict, amount=None, note: str | None = None) -> dict:
    """Resolving in the buyer's favour refunds the disputed payment."""
    capture = store.db.get("captures", dispute["disputed_transactions"][0]["seller_transaction_id"])
    remaining = amount_of(capture["amount"]) - amount_of(capture["refunded_amount"])
    refund_amount = min(amount if amount is not None else amount_of(dispute["dispute_amount"]), remaining)
    if refund_amount <= 0:
        raise PayPalError(422, "NOTHING_TO_REFUND", "This payment has already been refunded in full.", "refund_amount")
    refund = store.create_refund(capture, refund_amount, note or f"Dispute {dispute['dispute_id']} resolved")
    dispute["refund_id"] = refund["id"]
    return refund


BUYER_REASONS = {"MERCHANDISE_OR_SERVICE_NOT_AS_DESCRIBED", "MERCHANDISE_OR_SERVICE_NOT_RECEIVED", "INCORRECT_AMOUNT",
                 "DUPLICATE_TRANSACTION", "CREDIT_NOT_PROCESSED"}


def open_dispute(store: Store, body: dict) -> dict:
    """A buyer opens a case about one of their payments: the disputed amount can't be more than
    what's left after refunds, and a payment can only have one open case."""
    import random
    from datetime import timedelta

    capture = store.db.get("captures", str(body.get("capture_id") or ""))
    if not capture:
        raise not_found("capture", str(body.get("capture_id")))
    reason = body.get("reason")
    if reason not in BUYER_REASONS:
        raise PayPalError(400, "INVALID_PARAMETER_VALUE", f"reason must be one of {sorted(BUYER_REASONS)}.", "reason")
    message = str(body.get("message") or "").strip()
    if not message:
        raise PayPalError(400, "MISSING_REQUIRED_PARAMETER", "Describe the problem in message.", "message")
    remaining = amount_of(capture["amount"]) - amount_of(capture["refunded_amount"])
    amount = amount_of(body["amount"]) if body.get("amount") else remaining
    if amount <= 0 or amount > remaining:
        raise PayPalError(422, "INVALID_DISPUTE_AMOUNT", f"The amount must be between 0.01 and {remaining} (what's left of this payment).", "amount")
    if any(d["disputed_transactions"][0]["seller_transaction_id"] == capture["id"] and d["status"] not in CLOSED
           for d in store.db.all("disputes")):
        raise PayPalError(422, "DISPUTE_ALREADY_OPEN", "This payment already has an open case.", "capture_id")
    payer = store.customer((capture.get("payer") or {}).get("payer_id", ""))
    merchant = store.merchant()
    now = store.now()
    dispute_id = next(i for i in (f"PP-D-{random.randint(10000, 99999)}" for _ in range(100)) if not store.db.get("disputes", i))
    return store.db.put("disputes", {
        "dispute_id": dispute_id, "create_time": iso(now), "update_time": iso(now), "reason": reason,
        "status": "WAITING_FOR_SELLER_RESPONSE", "dispute_state": "REQUIRED_ACTION", "dispute_amount": money(amount),
        "dispute_life_cycle_stage": "INQUIRY", "dispute_channel": "INTERNAL",
        "seller_response_due_date": iso(now + timedelta(days=10)),
        "disputed_transactions": [{
            "seller_transaction_id": capture["id"], "create_time": capture["create_time"], "gross_amount": capture["amount"],
            "buyer": {"payer_id": payer["payer_id"], "name": f"{payer['given_name']} {payer['surname']}", "email": payer["email"]},
            "seller": {"email": merchant.get("email_address"), "name": merchant.get("business_name")}}],
        "messages": [{"posted_by": "BUYER", "time_posted": iso(now), "content": message}],
        "evidences": [],
    })


def shop_acted(store: Store, dispute: dict, action: dict) -> None:
    """Record what the shop did (refund or replacement). The case then waits for the customer."""
    dispute.pop("offer", None)
    dispute["seller_action"] = {**{k: v for k, v in action.items() if v is not None}, "time": iso(store.now())}


def buyer_closes(store: Store, dispute_id: str, body: dict) -> dict:
    """Only the customer closes a case: after the shop refunded or sent a replacement, or any time
    they're satisfied. The outcome records what the shop did."""
    dispute = get_open_dispute(store, dispute_id)
    note = str(body.get("message") or "").strip() or "I'm happy with how this was sorted out, so I'm closing this case."
    dispute.setdefault("messages", []).append({"posted_by": "BUYER", "time_posted": iso(store.now()), "content": f"✅ {note}"})
    action = dispute.get("seller_action") or {}
    if action.get("type") == "refund":
        outcome = {"outcome_code": "RESOLVED_BUYER_FAVOUR", "amount_refunded": action.get("amount")}
    elif action.get("type") == "replacement":
        outcome = {"outcome_code": "RESOLVED_WITH_REPLACEMENT", "carrier": action.get("carrier"), "tracking_number": action.get("tracking_number")}
    else:
        outcome = {"outcome_code": "CANCELED_BY_BUYER"}
    dispute["dispute_outcome"] = {**outcome, "closed_by": "BUYER"}
    return update(store, dispute, "RESOLVED", "RESOLVED")


def buyer_reopens(store: Store, dispute_id: str, body: dict) -> dict:
    """The shop acted (replacement / refund) but the customer says it didn't arrive: the turn goes
    back to the shop, with what it did kept in the history."""
    dispute = get_open_dispute(store, dispute_id)
    action = dispute.pop("seller_action", None)
    if not action:
        raise PayPalError(422, "NOTHING_TO_REOPEN", "The shop hasn't sent anything on this case yet.")
    dispute.setdefault("previous_actions", []).append(action)
    note = str(body.get("message") or "").strip() or "It hasn't arrived."
    dispute.setdefault("messages", []).append({"posted_by": "BUYER", "time_posted": iso(store.now()), "content": f"❌ {note}"})
    return update(store, dispute, "WAITING_FOR_SELLER_RESPONSE", "REQUIRED_ACTION")


def close_with_replacement(store: Store, dispute_id: str, body: dict) -> dict:
    """The shop sends a replacement (no money moves); the case waits for the customer to confirm."""
    dispute = get_open_dispute(store, dispute_id)
    shop_acted(store, dispute, {"type": "replacement", "carrier": body.get("carrier"), "tracking_number": body.get("tracking_number")})
    return update(store, dispute, "WAITING_FOR_BUYER_RESPONSE", "REQUIRED_OTHER_PARTY_ACTION")


def summary(d: dict) -> dict:
    keys = ("dispute_id", "create_time", "update_time", "reason", "status", "dispute_state",
            "dispute_amount", "dispute_life_cycle_stage", "disputed_transactions")
    return {k: d[k] for k in keys} | {"links": [{"href": f"/v1/customer/disputes/{d['dispute_id']}", "rel": "self"}]}


@router.get("")
def list_disputes(request: Request, dispute_state: str | None = None, disputed_transaction_id: str | None = None,
                  start_time: str | None = None, page_size: int = 10):
    """dispute_state may be a comma-separated list, e.g. REQUIRED_ACTION,UNDER_PAYPAL_REVIEW."""
    states = set(dispute_state.split(",")) if dispute_state else None
    items = []
    for d in sorted(store_of(request).db.all("disputes"), key=lambda d: d["create_time"], reverse=True):
        if states and d["dispute_state"] not in states:
            continue
        if disputed_transaction_id and d["disputed_transactions"][0]["seller_transaction_id"] != disputed_transaction_id:
            continue
        if start_time and d["create_time"] < start_time:
            continue
        items.append(summary(d))
    return {"items": items[: max(1, min(page_size, 50))]}


@router.get("/{dispute_id}")
def show_dispute_details(request: Request, dispute_id: str):
    return get_dispute(store_of(request), dispute_id)


@router.post("/{dispute_id}/accept-claim")
def accept_claim(request: Request, dispute_id: str, body: dict = Body(default={})):
    """The shop refunds the buyer: the disputed amount, or refund_amount (partial). PayMind rule: the
    case isn't closed by the shop; it waits for the customer to confirm (only the customer closes)."""
    store = store_of(request)
    dispute = get_open_dispute(store, dispute_id)
    amount = amount_of(body["refund_amount"]) if body.get("refund_amount") else None
    if amount is not None and (amount <= 0 or amount > amount_of(dispute["dispute_amount"])):
        raise PayPalError(422, "INVALID_REFUND_AMOUNT", "The refund must be between 0.01 and the disputed amount.", "refund_amount")
    refund = refund_buyer(store, dispute, amount, note=body.get("note"))
    shop_acted(store, dispute, {"type": "refund", "amount": refund["amount"], "refund_id": refund["id"], "note": body.get("note")})
    result = update(store, dispute, "WAITING_FOR_BUYER_RESPONSE", "REQUIRED_OTHER_PARTY_ACTION")
    return {**result, "refund": {"id": refund["id"], "amount": refund["amount"]}}  # what was actually refunded


@router.post("/{dispute_id}/make-offer")
def make_offer_to_resolve_dispute(request: Request, dispute_id: str, body: dict = Body(default={})):
    store = store_of(request)
    dispute = get_open_dispute(store, dispute_id)
    offer = amount_of(body.get("offer_amount")) if body.get("offer_amount") else None
    if offer is None or offer <= 0:
        raise PayPalError(400, "MISSING_REQUIRED_PARAMETER", "Include a positive offer_amount.", "offer_amount")
    if offer > amount_of(dispute["dispute_amount"]):
        raise PayPalError(422, "OFFER_AMOUNT_EXCEEDED", "The offer can't be more than the disputed amount.", "offer_amount")
    dispute["offer"] = {"offer_type": body.get("offer_type", "REFUND"), "offer_amount": money(offer), "note": body.get("note")}
    return update(store, dispute, "WAITING_FOR_BUYER_RESPONSE", "REQUIRED_OTHER_PARTY_ACTION")


@router.post("/{dispute_id}/accept-offer")
def accept_offer_to_resolve_dispute(request: Request, dispute_id: str, body: dict = Body(default={})):
    """Buyer accepts the seller's offer: the offer amount is refunded."""
    store = store_of(request)
    dispute = get_open_dispute(store, dispute_id)
    if not dispute.get("offer"):
        raise PayPalError(422, "NO_OFFER_TO_ACCEPT", "The seller hasn't made an offer on this dispute.")
    refund_buyer(store, dispute, amount_of(dispute["offer"]["offer_amount"]), body.get("note"))
    dispute["dispute_outcome"] = {"outcome_code": "RESOLVED_WITH_PAYOUT", "amount_refunded": dispute["offer"]["offer_amount"], "closed_by": "BUYER"}
    return update(store, dispute, "RESOLVED", "RESOLVED")


@router.post("/{dispute_id}/deny-offer")
def deny_offer_to_resolve_dispute(request: Request, dispute_id: str, body: dict = Body(default={})):
    store = store_of(request)
    dispute = get_open_dispute(store, dispute_id)
    if not dispute.pop("offer", None):
        raise PayPalError(422, "NO_OFFER_TO_DENY", "The seller hasn't made an offer on this dispute.")
    dispute["messages"].append({"posted_by": "BUYER", "time_posted": iso(store.now()), "content": body.get("note") or "Offer declined."})
    return update(store, dispute, "WAITING_FOR_SELLER_RESPONSE", "REQUIRED_ACTION")


@router.post("/{dispute_id}/send-message")
def send_message_about_dispute_to_other_party(request: Request, dispute_id: str, body: dict = Body(default={})):
    store = store_of(request)
    dispute = get_open_dispute(store, dispute_id)
    if not body.get("message"):
        raise PayPalError(400, "MISSING_REQUIRED_PARAMETER", "Include the message text.", "message")
    posted_by = actor(request)
    dispute["messages"].append({"posted_by": posted_by, "time_posted": iso(store.now()), "content": body["message"]})
    save(store, dispute)
    return {"links": [{"href": f"/v1/customer/disputes/{dispute_id}", "rel": "self", "method": "GET"}],
            "posted_by": posted_by}


@router.post("/{dispute_id}/provide-evidence")
def provide_evidence(request: Request, dispute_id: str, body: dict = Body(default={})):
    store = store_of(request)
    dispute = get_open_dispute(store, dispute_id)
    dispute["evidences"].append({"date": iso(store.now()), "details": body.get("input") or body})
    return update(store, dispute, "UNDER_REVIEW", "UNDER_PAYPAL_REVIEW")


@router.post("/{dispute_id}/escalate")
def escalate_dispute_to_claim(request: Request, dispute_id: str, body: dict = Body(default={})):
    store = store_of(request)
    dispute = get_open_dispute(store, dispute_id)
    if dispute["dispute_life_cycle_stage"] in ("CHARGEBACK", "PRE_ARBITRATION"):
        raise PayPalError(422, "CANNOT_ESCALATE", "This dispute is already past the claim stage.")
    dispute["dispute_life_cycle_stage"] = "CHARGEBACK"
    dispute["messages"].append({"posted_by": "BUYER", "time_posted": iso(store.now()), "content": body.get("note") or "Escalated to a claim."})
    return update(store, dispute, "UNDER_REVIEW", "UNDER_PAYPAL_REVIEW")
