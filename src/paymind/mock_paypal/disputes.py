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
    refund = store.create_refund(capture, refund_amount, note or f"Dispute {dispute['dispute_id']} resolved")
    dispute["refund_id"] = refund["id"]
    return refund


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
    """Seller accepts the buyer's claim: full refund, dispute resolved for the buyer."""
    store = store_of(request)
    dispute = get_open_dispute(store, dispute_id)
    refund_buyer(store, dispute, note=body.get("note"))
    dispute["dispute_outcome"] = {"outcome_code": "RESOLVED_BUYER_FAVOUR", "amount_refunded": dispute["dispute_amount"]}
    return update(store, dispute, "RESOLVED", "RESOLVED")


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
    dispute["dispute_outcome"] = {"outcome_code": "RESOLVED_WITH_PAYOUT", "amount_refunded": dispute["offer"]["offer_amount"]}
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
