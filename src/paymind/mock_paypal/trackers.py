"""Shipment tracking for PayPal payments (PayPal's Add Tracking API, v1). Reads and writes SQLite.

A tracker links a payment (transaction_id) to a carrier and tracking number, with a status.
Its ID is "<transaction_id>-<tracking_number>", as in PayPal.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Request

from .errors import PayPalError, not_found
from .store import Store, iso

router = APIRouter(prefix="/v1/shipping")

STATUSES = {"SHIPPED", "ON_HOLD", "DELIVERED", "CANCELLED"}


def store_of(request: Request) -> Store:
    return request.app.state.store


def check(tracker: dict) -> None:
    for field in ("transaction_id", "tracking_number", "carrier"):
        if not str(tracker.get(field) or "").strip():
            raise PayPalError(400, "MISSING_REQUIRED_PARAMETER", f"{field} is required.", field)
    if tracker.get("status") not in STATUSES:
        raise PayPalError(400, "INVALID_PARAMETER_VALUE", f"status must be one of {sorted(STATUSES)}.", "status")


@router.post("/trackers-batch")
def add_tracking_information_for_multiple_paypal_transactions(request: Request, body: dict = Body(...)):
    store = store_of(request)
    added = []
    for t in body.get("trackers") or []:
        check(t)
        if not store.db.get("captures", t["transaction_id"]):
            raise not_found("transaction", t["transaction_id"])
        tracker_id = f"{t['transaction_id']}-{t['tracking_number']}"
        now = iso(store.now())
        old = store.db.get("trackers", tracker_id) or {}
        added.append(store.db.put("trackers", {
            "id": tracker_id, "transaction_id": t["transaction_id"], "tracking_number": t["tracking_number"],
            "status": t["status"], "carrier": t["carrier"], "shipment_date": t.get("shipment_date") or now[:10],
            "create_time": old.get("create_time", now), "update_time": now}))
    return {"tracker_identifiers": [{"transaction_id": a["transaction_id"], "tracking_number": a["tracking_number"]} for a in added],
            "errors": []}


def trackers_for(store: Store, transaction_id: str | None) -> dict:
    """All trackers for one payment, newest first. Served at /mock/trackers: the Postman collection
    has no "list trackers" request, so it isn't dressed up as a PayPal API."""
    rows = [t for t in store.db.all("trackers") if not transaction_id or t["transaction_id"] == transaction_id]
    return {"trackers": sorted(rows, key=lambda t: t["update_time"], reverse=True)}


@router.get("/trackers/{tracking_id}")
def show_tracking_information(request: Request, tracking_id: str):
    tracker = store_of(request).db.get("trackers", tracking_id)
    if not tracker:
        raise not_found("tracker", tracking_id)
    return tracker


@router.put("/trackers/{tracking_id}")
def update_or_cancel_tracking_information_for_paypal_transaction(request: Request, tracking_id: str, body: dict = Body(...)):
    store = store_of(request)
    tracker = store.db.get("trackers", tracking_id)
    if not tracker:
        raise not_found("tracker", tracking_id)
    merged = {**tracker, **{k: v for k, v in body.items() if k in ("status", "carrier")}}
    check(merged)
    merged["update_time"] = iso(store.now())
    return store.db.put("trackers", merged)
