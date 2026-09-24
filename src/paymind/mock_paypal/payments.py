"""Payments (captures, refunds) and checkout orders. Reads and writes SQLite."""

from __future__ import annotations

from decimal import InvalidOperation

from fastapi import APIRouter, Body, Request

from .errors import PayPalError, not_found
from .store import Store, amount_of, iso, money

router = APIRouter()


def store_of(request: Request) -> Store:
    return request.app.state.store


def get_or_404(store: Store, table: str, kind: str, record_id: str) -> dict:
    record = store.db.get(table, record_id)
    if not record:
        raise not_found(kind, record_id)
    return record


# ---- captures and refunds ---------------------------------------------------

@router.get("/v2/payments/captures/{capture_id}")
def show_captured_payment_details(request: Request, capture_id: str):
    return get_or_404(store_of(request), "captures", "captured payment", capture_id)


@router.post("/v2/payments/captures/{capture_id}/refund", status_code=201)
def refund_captured_payment(request: Request, capture_id: str, body: dict = Body(default={})):
    """No amount = refund whatever is left. Partial refunds allowed up to the remaining amount."""
    store = store_of(request)
    capture = get_or_404(store, "captures", "captured payment", capture_id)
    remaining = amount_of(capture["amount"]) - amount_of(capture["refunded_amount"])
    if remaining <= 0:
        raise PayPalError(422, "CAPTURE_FULLY_REFUNDED", "This payment has already been fully refunded.")
    if body.get("amount"):
        try:
            amount = amount_of(body["amount"])
        except InvalidOperation:
            raise PayPalError(400, "INVALID_PARAMETER_VALUE", "Refund amount must be a number.", "amount.value")
        if amount <= 0:
            raise PayPalError(422, "INVALID_PARAMETER_VALUE", "Refund amount must be positive.", "amount.value")
        if amount > remaining:
            raise PayPalError(422, "REFUND_AMOUNT_EXCEEDED", f"Refund of {amount} is more than the {remaining} left to refund.", "amount.value")
    else:
        amount = remaining
    return store.create_refund(capture, amount, body.get("note_to_payer"))


@router.get("/v2/payments/refunds/{refund_id}")
def show_refund_details(request: Request, refund_id: str):
    return get_or_404(store_of(request), "refunds", "refund", refund_id)


# ---- orders -------------------------------------------------------------------

@router.post("/v2/checkout/orders", status_code=201)
def create_order(request: Request, body: dict = Body(default={})):
    store = store_of(request)
    units = body.get("purchase_units") or []
    if not units or not (units[0].get("amount") or {}).get("value"):
        raise PayPalError(400, "MISSING_REQUIRED_PARAMETER", "An order needs purchase_units[0].amount.", "purchase_units")
    try:
        total = amount_of(units[0]["amount"])
    except InvalidOperation:
        raise PayPalError(400, "INVALID_PARAMETER_VALUE", "Order amount must be a number.", "purchase_units[0].amount.value")
    if total <= 0:
        raise PayPalError(422, "INVALID_PARAMETER_VALUE", "Order amount must be positive.", "purchase_units[0].amount.value")
    order_id = store.new_id("orders")
    return store.db.put("orders", {
        "id": order_id,
        "intent": body.get("intent", "CAPTURE"),
        "status": "CREATED",
        "create_time": iso(store.now()),
        "purchase_units": [{"reference_id": units[0].get("reference_id", "default"),
                            "description": units[0].get("description"), "amount": money(total)}],
        "links": [{"href": f"https://www.sandbox.paypal.com/checkoutnow?token={order_id}", "rel": "approve", "method": "GET"}],
    })


@router.get("/v2/checkout/orders/{order_id}")
def show_order_details(request: Request, order_id: str):
    return get_or_404(store_of(request), "orders", "order", order_id)


@router.post("/v2/checkout/orders/{order_id}/capture", status_code=201)
def capture_payment_for_order(request: Request, order_id: str, body: dict = Body(default={})):
    """In the mock, the buyer is assumed to have approved: capturing takes the money."""
    store = store_of(request)
    order = get_or_404(store, "orders", "order", order_id)
    if order["status"] == "COMPLETED":
        raise PayPalError(422, "ORDER_ALREADY_CAPTURED", "This order has already been captured.")
    payer_id = (order.get("payer") or {}).get("payer_id") or store.db.all("customers")[0]["payer_id"]
    capture = store.create_capture(amount_of(order["purchase_units"][0]["amount"]), payer_id)
    order["status"] = "COMPLETED"
    order["payer"] = capture["payer"]
    order["purchase_units"][0]["payments"] = {"captures": [capture]}
    return store.db.put("orders", order)
