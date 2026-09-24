"""Invoicing endpoints: drafts, sending, reminders, payments, search. Reads and writes SQLite."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Body, Request

from .errors import PayPalError, not_found
from .store import CURRENCY, Store, amount_of, iso, line_item, money

router = APIRouter(prefix="/v2/invoicing")

SENDABLE = {"DRAFT", "SCHEDULED"}
OPEN = {"SENT", "UNPAID", "PARTIALLY_PAID", "SCHEDULED"}


def store_of(request: Request) -> Store:
    return request.app.state.store


def get_invoice(store: Store, invoice_id: str) -> dict:
    invoice = store.db.get("invoices", invoice_id)
    if not invoice:
        raise not_found("invoice", invoice_id)
    return invoice


def save(store: Store, invoice: dict) -> dict:
    invoice["detail"]["metadata"]["last_update_time"] = iso(store.now())
    return store.db.put("invoices", invoice)


def recipient_email(invoice: dict) -> str | None:
    recipients = invoice.get("primary_recipients") or []
    return (recipients[0].get("billing_info") or {}).get("email_address") if recipients else None


def parse_items(body: dict) -> list[dict]:
    """Accept PayPal's items[] or, leniently, a plain amount (what an agent often sends)."""
    items = body.get("items") or []
    if not items and body.get("amount"):
        value = body["amount"].get("value") if isinstance(body["amount"], dict) else body["amount"]
        items = [line_item(body.get("detail", {}).get("note") or "Invoice amount", str(value))]
    if not items:
        raise PayPalError(400, "MISSING_REQUIRED_PARAMETER", "An invoice needs at least one item (or an amount).", "items")
    cleaned = []
    for i, item in enumerate(items):
        try:
            price = Decimal(str((item.get("unit_amount") or {}).get("value")))
            qty = int(Decimal(str(item.get("quantity", "1"))))
        except (InvalidOperation, TypeError, ValueError):
            raise PayPalError(400, "INVALID_PARAMETER_VALUE", "Item price and quantity must be numbers.", f"items[{i}]")
        if price <= 0 or qty <= 0:
            raise PayPalError(422, "INVALID_PARAMETER_VALUE", "Item price and quantity must be positive.", f"items[{i}]")
        cleaned.append(line_item(item.get("name") or "Item", str(price), qty))
    return cleaned


def summary(invoice: dict) -> dict:
    return {k: invoice[k] for k in ("id", "status", "detail", "primary_recipients", "amount", "due_amount", "links")}


@router.post("/generate-next-invoice-number")
def generate_invoice_number(request: Request):
    return {"invoice_number": f"INV-{store_of(request).db.meta('next_invoice_number', 1001)}"}


@router.post("/invoices", status_code=201)
def create_draft_invoice(request: Request, body: dict = Body(default={})):
    store = store_of(request)
    items = parse_items(body)
    detail = body.get("detail") or {}
    if detail.get("currency_code", CURRENCY) != CURRENCY:
        raise PayPalError(422, "CURRENCY_NOT_SUPPORTED", f"This account only invoices in {CURRENCY}.", "detail.currency_code")
    return store.new_invoice(items, body.get("primary_recipients") or [], detail.get("note"), detail.get("invoice_number"))


@router.get("/invoices")
def list_invoices(request: Request, page: int = 1, page_size: int = 20, total_required: bool = False):
    invoices = sorted(store_of(request).db.all("invoices"), key=lambda i: i["detail"]["invoice_date"], reverse=True)
    page_size = max(1, min(page_size, 100))
    result = {"items": [summary(i) for i in invoices[(page - 1) * page_size : page * page_size]]}
    if total_required:
        result |= {"total_items": len(invoices), "total_pages": -(-len(invoices) // page_size)}
    return result


@router.get("/invoices/{invoice_id}")
def show_invoice_details(request: Request, invoice_id: str):
    return get_invoice(store_of(request), invoice_id)


@router.post("/invoices/{invoice_id}/send")
def send_invoice(request: Request, invoice_id: str, body: dict = Body(default={})):
    store = store_of(request)
    invoice = get_invoice(store, invoice_id)
    if invoice["status"] not in SENDABLE:
        raise PayPalError(422, "CANNOT_SEND_INVOICE", f"Invoice is {invoice['status']}; only draft or scheduled invoices can be sent.")
    if not recipient_email(invoice):
        raise PayPalError(422, "MISSING_RECIPIENT", "Add a recipient email before sending this invoice.", "primary_recipients")
    invoice["status"] = "SENT"
    save(store, invoice)
    return {"href": f"https://www.sandbox.paypal.com/invoice/p/#{invoice_id}", "rel": "payer-view", "method": "GET",
            "status": invoice["status"], "sent_to": recipient_email(invoice)}


@router.post("/invoices/{invoice_id}/remind", status_code=204)
def send_invoice_reminder(request: Request, invoice_id: str, body: dict = Body(default={})):
    store = store_of(request)
    invoice = get_invoice(store, invoice_id)
    if invoice["status"] not in OPEN:
        raise PayPalError(422, "CANNOT_REMIND_INVOICE", f"Invoice is {invoice['status']}; reminders only go out for unpaid sent invoices.")
    invoice.setdefault("reminders", []).append({"time": iso(store.now()), "note": body.get("note")})
    save(store, invoice)


@router.post("/invoices/{invoice_id}/cancel", status_code=204)
def cancel_sent_invoice(request: Request, invoice_id: str, body: dict = Body(default={})):
    store = store_of(request)
    invoice = get_invoice(store, invoice_id)
    if invoice["status"] not in OPEN:
        raise PayPalError(422, "CANNOT_CANCEL_INVOICE", f"Invoice is {invoice['status']}; only sent, unpaid invoices can be cancelled.")
    invoice["status"] = "CANCELLED"
    save(store, invoice)


@router.delete("/invoices/{invoice_id}", status_code=204)
def delete_invoice(request: Request, invoice_id: str):
    store = store_of(request)
    invoice = get_invoice(store, invoice_id)
    if invoice["status"] not in SENDABLE:
        raise PayPalError(422, "CANNOT_DELETE_INVOICE", f"Invoice is {invoice['status']}; only draft or scheduled invoices can be deleted.")
    store.db.delete("invoices", invoice_id)


@router.post("/invoices/{invoice_id}/payments")
def record_payment_for_invoice(request: Request, invoice_id: str, body: dict = Body(default={})):
    store = store_of(request)
    invoice = get_invoice(store, invoice_id)
    if invoice["status"] not in OPEN:
        raise PayPalError(422, "CANNOT_RECORD_PAYMENT", f"Invoice is {invoice['status']}; payments can only be recorded on sent, unpaid invoices.")
    due = amount_of(invoice["due_amount"])
    amount = amount_of(body["amount"]) if body.get("amount") else due
    if amount <= 0 or amount > due:
        raise PayPalError(422, "INVALID_PAYMENT_AMOUNT", f"Payment must be between 0.01 and the amount due ({due}).", "amount")
    payment_id = "EXTR-" + store.new_id()
    invoice["payments"]["transactions"].append({
        "payment_id": payment_id, "method": body.get("method", "CASH"),
        "payment_date": body.get("payment_date", store.now().date().isoformat()), "amount": money(amount),
    })
    invoice["payments"]["paid_amount"] = money(amount_of(invoice["payments"]["paid_amount"]) + amount)
    invoice["due_amount"] = money(due - amount)
    invoice["status"] = "MARKED_AS_PAID" if due - amount == 0 else "PARTIALLY_PAID"
    save(store, invoice)
    return {"payment_id": payment_id, "status": invoice["status"], "due_amount": invoice["due_amount"]}


@router.post("/search-invoices")
def search_for_invoices(request: Request, body: dict = Body(default={}), page: int = 1, page_size: int = 20):
    """Filters: recipient_email, recipient_first_name, recipient_last_name, invoice_number,
    status (list), total_amount_range {lower_amount, upper_amount}, invoice_date_range {start, end}."""
    results = []
    for inv in store_of(request).db.all("invoices"):
        billing = ((inv.get("primary_recipients") or [{}])[0].get("billing_info") or {})
        name = billing.get("name") or {}
        checks = [
            ("recipient_email", lambda v: (billing.get("email_address") or "").lower() == v.lower()),
            ("recipient_first_name", lambda v: (name.get("given_name") or "").lower() == v.lower()),
            ("recipient_last_name", lambda v: (name.get("surname") or "").lower() == v.lower()),
            ("invoice_number", lambda v: inv["detail"]["invoice_number"] == v),
            ("status", lambda v: inv["status"] in (v if isinstance(v, list) else [v])),
        ]
        if not all(check(body[key]) for key, check in checks if body.get(key)):
            continue
        if rng := body.get("total_amount_range"):
            total = amount_of(inv["amount"])
            if total < amount_of(rng.get("lower_amount")) or ("upper_amount" in rng and total > amount_of(rng["upper_amount"])):
                continue
        if dr := body.get("invoice_date_range"):
            day = inv["detail"]["invoice_date"]
            if (dr.get("start") and day < dr["start"][:10]) or (dr.get("end") and day > dr["end"][:10]):
                continue
        results.append(summary(inv))
    page_size = max(1, min(page_size, 100))
    return {"items": results[(page - 1) * page_size : page * page_size], "total_items": len(results),
            "total_pages": -(-len(results) // page_size)}
