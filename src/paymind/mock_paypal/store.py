"""Business helpers shared by the endpoints: IDs, money, and actions that touch
several tables at once (a capture adds a ledger row; a refund updates the
capture, adds a refund and a ledger row).

All data is read from and written to SQLite through `self.db`. There is no
generated or in-memory data: the starting data lives in data/mock/initial.db.
"""

from __future__ import annotations

import secrets
import string
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .db import Database

CURRENCY = "USD"
ID_CHARS = string.ascii_uppercase + string.digits


def money(value: Decimal | str | float) -> dict[str, str]:
    return {"currency_code": CURRENCY, "value": f"{Decimal(str(value)).quantize(Decimal('0.01'), ROUND_HALF_UP)}"}


def amount_of(obj: dict[str, Any] | None) -> Decimal:
    return Decimal(str((obj or {}).get("value", "0")))


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def payer_info(c: dict[str, str]) -> dict[str, Any]:
    return {
        "account_id": c["payer_id"],
        "payer_id": c["payer_id"],
        "email_address": c["email"],
        "payer_name": {"given_name": c["given_name"], "surname": c["surname"]},
    }


def line_item(name: str, price: str, quantity: int = 1) -> dict:
    return {"name": name, "quantity": str(quantity), "unit_amount": money(price)}


class Store:
    def __init__(self, db: Database):
        self.db = db

    # ---- ids and time -------------------------------------------------------

    def new_id(self, table: str | None = None, length: int = 17) -> str:
        """PayPal-style ID, e.g. 8MC585209K746392H, checked unique against the table."""
        while True:
            candidate = "".join(secrets.choice(ID_CHARS) for _ in range(length))
            if table is None or self.db.get(table, candidate) is None:
                return candidate

    def invoice_id(self) -> str:
        return "INV2-" + "-".join(self.new_id(length=4) for _ in range(4))

    def take_invoice_number(self) -> str:
        number = self.db.meta("next_invoice_number", 1001)
        self.db.set_meta("next_invoice_number", number + 1)
        return f"INV-{number}"

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)

    # ---- lookups ----------------------------------------------------------------

    def customer(self, payer_id: str) -> dict | None:
        return self.db.get("customers", payer_id)

    def merchant(self) -> dict:
        return self.db.meta("merchant", {})

    # ---- money movements (several tables at once) ---------------------------------

    def record_transaction(self, transaction_id: str, event_code: str, amount: Decimal, when: datetime,
                           payer: dict | None, invoice_id: str | None = None, status: str = "S") -> None:
        """Add a ledger row. Sales (T0006) are positive, refunds (T1107) negative."""
        fee = (abs(amount) * Decimal("0.029") + Decimal("0.30")).quantize(Decimal("0.01")) if amount > 0 else Decimal("0")
        self.db.add_transaction({
            "transaction_info": {
                "transaction_id": transaction_id,
                "transaction_event_code": event_code,
                "transaction_initiation_date": iso(when),
                "transaction_updated_date": iso(when),
                "transaction_amount": money(amount),
                "fee_amount": money(-fee),
                "transaction_status": status,
                **({"invoice_id": invoice_id} if invoice_id else {}),
            },
            "payer_info": payer_info(payer) if payer else {},
        })

    def create_capture(self, amount: Decimal, payer_id: str, invoice_id: str | None = None) -> dict:
        payer = self.customer(payer_id)
        when = self.now()
        capture = self.db.put("captures", {
            "id": self.new_id("captures"),
            "status": "COMPLETED",
            "amount": money(amount),
            "final_capture": True,
            "invoice_id": invoice_id,
            "create_time": iso(when),
            "update_time": iso(when),
            "payer": payer_info(payer) if payer else {},
            "seller_receivable_breakdown": {"gross_amount": money(amount)},
            "refunded_amount": money(0),
        })
        self.record_transaction(capture["id"], "T0006", amount, when, payer, invoice_id)
        return capture

    def create_refund(self, capture: dict, amount: Decimal, note: str | None) -> dict:
        """Refund part or all of a capture: updates the capture, adds a refund and a ledger row."""
        when = self.now()
        refund = self.db.put("refunds", {
            "id": self.new_id("refunds"),
            "status": "COMPLETED",
            "amount": money(amount),
            "capture_id": capture["id"],
            "note_to_payer": note,
            "create_time": iso(when),
            "update_time": iso(when),
        })
        refunded = amount_of(capture["refunded_amount"]) + amount
        capture["refunded_amount"] = money(refunded)
        capture["status"] = "REFUNDED" if refunded >= amount_of(capture["amount"]) else "PARTIALLY_REFUNDED"
        capture["update_time"] = iso(when)
        self.db.put("captures", capture)
        payer = self.customer((capture.get("payer") or {}).get("payer_id", ""))
        self.record_transaction(refund["id"], "T1107", -amount, when, payer, capture.get("invoice_id"))
        if capture["status"] == "REFUNDED":
            self.close_disputes_on(capture["id"], refund["id"], when)
        return refund

    def close_disputes_on(self, capture_id: str, refund_id: str, when) -> None:
        """A payment refunded in full leaves nothing to dispute: like PayPal, its open disputes close
        in the buyer's favour (whichever way the refund was made)."""
        for dispute in self.db.all("disputes"):
            tx = (dispute.get("disputed_transactions") or [{}])[0]
            if tx.get("seller_transaction_id") != capture_id or dispute.get("status") == "RESOLVED":
                continue
            dispute.update(status="RESOLVED", dispute_state="RESOLVED", refund_id=refund_id, update_time=iso(when),
                           dispute_outcome={"outcome_code": "RESOLVED_BUYER_FAVOUR", "amount_refunded": dispute["dispute_amount"]})
            dispute.pop("offer", None)
            self.db.put("disputes", dispute)

    def balance(self) -> Decimal:
        total = Decimal(self.db.meta("starting_balance", "0"))
        for row in self.db.transactions():
            info = row["transaction_info"]
            if info["transaction_status"] == "S":
                total += amount_of(info["transaction_amount"]) + amount_of(info["fee_amount"])
        return total

    # ---- invoices ---------------------------------------------------------------

    def new_invoice(self, items: list[dict], recipients: list[dict], note: str | None = None,
                    invoice_number: str | None = None, due_days: int = 10) -> dict:
        total = sum(Decimal(i["unit_amount"]["value"]) * Decimal(i.get("quantity", "1")) for i in items)
        when = self.now()
        invoice_id = self.invoice_id()
        merchant = self.merchant()
        return self.db.put("invoices", {
            "id": invoice_id,
            "status": "DRAFT",
            "detail": {
                "invoice_number": invoice_number or self.take_invoice_number(),
                "invoice_date": when.date().isoformat(),
                "currency_code": CURRENCY,
                "payment_term": {"due_date": (when.date() + timedelta(days=due_days)).isoformat()},
                **({"note": note} if note else {}),
                "metadata": {"create_time": iso(when), "last_update_time": iso(when)},
            },
            "invoicer": {"business_name": merchant.get("business_name"), "email_address": merchant.get("email_address")},
            "primary_recipients": recipients,
            "items": items,
            "amount": money(total),
            "due_amount": money(total),
            "payments": {"paid_amount": money(0), "transactions": []},
            "links": [{"href": f"/v2/invoicing/invoices/{invoice_id}", "rel": "self", "method": "GET"}],
        })
