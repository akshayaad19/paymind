"""A customer's own purchases from the shop, for "my earphones stopped working": which payment,
what was bought, how it was paid, what's been refunded, and whether a case is already open.

Built from the same PayPal data the shop sees (transaction search, 30 days per call, plus the
dispute list), filtered to the customer's payer_id. No LLM involved.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from ..agent.executor import Executor
from .database import User

DAYS = 90
SAME = {"earphone": "headphone", "earbud": "headphone", "headset": "headphone", "charger": "charger",
        "watch": "watch", "case": "case", "speaker": "speaker", "stand": "stand"}


def paid_how(row: dict) -> str:
    info = row["transaction_info"]
    if info.get("invoice_id"):
        return "invoice"
    return "in store" if row.get("store_info") else "online store"


def payer_label(row: dict) -> dict[str, str]:
    info = row.get("payer_info") or {}
    name = info.get("payer_name") or {}
    return {"name": f"{name.get('given_name', '')} {name.get('surname', '')}".strip(), "email": info.get("email_address", ""),
            "payer_id": info.get("payer_id", "")}


def purchases(executor: Executor, caller: User | None, payer_ok, item: str | None = None, now: datetime | None = None,
              days: int = DAYS) -> list[dict[str, Any]]:
    """Sales whose payer passes payer_ok(payer_info), newest first: payment_id, date, customer, items,
    amount, refunded, left (what can still be refunded or claimed), paid_how, open_case, shipments."""
    now = now or datetime.now(timezone.utc)
    rows, end = [], now
    while end > now - timedelta(days=days):
        start = max(end - timedelta(days=30), now - timedelta(days=days))
        found = executor.execute("list_transactions", {"start_date": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                       "end_date": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "page_size": 500}, caller=caller)
        if found.ok:
            rows += found.body.get("transaction_details", [])
        end = start
    chosen = [r for r in rows if r["transaction_info"]["transaction_event_code"] == "T0006" and payer_ok(r.get("payer_info") or {})]
    disputes = executor.execute("list_disputes", {"page_size": 50}, caller=caller)
    open_cases = {d["disputed_transactions"][0]["seller_transaction_id"]: d["dispute_id"]
                  for d in (disputes.body.get("items", []) if disputes.ok else []) if d.get("status") != "RESOLVED"}
    out = []
    for r in chosen:
        info = r["transaction_info"]
        capture = executor.execute("show_captured_payment_details", {"capture_id": info["transaction_id"]}, caller=caller)
        refunded = Decimal(((capture.body or {}).get("refunded_amount") or {}).get("value", "0")) if capture.ok else Decimal("0")
        amount = Decimal(info["transaction_amount"]["value"])
        items = [f"{i['item_name']} × {i.get('item_quantity', '1')}" for i in (r.get("cart_info") or {}).get("item_details", [])]
        out.append({"payment_id": info["transaction_id"], "date": info["transaction_initiation_date"], "customer": payer_label(r),
                    "items": items, "amount": f"{amount:.2f}", "refunded": f"{refunded:.2f}", "left": f"{amount - refunded:.2f}",
                    "paid_how": paid_how(r), "open_case": open_cases.get(info["transaction_id"]),
                    "shipments": shipments(info["transaction_id"], executor)})
    if item:  # everyday words for the shop's products
        words = [SAME.get(w.rstrip("s"), w.rstrip("s")) for w in item.lower().split()]
        out = [p for p in out if any(w.rstrip("s") in " ".join(p["items"]).lower() for w in words)] or out
    return sorted(out, key=lambda p: p["date"], reverse=True)


def customer_purchases(user: User, executor: Executor, item: str | None = None, now: datetime | None = None,
                       days: int = DAYS) -> list[dict[str, Any]]:
    """A customer's own purchases (matched on their payer_id only)."""
    rows = purchases(executor, user, lambda payer: payer.get("payer_id") == user.payer_id, item, now, days)
    for r in rows:
        r.pop("customer", None)  # it's them
    return rows


MAX_CUSTOMERS = 10


def customer_payments(query: str, executor: Executor, caller: User | None = None, item: str | None = None,
                      now: datetime | None = None, days: int = DAYS) -> dict[str, Any]:
    """Shop side, for "refund Rahul": every customer whose name, email or payer ID matches the query,
    each with their payments. Several matches means the user must be asked which one."""
    words = [w for w in query.lower().replace(",", " ").split() if w]

    def matches(payer: dict) -> bool:
        label = payer_label({"payer_info": payer})
        text = f"{label['name']} {label['email']} {label['payer_id']}".lower()
        return bool(words) and all(w in text for w in words)

    rows = purchases(executor, caller, matches, item, now, days)
    customers: dict[str, dict] = {}
    for r in rows:
        c = customers.setdefault(r["customer"]["payer_id"], {**r.pop("customer"), "payments": []})
        c["payments"].append(r)
    found = list(customers.values())[:MAX_CUSTOMERS]
    note = ("No customer matches; ask for their email." if not found else
            "Several customers match: ask the user which one (show name and email) before doing anything." if len(found) > 1 else
            "One customer. If they have several payments and the user didn't say which, ask (date, items, amount).")
    return {"customers": found, "note": note, "more": len(customers) > MAX_CUSTOMERS}


def shipments(payment_id: str, executor: Executor) -> list[dict[str, Any]]:
    """PayPal trackers for one payment (carrier, tracking number, status), newest first."""
    r = executor.client.get(f"{executor.base_url}/mock/trackers", params={"transaction_id": payment_id})
    if r.status_code >= 300:
        return []
    keep = ("tracking_number", "carrier", "status", "shipment_date", "update_time")
    return [{k: t.get(k) for k in keep} for t in r.json().get("trackers", [])]


def purchase_details(payment_id: str, executor: Executor, user: User | None = None) -> dict[str, Any] | None:
    """What a dispute is about: items, amount, when and how it was paid, refunds, shipments.
    Callers check access to the dispute first."""
    capture = executor.execute("show_captured_payment_details", {"capture_id": payment_id}, caller=user)
    if not capture.ok:
        return None
    when = datetime.fromisoformat(capture.body["create_time"].replace("Z", "+00:00"))
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    found = executor.execute("list_transactions", {"start_date": (when - timedelta(hours=1)).strftime(fmt),
                                                   "end_date": (when + timedelta(hours=1)).strftime(fmt),
                                                   "transaction_id": payment_id}, caller=user)
    row = next(iter(found.body.get("transaction_details", [])), None) if found.ok else None
    items = [f"{i['item_name']} × {i.get('item_quantity', '1')}" for i in ((row or {}).get("cart_info") or {}).get("item_details", [])]
    return {"payment_id": payment_id, "date": capture.body["create_time"], "items": items,
            "customer": payer_label(row) if row else None,
            "amount": capture.body["amount"]["value"], "refunded": (capture.body.get("refunded_amount") or {}).get("value", "0.00"),
            "paid_how": paid_how(row) if row else "online store", "shipments": shipments(payment_id, executor)}
