"""What's new for a user: dispute messages from the other side they haven't seen.

One rule, shared by the page (the "N new" badge) and the assistant (the
check_updates tool, and reading a thread in chat), so they always agree:

  - a customer's "other side" is the shop (SELLER); the shop's is the customer (BUYER)
  - threads only grow, so everything after the number of messages a user has
    seen (dispute_reads.seen_count) is new
  - resolved disputes never count as having new messages
"""

from __future__ import annotations

from typing import Any

from ..agent.executor import Executor
from ..agent.scope import filter_results
from .database import AppDatabase, User


def other_side(user: User) -> str:
    return "SELLER" if user.is_customer else "BUYER"


def unread(user: User, dispute: dict, seen: int) -> list[dict]:
    """Messages from the other side after the first `seen` messages. Resolved disputes are closed
    cases, so nothing on them is flagged as new."""
    if dispute.get("status") == "RESOLVED":
        return []
    return [m for m in (dispute.get("messages") or [])[seen:] if m.get("posted_by") == other_side(user)]


def mark_seen(appdb: AppDatabase, user: User, dispute: dict) -> None:
    """The user has now seen this dispute's whole thread."""
    appdb.mark_dispute_read(user.user_id, dispute["dispute_id"], len(dispute.get("messages") or []))


def dispute_updates(user: User, executor: Executor, appdb: AppDatabase) -> list[dict[str, Any]]:
    """Every dispute this user may see, with how many unread messages it has (newest dispute first)."""
    listed = executor.execute("list_disputes", {"page_size": 50}, caller=user)
    if not listed.ok:
        raise RuntimeError(listed.error)
    reads = appdb.dispute_reads(user.user_id)
    rows = []
    for item in filter_results(user, "list_disputes", listed.body)["items"]:
        full = executor.execute("show_dispute_details", {"dispute_id": item["dispute_id"]}, caller=user)
        dispute = full.body if full.ok else item
        new = unread(user, dispute, reads.get(item["dispute_id"], 0))
        tx = (dispute.get("disputed_transactions") or [{}])[0]
        rows.append({
            "dispute_id": item["dispute_id"],
            "status": dispute.get("status"),
            "amount": dispute.get("dispute_amount"),
            "with": ((tx.get("seller") if user.is_customer else tx.get("buyer")) or {}).get("name"),
            "message_count": len(dispute.get("messages") or []),
            "unread": len(new),
            "new_messages": [{"time": m["time_posted"], "text": m["content"]} for m in new],
        })
    return rows


WAITING_DAYS = 2  # after this long without an answer, suggest a reminder


def _parse(ts: str):
    from datetime import datetime

    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def open_refund_request(dispute: dict, requests: dict[str, dict]) -> dict | None:
    """The customer's refund request while it's the shop's turn. Refunding resolves the dispute and an
    offer hands the turn to the customer, so either one clears it (if the customer declines the offer,
    the turn and the request come back)."""
    request = requests.get(dispute.get("dispute_id"))
    if not request or dispute.get("status") not in ("WAITING_FOR_SELLER_RESPONSE", "OPEN"):
        return None
    return {k: request.get(k, "refund") if k == "wants" else request[k] for k in ("amount", "wants", "message", "time")}


def whats_new(user: User, executor: Executor, appdb: AppDatabase, now=None, waiting_days: int = WAITING_DAYS) -> list[dict[str, Any]]:
    """Things that need the user's attention on open disputes, most urgent first:

      invoice_overdue / invoice_due  unpaid invoices: past their due date (both sides; the shop sees
                    which customer owes), or due within INVOICE_SOON_DAYS days (customer)
      case_closed   the OTHER side closed a case in the last CLOSED_DAYS days (refund, offer accepted, replacement,
                    or the customer marked it resolved); shown until this user opens the case
      refunded      (customer) the shop refunded one of their payments in the last CLOSED_DAYS days
      refund_requested  (shop) the customer formally asked for a refund of the disputed amount; stays until the shop
                    refunds or makes an offer
      new_message   the other side wrote and the user hasn't seen it
      needs_reply   the other side wrote, the user has seen it, but hasn't answered
      no_reply_yet  the user wrote `waiting_days`+ days ago and the other side hasn't answered

      new_po / po_send_invoice / po_ship / po_not_received   (shop) review a PO, send its invoice,
                    ship a paid order (with its delivery date), follow up a "not received" report
      po_accepted / po_rejected / po_pay / po_shipped / po_confirm   (customer) the shop's decision,
                    an invoice to pay, a shipment with tracking, "has it arrived?" once the date comes
      action_needed PayPal says it's this user's turn to act (the shop must respond, or the customer
                    must answer an offer). Shown on every login until the dispute's status changes;
                    a message alone doesn't clear it. Carries PayPal's response deadline.

    Every item also says whether it's the user's turn (action_needed) and, if so, the deadline.
    Plain code over the same data as the badge; no LLM involved.
    """
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    listed = executor.execute("list_disputes", {"page_size": 50}, caller=user)
    if not listed.ok:
        raise RuntimeError(listed.error)
    reads = appdb.dispute_reads(user.user_id)
    read_at = appdb.dispute_read_times(user.user_id)
    requests = appdb.refund_requests()
    items = []
    for row in filter_results(user, "list_disputes", listed.body)["items"]:
        if row.get("status") == "RESOLVED":
            closed = closed_by_other_side(user, row, executor, read_at.get(row["dispute_id"]), now)
            if closed:
                items.append(closed)
            continue
        full = executor.execute("show_dispute_details", {"dispute_id": row["dispute_id"]}, caller=user)
        if not full.ok:
            continue
        dispute = full.body
        messages = sorted(dispute.get("messages") or [], key=lambda m: m["time_posted"])
        last = messages[-1] if messages else {"posted_by": "", "time_posted": dispute.get("create_time", ""), "content": ""}
        tx = (dispute.get("disputed_transactions") or [{}])[0]
        other_name = ((tx.get("seller") if user.is_customer else tx.get("buyer")) or {}).get("name") or "the other side"
        base = {
            "dispute_id": dispute["dispute_id"], "reason": dispute.get("reason"), "amount": dispute.get("dispute_amount"),
            "with": other_name, "time": last["time_posted"], "text": last["content"],
        }
        if dispute.get("offer"):  # the shop's offer waiting for the customer's answer
            base["offer"] = {"amount": dispute["offer"].get("offer_amount"), "type": dispute["offer"].get("offer_type"),
                             "note": dispute["offer"].get("note")}
        my_turn_status = "WAITING_FOR_BUYER_RESPONSE" if user.is_customer else "WAITING_FOR_SELLER_RESPONSE"
        if dispute.get("status") == my_turn_status:
            due = dispute.get("seller_response_due_date") if not user.is_customer else dispute.get("buyer_response_due_date")
            base["action_needed"] = True
            base["due_date"] = due
            base["days_left"] = (_parse(due) - now).days if due else None
        request = open_refund_request(dispute, requests)
        if request:
            base["refund_request"] = request
        if request and not user.is_customer:  # a clear to-do for the shop, above a plain "new message"
            new = unread(user, dispute, reads.get(dispute["dispute_id"], 0))
            items.append({**base, "kind": "refund_requested", "count": len(new), "time": request["time"],
                          "text": request["message"]})
        elif last["posted_by"] == other_side(user):
            new = unread(user, dispute, reads.get(dispute["dispute_id"], 0))
            items.append({**base, "kind": "new_message" if new else "needs_reply", "count": len(new)})
        elif base.get("action_needed"):
            items.append({**base, "kind": "action_needed"})
        elif (now - _parse(last["time_posted"])).days >= waiting_days:
            items.append({**base, "kind": "no_reply_yet", "days": (now - _parse(last["time_posted"])).days})
    items += po_updates(user, appdb, now, executor)
    items += invoice_updates(user, executor, now)
    if user.is_customer:
        items += refund_updates(user, executor, now)
    order = {"case_closed": 0, "refunded": 1, "invoice_overdue": 1, "invoice_due": 2, "refund_requested": 0, "new_message": 0, "action_needed": 1, "po_not_received": 1, "po_confirm": 1, "po_ship": 1, "new_po": 1,
             "po_pay": 2, "po_send_invoice": 2, "needs_reply": 2, "po_shipped": 3, "po_accepted": 3, "po_rejected": 3, "no_reply_yet": 4}
    # overdue / closest deadline first, then by kind, then oldest first
    return sorted(items, key=lambda i: (i.get("days_left") is None, i.get("days_left") or 0, order[i["kind"]], i["time"]))


CLOSED_DAYS = 14  # how long a closed case or a refund stays in What's new

OUTCOMES = {  # outcome_code -> how it ended, in words (the amount is added where there is one)
    "RESOLVED_BUYER_FAVOUR": "refunded",
    "RESOLVED_WITH_PAYOUT": "settled with the offer",
    "RESOLVED_WITH_REPLACEMENT": "replacement sent",
    "CANCELED_BY_BUYER": "marked resolved by the customer",
    "RESOLVED_SELLER_FAVOUR": "closed in the shop's favour",
}


def closed_by_other_side(user: User, row: dict, executor: Executor, last_read: str | None, now) -> dict | None:
    """A case the other side closed recently that this user hasn't opened since."""
    closed_at = row.get("update_time") or ""
    if not closed_at or (now - _parse(closed_at)).days > CLOSED_DAYS or (last_read and last_read >= closed_at):
        return None
    full = executor.execute("show_dispute_details", {"dispute_id": row["dispute_id"]}, caller=user)
    outcome = (full.body or {}).get("dispute_outcome") or {} if full.ok else {}
    mine = "BUYER" if user.is_customer else "SELLER"
    if not outcome.get("closed_by") or outcome["closed_by"] == mine:
        return None
    tx = (row.get("disputed_transactions") or [{}])[0]
    other = ((tx.get("seller") if user.is_customer else tx.get("buyer")) or {}).get("name") or "the other side"
    return {"kind": "case_closed", "dispute_id": row["dispute_id"], "reason": row.get("reason"), "amount": row.get("dispute_amount"),
            "with": other, "time": closed_at, "text": "", "outcome": OUTCOMES.get(outcome.get("outcome_code"), "closed"),
            "amount_refunded": outcome.get("amount_refunded"), "carrier": outcome.get("carrier"),
            "tracking_number": outcome.get("tracking_number")}


def refund_updates(user: User, executor: Executor, now) -> list[dict[str, Any]]:
    """Refunds the shop made on this customer's payments in the last CLOSED_DAYS days."""
    from datetime import timedelta

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    found = executor.execute("list_transactions", {"start_date": (now - timedelta(days=CLOSED_DAYS)).strftime(fmt),
                                                   "end_date": now.strftime(fmt), "transaction_type": "T1107"}, caller=user)
    if not found.ok:
        return []
    items = []
    for r in found.body.get("transaction_details", []):
        if (r.get("payer_info") or {}).get("payer_id") != user.payer_id:
            continue
        info = r["transaction_info"]
        refund = executor.execute("show_refund_details", {"refund_id": info["transaction_id"]}, caller=user)
        note = (refund.body or {}).get("note_to_payer") if refund.ok else None
        items.append({"kind": "refunded", "refund_id": info["transaction_id"], "time": info["transaction_initiation_date"],
                      "amount": {"currency_code": "USD", "value": info["transaction_amount"]["value"].lstrip("-")},
                      "with": "PayMind Demo Store", "text": note or ""})
    return items


INVOICE_SOON_DAYS = 3  # a customer is reminded this many days before an invoice is due
UNPAID = ("SENT", "UNPAID", "PARTIALLY_PAID", "SCHEDULED")


def shop_name(inv: dict, user: User, executor: Executor) -> str:
    """The business that sent the invoice (search results leave it out, the details have it)."""
    invoicer = inv.get("invoicer") or {}
    if not invoicer:
        details = executor.execute("show_invoice_details", {"invoice_id": inv["id"]}, caller=user)
        invoicer = (details.body or {}).get("invoicer") or {} if details.ok else {}
    return invoicer.get("business_name") or "the shop"


def invoice_updates(user: User, executor: Executor, now) -> list[dict[str, Any]]:
    """Unpaid invoices that need attention: overdue (both sides), or due soon (customer)."""
    if user.is_customer:
        found = executor.execute("search_for_invoices", {"recipient_email": user.email}, caller=user)
    else:
        found = executor.execute("list_invoices", {"page_size": 100}, caller=user)
    if not found.ok:
        return []
    items = []
    for inv in found.body.get("items", []):
        due = ((inv.get("detail") or {}).get("payment_term") or {}).get("due_date")
        if inv.get("status") not in UNPAID or not due:
            continue
        days_left = (_parse(due + "T23:59:59Z") - now).days
        if days_left >= 0 and (not user.is_customer or days_left > INVOICE_SOON_DAYS):
            continue
        billing = ((inv.get("primary_recipients") or [{}])[0].get("billing_info") or {})
        name = " ".join(filter(None, [(billing.get("name") or {}).get("given_name"), (billing.get("name") or {}).get("surname")]))
        items.append({
            "kind": "invoice_overdue" if days_left < 0 else "invoice_due", "invoice_id": inv["id"],
            "invoice_number": (inv.get("detail") or {}).get("invoice_number"), "amount": inv.get("due_amount") or inv.get("amount"),
            "due_date": due, "days_left": days_left,
            "with": shop_name(inv, user, executor) if user.is_customer else name or billing.get("email_address") or "a customer",
            "time": due, "text": "",
        })
    return items


PO_RECENT_DAYS = 14  # how long a customer keeps seeing "your PO was accepted / rejected"


def sync_po_payment(po: dict, executor: Executor, appdb: AppDatabase) -> dict:
    """Move a PO forward when its PayPal invoice changes: sent → invoiced, paid → paid (processing)."""
    if po["status"] not in ("accepted", "invoiced") or not po.get("invoice_id"):
        return po
    result = executor.execute("show_invoice_details", {"invoice_id": po["invoice_id"]})
    if not result.ok:
        return po
    status = result.body.get("status")
    if status in ("PAID", "MARKED_AS_PAID"):
        paid = ((result.body.get("payments") or {}).get("transactions") or [{}])[-1].get("payment_date")
        return appdb.update_po(po["po_id"], status="paid", paid_at=paid or result.body["detail"]["metadata"]["last_update_time"])
    if status in ("SENT", "UNPAID", "PARTIALLY_PAID") and po["status"] == "accepted":
        return appdb.update_po(po["po_id"], status="invoiced")
    return po


def _days_left(date_str: str | None, now) -> int | None:
    from datetime import date

    return (date.fromisoformat(date_str) - now.date()).days if date_str else None


def po_updates(user: User, appdb: AppDatabase, now, executor: Executor | None = None) -> list[dict[str, Any]]:
    """Purchase orders that need attention, for the shop or for the customer (see whats_new)."""
    from datetime import timedelta

    pos = appdb.list_pos(statuses=("submitted", "accepted", "invoiced", "paid", "not_received")) if not user.is_customer \
        else appdb.list_pos(user_id=user.user_id)
    if executor:
        pos = [sync_po_payment(p, executor, appdb) for p in pos]
    since = (now - timedelta(days=PO_RECENT_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    items = []
    for po in pos:
        owner = appdb.get_user(po["user_id"])
        base = {"po_id": po["po_id"], "time": po["updated_at"], "customer_po_ref": po.get("customer_po_ref"),
                "expected_date": po.get("expected_date"), "days_left": _days_left(po.get("expected_date"), now),
                "invoice_id": po.get("invoice_id"), "with": (owner.name if owner else "a customer") if not user.is_customer else "the shop",
                "text": ", ".join(f'{i["quantity"]} × {i["name"]}' for i in po["items"][:3])}
        s = po["status"]
        if not user.is_customer:
            kind = {"submitted": "new_po", "accepted": "po_send_invoice", "paid": "po_ship", "not_received": "po_not_received"}.get(s)
            if kind:
                items.append({**base, "kind": kind, "items": len(po["items"]), "requested_date": po.get("requested_date"),
                              **({"text": po.get("not_received_note") or "No details given."} if kind == "po_not_received" else {})})
        else:
            if s in ("accepted", "rejected") and po["updated_at"] >= since:
                items.append({**base, "kind": f"po_{s}", **({"text": po.get("reject_reason") or ""} if s == "rejected" else {})})
            elif s == "invoiced":
                items.append({**base, "kind": "po_pay"})
            elif s == "paid" and base["days_left"] is not None and base["days_left"] < 0:
                items.append({**base, "kind": "po_confirm"})       # due but not shipped: did it come anyway?
            elif s == "shipped":
                arrived_by_now = base["days_left"] is not None and base["days_left"] <= 0
                items.append({**base, "kind": "po_confirm" if arrived_by_now else "po_shipped",
                              "carrier": po.get("carrier"), "tracking_number": po.get("tracking_number")})
    return items
