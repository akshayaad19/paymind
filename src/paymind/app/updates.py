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


def whats_new(user: User, executor: Executor, appdb: AppDatabase, now=None, waiting_days: int = WAITING_DAYS) -> list[dict[str, Any]]:
    """Things that need the user's attention on open disputes, most urgent first:

      new_message   the other side wrote and the user hasn't seen it
      needs_reply   the other side wrote, the user has seen it, but hasn't answered
      no_reply_yet  the user wrote `waiting_days`+ days ago and the other side hasn't answered

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
    items = []
    for row in filter_results(user, "list_disputes", listed.body)["items"]:
        if row.get("status") == "RESOLVED":
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
        my_turn_status = "WAITING_FOR_BUYER_RESPONSE" if user.is_customer else "WAITING_FOR_SELLER_RESPONSE"
        if dispute.get("status") == my_turn_status:
            due = dispute.get("seller_response_due_date") if not user.is_customer else dispute.get("buyer_response_due_date")
            base["action_needed"] = True
            base["due_date"] = due
            base["days_left"] = (_parse(due) - now).days if due else None
        if last["posted_by"] == other_side(user):
            new = unread(user, dispute, reads.get(dispute["dispute_id"], 0))
            items.append({**base, "kind": "new_message" if new else "needs_reply", "count": len(new)})
        elif base.get("action_needed"):
            items.append({**base, "kind": "action_needed"})
        elif (now - _parse(last["time_posted"])).days >= waiting_days:
            items.append({**base, "kind": "no_reply_yet", "days": (now - _parse(last["time_posted"])).days})
    order = {"new_message": 0, "action_needed": 1, "needs_reply": 2, "no_reply_yet": 3}
    # overdue / closest deadline first, then by kind, then oldest first
    return sorted(items, key=lambda i: (i.get("days_left") is None, i.get("days_left") or 0, order[i["kind"]], i["time"]))
