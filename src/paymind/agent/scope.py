"""Customer data scope: a customer may only see and act on their own records.

Accountants see everything, so nothing here applies to them.

Two checks, both using real PayPal data through the executor:
  before  a call that names a record (dispute_id, invoice_id, capture_id, order_id,
          refund_id), look the record up and check it belongs to the customer.
          Only read calls are made here, so running it twice is harmless.
  after   list results are filtered down to the customer's own records.

Ownership: disputes, captures, orders → payer_id; invoices → recipient email;
refunds → the payer of the refunded capture.
"""

from __future__ import annotations

from typing import Any

from ..app.database import User
from .executor import Executor


def _payer(record: dict | None) -> str | None:
    return ((record or {}).get("payer") or {}).get("payer_id")


def _dispute_buyer(dispute: dict | None) -> str | None:
    txs = (dispute or {}).get("disputed_transactions") or [{}]
    return (txs[0].get("buyer") or {}).get("payer_id")


def _invoice_email(invoice: dict | None) -> str | None:
    recipients = (invoice or {}).get("primary_recipients") or [{}]
    return ((recipients[0].get("billing_info") or {}).get("email_address") or "").lower() or None


def _owns(user: User, param: str, value: str, executor: Executor) -> bool | None:
    """True/False if ownership could be checked, None if the record couldn't be fetched."""
    lookups = {
        "dispute_id": ("show_dispute_details", lambda r: _dispute_buyer(r) == user.payer_id),
        "capture_id": ("show_captured_payment_details", lambda r: _payer(r) == user.payer_id),
        "order_id": ("show_order_details", lambda r: _payer(r) == user.payer_id),
        "invoice_id": ("show_invoice_details", lambda r: _invoice_email(r) == user.email.lower()),
    }
    if param == "refund_id":
        refund = executor.execute("show_refund_details", {"refund_id": value}, caller=user)
        if not refund.ok:
            return None
        return _owns(user, "capture_id", refund.body.get("capture_id", ""), executor)
    if param not in lookups:
        return None
    tool, check = lookups[param]
    result = executor.execute(tool, {param: value}, caller=user)
    return check(result.body) if result.ok else None


def check_access(user: User, params: dict[str, Any], executor: Executor) -> str | None:
    """Return a reason if the customer may not touch a record named in params, else None."""
    if not user.is_customer:
        return None
    for param in ("dispute_id", "invoice_id", "capture_id", "order_id", "refund_id"):
        if params.get(param) and _owns(user, param, str(params[param]), executor) is False:
            return f"{param} {params[param]} does not belong to this customer"
    return None


def filter_results(user: User, tool: str, body: Any) -> Any:
    """Keep only the customer's own records in list results."""
    if not user.is_customer or not isinstance(body, dict) or not isinstance(body.get("items"), list):
        return body
    keep = None
    if tool == "list_disputes":
        keep = lambda item: _dispute_buyer(item) == user.payer_id  # noqa: E731
    elif tool in ("list_invoices", "search_for_invoices"):
        keep = lambda item: _invoice_email(item) == user.email.lower()  # noqa: E731
    if keep is None:
        return body
    items = [i for i in body["items"] if keep(i)]
    return {**body, "items": items, **({"total_items": len(items)} if "total_items" in body else {})}
