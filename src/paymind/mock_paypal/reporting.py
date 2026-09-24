"""Transaction search and balances, computed from the store's ledger."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Request

from .errors import PayPalError
from .store import CURRENCY, Store, iso, money

router = APIRouter(prefix="/v1/reporting")

MAX_RANGE = timedelta(days=31)  # PayPal's real limit for one transaction search


def parse_time(value: str, field: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise PayPalError(400, "INVALID_PARAMETER_SYNTAX", f"{field} must be an ISO 8601 date-time, e.g. 2026-08-01T00:00:00Z.", field)


@router.get("/transactions")
def list_transactions(request: Request, start_date: str | None = None, end_date: str | None = None,
                      transaction_id: str | None = None, transaction_type: str | None = None,
                      transaction_status: str | None = None, page_size: int = 100, page: int = 1):
    """Sales are event code T0006 (positive), refunds T1107 (negative)."""
    store: Store = request.app.state.store
    if not start_date or not end_date:
        raise PayPalError(400, "MISSING_REQUIRED_PARAMETER", "start_date and end_date are required.", "start_date" if not start_date else "end_date")
    start, end = parse_time(start_date, "start_date"), parse_time(end_date, "end_date")
    if end < start:
        raise PayPalError(400, "INVALID_PARAMETER_VALUE", "end_date must be after start_date.", "end_date")
    if end - start > MAX_RANGE:
        raise PayPalError(400, "INVALID_REQUEST", "Date range can't be more than 31 days. Split it into smaller ranges.", "end_date")

    rows = []
    for row in store.db.transactions():
        info = row["transaction_info"]
        when = parse_time(info["transaction_initiation_date"], "transaction_initiation_date")
        if not (start <= when <= end):
            continue
        if transaction_id and info["transaction_id"] != transaction_id:
            continue
        if transaction_type and info["transaction_event_code"] != transaction_type:
            continue
        if transaction_status and info["transaction_status"] != transaction_status:
            continue
        rows.append(row)
    rows.sort(key=lambda r: r["transaction_info"]["transaction_initiation_date"])
    page_size = max(1, min(page_size, 500))
    return {
        "transaction_details": rows[(page - 1) * page_size : page * page_size],
        "account_number": "PAYMIND-DEMO",
        "start_date": iso(start),
        "end_date": iso(end),
        "last_refreshed_datetime": iso(store.now()),
        "page": page,
        "total_items": len(rows),
        "total_pages": max(1, -(-len(rows) // page_size)),
    }


@router.get("/balances")
def list_all_balances(request: Request, currency_code: str | None = None, as_of_time: str | None = None):
    store: Store = request.app.state.store
    if currency_code and currency_code != CURRENCY:  # this account only holds USD
        return {"balances": [], "account_id": "PAYMIND-DEMO", "as_of_time": iso(store.now())}
    total = store.balance()
    return {
        "balances": [{
            "currency": CURRENCY,
            "primary": True,
            "total_balance": money(total),
            "available_balance": money(total),
            "withheld_balance": money(0),
        }],
        "account_id": "PAYMIND-DEMO",
        "as_of_time": iso(store.now()),
        "last_refresh_time": iso(store.now()),
    }
