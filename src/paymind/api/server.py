"""PayMind API + web app.

Every /api route except login needs a valid JWT (see auth.py). Roles are
checked on the server for every call; the web page only decides what to show.

  POST /api/auth/login          email + password → token
  POST /api/auth/logout         end the session: this token and every older one stop working
  GET  /api/me                  who am I
  POST /api/chat                send a message to the agent
  POST /api/chat/confirm        answer a pending yes/no
  POST /api/chat/stream, /api/chat/confirm/stream   the same, streamed (Server-Sent Events): the answer's
                                words as Gemini writes them, then a final "done" event with the full result
  GET  /api/paypal/overview     account data: everything for accountants, own records for customers
  GET  /api/paypal/transactions accountant only: transactions for a day / week / month, with totals
  GET  /api/invoices/{id}/pdf   download an invoice as PDF (own invoices only for customers)
  POST /api/invoices/{id}/send  accountants: send a draft invoice to its recipient
  POST /api/pos/read            customers: upload a PO (photo/scan, often handwritten); the AI reads it into a draft
  POST /api/pos, PUT /api/pos/{id}, POST /api/pos/{id}/submit   customers: type / edit / send a PO
  GET  /api/pos, /api/pos/{id}, /api/pos/{id}/document         own POs (customers) or all sent POs (accountants)
  POST /api/pos/{id}/accept     accountants: prices + delivery date → draft PayPal invoice; /reject with a reason
  POST /api/invoices/{id}/pay   customers pay their invoice (mock: simulated PayPal checkout) → PO paid
  POST /api/pos/{id}/ship       accountants: carrier + tracking number → shipped
  POST /api/pos/{id}/delivered | /not-received   customers confirm delivery or report it missing
  GET  /api/disputes/{id}       a dispute and its message thread (own disputes only for customers)
  POST /api/disputes/{id}/messages  send a message to the other side of the dispute
  POST /api/disputes/{id}/resolve   accountants: refund in full, or offer a partial refund
  GET  /api/whats-new           new messages / replies owed / no reply yet, on open disputes
  GET  /api/audit               the caller's own audit log

Run (mock PayPal must be running on port 8000):
    uvicorn paymind.api.server:create_app --factory --port 8001
    open http://localhost:8001
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.executor import Executor, ToolRegistry
from ..agent.scope import check_access, filter_results
from ..app.database import AppDatabase, User
from ..app.updates import dispute_updates, mark_seen, open_refund_request, sync_po_payment, whats_new
from .auth import TOKEN_TTL, create_token, current_user, require_role, secret_key

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / "static"
UPLOADS = ROOT / "data/app/uploads"
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


@dataclass
class Services:
    appdb: AppDatabase
    executor: Executor
    registry: ToolRegistry
    search: Callable[..., list[tuple[str, str]]]
    agent_factory: Callable[[], Any]          # built on first chat (loads models, connects to Gemini)
    mock_base_url: str = "http://localhost:8000"
    po_reader: Callable[[bytes, str], Any] | None = None   # reads uploaded purchase orders (Gemini)
    uploads: Path = UPLOADS
    _agent: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def agent(self):
        with self._lock:
            if self._agent is None:
                self._agent = self.agent_factory()
            return self._agent


def default_services() -> Services:
    from dotenv import load_dotenv

    from ..agent.factory import build_agent, qdrant_search

    load_dotenv(ROOT / ".env")
    registry = ToolRegistry()
    executor = Executor(registry)
    from ..agent.factory import agent_models
    from ..app.po_reader import gemini_reader

    return Services(appdb=AppDatabase(), executor=executor, registry=registry,
                    search=qdrant_search(registry), agent_factory=build_agent, mock_base_url=executor.base_url,
                    po_reader=gemini_reader(agent_models()))


# ---- request bodies ------------------------------------------------------------------

class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class MessageIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class POItem(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    quantity: int = Field(ge=1, le=100000)
    unit_price: str | None = None


class POIn(BaseModel):
    items: list[POItem] = Field(default_factory=list, max_length=100)
    customer_po_ref: str | None = Field(default=None, max_length=60)
    requested_date: str | None = None
    notes: str | None = Field(default=None, max_length=1000)


class POAccept(BaseModel):
    items: list[POItem] = Field(min_length=1, max_length=100)
    expected_date: str
    note: str | None = Field(default=None, max_length=500)


class POShip(BaseModel):
    carrier: str = Field(min_length=2, max_length=60)
    tracking_number: str = Field(min_length=3, max_length=60)


class PONotReceived(BaseModel):
    note: str | None = Field(default=None, max_length=500)


class POReject(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class ResolveIn(BaseModel):
    action: str = Field(pattern="^(refund|offer)$")
    amount: str | None = None                                  # offer: the partial refund offered
    note: str | None = Field(default=None, max_length=1000)


class ConfirmIn(BaseModel):
    session_id: str
    approve: bool


def public(user: User) -> dict:
    return {"user_id": user.user_id, "name": user.name, "email": user.email, "role": user.role, "payer_id": user.payer_id}


def thread_for(user: User, session_id: str) -> str:
    """Chat threads are namespaced by user, so nobody can open another user's conversation."""
    if not SESSION_ID.match(session_id):
        raise HTTPException(400, "Invalid session id.")
    return f"{user.user_id}__{session_id}"


def create_app(services: Services | None = None, jwt_secret: str | None = None) -> FastAPI:
    services = services or default_services()
    app = FastAPI(title="PayMind API", description="Chat with your PayPal business account. Log in to get a token.")
    app.state.services = services
    app.state.jwt_secret = jwt_secret or secret_key()

    # ---- auth ------------------------------------------------------------------------

    @app.post("/api/auth/login", tags=["auth"])
    def login(body: LoginIn):
        user = services.appdb.authenticate(body.email, body.password)
        if user is None:
            raise HTTPException(401, "Wrong email or password.")
        _, version = services.appdb.user_for_token(user.user_id)
        return {"access_token": create_token(user, app.state.jwt_secret, version=version), "token_type": "bearer",
                "expires_in": int(TOKEN_TTL.total_seconds()), "user": public(user)}

    @app.post("/api/auth/logout", tags=["auth"])
    def logout(user: User = Depends(current_user)):
        services.appdb.revoke_tokens(user.user_id)
        return {"ok": True}

    @app.get("/api/me", tags=["auth"])
    def me(user: User = Depends(current_user)):
        return public(user)

    # ---- chat ------------------------------------------------------------------------

    def agent_reply(session_id: str, reply) -> dict:
        # Which tools ran and why is developer information: it's in the LangSmith trace, not here.
        return {"session_id": session_id, "reply": reply.text, "confirmation": reply.confirmation,
                "receipt": getattr(reply, "receipt", None)}  # built by code from what actually ran

    @app.post("/api/chat", tags=["chat"])
    def chat(body: ChatIn, user: User = Depends(current_user)):
        session_id = body.session_id or uuid.uuid4().hex[:12]
        reply = services.agent().send(user.user_id, thread_for(user, session_id), body.message.strip())
        return agent_reply(session_id, reply)

    @app.post("/api/chat/confirm", tags=["chat"])
    def confirm(body: ConfirmIn, user: User = Depends(current_user)):
        reply = services.agent().answer(user.user_id, thread_for(user, body.session_id), body.approve)
        return agent_reply(body.session_id, reply)

    def event_stream(events, session_id: str):
        """Server-Sent Events: one `data: {json}` line per event. Errors end the stream with an error event."""
        try:
            for event in events:
                if event["type"] == "done":
                    event = {**event, "session_id": session_id}
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # the page shows it like any failed request
            yield f"data: {json.dumps({'type': 'error', 'detail': f'Something went wrong: {type(exc).__name__}'})}\n\n"

    def sse(generator) -> StreamingResponse:
        return StreamingResponse(generator, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/chat/stream", tags=["chat"])
    def chat_stream(body: ChatIn, user: User = Depends(current_user)):
        session_id = body.session_id or uuid.uuid4().hex[:12]
        events = services.agent().stream(user.user_id, thread_for(user, session_id), text=body.message.strip())
        return sse(event_stream(events, session_id))

    @app.post("/api/chat/confirm/stream", tags=["chat"])
    def confirm_stream(body: ConfirmIn, user: User = Depends(current_user)):
        events = services.agent().stream(user.user_id, thread_for(user, body.session_id), approve=body.approve)
        return sse(event_stream(events, body.session_id))

    # ---- PayPal account data (through the same executor the agent uses) ------------------------

    def call(tool: str, params: dict, user: User) -> Any:
        result = services.executor.execute(tool, params, caller=user)
        if not result.ok:
            raise HTTPException(502, f"PayPal: {result.error}")
        return result.body

    @app.get("/api/paypal/overview", tags=["paypal"])
    def overview(user: User = Depends(current_user)):
        disputes = filter_results(user, "list_disputes", call("list_disputes", {"page_size": 50}, user))["items"]
        counts = {u["dispute_id"]: u for u in dispute_updates(user, services.executor, services.appdb)}
        requests = services.appdb.refund_requests()
        for d in disputes:  # same "new" rule as the assistant's check_updates tool
            d["unread"] = counts.get(d["dispute_id"], {}).get("unread", 0)
            d["refund_request"] = open_refund_request(d, requests)  # "Refund requested" until refunded or an offer is made
            d["message_count"] = counts.get(d["dispute_id"], {}).get("message_count", 0)
        if user.is_customer:
            invoices = call("search_for_invoices", {"recipient_email": user.email}, user)["items"]
            return {"role": "customer", "disputes": disputes, "invoices": invoices}
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        tx = call("list_transactions", {"start_date": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        "end_date": end.strftime("%Y-%m-%dT%H:%M:%SZ")}, user)["transaction_details"]
        balance = call("list_all_balances", {}, user)["balances"][0]["total_balance"]
        invoices = call("list_invoices", {"page_size": 100}, user)["items"]
        return {"role": "accountant", "balance": balance, "disputes": disputes, "invoices": invoices,
                "transactions": list(reversed(tx))}

    @app.get("/api/paypal/transactions", tags=["paypal"])
    def transactions(start: datetime, end: datetime, user: User = Depends(require_role("accountant"))):
        """Transactions between start and end (the page sends the user's local day / week / month),
        with totals. PayPal allows at most 31 days per search."""
        if start.tzinfo is None or end.tzinfo is None:
            raise HTTPException(400, "start and end need a time zone, e.g. 2026-09-01T00:00:00+05:30.")
        if end <= start or end - start > timedelta(days=31, seconds=1):
            raise HTTPException(400, "Pick a period of up to 31 days.")
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        rows = call("list_transactions", {"start_date": start.astimezone(timezone.utc).strftime(fmt),
                                          "end_date": end.astimezone(timezone.utc).strftime(fmt), "page_size": 500},
                    user)["transaction_details"]
        sales = sum(float(r["transaction_info"]["transaction_amount"]["value"]) for r in rows
                    if r["transaction_info"]["transaction_event_code"] == "T0006")
        refunds = sum(float(r["transaction_info"]["transaction_amount"]["value"]) for r in rows
                      if r["transaction_info"]["transaction_event_code"] == "T1107")
        fees = sum(float(r["transaction_info"]["fee_amount"]["value"]) for r in rows)
        return {"start": start.isoformat(), "end": end.isoformat(), "transactions": list(reversed(rows)),
                "totals": {"count": len(rows), "sales": round(sales, 2), "refunds": round(refunds, 2),
                           "fees": round(fees, 2), "net": round(sales + refunds + fees, 2)}}

    # ---- invoices: download and send -------------------------------------------------------------------

    def own_invoice(user: User, invoice_id: str) -> dict:
        """The invoice, if this user may see it. Other customers' invoices look like they don't exist."""
        if check_access(user, {"invoice_id": invoice_id}, services.executor):
            raise HTTPException(404, "Invoice not found.")
        result = services.executor.execute("show_invoice_details", {"invoice_id": invoice_id}, caller=user)
        if result.status_code == 404:
            raise HTTPException(404, "Invoice not found.")
        if not result.ok:
            raise HTTPException(502, f"PayPal: {result.error}")
        return result.body

    @app.get("/api/invoices/{invoice_id}/pdf", tags=["invoices"])
    def download_invoice(invoice_id: str, user: User = Depends(current_user)):
        from .invoice_pdf import invoice_pdf

        invoice = own_invoice(user, invoice_id)
        number = re.sub(r"[^A-Za-z0-9_-]", "", (invoice.get("detail") or {}).get("invoice_number") or invoice_id)
        return Response(invoice_pdf(invoice), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{number or "invoice"}.pdf"'})

    @app.post("/api/invoices/{invoice_id}/send", tags=["invoices"])
    def send_invoice(invoice_id: str, user: User = Depends(require_role("accountant"))):
        """Accountants send a draft to its recipient. The page asks for a second click to confirm."""
        own_invoice(user, invoice_id)
        params = {"invoice_id": invoice_id}
        card = services.registry.get("send_invoice") or {}
        result = services.executor.execute("send_invoice", params, caller=user)
        services.appdb.log_action(user, "send_invoice", params, "success" if result.ok else "failed",
                                  method=card.get("method"), path=card.get("path"), http_status=result.status_code,
                                  result_summary=(result.error or "sent from the invoices page")[:300],
                                  confirmed=True, request_id=result.request_id)
        if not result.ok:
            raise HTTPException(422 if result.status_code == 422 else 502, result.error)
        for po in services.appdb.list_pos(statuses=("accepted",)):
            if po.get("invoice_id") == invoice_id:
                sync_po_payment(po, services.executor, services.appdb)
        return own_invoice(user, invoice_id)

    # ---- purchase orders ----------------------------------------------------------------------------------
    # Customers send POs (typed, or read from an uploaded, often handwritten, document) and always review
    # what was read before submitting. The shop accepts (sets prices + delivery date, which creates a draft
    # PayPal invoice) or rejects with a reason.

    def po_or_404(user: User, po_id: str) -> dict:
        po = services.appdb.get_po(po_id)
        if po:
            po = sync_po_payment(po, services.executor, services.appdb)  # paid in PayPal → processing
        if not po or (user.is_customer and po["user_id"] != user.user_id) or (not user.is_customer and po["status"] == "draft"):
            raise HTTPException(404, "Purchase order not found.")  # others' POs, and customers' unsent drafts, look absent
        return po

    def po_view(po: dict) -> dict:
        owner = services.appdb.get_user(po["user_id"])
        items = po["items"]
        priced = all(i.get("unit_price") for i in items)
        total = sum(Decimal(i["unit_price"]) * i["quantity"] for i in items) if items and priced else None
        return {**{k: v for k, v in po.items() if k != "document_path"}, "has_document": bool(po.get("document_path")),
                "customer": {"name": owner.name, "email": owner.email} if owner else None,
                "total": f"{total:.2f}" if total is not None else None}

    def po_fields(body: POIn) -> dict:
        from ..app.po_reader import clean_date, clean_items

        return {"items": clean_items([i.model_dump() for i in body.items]), "customer_po_ref": (body.customer_po_ref or "").strip() or None,
                "requested_date": clean_date(body.requested_date), "notes": (body.notes or "").strip() or None}

    @app.post("/api/pos/read", tags=["purchase orders"])
    async def read_po_document(file: UploadFile = File(...), user: User = Depends(require_role("customer"))):
        """Upload a PO document (JPG, PNG or PDF, up to 10 MB). Returns a DRAFT for the customer to check."""
        from ..app.po_reader import ALLOWED_TYPES, MAX_BYTES, clean_date, clean_items

        data = await file.read()
        mime = (file.content_type or "").lower()
        if mime not in ALLOWED_TYPES:
            raise HTTPException(415, "Upload a JPG, PNG or PDF.")
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "The file is larger than 10 MB.")
        if not data:
            raise HTTPException(400, "The file is empty.")
        services.uploads.mkdir(parents=True, exist_ok=True)
        path = services.uploads / f"{uuid.uuid4().hex}{ALLOWED_TYPES[mime]}"
        path.write_bytes(data)
        read_ok, unclear, extracted = True, [], None
        try:
            result = services.po_reader(data, mime)
            extracted, unclear = result.model_dump(), result.unclear
        except Exception:  # AI busy / out of quota / unreadable: the customer fills the form by hand
            read_ok = False
        po = services.appdb.create_po(
            user.user_id, clean_items((extracted or {}).get("items") or []),
            customer_po_ref=(extracted or {}).get("customer_po_ref"),
            requested_date=clean_date((extracted or {}).get("requested_delivery_date")),
            notes=(extracted or {}).get("notes"), document_path=str(path), document_type=mime, extracted=extracted)
        return {"po": po_view(po), "read_ok": read_ok, "unclear": unclear}

    @app.post("/api/pos", tags=["purchase orders"])
    def create_po(body: POIn, user: User = Depends(require_role("customer"))):
        """Type a PO in by hand (also works when the AI is unavailable). Starts as a draft."""
        f = po_fields(body)
        return po_view(services.appdb.create_po(user.user_id, f.pop("items"), **f))

    @app.put("/api/pos/{po_id}", tags=["purchase orders"])
    def edit_po(po_id: str, body: POIn, user: User = Depends(require_role("customer"))):
        po = po_or_404(user, po_id)
        if po["status"] != "draft":
            raise HTTPException(409, "Only drafts can be edited.")
        return po_view(services.appdb.update_po(po_id, **po_fields(body)))

    @app.post("/api/pos/{po_id}/submit", tags=["purchase orders"])
    def submit_po(po_id: str, user: User = Depends(require_role("customer"))):
        po = po_or_404(user, po_id)
        if po["status"] != "draft":
            raise HTTPException(409, "This PO was already sent.")
        if not po["items"]:
            raise HTTPException(422, "Add at least one item before sending.")
        services.appdb.log_action(user, "submit_purchase_order", {"po_id": po_id, "items": len(po["items"])}, "success",
                                  result_summary=f"{po_id} sent to the shop", confirmed=True)
        return po_view(services.appdb.update_po(po_id, status="submitted"))

    @app.get("/api/pos", tags=["purchase orders"])
    def list_pos(user: User = Depends(current_user)):
        sync = lambda p: sync_po_payment(p, services.executor, services.appdb)  # noqa: E731
        if user.is_customer:
            return {"items": [po_view(sync(p)) for p in services.appdb.list_pos(user_id=user.user_id)]}
        return {"items": [po_view(sync(p)) for p in services.appdb.list_pos(
            statuses=("submitted", "accepted", "invoiced", "paid", "shipped", "not_received", "delivered", "rejected"))]}

    @app.get("/api/pos/{po_id}", tags=["purchase orders"])
    def get_po(po_id: str, user: User = Depends(current_user)):
        return po_view(po_or_404(user, po_id))

    @app.get("/api/pos/{po_id}/document", tags=["purchase orders"])
    def po_document(po_id: str, user: User = Depends(current_user)):
        po = po_or_404(user, po_id)
        path = Path(po.get("document_path") or "")
        if not po.get("document_path") or not path.is_file() or services.uploads.resolve() not in path.resolve().parents:
            raise HTTPException(404, "No document for this PO.")
        return FileResponse(path, media_type=po["document_type"])

    @app.post("/api/pos/{po_id}/accept", tags=["purchase orders"])
    def accept_po(po_id: str, body: POAccept, user: User = Depends(require_role("accountant"))):
        """Set prices and the delivery date; creates a draft PayPal invoice for the customer (send it from Invoices)."""
        from ..app.po_reader import clean_date, clean_items

        po = po_or_404(user, po_id)
        if po["status"] != "submitted":
            raise HTTPException(409, f"This PO is already {po['status']}.")
        items = clean_items([i.model_dump() for i in body.items])
        if not items or any(not i["unit_price"] for i in items):
            raise HTTPException(422, "Every item needs a quantity and a unit price.")
        expected = clean_date(body.expected_date)
        if not expected or expected < datetime.now(timezone.utc).date().isoformat():
            raise HTTPException(422, "Pick a delivery date from today onwards.")
        customer = services.appdb.get_user(po["user_id"])
        ref = po.get("customer_po_ref") or po_id
        params = {
            "primary_recipients": [{"billing_info": {"name": {"given_name": customer.name.split()[0], "surname": " ".join(customer.name.split()[1:])},
                                                      "email_address": customer.email}}],
            "items": [{"name": i["name"], "quantity": str(i["quantity"]), "unit_amount": {"currency_code": "USD", "value": i["unit_price"]}}
                      for i in items],
            "detail": {"currency_code": "USD", "note": f"For your purchase order {ref}. Expected delivery: {expected}."
                       + (f" {body.note.strip()}" if body.note and body.note.strip() else "")},
        }
        result = services.executor.execute("create_draft_invoice", params, caller=user)
        services.appdb.log_action(user, "accept_purchase_order", {"po_id": po_id, "expected_date": expected},
                                  "success" if result.ok else "failed", http_status=result.status_code,
                                  result_summary=(result.error or f"{po_id} accepted, draft invoice {result.body.get('id')}")[:300],
                                  confirmed=True, request_id=result.request_id)
        if not result.ok:
            raise HTTPException(502, f"PayPal: {result.error}")
        return po_view(services.appdb.update_po(po_id, status="accepted", items=items, expected_date=expected,
                                                invoice_id=result.body["id"]))

    @app.post("/api/pos/{po_id}/ship", tags=["purchase orders"])
    def ship_po(po_id: str, body: POShip, user: User = Depends(require_role("accountant"))):
        """The shop ships a paid order (or ships again after a 'not received' report) with tracking details."""
        po = po_or_404(user, po_id)
        if po["status"] not in ("paid", "not_received"):
            raise HTTPException(409, "Only paid orders (or ones reported as not received) can be shipped.")
        services.appdb.log_action(user, "ship_purchase_order", {"po_id": po_id, "carrier": body.carrier, "tracking_number": body.tracking_number},
                                  "success", result_summary=f"{po_id} shipped with {body.carrier} {body.tracking_number}", confirmed=True)
        return po_view(services.appdb.update_po(po_id, status="shipped", carrier=body.carrier.strip(),
                                                tracking_number=body.tracking_number.strip(),
                                                shipped_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))

    @app.post("/api/pos/{po_id}/delivered", tags=["purchase orders"])
    def po_delivered(po_id: str, user: User = Depends(require_role("customer"))):
        po = po_or_404(user, po_id)
        if po["status"] not in ("shipped", "not_received", "paid"):
            raise HTTPException(409, "This order isn't on its way yet.")
        services.appdb.log_action(user, "confirm_delivery", {"po_id": po_id}, "success", result_summary=f"{po_id} received", confirmed=True)
        return po_view(services.appdb.update_po(po_id, status="delivered", delivered_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))

    @app.post("/api/pos/{po_id}/not-received", tags=["purchase orders"])
    def po_not_received(po_id: str, body: PONotReceived, user: User = Depends(require_role("customer"))):
        po = po_or_404(user, po_id)
        if po["status"] not in ("shipped", "paid"):
            raise HTTPException(409, "This order isn't on its way yet.")
        note = (body.note or "").strip() or None
        services.appdb.log_action(user, "report_not_received", {"po_id": po_id, "note": note}, "success",
                                  result_summary=f"{po_id} reported as not received", confirmed=True)
        return po_view(services.appdb.update_po(po_id, status="not_received", not_received_note=note,
                                                not_received_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))

    @app.post("/api/pos/{po_id}/reject", tags=["purchase orders"])
    def reject_po(po_id: str, body: POReject, user: User = Depends(require_role("accountant"))):
        po = po_or_404(user, po_id)
        if po["status"] != "submitted":
            raise HTTPException(409, f"This PO is already {po['status']}.")
        services.appdb.log_action(user, "reject_purchase_order", {"po_id": po_id}, "success",
                                  result_summary=f"{po_id} rejected: {body.reason.strip()}"[:300], confirmed=True)
        return po_view(services.appdb.update_po(po_id, status="rejected", reject_reason=body.reason.strip()))

    @app.post("/api/invoices/{invoice_id}/pay", tags=["invoices"])
    def pay_invoice(invoice_id: str, user: User = Depends(require_role("customer"))):
        """Customers pay their own invoice. With real PayPal this happens on PayPal's checkout page;
        with the mock it's simulated by POST /mock/invoices/{id}/pay."""
        invoice = own_invoice(user, invoice_id)
        if invoice["status"] not in ("SENT", "UNPAID", "PARTIALLY_PAID"):
            raise HTTPException(409, "This invoice isn't waiting for payment.")
        r = services.executor.client.post(f"{services.executor.base_url}/mock/invoices/{invoice_id}/pay")
        ok = r.status_code < 300
        services.appdb.log_action(user, "pay_invoice", {"invoice_id": invoice_id, "amount": invoice["due_amount"]["value"]},
                                  "success" if ok else "failed", http_status=r.status_code,
                                  result_summary=("paid with PayPal" if ok else r.text)[:300], confirmed=True)
        if not ok:
            raise HTTPException(502, "Payment failed. Please try again.")
        for po in services.appdb.list_pos(user_id=user.user_id, statuses=("accepted", "invoiced")):
            if po.get("invoice_id") == invoice_id:
                sync_po_payment(po, services.executor, services.appdb)
        return own_invoice(user, invoice_id)

    # ---- dispute conversations ----------------------------------------------------------------------

    def own_dispute(user: User, dispute_id: str) -> dict:
        """The dispute, if this user may see it. Other people's disputes look like they don't exist."""
        if check_access(user, {"dispute_id": dispute_id}, services.executor):
            raise HTTPException(404, "Dispute not found.")
        result = services.executor.execute("show_dispute_details", {"dispute_id": dispute_id}, caller=user)
        if result.status_code == 404:
            raise HTTPException(404, "Dispute not found.")
        if not result.ok:
            raise HTTPException(502, f"PayPal: {result.error}")
        return result.body

    def conversation(dispute: dict) -> dict:
        dispute["refund_request"] = open_refund_request(dispute, services.appdb.refund_requests())
        tx = (dispute.get("disputed_transactions") or [{}])[0]
        names = {"BUYER": (tx.get("buyer") or {}).get("name", "Customer"), "SELLER": (tx.get("seller") or {}).get("name", "Shop")}
        messages = [{"from": m["posted_by"], "name": names.get(m["posted_by"], m["posted_by"]),
                     "time": m["time_posted"], "text": m["content"]}
                    for m in sorted(dispute.get("messages", []), key=lambda m: m["time_posted"])]
        return {"dispute": dispute, "messages": messages, "can_reply": dispute.get("status") != "RESOLVED",
                "can_resolve": dispute.get("status") in ("WAITING_FOR_SELLER_RESPONSE", "OPEN"),
                "offer": dispute.get("offer")}

    @app.get("/api/disputes/{dispute_id}", tags=["disputes"])
    def open_dispute(dispute_id: str, user: User = Depends(current_user)):
        dispute = own_dispute(user, dispute_id)
        mark_seen(services.appdb, user, dispute)
        return conversation(dispute)

    @app.post("/api/disputes/{dispute_id}/messages", tags=["disputes"])
    def send_dispute_message(dispute_id: str, body: MessageIn, user: User = Depends(current_user)):
        """The user typed and sent the message themselves, so no extra confirmation is asked."""
        own_dispute(user, dispute_id)
        params = {"dispute_id": dispute_id, "message": body.message.strip()}
        result = services.executor.execute("send_message_about_dispute_to_other_party", params, caller=user)
        card = services.registry.get("send_message_about_dispute_to_other_party") or {}
        services.appdb.log_action(user, card.get("name", "send_message"), params, "success" if result.ok else "failed",
                                  method=card.get("method"), path=card.get("path"), http_status=result.status_code,
                                  result_summary=(result.error or "message sent from the dispute page")[:300],
                                  confirmed=True, request_id=result.request_id)
        if not result.ok:
            raise HTTPException(422 if result.status_code == 422 else 502, result.error)
        dispute = own_dispute(user, dispute_id)
        mark_seen(services.appdb, user, dispute)
        return conversation(dispute)

    @app.get("/api/whats-new", tags=["disputes"])
    def whats_new_for_me(user: User = Depends(current_user)):
        """New messages, replies you owe, and messages still waiting for an answer (no LLM involved)."""
        return {"items": whats_new(user, services.executor, services.appdb)}

    @app.post("/api/disputes/{dispute_id}/resolve", tags=["disputes"])
    def resolve_dispute(dispute_id: str, body: ResolveIn, user: User = Depends(require_role("accountant"))):
        """The shop resolves a dispute with the customer: refund in full, or offer a partial refund
        (the customer accepts or declines in the chat). Disputes stay between the shop and the
        customer; there's no PayPal review in PayMind."""
        dispute = own_dispute(user, dispute_id)
        if dispute.get("status") == "RESOLVED":
            raise HTTPException(409, "This dispute is already resolved.")
        note = (body.note or "").strip() or None
        if body.action == "refund":
            tool, params = "accept_claim", {"dispute_id": dispute_id, **({"note": note} if note else {})}
        elif body.action == "offer":
            try:
                amount = Decimal(str(body.amount or "0"))
            except Exception:
                raise HTTPException(422, "The offer amount must be a number.")
            if amount <= 0 or amount > Decimal(dispute["dispute_amount"]["value"]):
                raise HTTPException(422, f"Offer between 0.01 and the disputed {dispute['dispute_amount']['value']}.")
            tool, params = "make_offer_to_resolve_dispute", {
                "dispute_id": dispute_id, "offer_type": "REFUND",
                "offer_amount": {"currency_code": dispute["dispute_amount"]["currency_code"], "value": f"{amount:.2f}"},
                **({"note": note} if note else {})}
        card = services.registry.get(tool) or {}
        result = services.executor.execute(tool, params, caller=user)
        services.appdb.log_action(user, tool, params, "success" if result.ok else "failed", method=card.get("method"),
                                  path=card.get("path"), http_status=result.status_code,
                                  result_summary=(result.error or f"dispute {dispute_id}: {body.action}")[:300],
                                  confirmed=True, request_id=result.request_id)
        if not result.ok:
            raise HTTPException(422 if result.status_code == 422 else 502, result.error)
        return conversation(own_dispute(user, dispute_id))

    # ---- audit --------------------------------------------------------------------------

    @app.get("/api/audit", tags=["audit"])
    def audit(limit: int = 50, user: User = Depends(current_user)):
        return {"items": services.appdb.recent_actions(user.user_id, limit=limit)}

    # ---- web page ---------------------------------------------------------------------------------

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def no_cache_for_the_page(request: Request, call_next):
        """Always load the latest page, script and styles (a normal refresh shows changes)."""
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app
