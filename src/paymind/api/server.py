"""PayMind API + web app.

Every /api route except login needs a valid JWT (see auth.py). Roles are
checked on the server for every call; the web page only decides what to show.

  POST /api/auth/login          email + password → token
  GET  /api/me                  who am I
  POST /api/chat                send a message to the agent
  POST /api/chat/confirm        answer a pending yes/no
  GET  /api/paypal/overview     account data: everything for accountants, own records for customers
  GET  /api/paypal/transactions accountant only: transactions for a day / week / month, with totals
  GET  /api/audit               the caller's own audit log

Run (mock PayPal must be running on port 8000):
    uvicorn paymind.api.server:create_app --factory --port 8001
    open http://localhost:8001
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent.executor import Executor, ToolRegistry
from ..agent.scope import filter_results
from ..app.database import AppDatabase, User
from .auth import TOKEN_TTL, create_token, current_user, require_role, secret_key

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / "static"
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


@dataclass
class Services:
    appdb: AppDatabase
    executor: Executor
    registry: ToolRegistry
    search: Callable[..., list[tuple[str, str]]]
    agent_factory: Callable[[], Any]          # built on first chat (loads models, connects to Gemini)
    mock_base_url: str = "http://localhost:8000"
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
    return Services(appdb=AppDatabase(), executor=executor, registry=registry,
                    search=qdrant_search(registry), agent_factory=build_agent, mock_base_url=executor.base_url)


# ---- request bodies ------------------------------------------------------------------

class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


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
        return {"access_token": create_token(user, app.state.jwt_secret), "token_type": "bearer",
                "expires_in": int(TOKEN_TTL.total_seconds()), "user": public(user)}

    @app.get("/api/me", tags=["auth"])
    def me(user: User = Depends(current_user)):
        return public(user)

    # ---- chat ------------------------------------------------------------------------

    def agent_reply(session_id: str, reply) -> dict:
        # Which tools ran and why is developer information: it's in the LangSmith trace, not here.
        return {"session_id": session_id, "reply": reply.text, "confirmation": reply.confirmation}

    @app.post("/api/chat", tags=["chat"])
    def chat(body: ChatIn, user: User = Depends(current_user)):
        session_id = body.session_id or uuid.uuid4().hex[:12]
        reply = services.agent().send(user.user_id, thread_for(user, session_id), body.message.strip())
        return agent_reply(session_id, reply)

    @app.post("/api/chat/confirm", tags=["chat"])
    def confirm(body: ConfirmIn, user: User = Depends(current_user)):
        reply = services.agent().answer(user.user_id, thread_for(user, body.session_id), body.approve)
        return agent_reply(body.session_id, reply)

    # ---- PayPal account data (through the same executor the agent uses) ------------------------

    def call(tool: str, params: dict, user: User) -> Any:
        result = services.executor.execute(tool, params, caller=user)
        if not result.ok:
            raise HTTPException(502, f"PayPal: {result.error}")
        return result.body

    @app.get("/api/paypal/overview", tags=["paypal"])
    def overview(user: User = Depends(current_user)):
        disputes = filter_results(user, "list_disputes", call("list_disputes", {"page_size": 50}, user))["items"]
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
