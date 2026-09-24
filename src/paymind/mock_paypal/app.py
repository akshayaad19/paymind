"""Mock PayPal server.

Answers the agent's API calls on the same paths as PayPal:

  - ~27 endpoints keep real state (invoices, captures/refunds, orders,
    disputes, transactions, balance): a refund changes the payment, adds a
    refund and lowers the balance.
  - Every other tool in tools.json replays PayPal's own example response
    from example_responses.json, so all 112 tools answer.

Extras for testing the agent:
  - Idempotency: a write sent again with the same PayPal-Request-Id header
    returns the first result instead of running twice.
  - Failures on demand: header `X-Mock-Fail: 500 | 503 | timeout`, or
    MOCK_FAILURE_RATE=0.2 for random 503s. `timeout` does the work, then
    responds late: the "it worked but the client timed out" case that
    idempotency protects against.
  - All data lives in SQLite (see db.py). The server works on
    data/mock/paypal_mock.db, created from data/mock/initial.db on first start.
    Each request is one database transaction: a failed request writes nothing.
  - POST /mock/reset copies initial.db over the working database.

Run:
    uvicorn paymind.mock_paypal.app:create_app --factory --port 8000
    open http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from . import disputes, invoices, payments, reporting
from .db import Database, prepare
from .errors import PayPalError, error_body, paypal_error_handler
from .store import Store

ROOT = Path(__file__).resolve().parents[3]
INITIAL_DB = ROOT / "data/mock/initial.db"
STATEFUL_ROUTERS = [invoices.router, payments.router, disputes.router, reporting.router]


def first_success_example(examples: list[dict]) -> tuple[int, object] | None:
    for ex in examples:
        code = ex.get("code") or 0
        if 200 <= code < 300:
            return code, ex.get("body")
    return None


def stateful_routes() -> set[tuple[str, str]]:
    """(method, path) of every hand-written handler. Read from our routers, since
    newer FastAPI versions don't list included routes in app.routes."""
    return {(m, r.path) for router in STATEFUL_ROUTERS for r in router.routes for m in getattr(r, "methods", set())}


def add_example_routes(app: FastAPI, tools: list[dict], examples: dict[str, list[dict]]) -> list[str]:
    """Register a replay route for every tool that has no stateful handler."""
    taken = stateful_routes()
    added = []
    for tool in tools:
        key = (tool["method"], tool["path"])
        if key in taken or tool.get("eval_only"):
            continue
        example = first_success_example(examples.get(tool["name"], []))
        status, body = example if example else (200, {})

        async def replay(status=status, body=body, name=tool["name"]):
            headers = {"X-Mock-Source": f"example:{name}"}
            if status == 204 or body is None:
                return Response(status_code=status, headers=headers)
            return JSONResponse(body, status_code=status, headers=headers)

        app.add_api_route(tool["path"], replay, methods=[tool["method"]], name=tool["name"], tags=["example replay"])
        taken.add(key)
        added.append(tool["name"])
    return added


def create_app(db_path: str | Path | None = None, initial_db: Path = INITIAL_DB,
               tools_path: Path | None = None, examples_path: Path | None = None,
               failure_rate: float | None = None, slow_seconds: float | None = None) -> FastAPI:
    app = FastAPI(title="Mock PayPal", description="Stateful mock of PayPal's REST APIs for the PayMind agent.")
    db_path = Path(db_path or os.getenv("MOCK_DB_PATH") or ROOT / "data/mock/paypal_mock.db")
    prepare(db_path, initial_db)
    app.state.db_path, app.state.initial_db = db_path, initial_db
    app.state.store = Store(Database(db_path))
    app.state.lock = asyncio.Lock()  # one request at a time: simple and safe for a single-connection mock
    app.state.failure_rate = float(os.getenv("MOCK_FAILURE_RATE", "0")) if failure_rate is None else failure_rate
    app.state.slow_seconds = float(os.getenv("MOCK_SLOW_SECONDS", "10")) if slow_seconds is None else slow_seconds
    app.add_exception_handler(PayPalError, paypal_error_handler)

    @app.middleware("http")
    async def transaction_per_request(request: Request, call_next):
        if request.url.path.startswith(("/mock", "/docs", "/openapi", "/redoc")):
            return await call_next(request)

        fail = (request.headers.get("x-mock-fail") or "").lower()
        if not fail and app.state.failure_rate and random.random() < app.state.failure_rate:
            fail = "503"
        if fail in ("500", "503"):
            code = int(fail)
            return JSONResponse(error_body(code, "SIMULATED_FAILURE", "Simulated server failure (X-Mock-Fail)."), status_code=code)

        async with app.state.lock:
            db = app.state.store.db
            request_id = request.headers.get("paypal-request-id")
            is_write = request.method != "GET"

            if is_write and request_id:
                cached = db.idempotent_response(request.method, request.url.path, request_id)
                if cached:
                    status, content = cached
                    return Response(content, status_code=status, media_type="application/json",
                                    headers={"X-Mock-Idempotent-Replay": "true"})

            db.begin()
            try:
                response = await call_next(request)
                content = b"".join([chunk async for chunk in response.body_iterator])
            except Exception:
                db.rollback()
                raise
            remember = is_write and request_id and response.status_code < 500
            if response.status_code < 400:
                if remember:
                    db.save_idempotent_response(request.method, request.url.path, request_id, response.status_code, content)
                db.commit()  # the whole request's changes are saved together
            else:
                db.rollback()  # a failed request leaves the data untouched...
                if remember:  # ...but a retry with the same request ID gets the same answer
                    db.begin()
                    db.save_idempotent_response(request.method, request.url.path, request_id, response.status_code, content)
                    db.commit()

        response = Response(content, status_code=response.status_code, media_type=response.media_type,
                            headers={k: v for k, v in response.headers.items() if k.lower() != "content-length"})
        if fail == "timeout":
            await asyncio.sleep(app.state.slow_seconds)  # the work is saved; the answer is just late
        return response

    for router in STATEFUL_ROUTERS:
        app.include_router(router, tags=["stateful"])

    tools = json.loads((tools_path or ROOT / "data/tools/tools.json").read_text())
    examples = json.loads((examples_path or ROOT / "data/mock/example_responses.json").read_text())
    app.state.example_tools = add_example_routes(app, tools, examples)

    @app.post("/mock/reset", tags=["mock"])
    async def reset():
        """Throw away all changes: copy initial.db over the working database."""
        async with app.state.lock:
            app.state.store.db.close()
            shutil.copyfile(app.state.initial_db, app.state.db_path)
            app.state.store = Store(Database(app.state.db_path))
        return {"status": "reset", "from": app.state.initial_db.name}

    @app.get("/mock/summary", tags=["mock"])
    def summary():
        store = app.state.store
        invoices_ = store.db.all("invoices")
        return {
            "database": str(app.state.db_path),
            "customers": [c["payer_id"] for c in store.db.all("customers")],
            "captures": store.db.count("captures"), "refunds": store.db.count("refunds"), "orders": store.db.count("orders"),
            "invoices": {status: sum(i["status"] == status for i in invoices_) for status in sorted({i["status"] for i in invoices_})},
            "disputes": [{"id": d["dispute_id"], "buyer": d["disputed_transactions"][0]["buyer"]["payer_id"],
                          "status": d["status"], "amount": d["dispute_amount"]["value"]} for d in store.db.all("disputes")],
            "balance": str(store.balance()),
            "example_replay_tools": len(app.state.example_tools),
        }

    return app
