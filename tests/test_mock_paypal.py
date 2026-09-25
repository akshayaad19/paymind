import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paymind.mock_paypal.app import INITIAL_DB, create_app, stateful_routes

TOOLS = json.loads((Path(__file__).resolve().parents[1] / "data/tools/tools.json").read_text())


@pytest.fixture
def db_file(tmp_path):
    return tmp_path / "mock.db"  # each test works on its own copy of initial.db


@pytest.fixture
def app(db_file):
    return create_app(db_path=db_file, slow_seconds=0)


@pytest.fixture
def client(app):
    return TestClient(app)


def usd(value):
    return {"currency_code": "USD", "value": value}


def issue(response):
    return response.json()["details"][0]["issue"]


def completed_capture(app):
    return next(c for c in app.state.store.db.all("captures") if c["status"] == "COMPLETED")


def refunds_for(app, capture_id):
    return [r for r in app.state.store.db.all("refunds") if r["capture_id"] == capture_id]


# ---- starting data + routing ---------------------------------------------------

def test_initial_database_contents():
    conn = sqlite3.connect(INITIAL_DB)
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("customers", "captures", "refunds", "orders", "invoices", "disputes", "transactions")}
    assert counts == {"customers": 6, "captures": 47, "refunds": 4, "orders": 2, "invoices": 8, "disputes": 4, "transactions": 51}


def test_initial_data_is_consistent():
    """No payment is refunded beyond its amount, and every refund has a matching ledger row."""
    conn = sqlite3.connect(INITIAL_DB)
    captures = {i: json.loads(d) for i, d in conn.execute("SELECT id, data FROM captures")}
    refunds = [json.loads(d) for (d,) in conn.execute("SELECT data FROM refunds")]
    ledger = {json.loads(d)["transaction_info"]["transaction_id"] for (d,) in conn.execute("SELECT data FROM transactions")}
    for c in captures.values():
        assert float(c["refunded_amount"]["value"]) <= float(c["amount"]["value"]), c["id"]
    for r in refunds:
        assert r["capture_id"] in captures and r["id"] in ledger
        total = sum(float(x["amount"]["value"]) for x in refunds if x["capture_id"] == r["capture_id"])
        assert abs(total - float(captures[r["capture_id"]]["refunded_amount"]["value"])) < 0.001


def test_every_stateful_route_is_a_real_tool_path():
    tool_routes = {(t["method"], t["path"]) for t in TOOLS}
    assert stateful_routes() <= tool_routes
    assert len(stateful_routes()) == 27


def test_every_tool_is_answered(app):
    stateful = stateful_routes()
    for tool in TOOLS:
        routed = (tool["method"], tool["path"]) in stateful or any(
            t["method"] == tool["method"] and t["path"] == tool["path"] and t["name"] in app.state.example_tools for t in TOOLS
        )
        assert routed, tool["name"]


def test_example_replay(client):
    r = client.get("/v1/notifications/webhooks")
    assert r.status_code == 200
    assert r.headers["x-mock-source"] == "example:list_webhooks"
    assert "webhooks" in r.json()


# ---- the brief's example questions ----------------------------------------------

def test_open_dispute_from_user_123(client):
    items = client.get("/v1/customer/disputes", params={"dispute_state": "REQUIRED_ACTION"}).json()["items"]
    buyers = [d["disputed_transactions"][0]["buyer"]["payer_id"] for d in items]
    assert buyers == ["user_123"]
    assert items[0]["dispute_amount"] == usd("40.00")


def test_sales_last_month(client):
    r = client.get("/v1/reporting/transactions", params={
        "start_date": "2026-08-01T00:00:00Z", "end_date": "2026-08-31T23:59:59Z", "transaction_type": "T0006"})
    assert r.status_code == 200
    assert r.json()["total_items"] > 0
    assert all(float(t["transaction_info"]["transaction_amount"]["value"]) > 0 for t in r.json()["transaction_details"])


def test_transaction_search_limits(client):
    too_long = client.get("/v1/reporting/transactions", params={"start_date": "2026-06-01T00:00:00Z", "end_date": "2026-08-31T00:00:00Z"})
    assert too_long.status_code == 400
    missing = client.get("/v1/reporting/transactions", params={"start_date": "2026-08-01T00:00:00Z"})
    assert missing.status_code == 400 and issue(missing) == "MISSING_REQUIRED_PARAMETER"


def test_invoice_lifecycle_create_send_pay(client):
    draft = client.post("/v2/invoicing/invoices", json={
        "primary_recipients": [{"billing_info": {"email_address": "john@x.com"}}],
        "items": [{"name": "Consulting", "quantity": "1", "unit_amount": usd("50.00")}],
    })
    assert draft.status_code == 201
    invoice_id = draft.json()["id"]
    assert draft.json()["status"] == "DRAFT" and draft.json()["amount"] == usd("50.00")

    assert client.post(f"/v2/invoicing/invoices/{invoice_id}/send", json={}).json()["status"] == "SENT"
    again = client.post(f"/v2/invoicing/invoices/{invoice_id}/send", json={})
    assert again.status_code == 422 and issue(again) == "CANNOT_SEND_INVOICE"

    assert client.post(f"/v2/invoicing/invoices/{invoice_id}/remind", json={}).status_code == 204
    partial = client.post(f"/v2/invoicing/invoices/{invoice_id}/payments", json={"method": "CASH", "amount": usd("20.00")})
    assert partial.json()["status"] == "PARTIALLY_PAID"
    full = client.post(f"/v2/invoicing/invoices/{invoice_id}/payments", json={"method": "CASH", "amount": usd("30.00")})
    assert full.json()["status"] == "MARKED_AS_PAID"

    found = client.post("/v2/invoicing/search-invoices", json={"recipient_email": "john@x.com"}).json()
    assert invoice_id in [i["id"] for i in found["items"]]


def test_invoice_errors(client):
    no_items = client.post("/v2/invoicing/invoices", json={})
    assert no_items.status_code == 400 and issue(no_items) == "MISSING_REQUIRED_PARAMETER"
    no_recipient = client.post("/v2/invoicing/invoices", json={"items": [{"name": "X", "unit_amount": usd("5.00")}]}).json()["id"]
    r = client.post(f"/v2/invoicing/invoices/{no_recipient}/send", json={})
    assert r.status_code == 422 and issue(r) == "MISSING_RECIPIENT"
    assert client.get("/v2/invoicing/invoices/INV2-NOPE").status_code == 404


# ---- payments ------------------------------------------------------------------

def test_refund_updates_capture_and_balance(app, client):
    capture = completed_capture(app)
    total = float(capture["amount"]["value"])
    before = float(client.get("/v1/reporting/balances").json()["balances"][0]["total_balance"]["value"])

    partial = client.post(f"/v2/payments/captures/{capture['id']}/refund", json={"amount": usd("1.00")})
    assert partial.status_code == 201
    assert client.get(f"/v2/payments/captures/{capture['id']}").json()["status"] == "PARTIALLY_REFUNDED"
    assert client.get(f"/v2/payments/refunds/{partial.json()['id']}").json()["amount"] == usd("1.00")

    rest = client.post(f"/v2/payments/captures/{capture['id']}/refund", json={})
    assert float(rest.json()["amount"]["value"]) == pytest.approx(total - 1)
    assert client.get(f"/v2/payments/captures/{capture['id']}").json()["status"] == "REFUNDED"

    after = float(client.get("/v1/reporting/balances").json()["balances"][0]["total_balance"]["value"])
    assert after == pytest.approx(before - total)

    again = client.post(f"/v2/payments/captures/{capture['id']}/refund", json={})
    assert again.status_code == 422 and issue(again) == "CAPTURE_FULLY_REFUNDED"


def test_refund_more_than_paid(app, client):
    capture = completed_capture(app)
    r = client.post(f"/v2/payments/captures/{capture['id']}/refund", json={"amount": usd("99999.00")})
    assert r.status_code == 422 and issue(r) == "REFUND_AMOUNT_EXCEEDED"


def test_order_create_and_capture(client):
    order = client.post("/v2/checkout/orders", json={"intent": "CAPTURE", "purchase_units": [{"amount": usd("25.00")}]}).json()
    assert order["status"] == "CREATED"
    captured = client.post(f"/v2/checkout/orders/{order['id']}/capture", json={}).json()
    assert captured["status"] == "COMPLETED"
    capture_id = captured["purchase_units"][0]["payments"]["captures"][0]["id"]
    assert client.get(f"/v2/payments/captures/{capture_id}").status_code == 200
    assert client.post(f"/v2/checkout/orders/{order['id']}/capture", json={}).status_code == 422


# ---- disputes ------------------------------------------------------------------

def open_dispute_id(client, payer_id="user_123"):
    items = client.get("/v1/customer/disputes").json()["items"]
    return next(d["dispute_id"] for d in items if d["disputed_transactions"][0]["buyer"]["payer_id"] == payer_id)


def test_accept_claim_refunds_buyer(client):
    dispute_id = open_dispute_id(client)
    r = client.post(f"/v1/customer/disputes/{dispute_id}/accept-claim", json={"note": "Sorry about that"})
    assert r.json()["status"] == "RESOLVED"
    dispute = client.get(f"/v1/customer/disputes/{dispute_id}").json()
    assert dispute["dispute_outcome"]["outcome_code"] == "RESOLVED_BUYER_FAVOUR"
    assert client.get(f"/v2/payments/refunds/{dispute['refund_id']}").json()["amount"] == usd("40.00")
    again = client.post(f"/v1/customer/disputes/{dispute_id}/accept-claim", json={})
    assert again.status_code == 422 and issue(again) == "DISPUTE_ALREADY_RESOLVED"


def test_offer_then_buyer_accepts(client):
    dispute_id = open_dispute_id(client)
    too_big = client.post(f"/v1/customer/disputes/{dispute_id}/make-offer", json={"offer_amount": usd("999.00")})
    assert too_big.status_code == 422
    offer = client.post(f"/v1/customer/disputes/{dispute_id}/make-offer", json={"offer_amount": usd("20.00"), "offer_type": "REFUND"})
    assert offer.json()["status"] == "WAITING_FOR_BUYER_RESPONSE"
    accepted = client.post(f"/v1/customer/disputes/{dispute_id}/accept-offer", json={})
    assert accepted.json()["status"] == "RESOLVED"


def test_dispute_message_and_evidence(client):
    dispute_id = open_dispute_id(client)
    assert client.post(f"/v1/customer/disputes/{dispute_id}/send-message", json={}).status_code == 400
    client.post(f"/v1/customer/disputes/{dispute_id}/send-message", json={"message": "Tracking: 1Z999"})
    evidence = client.post(f"/v1/customer/disputes/{dispute_id}/provide-evidence", json={"input": {"notes": "Delivered 12 Sep"}})
    assert evidence.json()["status"] == "UNDER_REVIEW"
    dispute = client.get(f"/v1/customer/disputes/{dispute_id}").json()
    assert dispute["messages"][-1]["content"] == "Tracking: 1Z999"
    assert len(dispute["evidences"]) == 1


# ---- idempotency and failures ------------------------------------------------------

def test_same_request_id_refunds_only_once(app, client):
    capture = completed_capture(app)
    headers = {"PayPal-Request-Id": "req_7f3a"}
    body = {"amount": usd("1.00")}
    first = client.post(f"/v2/payments/captures/{capture['id']}/refund", json=body, headers=headers)
    second = client.post(f"/v2/payments/captures/{capture['id']}/refund", json=body, headers=headers)
    assert first.json()["id"] == second.json()["id"]
    assert second.headers["x-mock-idempotent-replay"] == "true"
    assert len(refunds_for(app, capture["id"])) == 1

    different = client.post(f"/v2/payments/captures/{capture['id']}/refund", json=body, headers={"PayPal-Request-Id": "req_other"})
    assert different.json()["id"] != first.json()["id"]


def test_timeout_still_does_the_work(app, client):
    capture = completed_capture(app)
    headers = {"PayPal-Request-Id": "req_slow", "X-Mock-Fail": "timeout"}
    client.post(f"/v2/payments/captures/{capture['id']}/refund", json={"amount": usd("1.00")}, headers=headers)
    retry = client.post(f"/v2/payments/captures/{capture['id']}/refund", json={"amount": usd("1.00")},
                        headers={"PayPal-Request-Id": "req_slow"})
    assert retry.headers["x-mock-idempotent-replay"] == "true"
    assert len(refunds_for(app, capture["id"])) == 1


@pytest.mark.parametrize("code", [500, 503])
def test_injected_failures_do_nothing(app, client, code):
    capture = completed_capture(app)
    r = client.post(f"/v2/payments/captures/{capture['id']}/refund", json={}, headers={"X-Mock-Fail": str(code)})
    assert r.status_code == code and issue(r) == "SIMULATED_FAILURE"
    assert client.get(f"/v2/payments/captures/{capture['id']}").json()["status"] == "COMPLETED"


def test_random_failure_rate(db_file):
    client = TestClient(create_app(db_path=db_file, failure_rate=1.0))
    assert client.get("/v1/reporting/balances").status_code == 503
    assert client.get("/mock/summary").status_code == 200  # mock admin routes never fail


def test_reset(app, client):
    capture = completed_capture(app)
    client.post(f"/v2/payments/captures/{capture['id']}/refund", json={})
    client.post("/mock/reset")
    assert client.get(f"/v2/payments/captures/{capture['id']}").json()["status"] == "COMPLETED"


# ---- SQLite is the source of truth ---------------------------------------------------

def test_changes_survive_a_restart(db_file):
    first = create_app(db_path=db_file)
    client = TestClient(first)
    capture = completed_capture(first)
    client.post(f"/v2/payments/captures/{capture['id']}/refund", json={}, headers={"PayPal-Request-Id": "req_persist"})
    invoice_id = client.post("/v2/invoicing/invoices", json={"items": [{"name": "X", "unit_amount": usd("5.00")}]}).json()["id"]
    balance = client.get("/v1/reporting/balances").json()["balances"]

    restarted = TestClient(create_app(db_path=db_file))  # same database file, brand-new server
    assert restarted.get(f"/v2/payments/captures/{capture['id']}").json()["status"] == "REFUNDED"
    assert restarted.get(f"/v2/invoicing/invoices/{invoice_id}").status_code == 200
    assert restarted.get("/v1/reporting/balances").json()["balances"] == balance
    retry = restarted.post(f"/v2/payments/captures/{capture['id']}/refund", json={}, headers={"PayPal-Request-Id": "req_persist"})
    assert retry.headers["x-mock-idempotent-replay"] == "true"  # no double refund, even after a restart


def test_writes_go_straight_to_the_database(app, client, db_file):
    capture = completed_capture(app)
    client.post(f"/v2/payments/captures/{capture['id']}/refund", json={"amount": usd("1.00")})
    row = sqlite3.connect(db_file).execute("SELECT data FROM captures WHERE id = ?", (capture["id"],)).fetchone()
    assert json.loads(row[0])["status"] == "PARTIALLY_REFUNDED"


def test_edits_made_in_the_database_are_seen(app, client, db_file):
    conn = sqlite3.connect(db_file)
    dispute_id = open_dispute_id(client)
    data = json.loads(conn.execute("SELECT data FROM disputes WHERE id = ?", (dispute_id,)).fetchone()[0])
    data["dispute_amount"] = usd("41.00")
    conn.execute("UPDATE disputes SET data = ? WHERE id = ?", (json.dumps(data), dispute_id))
    conn.commit()
    assert client.get(f"/v1/customer/disputes/{dispute_id}").json()["dispute_amount"] == usd("41.00")


def test_failed_request_writes_nothing(app, client, db_file):
    capture = completed_capture(app)
    before = client.get("/mock/summary").json()
    too_much = client.post(f"/v2/payments/captures/{capture['id']}/refund", json={"amount": usd("99999.00")})
    assert too_much.status_code == 422
    assert client.get("/mock/summary").json() == before


def test_initial_db_is_never_changed(app, client):
    before = INITIAL_DB.read_bytes()
    client.post(f"/v2/payments/captures/{completed_capture(app)['id']}/refund", json={})
    client.post("/mock/reset")
    assert INITIAL_DB.read_bytes() == before


def test_new_ids_are_unique(client):
    ids = {client.post("/v2/checkout/orders", json={"purchase_units": [{"amount": usd("5.00")}]}).json()["id"] for _ in range(20)}
    assert len(ids) == 20


def test_reset_command_copies_initial_db_when_server_is_down(tmp_path):
    from paymind.mock_paypal.reset import reset
    working = tmp_path / "paypal_mock.db"
    working.write_bytes(b"changed")
    message = reset("http://127.0.0.1:9", working_db=working)  # nothing listens on port 9
    assert "copied" in message and working.read_bytes() == INITIAL_DB.read_bytes()


def test_buyer_pays_a_sent_invoice(app, client):
    sent = next(i for i in app.state.store.db.all("invoices") if i["status"] == "SENT")
    before = float(client.get("/v1/reporting/balances").json()["balances"][0]["total_balance"]["value"])
    paid = client.post(f"/mock/invoices/{sent['id']}/pay").json()
    assert paid["status"] == "PAID" and paid["due_amount"]["value"] == "0.00"
    capture = client.get(f"/v2/payments/captures/{paid['payments']['transactions'][-1]['payment_id']}").json()
    assert capture["invoice_id"] == sent["id"] and capture["status"] == "COMPLETED"
    after = float(client.get("/v1/reporting/balances").json()["balances"][0]["total_balance"]["value"])
    assert after > before
    again = client.post(f"/mock/invoices/{sent['id']}/pay")
    assert again.status_code == 422 and again.json()["details"][0]["issue"] == "CANNOT_PAY_INVOICE"
    draft = next(i for i in app.state.store.db.all("invoices") if i["status"] == "DRAFT")
    assert client.post(f"/mock/invoices/{draft['id']}/pay").status_code == 422
