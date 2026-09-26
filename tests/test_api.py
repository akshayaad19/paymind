"""PayMind API: JWT login, role checks and data scoping.

Uses the real app database, executor and mock PayPal (in-process); the agent
and tool search are fakes so no Gemini or Qdrant is needed."""

import json
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from paymind.agent.executor import Executor, ToolRegistry
from paymind.agent.graph import Reply
from paymind.api.auth import create_token
from paymind.api.server import Services, create_app
from paymind.app.database import AppDatabase
from paymind.mock_paypal.app import create_app as create_mock

SECRET = "test-secret-" + "x" * 40
REGISTRY = ToolRegistry()
LOGINS = {
    "asha": ("asha@paymind-demo.example", "asha-demo-123"),
    "rahul": ("rahul.sharma@example.com", "rahul-demo-123"),
    "priya": ("priya.nair@example.com", "priya-demo-123"),
}


class FakeAgent:
    def __init__(self):
        self.calls = []

    def send(self, user_id, thread, text):
        self.calls.append(("send", user_id, thread, text))
        if "refund" in text:
            return Reply(None, {"question": "Refund captured payment for 5 USD?", "tool": "refund_captured_payment", "params": {}, "large_amount": False})
        return Reply(f"echo: {text}", None)

    def answer(self, user_id, thread, approve):
        self.calls.append(("answer", user_id, thread, approve))
        return Reply("Done." if approve else "Cancelled.", None)

    def stream(self, user_id, thread, text=None, approve=None):
        reply = self.send(user_id, thread, text) if approve is None else self.answer(user_id, thread, approve)
        if reply.text:
            yield {"type": "restart"}
            for word in reply.text.split(" "):
                yield {"type": "token", "text": word + " "}
        yield {"type": "done", "reply": reply.text, "confirmation": reply.confirmation, "receipt": None}



def fake_search(query, role=None, k=5, include_eval_only=True):
    names = [n for n in ("refund_captured_payment", "list_disputes", "show_dispute_details")
             if role in REGISTRY.get(n)["allowed_roles"]]
    return [(n, REGISTRY.get(n)["description"]) for n in names][:k]


def fake_po_reader(data, mime):
    """Stands in for Gemini reading a handwritten PO."""
    from paymind.app.po_reader import POExtraction, POLine
    if b"unreadable" in data:
        raise RuntimeError("503 model busy")
    return POExtraction(customer_po_ref="RS-2026-07", requested_delivery_date="2030-10-05", notes="Deliver to back door",
                        items=[POLine(name="Wireless Headphones", quantity=2, unit_price="$79.99"), POLine(name="Phone Case", quantity=3)],
                        unclear=["quantity on line 2 could be 3 or 8"])


@pytest.fixture
def setup(tmp_path):
    mock = create_mock(db_path=tmp_path / "mock.db", slow_seconds=0)
    executor = Executor(REGISTRY, base_url="http://testserver", client=TestClient(mock), sleep=lambda s: None)
    agent = FakeAgent()
    services = Services(appdb=AppDatabase(tmp_path / "app.db"), executor=executor, registry=REGISTRY,
                        search=fake_search, agent_factory=lambda: agent, mock_base_url="http://testserver",
                        po_reader=fake_po_reader, uploads=tmp_path / "uploads")
    return TestClient(create_app(services, jwt_secret=SECRET)), agent, services


def login(client, who):
    email, password = LOGINS[who]
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


# ---- login and tokens ---------------------------------------------------------------------

def test_login_returns_a_signed_token(setup):
    client, _, _ = setup
    r = client.post("/api/auth/login", json={"email": "asha@paymind-demo.example", "password": "asha-demo-123"})
    body = r.json()
    claims = jwt.decode(body["access_token"], SECRET, algorithms=["HS256"], issuer="paymind")
    assert claims["sub"] == "u_asha" and claims["role"] == "accountant"
    assert body["user"]["role"] == "accountant" and "password" not in str(body["user"])


def test_wrong_password_and_unknown_email_look_the_same(setup):
    client, _, _ = setup
    a = client.post("/api/auth/login", json={"email": "asha@paymind-demo.example", "password": "nope"})
    b = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "nope"})
    assert a.status_code == b.status_code == 401 and a.json() == b.json()


@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer not-a-jwt"},
    {"Authorization": "Basic abc"},
])
def test_protected_routes_need_a_valid_token(setup, headers):
    client, _, _ = setup
    assert client.get("/api/me", headers=headers).status_code == 401


def test_expired_token_is_rejected(setup):
    client, _, services = setup
    old = create_token(services.appdb.get_user("u_asha"), SECRET, now=datetime.now(timezone.utc) - timedelta(hours=9))
    r = client.get("/api/me", headers={"Authorization": f"Bearer {old}"})
    assert r.status_code == 401 and "expired" in r.json()["detail"]


def test_forged_token_is_rejected(setup):
    """A customer can't make themselves an accountant: changing the token breaks the signature."""
    client, _, services = setup
    forged = jwt.encode({"sub": "u_asha", "role": "accountant", "iss": "paymind", "iat": 0, "exp": 9999999999},
                        "attacker-secret-" + "y" * 40, algorithm="HS256")
    assert client.get("/api/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_role_comes_from_the_database_not_the_token(setup):
    client, _, services = setup
    rahul = services.appdb.get_user("u_rahul")
    token = create_token(rahul, SECRET)
    # even if a token claimed "accountant", the server uses the stored role
    assert client.get("/api/me", headers={"Authorization": f"Bearer {token}"}).json()["role"] == "customer"


def test_logout_kills_the_token_and_any_copy(setup):
    """Logout raises the user's token_version: a copy taken before logout is rejected too."""
    client, _, _ = setup
    headers = login(client, "asha")
    copy = dict(headers)  # e.g. copied from dev tools
    assert client.post("/api/auth/logout", headers=headers).json() == {"ok": True}
    r = client.get("/api/me", headers=copy)
    assert r.status_code == 401 and "logged out" in r.json()["detail"]
    assert client.get("/api/me", headers=login(client, "asha")).status_code == 200  # a new login works


def test_logout_only_affects_that_user(setup):
    client, _, _ = setup
    asha, rahul = login(client, "asha"), login(client, "rahul")
    client.post("/api/auth/logout", headers=asha)
    assert client.get("/api/me", headers=rahul).status_code == 200


def test_password_change_revokes_old_tokens(setup):
    client, _, services = setup
    headers = login(client, "rahul")
    services.appdb.set_password("u_rahul", "new-secret-1")
    assert client.get("/api/me", headers=headers).status_code == 401


def test_token_without_version_is_rejected(setup):
    """Tokens from before versioning (no "ver") are refused, not treated as current."""
    client, _, _ = setup
    old_style = jwt.encode({"sub": "u_asha", "role": "accountant", "iss": "paymind", "iat": int(datetime.now(timezone.utc).timestamp()),
                            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())}, SECRET, algorithm="HS256")
    assert client.get("/api/me", headers={"Authorization": f"Bearer {old_style}"}).status_code == 401


# ---- role checks ---------------------------------------------------------------------------------

def test_demo_reset_is_not_in_the_app(setup):
    """Resetting demo data is a developer command (python -m paymind.mock_paypal.reset), not an app feature."""
    client, _, _ = setup
    assert client.post("/api/paypal/reset", headers=login(client, "asha")).status_code in (404, 405)


# ---- data scoping ----------------------------------------------------------------------------------

def test_accountant_sees_the_whole_account(setup):
    client, _, _ = setup
    data = client.get("/api/paypal/overview", headers=login(client, "asha")).json()
    assert data["role"] == "accountant" and data["balance"]["currency_code"] == "USD"
    assert len(data["disputes"]) == 4 and data["transactions"]


def test_customer_sees_only_own_records(setup):
    client, _, _ = setup
    data = client.get("/api/paypal/overview", headers=login(client, "rahul")).json()
    assert data["role"] == "customer" and "balance" not in data and "transactions" not in data
    assert {d["disputed_transactions"][0]["buyer"]["payer_id"] for d in data["disputes"]} == {"user_123"}
    assert all(i["primary_recipients"][0]["billing_info"]["email_address"] == "rahul.sharma@example.com" for i in data["invoices"])


def test_audit_log_is_per_user(setup):
    client, _, services = setup
    services.appdb.log_action(services.appdb.get_user("u_asha"), "list_disputes", {}, "success")
    assert len(client.get("/api/audit", headers=login(client, "asha")).json()["items"]) == 1
    assert client.get("/api/audit", headers=login(client, "rahul")).json()["items"] == []


# ---- chat ---------------------------------------------------------------------------------------------

def test_chat_and_confirmation(setup):
    client, agent, _ = setup
    asha = login(client, "asha")
    first = client.post("/api/chat", json={"message": "hello"}, headers=asha).json()
    assert first["reply"] == "echo: hello" and "trace" not in first  # tool details live in LangSmith, not the app
    pending = client.post("/api/chat", json={"message": "refund 5", "session_id": first["session_id"]}, headers=asha).json()
    assert pending["reply"] is None and pending["confirmation"]["question"].startswith("Refund")
    done = client.post("/api/chat/confirm", json={"session_id": first["session_id"], "approve": True}, headers=asha).json()
    assert done["reply"] == "Done."


def sse_events(response):
    return [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]


def test_chat_streams_words_then_the_final_result(setup):
    client, _, _ = setup
    asha = login(client, "asha")
    r = client.post("/api/chat/stream", json={"message": "hello there"}, headers=asha)
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r)
    assert "".join(e["text"] for e in events if e["type"] == "token") == "echo: hello there "
    done = events[-1]
    assert done["type"] == "done" and done["reply"] == "echo: hello there" and done["session_id"]
    pending = sse_events(client.post("/api/chat/stream", json={"message": "refund 5", "session_id": done["session_id"]}, headers=asha))
    assert pending[-1]["confirmation"]["question"].startswith("Refund") and not [e for e in pending if e["type"] == "token"]
    final = sse_events(client.post("/api/chat/confirm/stream", json={"session_id": done["session_id"], "approve": True}, headers=asha))
    assert final[-1]["reply"] == "Done."


def test_chat_stream_needs_login(setup):
    client, _, _ = setup
    assert client.post("/api/chat/stream", json={"message": "hi"}).status_code == 401


def test_chat_threads_are_private_per_user(setup):
    """The same session id from two users maps to two different conversations."""
    client, agent, _ = setup
    client.post("/api/chat", json={"message": "a", "session_id": "shared1"}, headers=login(client, "asha"))
    client.post("/api/chat", json={"message": "b", "session_id": "shared1"}, headers=login(client, "rahul"))
    threads = {c[2] for c in agent.calls}
    assert threads == {"u_asha__shared1", "u_rahul__shared1"}


def test_bad_session_id_rejected(setup):
    client, _, _ = setup
    r = client.post("/api/chat", json={"message": "x", "session_id": "../../etc"}, headers=login(client, "asha"))
    assert r.status_code == 400


def test_web_page_is_served(setup):
    client, _, _ = setup
    assert "PayMind" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_tool_search_is_not_exposed_to_users(setup):
    """Which tools were picked is developer information: it lives in tracing, not in the app."""
    client, _, _ = setup
    assert client.get("/api/tools/search?q=refund", headers=login(client, "asha")).status_code == 404


def test_system_status_is_not_exposed_to_users(setup):
    """System health is developer information: it's in LangSmith (traces, errors, monitoring)."""
    client, _, _ = setup
    assert client.get("/api/health", headers=login(client, "asha")).status_code == 404


# ---- transactions by period -------------------------------------------------------------------------

def tx(client, headers, start, end):
    return client.get("/api/paypal/transactions", params={"start": start, "end": end}, headers=headers)


def test_transactions_for_a_month_with_totals(setup):
    client, _, _ = setup
    r = tx(client, login(client, "asha"), "2026-09-01T00:00:00+05:30", "2026-09-30T23:59:59+05:30").json()
    t = r["totals"]
    kinds = [row["transaction_info"]["transaction_event_code"] for row in r["transactions"]]
    assert t["count"] == len(kinds) and "T1107" in kinds and "T0006" in kinds  # sales and the recent refunds
    assert t["refunds"] < 0 < t["sales"]
    assert t["net"] == round(t["sales"] + t["refunds"] + t["fees"], 2)
    dates = [row["transaction_info"]["transaction_initiation_date"] for row in r["transactions"]]
    assert dates == sorted(dates, reverse=True)  # newest first


def test_transactions_for_a_single_day(setup):
    client, _, services = setup
    asha = login(client, "asha")
    month = tx(client, asha, "2026-09-01T00:00:00+00:00", "2026-09-30T23:59:59+00:00").json()
    refund_day = next(r["transaction_info"]["transaction_initiation_date"][:10] for r in month["transactions"]
                      if r["transaction_info"]["transaction_event_code"] == "T1107")
    day = tx(client, asha, f"{refund_day}T00:00:00+00:00", f"{refund_day}T23:59:59+00:00").json()
    assert day["transactions"] and all(r["transaction_info"]["transaction_initiation_date"].startswith(refund_day) for r in day["transactions"])
    assert any(r["transaction_info"]["transaction_event_code"] == "T1107" for r in day["transactions"])
    assert day["totals"]["count"] < month["totals"]["count"]


def test_transactions_rules(setup):
    client, _, _ = setup
    asha = login(client, "asha")
    assert tx(client, asha, "2026-06-01T00:00:00+05:30", "2026-09-01T00:00:00+05:30").status_code == 400  # > 31 days
    assert tx(client, asha, "2026-09-10T00:00:00+05:30", "2026-09-01T00:00:00+05:30").status_code == 400  # end before start
    assert tx(client, asha, "2026-09-01T00:00:00", "2026-09-02T00:00:00").status_code == 400             # no time zone
    assert tx(client, login(client, "rahul"), "2026-09-01T00:00:00+05:30", "2026-09-02T00:00:00+05:30").status_code == 403


# ---- dispute conversations ----------------------------------------------------------------------------

def dispute_of(services, payer_id):
    rows = services.executor.execute("list_disputes", {}).body["items"]
    return next(d["dispute_id"] for d in rows if d["disputed_transactions"][0]["buyer"]["payer_id"] == payer_id)


def test_customer_and_shop_message_each_other(setup):
    client, _, services = setup
    rahul, asha = login(client, "rahul"), login(client, "asha")
    dispute = dispute_of(services, "user_123")

    thread = client.get(f"/api/disputes/{dispute}", headers=rahul).json()
    assert thread["can_reply"] and thread["messages"][0]["from"] == "BUYER"   # his opening claim

    sent = client.post(f"/api/disputes/{dispute}/messages", json={"message": "Any update on my order?"}, headers=rahul).json()
    assert sent["messages"][-1] == {**sent["messages"][-1], "from": "BUYER", "text": "Any update on my order?"}

    # Asha sees one new message from the customer, then it's cleared once she opens the thread
    unread = {d["dispute_id"]: d["unread"] for d in client.get("/api/paypal/overview", headers=asha).json()["disputes"]}
    assert unread[dispute] >= 1
    client.get(f"/api/disputes/{dispute}", headers=asha)
    assert {d["dispute_id"]: d["unread"] for d in client.get("/api/paypal/overview", headers=asha).json()["disputes"]}[dispute] == 0

    reply = client.post(f"/api/disputes/{dispute}/messages", json={"message": "Shipped 20 Sep, tracking 1Z999."}, headers=asha).json()
    assert reply["messages"][-1]["from"] == "SELLER" and reply["messages"][-1]["name"] == "PayMind Demo Store"
    rahul_view = {d["dispute_id"]: d["unread"] for d in client.get("/api/paypal/overview", headers=rahul).json()["disputes"]}
    assert rahul_view[dispute] == 1   # the shop's reply is new for Rahul

    log = services.appdb.recent_actions("u_rahul")
    assert log[0]["tool"] == "send_message_about_dispute_to_other_party" and log[0]["status"] == "success"


def test_customer_cannot_open_or_message_someone_elses_dispute(setup):
    client, _, services = setup
    rahul = login(client, "rahul")
    priyas = dispute_of(services, "user_456")
    assert client.get(f"/api/disputes/{priyas}", headers=rahul).status_code == 404
    assert client.post(f"/api/disputes/{priyas}/messages", json={"message": "hi"}, headers=rahul).status_code == 404


def test_resolved_dispute_is_read_only(setup):
    client, _, services = setup
    asha = login(client, "asha")
    resolved = next(d["dispute_id"] for d in services.executor.execute("list_disputes", {}).body["items"] if d["status"] == "RESOLVED")
    assert client.get(f"/api/disputes/{resolved}", headers=asha).json()["can_reply"] is False
    assert client.post(f"/api/disputes/{resolved}/messages", json={"message": "hi"}, headers=asha).status_code == 422


def test_unknown_dispute(setup):
    client, _, _ = setup
    assert client.get("/api/disputes/PP-D-00000", headers=login(client, "asha")).status_code == 404


def test_whats_new_endpoint(setup):
    client, _, services = setup
    items = client.get("/api/whats-new", headers=login(client, "asha")).json()["items"]
    assert items and {i["kind"] for i in items} <= {"new_message", "needs_reply", "no_reply_yet", "invoice_overdue", "invoice_due"}
    rahul_items = client.get("/api/whats-new", headers=login(client, "rahul")).json()["items"]
    assert all(i["with"] == "PayMind Demo Store" for i in rahul_items)   # only his own dispute, with the shop
    assert client.get("/api/whats-new").status_code == 401


# ---- invoices: download and send ----------------------------------------------------------------------

def invoice_where(services, status=None, email=None):
    def recipient(i):
        return ((i.get("primary_recipients") or [{}])[0].get("billing_info") or {}).get("email_address")
    rows = services.executor.execute("list_invoices", {"page_size": 100}).body["items"]
    return next(i for i in rows if (status is None or i["status"] == status) and (email is None or recipient(i) == email))


def test_download_invoice_pdf(setup):
    client, _, services = setup
    inv = invoice_where(services, email="rahul.sharma@example.com")
    r = client.get(f"/api/invoices/{inv['id']}/pdf", headers=login(client, "rahul"))
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF") and f'{inv["detail"]["invoice_number"]}.pdf' in r.headers["content-disposition"]
    assert client.get(f"/api/invoices/{inv['id']}/pdf", headers=login(client, "asha")).status_code == 200


def test_customer_cannot_download_someone_elses_invoice(setup):
    client, _, services = setup
    other = invoice_where(services, email="maria@acme.example")
    assert client.get(f"/api/invoices/{other['id']}/pdf", headers=login(client, "rahul")).status_code == 404
    assert client.get("/api/invoices/INV2-NOPE/pdf", headers=login(client, "asha")).status_code == 404


def test_accountant_sends_a_draft(setup):
    client, _, services = setup
    draft = invoice_where(services, status="DRAFT", email="john@x.com")
    sent = client.post(f"/api/invoices/{draft['id']}/send", headers=login(client, "asha"))
    assert sent.status_code == 200 and sent.json()["status"] == "SENT"
    again = client.post(f"/api/invoices/{draft['id']}/send", headers=login(client, "asha"))
    assert again.status_code == 422 and "CANNOT_SEND_INVOICE" in again.json()["detail"]
    log = services.appdb.recent_actions("u_asha")
    assert [(a["tool"], a["status"]) for a in log[:2]] == [("send_invoice", "failed"), ("send_invoice", "success")]


def test_customers_cannot_send_invoices(setup):
    client, _, services = setup
    draft = invoice_where(services, status="DRAFT", email="john@x.com")
    assert client.post(f"/api/invoices/{draft['id']}/send", headers=login(client, "rahul")).status_code == 403


# ---- purchase orders ----------------------------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"fake image bytes"


def upload(client, headers, data=PNG, name="po.png", mime="image/png"):
    return client.post("/api/pos/read", files={"file": (name, data, mime)}, headers=headers)


def test_customer_uploads_checks_and_sends_a_po(setup):
    client, _, services = setup
    rahul = login(client, "rahul")
    res = upload(client, rahul).json()
    po = res["po"]
    assert res["read_ok"] and res["unclear"] == ["quantity on line 2 could be 3 or 8"]
    assert po["status"] == "draft" and po["customer_po_ref"] == "RS-2026-07" and po["requested_date"] == "2030-10-05"
    assert po["items"] == [{"name": "Wireless Headphones", "quantity": 2, "unit_price": "79.99"},   # "$79.99" cleaned
                           {"name": "Phone Case", "quantity": 3, "unit_price": None}]
    assert client.get(f"/api/pos/{po['po_id']}/document", headers=rahul).content == PNG

    # the customer fixes the unclear quantity, then sends it
    items = [{"name": "Wireless Headphones", "quantity": 2, "unit_price": "79.99"}, {"name": "Phone Case", "quantity": 8}]
    edited = client.put(f"/api/pos/{po['po_id']}", json={"items": items, "customer_po_ref": "RS-2026-07",
                                                           "requested_date": "2030-10-05"}, headers=rahul).json()
    assert edited["items"][1]["quantity"] == 8
    sent = client.post(f"/api/pos/{po['po_id']}/submit", headers=rahul).json()
    assert sent["status"] == "submitted"
    assert client.put(f"/api/pos/{po['po_id']}", json={"items": items}, headers=rahul).status_code == 409   # locked once sent


def test_shop_accepts_a_po_and_an_invoice_is_created(setup):
    client, _, services = setup
    rahul, asha = login(client, "rahul"), login(client, "asha")
    po = upload(client, rahul).json()["po"]
    client.post(f"/api/pos/{po['po_id']}/submit", headers=rahul)

    new = [i for i in client.get("/api/whats-new", headers=asha).json()["items"] if i["kind"] == "new_po"]
    assert new and new[0]["po_id"] == po["po_id"] and new[0]["with"] == "Rahul Sharma"

    unpriced = [{"name": "Wireless Headphones", "quantity": 2, "unit_price": "79.99"}, {"name": "Phone Case", "quantity": 3}]
    assert client.post(f"/api/pos/{po['po_id']}/accept", json={"items": unpriced, "expected_date": "2030-10-03"}, headers=asha).status_code == 422
    assert client.post(f"/api/pos/{po['po_id']}/accept", json={"items": [{**unpriced[0]}], "expected_date": "2020-01-01"}, headers=asha).status_code == 422

    priced = [unpriced[0], {**unpriced[1], "unit_price": "19.99"}]
    accepted = client.post(f"/api/pos/{po['po_id']}/accept", json={"items": priced, "expected_date": "2030-10-03"}, headers=asha).json()
    assert accepted["status"] == "accepted" and accepted["expected_date"] == "2030-10-03" and accepted["total"] == "219.95"

    invoice = services.executor.execute("show_invoice_details", {"invoice_id": accepted["invoice_id"]}).body
    assert invoice["status"] == "DRAFT" and invoice["amount"]["value"] == "219.95"
    assert invoice["primary_recipients"][0]["billing_info"]["email_address"] == "rahul.sharma@example.com"
    assert "RS-2026-07" in invoice["detail"]["note"] and "2030-10-03" in invoice["detail"]["note"]
    assert client.get(f"/api/invoices/{accepted['invoice_id']}/pdf", headers=rahul).status_code == 200   # Rahul can download it

    mine = client.get("/api/whats-new", headers=rahul).json()["items"]
    assert any(i["kind"] == "po_accepted" and i["expected_date"] == "2030-10-03" for i in mine)
    assert client.post(f"/api/pos/{po['po_id']}/accept", json={"items": priced, "expected_date": "2030-10-03"}, headers=asha).status_code == 409


def test_shop_declines_a_po(setup):
    client, _, _ = setup
    rahul, asha = login(client, "rahul"), login(client, "asha")
    po = upload(client, rahul).json()["po"]
    client.post(f"/api/pos/{po['po_id']}/submit", headers=rahul)
    declined = client.post(f"/api/pos/{po['po_id']}/reject", json={"reason": "Out of stock until November"}, headers=asha).json()
    assert declined["status"] == "rejected"
    assert any(i["kind"] == "po_rejected" and "November" in i["text"] for i in client.get("/api/whats-new", headers=rahul).json()["items"])


def test_po_privacy_and_roles(setup):
    client, _, _ = setup
    rahul, priya, asha = login(client, "rahul"), login(client, "priya"), login(client, "asha")
    po = upload(client, rahul).json()["po"]
    assert client.get(f"/api/pos/{po['po_id']}", headers=priya).status_code == 404          # another customer
    assert client.get(f"/api/pos/{po['po_id']}/document", headers=priya).status_code == 404
    assert client.get(f"/api/pos/{po['po_id']}", headers=asha).status_code == 404           # shop can't see unsent drafts
    assert upload(client, asha).status_code == 403                                          # only customers send POs
    client.post(f"/api/pos/{po['po_id']}/submit", headers=rahul)
    assert client.get(f"/api/pos/{po['po_id']}", headers=asha).status_code == 200
    assert client.post(f"/api/pos/{po['po_id']}/accept", json={"items": [{"name": "x", "quantity": 1, "unit_price": "1"}],
                                                                "expected_date": "2030-01-01"}, headers=rahul).status_code == 403


def test_po_upload_rules_and_fallback(setup):
    client, _, _ = setup
    rahul = login(client, "rahul")
    assert upload(client, rahul, b"hello", "po.txt", "text/plain").status_code == 415
    assert upload(client, rahul, b"", "po.png", "image/png").status_code == 400
    fallback = upload(client, rahul, PNG + b"unreadable").json()                              # AI unavailable
    assert fallback["read_ok"] is False and fallback["po"]["items"] == [] and fallback["po"]["status"] == "draft"
    typed = client.post("/api/pos", json={"items": [{"name": "USB-C Charger", "quantity": 4}]}, headers=rahul).json()
    assert typed["status"] == "draft" and not typed["has_document"]
    empty = client.post("/api/pos", json={"items": []}, headers=rahul).json()
    assert client.post(f"/api/pos/{empty['po_id']}/submit", headers=rahul).status_code == 422


# ---- order journey: pay → ship → delivered / not received ----------------------------------------------

def accepted_po(client, expected="2030-10-03"):
    rahul, asha = login(client, "rahul"), login(client, "asha")
    po = upload(client, rahul).json()["po"]
    client.post(f"/api/pos/{po['po_id']}/submit", headers=rahul)
    items = [{"name": "Wireless Headphones", "quantity": 2, "unit_price": "79.99"}, {"name": "Phone Case", "quantity": 3, "unit_price": "19.99"}]
    return client.post(f"/api/pos/{po['po_id']}/accept", json={"items": items, "expected_date": expected}, headers=asha).json(), rahul, asha


def kinds_for(client, headers, po_id):
    return [i["kind"] for i in client.get("/api/whats-new", headers=headers).json()["items"] if i.get("po_id") == po_id]


def test_full_order_journey(setup):
    client, _, services = setup
    po, rahul, asha = accepted_po(client)
    pid, inv = po["po_id"], po["invoice_id"]
    assert kinds_for(client, asha, pid) == ["po_send_invoice"]

    assert client.post(f"/api/invoices/{inv}/pay", headers=rahul).status_code == 409        # draft: nothing to pay yet
    client.post(f"/api/invoices/{inv}/send", headers=asha)
    assert client.get(f"/api/pos/{pid}", headers=rahul).json()["status"] == "invoiced"
    assert kinds_for(client, rahul, pid) == ["po_pay"]

    paid = client.post(f"/api/invoices/{inv}/pay", headers=rahul).json()
    assert paid["status"] == "PAID"
    now = client.get(f"/api/pos/{pid}", headers=rahul).json()
    assert now["status"] == "paid" and now["paid_at"]
    assert kinds_for(client, asha, pid) == ["po_ship"]                                      # shop: ship it by the date
    ship = next(i for i in client.get("/api/whats-new", headers=asha).json()["items"] if i.get("po_id") == pid)
    assert ship["expected_date"] == "2030-10-03" and ship["days_left"] > 0

    assert client.post(f"/api/pos/{pid}/ship", json={"carrier": "FedEx", "tracking_number": "1Z999"}, headers=rahul).status_code == 403
    shipped = client.post(f"/api/pos/{pid}/ship", json={"carrier": "FedEx", "tracking_number": "1Z999"}, headers=asha).json()
    assert shipped["status"] == "shipped" and shipped["tracking_number"] == "1Z999"
    track = next(i for i in client.get("/api/whats-new", headers=rahul).json()["items"] if i.get("po_id") == pid)
    assert track["kind"] == "po_shipped" and track["carrier"] == "FedEx"                   # date not reached: tracking

    assert client.post(f"/api/pos/{pid}/delivered", headers=asha).status_code == 403         # only the customer confirms
    done = client.post(f"/api/pos/{pid}/delivered", headers=rahul).json()
    assert done["status"] == "delivered" and done["delivered_at"]
    assert kinds_for(client, rahul, pid) == [] and kinds_for(client, asha, pid) == []        # nothing left to do
    tools = [a["tool"] for a in services.appdb.recent_actions("u_rahul", limit=10)]
    assert {"pay_invoice", "confirm_delivery", "submit_purchase_order"} <= set(tools)


def test_delivery_date_reached_asks_the_customer(setup):
    from datetime import date, timedelta
    client, _, _ = setup
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    po, rahul, asha = accepted_po(client, expected=tomorrow)
    pid = po["po_id"]
    client.post(f"/api/invoices/{po['invoice_id']}/send", headers=asha)
    client.post(f"/api/invoices/{po['invoice_id']}/pay", headers=rahul)
    client.post(f"/api/pos/{pid}/ship", json={"carrier": "DHL", "tracking_number": "JD014600"}, headers=asha)
    # pretend the delivery date has come
    client.app.state.services.appdb.update_po(pid, expected_date=date.today().isoformat())
    assert kinds_for(client, rahul, pid) == ["po_confirm"]                                  # "has it arrived?"


def test_not_received_goes_back_to_the_shop(setup):
    client, _, _ = setup
    po, rahul, asha = accepted_po(client)
    pid = po["po_id"]
    client.post(f"/api/invoices/{po['invoice_id']}/send", headers=asha)
    client.post(f"/api/invoices/{po['invoice_id']}/pay", headers=rahul)
    client.post(f"/api/pos/{pid}/ship", json={"carrier": "FedEx", "tracking_number": "1Z999"}, headers=asha)
    missing = client.post(f"/api/pos/{pid}/not-received", json={"note": "Tracking says delivered but nothing came"}, headers=rahul).json()
    assert missing["status"] == "not_received"
    alert = next(i for i in client.get("/api/whats-new", headers=asha).json()["items"] if i.get("po_id") == pid)
    assert alert["kind"] == "po_not_received" and "nothing came" in alert["text"]
    reship = client.post(f"/api/pos/{pid}/ship", json={"carrier": "FedEx", "tracking_number": "1Z888"}, headers=asha).json()
    assert reship["status"] == "shipped" and reship["tracking_number"] == "1Z888"             # shop ships again


def test_order_steps_in_the_wrong_order_are_refused(setup):
    client, _, _ = setup
    po, rahul, asha = accepted_po(client)
    pid = po["po_id"]
    assert client.post(f"/api/pos/{pid}/ship", json={"carrier": "FedEx", "tracking_number": "1Z999"}, headers=asha).status_code == 409  # not paid
    assert client.post(f"/api/pos/{pid}/delivered", headers=rahul).status_code == 409                                                  # not shipped
    assert client.post(f"/api/invoices/{po['invoice_id']}/pay", headers=login(client, "priya")).status_code == 404                     # not hers


# ---- resolving a dispute (shop) -------------------------------------------------------------------------

def test_resolve_refund_clears_the_reminder(setup):
    client, _, services = setup
    asha = login(client, "asha")
    rid = dispute_of(services, "user_123")
    assert any(i.get("dispute_id") == rid for i in client.get("/api/whats-new", headers=asha).json()["items"])
    assert client.get(f"/api/disputes/{rid}", headers=asha).json()["can_resolve"] is True
    done = client.post(f"/api/disputes/{rid}/resolve", json={"action": "refund", "note": "Sorry!"}, headers=asha).json()
    assert done["dispute"]["status"] == "RESOLVED" and done["can_resolve"] is False and done["can_reply"] is False
    assert not any(i.get("dispute_id") == rid for i in client.get("/api/whats-new", headers=asha).json()["items"])
    assert services.appdb.recent_actions("u_asha")[0]["tool"] == "accept_claim"
    assert client.post(f"/api/disputes/{rid}/resolve", json={"action": "refund"}, headers=asha).status_code == 409


def test_resolve_with_an_offer_moves_the_turn_to_the_customer(setup):
    client, _, services = setup
    asha, rahul = login(client, "asha"), login(client, "rahul")
    rid = dispute_of(services, "user_123")
    assert client.post(f"/api/disputes/{rid}/resolve", json={"action": "offer", "amount": "999"}, headers=asha).status_code == 422
    res = client.post(f"/api/disputes/{rid}/resolve", json={"action": "offer", "amount": "20"}, headers=asha).json()
    assert res["dispute"]["status"] == "WAITING_FOR_BUYER_RESPONSE" and res["offer"]["offer_amount"]["value"] == "20.00"
    mine = {i["dispute_id"]: i for i in client.get("/api/whats-new", headers=asha).json()["items"] if i.get("dispute_id")}
    assert rid not in mine or not mine[rid].get("action_needed")            # no longer the shop's turn
    his = {i["dispute_id"]: i for i in client.get("/api/whats-new", headers=rahul).json()["items"] if i.get("dispute_id")}
    assert his[rid].get("action_needed")                                    # now Rahul's turn


def test_disputes_stay_between_shop_and_customer(setup):
    """No PayPal review: 'send evidence' isn't a resolution, and the PayPal-review tools are switched off."""
    client, _, services = setup
    asha = login(client, "asha")
    rid = dispute_of(services, "user_123")
    assert client.post(f"/api/disputes/{rid}/resolve", json={"action": "evidence"}, headers=asha).status_code == 422
    off = {"escalate_dispute_to_claim", "provide_evidence", "provide_supporting_information_for_dispute",
           "appeal_dispute", "settle_dispute", "update_dispute_status"}
    assert all(REGISTRY.get(n)["allowed_roles"] == [] and REGISTRY.get(n)["disabled"] for n in off)


def test_only_the_shop_can_resolve(setup):
    client, _, services = setup
    rid = dispute_of(services, "user_123")
    assert client.post(f"/api/disputes/{rid}/resolve", json={"action": "refund"}, headers=login(client, "rahul")).status_code == 403


def test_customer_sees_the_shops_offer_in_whats_new(setup):
    client, _, services = setup
    asha, rahul = login(client, "asha"), login(client, "rahul")
    rid = dispute_of(services, "user_123")
    client.post(f"/api/disputes/{rid}/resolve", json={"action": "offer", "amount": "20", "note": "Sorry for the delay"}, headers=asha)
    item = next(i for i in client.get("/api/whats-new", headers=rahul).json()["items"] if i.get("dispute_id") == rid)
    assert item["action_needed"] and item["offer"]["amount"]["value"] == "20.00" and item["offer"]["note"] == "Sorry for the delay"


# ---- dispute photos and replacements ------------------------------------------------------------

PHOTO = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def rahuls_dispute(client, headers):
    return next(d["dispute_id"] for d in client.get("/api/paypal/overview", headers=headers).json()["disputes"])


def test_customer_attaches_a_photo_the_shop_can_see(setup):
    client, _, _ = setup
    rahul, asha, priya = login(client, "rahul"), login(client, "asha"), login(client, "priya")
    dispute_id = rahuls_dispute(client, rahul)
    r = client.post(f"/api/disputes/{dispute_id}/photos", headers=rahul, files={"file": ("broken.png", PHOTO, "image/png")})
    assert r.status_code == 200
    body = r.json()
    photo_id = body["photos"][0]["photo_id"]
    assert body["photos"][0]["by"] == "BUYER" and body["messages"][-1]["text"].startswith("📎")
    assert client.get(f"/api/disputes/{dispute_id}/photos/{photo_id}", headers=asha).content == PHOTO    # the shop sees it
    assert client.get(f"/api/disputes/{dispute_id}/photos/{photo_id}", headers=priya).status_code == 404  # other customers don't


def test_photo_upload_checks_type_and_owner(setup):
    client, _, _ = setup
    rahul, priya = login(client, "rahul"), login(client, "priya")
    dispute_id = rahuls_dispute(client, rahul)
    assert client.post(f"/api/disputes/{dispute_id}/photos", headers=rahul,
                       files={"file": ("x.pdf", b"%PDF", "application/pdf")}).status_code == 415
    assert client.post(f"/api/disputes/{dispute_id}/photos", headers=priya,
                       files={"file": ("x.png", PHOTO, "image/png")}).status_code == 404


def test_shop_resolves_with_a_replacement(setup):
    client, _, _ = setup
    rahul, asha = login(client, "rahul"), login(client, "asha")
    dispute_id = rahuls_dispute(client, rahul)
    missing = client.post(f"/api/disputes/{dispute_id}/resolve", headers=asha, json={"action": "replacement", "carrier": "Blue Dart"})
    assert missing.status_code == 422
    r = client.post(f"/api/disputes/{dispute_id}/resolve", headers=asha,
                    json={"action": "replacement", "carrier": "Blue Dart", "tracking_number": "BD123456789IN"})
    d = r.json()["dispute"]
    assert d["status"] == "RESOLVED" and d["dispute_outcome"]["outcome_code"] == "RESOLVED_WITH_REPLACEMENT"
    assert "BD123456789IN" in r.json()["messages"][-1]["text"]
    assert client.post(f"/api/disputes/{dispute_id}/resolve", headers=rahul, json={"action": "replacement",
                       "carrier": "x" * 3, "tracking_number": "y" * 5}).status_code == 403  # customers can't


def test_case_shows_the_purchase_and_its_shipment(setup):
    client, _, _ = setup
    rahul, asha = login(client, "rahul"), login(client, "asha")
    dispute_id = rahuls_dispute(client, rahul)
    p = client.get(f"/api/disputes/{dispute_id}", headers=rahul).json()["purchase"]
    assert p["items"] == ["Wireless Headphones × 1"] and p["amount"] == "79.99" and p["paid_how"] == "online store"
    assert p["shipments"][0]["carrier"] == "Blue Dart" and p["shipments"][0]["status"] == "SHIPPED"   # seeded: shipped 19 Sept


def test_shop_records_tracking_both_sides_see_it(setup):
    client, _, _ = setup
    rahul, asha = login(client, "rahul"), login(client, "asha")
    dispute_id = rahuls_dispute(client, rahul)
    r = client.post(f"/api/disputes/{dispute_id}/tracking", headers=asha,
                    json={"carrier": "Blue Dart", "tracking_number": "BD240919RS1IN", "status": "DELIVERED"})
    assert r.status_code == 200 and r.json()["purchase"]["shipments"][0]["status"] == "DELIVERED"
    assert client.get(f"/api/disputes/{dispute_id}", headers=rahul).json()["purchase"]["shipments"][0]["status"] == "DELIVERED"
    assert client.post(f"/api/disputes/{dispute_id}/tracking", headers=rahul,
                       json={"carrier": "X Co", "tracking_number": "123", "status": "SHIPPED"}).status_code == 403


# ---- closing a case, and telling the other side ---------------------------------------------------

def whats_new_kinds(client, headers):
    return [(i["kind"], i.get("dispute_id")) for i in client.get("/api/whats-new", headers=headers).json()["items"]]


def test_customer_closes_case_and_shop_is_told_until_it_looks(setup):
    client, _, _ = setup
    rahul, asha = login(client, "rahul"), login(client, "asha")
    dispute_id = rahuls_dispute(client, rahul)
    r = client.post(f"/api/disputes/{dispute_id}/close", headers=rahul, json={"message": "Got it, thanks!"})
    d = r.json()["dispute"]
    assert d["status"] == "RESOLVED" and d["dispute_outcome"]["outcome_code"] == "CANCELED_BY_BUYER"
    assert ("case_closed", dispute_id) in whats_new_kinds(client, asha)            # the shop is told
    assert ("case_closed", dispute_id) not in whats_new_kinds(client, rahul)       # not the one who closed it
    client.get(f"/api/disputes/{dispute_id}", headers=asha)                         # Asha opens it...
    assert ("case_closed", dispute_id) not in whats_new_kinds(client, asha)        # ...and it's no longer new
    assert client.post(f"/api/disputes/{dispute_id}/close", headers=rahul).status_code == 409
    assert client.post(f"/api/disputes/{dispute_id}/close", headers=asha).status_code == 403  # customers only


def test_shop_refund_tells_the_customer(setup):
    client, _, _ = setup
    rahul, asha = login(client, "rahul"), login(client, "asha")
    dispute_id = rahuls_dispute(client, rahul)
    client.post(f"/api/disputes/{dispute_id}/resolve", headers=asha, json={"action": "refund"})
    kinds = whats_new_kinds(client, rahul)
    assert ("case_closed", dispute_id) in kinds and any(k == "refunded" for k, _ in kinds)
    item = next(i for i in client.get("/api/whats-new", headers=rahul).json()["items"] if i["kind"] == "case_closed")
    assert item["outcome"] == "refunded" and item["amount_refunded"]["value"] == "79.99"
