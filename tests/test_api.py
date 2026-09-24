"""PayMind API: JWT login, role checks and data scoping.

Uses the real app database, executor and mock PayPal (in-process); the agent
and tool search are fakes so no Gemini or Qdrant is needed."""

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



def fake_search(query, role=None, k=5, include_eval_only=True):
    names = [n for n in ("refund_captured_payment", "list_disputes", "show_dispute_details")
             if role in REGISTRY.get(n)["allowed_roles"]]
    return [(n, REGISTRY.get(n)["description"]) for n in names][:k]


@pytest.fixture
def setup(tmp_path):
    mock = create_mock(db_path=tmp_path / "mock.db", slow_seconds=0)
    executor = Executor(REGISTRY, base_url="http://testserver", client=TestClient(mock), sleep=lambda s: None)
    agent = FakeAgent()
    services = Services(appdb=AppDatabase(tmp_path / "app.db"), executor=executor, registry=REGISTRY,
                        search=fake_search, agent_factory=lambda: agent, mock_base_url="http://testserver")
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
    client, _, _ = setup
    asha = login(client, "asha")
    day = tx(client, asha, "2026-09-18T00:00:00+05:30", "2026-09-18T23:59:59+05:30").json()
    assert all(row["transaction_info"]["transaction_initiation_date"].startswith(("2026-09-17T18", "2026-09-17T19", "2026-09-17T2", "2026-09-18"))
               for row in day["transactions"])
    assert any(row["transaction_info"]["transaction_event_code"] == "T1107" for row in day["transactions"])  # the 18 Sep refund


def test_transactions_rules(setup):
    client, _, _ = setup
    asha = login(client, "asha")
    assert tx(client, asha, "2026-06-01T00:00:00+05:30", "2026-09-01T00:00:00+05:30").status_code == 400  # > 31 days
    assert tx(client, asha, "2026-09-10T00:00:00+05:30", "2026-09-01T00:00:00+05:30").status_code == 400  # end before start
    assert tx(client, asha, "2026-09-01T00:00:00", "2026-09-02T00:00:00").status_code == 400             # no time zone
    assert tx(client, login(client, "rahul"), "2026-09-01T00:00:00+05:30", "2026-09-02T00:00:00+05:30").status_code == 403
