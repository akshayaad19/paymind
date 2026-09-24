import httpx
import pytest
from fastapi.testclient import TestClient

from paymind.agent.executor import Executor, ToolRegistry, build_request, describe_error
from paymind.mock_paypal.app import create_app

REGISTRY = ToolRegistry()


def usd(value):
    return {"currency_code": "USD", "value": value}


@pytest.fixture
def mock_app(tmp_path):
    return create_app(db_path=tmp_path / "mock.db", slow_seconds=0)


@pytest.fixture
def executor(mock_app):
    """Executor talking to a real mock PayPal server (in-process)."""
    return Executor(REGISTRY, base_url="http://testserver", client=TestClient(mock_app), sleep=lambda s: None)


def completed_capture_id(mock_app):
    return next(c["id"] for c in mock_app.state.store.db.all("captures") if c["status"] == "COMPLETED")


# ---- building requests ------------------------------------------------------------

def test_params_go_to_path_query_and_body():
    req = build_request(REGISTRY.get("refund_captured_payment"), {"capture_id": "ABC 1/2", "amount": usd("5.00")})
    assert (req.method, req.path) == ("POST", "/v2/payments/captures/ABC%201%2F2/refund")  # path values are escaped
    assert req.body == {"amount": usd("5.00")} and req.query == {}

    req = build_request(REGISTRY.get("list_disputes"), {"dispute_state": "REQUIRED_ACTION", "page_size": 5})
    assert req.path == "/v1/customer/disputes"
    assert req.query == {"dispute_state": "REQUIRED_ACTION", "page_size": 5}
    assert req.body is None  # GET sends no body


def test_missing_path_param_is_an_error():
    with pytest.raises(ValueError, match="capture_id"):
        build_request(REGISTRY.get("refund_captured_payment"), {})


def test_describe_error():
    body = {"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "REFUND_AMOUNT_EXCEEDED", "description": "Too much.", "field": "amount.value"}]}
    assert describe_error(422, body) == "422 REFUND_AMOUNT_EXCEEDED: Too much. (field: amount.value)"
    assert "did not respond" in describe_error(None, None)


# ---- against the mock server -------------------------------------------------------

def test_read_tool(executor):
    result = executor.execute("list_disputes", {"dispute_state": "REQUIRED_ACTION"})
    assert result.ok and result.status_code == 200
    assert result.body["items"][0]["disputed_transactions"][0]["buyer"]["payer_id"] == "user_123"
    assert result.request_id is None  # reads don't need one
    assert result.attempts == 1


def test_write_tool_gets_request_id(executor, mock_app):
    result = executor.execute("refund_captured_payment", {"capture_id": completed_capture_id(mock_app), "amount": usd("1.00")})
    assert result.ok and result.status_code == 201
    assert result.request_id.startswith("pm-")


def test_two_step_invoice(executor):
    draft = executor.execute("create_draft_invoice", {
        "primary_recipients": [{"billing_info": {"email_address": "john@x.com"}}],
        "items": [{"name": "Consulting", "quantity": "1", "unit_amount": usd("50.00")}],
    })
    sent = executor.execute("send_invoice", {"invoice_id": draft.body["id"]})
    assert sent.ok and sent.body["status"] == "SENT"


def test_paypal_error_becomes_one_sentence(executor, mock_app):
    result = executor.execute("refund_captured_payment", {"capture_id": completed_capture_id(mock_app), "amount": usd("99999.00")})
    assert not result.ok and result.status_code == 422
    assert result.error.startswith("422 REFUND_AMOUNT_EXCEEDED")
    assert result.attempts == 1  # 4xx is never retried


def test_same_request_id_never_refunds_twice(executor, mock_app):
    capture_id = completed_capture_id(mock_app)
    first = executor.execute("refund_captured_payment", {"capture_id": capture_id, "amount": usd("1.00")}, request_id="pm-fixed")
    second = executor.execute("refund_captured_payment", {"capture_id": capture_id, "amount": usd("1.00")}, request_id="pm-fixed")
    assert first.body["id"] == second.body["id"] and second.replayed
    assert len([r for r in mock_app.state.store.db.all("refunds") if r["capture_id"] == capture_id]) == 1


def test_example_replay_tool(executor):
    result = executor.execute("list_webhooks", {})
    assert result.ok and "webhooks" in result.body


def test_unknown_tool(executor):
    with pytest.raises(KeyError):
        executor.execute("book_a_flight", {})


# ---- retries (fake connection) --------------------------------------------------------

def scripted_client(steps):
    """A fake PayPal: each step is a status code or an exception to raise. Records request ids."""
    seen = []

    def handler(request):
        seen.append(request.headers.get("PayPal-Request-Id"))
        step = steps[len(seen) - 1]
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, json={"id": "R1"} if step < 300 else {"name": "ERR", "details": [{"issue": "X", "description": "bad"}]})

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def run(steps, tool="refund_captured_payment", params=None):
    client, seen = scripted_client(steps)
    waits = []
    ex = Executor(REGISTRY, base_url="http://paypal", client=client, sleep=waits.append)
    result = ex.execute(tool, params or {"capture_id": "ABC"})
    return result, seen, waits


def test_timeout_then_success_reuses_the_same_request_id():
    result, seen, waits = run([httpx.ReadTimeout("slow"), 503, 201])
    assert result.ok and result.attempts == 3
    assert len(set(seen)) == 1 and seen[0] == result.request_id  # same id every time: no double refund
    assert waits == [1.0, 2.0]


def test_gives_up_after_all_retries():
    result, seen, waits = run([httpx.ConnectError("down")] * 4)
    assert not result.ok and result.status_code is None and result.attempts == 4
    assert "may or may not have happened" in result.error
    assert waits == [1.0, 2.0, 4.0]


def test_client_errors_are_not_retried():
    result, seen, _ = run([404])
    assert not result.ok and result.attempts == 1 and result.error.startswith("404 X")


def test_rate_limit_is_retried():
    result, _, _ = run([429, 200], tool="list_disputes", params={})
    assert result.ok and result.attempts == 2


def test_result_records_caller_and_role_check(executor):
    from paymind.app.database import User
    asha = User("u_asha", "Asha", "a@x.com", "accountant", None)
    rahul = User("u_rahul", "Rahul", "r@x.com", "customer", "user_123")
    ok = executor.execute("list_disputes", {}, caller=rahul)
    assert ok.caller == {"user_id": "u_rahul", "role": "customer"}
    assert set(ok.allowed_roles) == {"customer", "accountant"} and ok.role_allowed is True
    refund_card_roles = REGISTRY.get("list_transactions")["allowed_roles"]
    r = executor.execute("list_transactions", {"start_date": "2026-08-01T00:00:00Z", "end_date": "2026-08-02T00:00:00Z"}, caller=rahul)
    assert r.allowed_roles == refund_card_roles and r.role_allowed is ("customer" in refund_card_roles)
    assert executor.execute("list_disputes", {}).role_allowed is None  # no caller given
    assert executor.execute("list_disputes", {}, caller=asha).role_allowed is True
