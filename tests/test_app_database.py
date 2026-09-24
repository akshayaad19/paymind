import sqlite3

import pytest

from paymind.app.database import INITIAL_DB, AppDatabase


@pytest.fixture
def db(tmp_path):
    return AppDatabase(tmp_path / "app.db")  # own copy of initial.db


def test_starting_users(db):
    users = {u.user_id: u for u in db.list_users()}
    assert set(users) == {"u_asha", "u_rahul", "u_priya"}
    assert users["u_asha"].role == "accountant" and users["u_asha"].payer_id is None
    assert users["u_rahul"].is_customer and users["u_rahul"].payer_id == "user_123"


def test_customer_must_have_payer_id(db):
    with pytest.raises(ValueError):
        db.add_user("u_x", "X", "x@example.com", "customer")
    with pytest.raises(sqlite3.IntegrityError):  # the database enforces it too
        db.conn.execute("INSERT INTO users VALUES ('u_y', 'Y', 'y@example.com', 'customer', NULL, 'now')")


def test_unknown_role_rejected(db):
    with pytest.raises(ValueError):
        db.add_user("u_x", "X", "x@example.com", "admin")


def test_audit_log_is_per_user_and_newest_first(db):
    asha, rahul = db.get_user("u_asha"), db.get_user("u_rahul")
    db.log_action(asha, "list_disputes", {"dispute_state": "REQUIRED_ACTION"}, "success", http_status=200)
    db.log_action(asha, "refund_captured_payment", {"capture_id": "ABC", "amount": "40.00"}, "pending_confirmation")
    db.log_action(rahul, "show_invoice_details", {"invoice_id": "INV2-1"}, "success", http_status=200)

    asha_log = db.recent_actions("u_asha")
    assert [a["tool"] for a in asha_log] == ["refund_captured_payment", "list_disputes"]
    assert asha_log[0]["params"] == {"capture_id": "ABC", "amount": "40.00"}
    assert [a["tool"] for a in db.recent_actions("u_rahul")] == ["show_invoice_details"]  # can't see Asha's


def test_audit_log_filters(db):
    asha = db.get_user("u_asha")
    db.log_action(asha, "send_invoice", {}, "failed", http_status=422)
    db.log_action(asha, "send_invoice", {}, "success", http_status=200, confirmed=True)
    db.log_action(asha, "list_invoices", {}, "success")
    assert len(db.recent_actions("u_asha", tool="send_invoice")) == 2
    assert [a["status"] for a in db.recent_actions("u_asha", tool="send_invoice", status="success")] == ["success"]
    assert db.recent_actions("u_asha", limit=1)[0]["tool"] == "list_invoices"


def test_invalid_status_rejected(db):
    with pytest.raises(ValueError):
        db.log_action(db.get_user("u_asha"), "x", {}, "maybe")


def test_initial_db_untouched(db):
    before = INITIAL_DB.read_bytes()
    db.log_action(db.get_user("u_asha"), "list_invoices", {}, "success")
    db.add_user("u_new", "New", "new@example.com", "accountant")
    assert INITIAL_DB.read_bytes() == before
