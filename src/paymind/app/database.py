"""Our app's own database: who uses the assistant, and what they did.

Separate from the mock PayPal database. PayPal only knows the shop's account;
knowing which person is chatting, their role, and recording every action is
the app's job.

Tables:
  users      one row per person who can log in. role = accountant | customer.
             A customer is linked to their PayPal payer_id, so they only ever
             see their own records.
  audit_log  one row per tool call the agent tries: who, which tool, the
             parameters, the outcome. System Search reads it to answer
             "what's the status of my last request?".

Files (same pattern as the mock):
  data/app/initial.db  starting data: demo users (committed)
  data/app/app.db      working copy (git-ignored), created from initial.db
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
INITIAL_DB = ROOT / "data/app/initial.db"
DEFAULT_DB = ROOT / "data/app/app.db"

ROLES = ("customer", "accountant")
STATUSES = ("success", "failed", "pending_confirmation", "declined", "blocked")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    role        TEXT NOT NULL CHECK (role IN ('customer', 'accountant')),
    payer_id    TEXT,                       -- PayPal payer_id; required for customers
    created_at  TEXT NOT NULL,
    CHECK (role = 'accountant' OR payer_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    time            TEXT NOT NULL,
    session_id      TEXT,
    user_id         TEXT NOT NULL REFERENCES users(user_id),
    role            TEXT NOT NULL,
    tool            TEXT NOT NULL,
    method          TEXT,
    path            TEXT,
    params          TEXT NOT NULL,          -- JSON
    status          TEXT NOT NULL CHECK (status IN ('success', 'failed', 'pending_confirmation', 'declined', 'blocked')),
    http_status     INTEGER,
    result_summary  TEXT,
    confirmed       INTEGER NOT NULL DEFAULT 0,
    request_id      TEXT                    -- PayPal-Request-Id used for the call
);
CREATE INDEX IF NOT EXISTS audit_user_time ON audit_log (user_id, time DESC);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    email: str
    role: str
    payer_id: str | None

    @property
    def is_customer(self) -> bool:
        return self.role == "customer"


class AppDatabase:
    def __init__(self, path: str | Path = DEFAULT_DB, initial: Path = INITIAL_DB):
        path = Path(path)
        if not path.exists() and initial.exists() and path != initial:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(initial, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---- users -------------------------------------------------------------

    def add_user(self, user_id: str, name: str, email: str, role: str, payer_id: str | None = None) -> User:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if role == "customer" and not payer_id:
            raise ValueError("a customer must be linked to a PayPal payer_id")
        with self.conn:
            self.conn.execute(
                "INSERT INTO users (user_id, name, email, role, payer_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, name, email, role, payer_id, now_iso()),
            )
        return User(user_id, name, email, role, payer_id)

    def get_user(self, user_id: str) -> User | None:
        row = self.conn.execute("SELECT user_id, name, email, role, payer_id FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return User(**dict(row)) if row else None

    def list_users(self) -> list[User]:
        rows = self.conn.execute("SELECT user_id, name, email, role, payer_id FROM users ORDER BY role, name")
        return [User(**dict(r)) for r in rows]

    # ---- audit log ---------------------------------------------------------------

    def log_action(
        self, user: User, tool: str, params: dict[str, Any], status: str, *, session_id: str | None = None,
        method: str | None = None, path: str | None = None, http_status: int | None = None,
        result_summary: str | None = None, confirmed: bool = False, request_id: str | None = None,
    ) -> int:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self.conn:
            cur = self.conn.execute(
                """INSERT INTO audit_log (time, session_id, user_id, role, tool, method, path, params, status,
                                          http_status, result_summary, confirmed, request_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now_iso(), session_id, user.user_id, user.role, tool, method, path, json.dumps(params), status,
                 http_status, result_summary, int(confirmed), request_id),
            )
        return cur.lastrowid

    def recent_actions(
        self, user_id: str, *, limit: int = 10, tool: str | None = None,
        status: str | None = None, since: str | None = None,
    ) -> list[dict[str, Any]]:
        """A user's own history, newest first. Fixed filters only (no free-form SQL),
        and always scoped to one user, so nobody reads someone else's log."""
        sql = "SELECT * FROM audit_log WHERE user_id = ?"
        args: list[Any] = [user_id]
        if tool:
            sql += " AND tool = ?"
            args.append(tool)
        if status:
            sql += " AND status = ?"
            args.append(status)
        if since:
            sql += " AND time >= ?"
            args.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, min(limit, 100)))
        rows = self.conn.execute(sql, args).fetchall()
        return [dict(r) | {"params": json.loads(r["params"]), "confirmed": bool(r["confirmed"])} for r in rows]
