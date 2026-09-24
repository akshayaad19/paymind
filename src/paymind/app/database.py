"""Our app's own database: who uses the assistant, and what they did.

Separate from the mock PayPal database. PayPal only knows the shop's account;
knowing which person is chatting, their role, and recording every action is
the app's job.

Tables:
  users      one row per person who can log in. role = accountant | customer.
             A customer is linked to their PayPal payer_id, so they only ever
             see their own records. Passwords are stored only as salted
             scrypt hashes, never in plain text.
  audit_log  one row per tool call the agent tries: who, which tool, the
             parameters, the outcome. System Search reads it to answer
             "what's the status of my last request?".
  dispute_reads  how many messages of each dispute's thread each user has
             seen, so the page can flag new messages from the other side.

Files (same pattern as the mock):
  data/app/initial.db  starting data: demo users (committed)
  data/app/app.db      working copy (git-ignored), created from initial.db
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
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
    password_hash TEXT,                     -- scrypt$salt$hash; NULL = can't log in
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

CREATE TABLE IF NOT EXISTS dispute_reads (   -- how much of each dispute's thread each user has seen
    user_id     TEXT NOT NULL REFERENCES users(user_id),
    dispute_id  TEXT NOT NULL,
    last_read   TEXT NOT NULL,
    seen_count  INTEGER NOT NULL DEFAULT 0,  -- messages seen; threads only grow, so the rest are new
    PRIMARY KEY (user_id, dispute_id)
);
"""


SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored or not stored.startswith("scrypt$"):
        return False
    _, salt_hex, digest_hex = stored.split("$")
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)


# Used when the email doesn't exist, so a wrong email takes as long as a wrong password
# (timing doesn't reveal which emails have accounts).
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


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
        columns = {r[1] for r in self.conn.execute("PRAGMA table_info(dispute_reads)")}
        if "seen_count" not in columns:  # databases created before this column existed
            self.conn.execute("ALTER TABLE dispute_reads ADD COLUMN seen_count INTEGER NOT NULL DEFAULT 0")

    def close(self) -> None:
        self.conn.close()

    # ---- users -------------------------------------------------------------

    def add_user(self, user_id: str, name: str, email: str, role: str, payer_id: str | None = None,
                 password: str | None = None) -> User:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if role == "customer" and not payer_id:
            raise ValueError("a customer must be linked to a PayPal payer_id")
        with self.conn:
            self.conn.execute(
                "INSERT INTO users (user_id, name, email, role, payer_id, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, name, email.lower(), role, payer_id, hash_password(password) if password else None, now_iso()),
            )
        return User(user_id, name, email.lower(), role, payer_id)

    def set_password(self, user_id: str, password: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (hash_password(password), user_id))

    def authenticate(self, email: str, password: str) -> User | None:
        """The user if email + password match, else None. Same work either way (no timing hint)."""
        row = self.conn.execute(
            "SELECT user_id, name, email, role, payer_id, password_hash FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        stored = row["password_hash"] if row else _DUMMY_HASH
        ok = verify_password(password, stored)
        if not row or not ok:
            return None
        return User(row["user_id"], row["name"], row["email"], row["role"], row["payer_id"])

    def get_user(self, user_id: str) -> User | None:
        row = self.conn.execute("SELECT user_id, name, email, role, payer_id FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return User(**dict(row)) if row else None

    def list_users(self) -> list[User]:
        rows = self.conn.execute("SELECT user_id, name, email, role, payer_id FROM users ORDER BY role, name")
        return [User(**dict(r)) for r in rows]

    # ---- dispute message read markers -----------------------------------------

    def mark_dispute_read(self, user_id: str, dispute_id: str, seen_count: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO dispute_reads (user_id, dispute_id, last_read, seen_count) VALUES (?, ?, ?, ?)",
                (user_id, dispute_id, now_iso(), seen_count))

    def dispute_reads(self, user_id: str) -> dict[str, int]:
        """dispute_id -> number of messages this user has seen."""
        return dict(self.conn.execute("SELECT dispute_id, seen_count FROM dispute_reads WHERE user_id = ?", (user_id,)).fetchall())

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
