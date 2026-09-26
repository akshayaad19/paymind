"""SQLite storage for the mock PayPal server. The database is the source of truth.

Every request reads and writes rows here directly; nothing is kept in memory
between requests. Each record is one row, stored as PayPal-shaped JSON.

Files:
  data/mock/initial.db      starting data (committed). Edit it with any SQLite viewer.
  data/mock/paypal_mock.db  working copy the server changes (git-ignored).
                            Created from initial.db on first start; POST /mock/reset
                            copies initial.db over it again.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

# table -> field inside the record that holds its ID
RECORD_TABLES = {
    "customers": "payer_id",
    "captures": "id",
    "refunds": "id",
    "orders": "id",
    "invoices": "id",
    "disputes": "dispute_id",
    "trackers": "id",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS transactions (seq INTEGER PRIMARY KEY AUTOINCREMENT, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS idempotency (
    method TEXT NOT NULL, path TEXT NOT NULL, request_id TEXT NOT NULL,
    status INTEGER NOT NULL, body BLOB NOT NULL,
    PRIMARY KEY (method, path, request_id)
);
""" + "".join(f"CREATE TABLE IF NOT EXISTS {t} (id TEXT PRIMARY KEY, data TEXT NOT NULL);\n" for t in RECORD_TABLES)


def prepare(db_path: Path, initial_path: Path) -> None:
    """Create the working database from the starting data if it doesn't exist yet."""
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(initial_path, db_path)


class Database:
    def __init__(self, path: str | Path):
        # autocommit mode: the server opens and closes transactions itself (one per request)
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.executescript(SCHEMA)

    # ---- transactions ----------------------------------------------------

    def begin(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")

    def commit(self) -> None:
        self.conn.execute("COMMIT")

    def rollback(self) -> None:
        self.conn.execute("ROLLBACK")

    def close(self) -> None:
        self.conn.close()

    # ---- records (captures, refunds, orders, invoices, disputes) -----------

    def get(self, table: str, record_id: str) -> dict | None:
        row = self.conn.execute(f"SELECT data FROM {table} WHERE id = ?", (record_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, table: str) -> list[dict]:
        return [json.loads(d) for (d,) in self.conn.execute(f"SELECT data FROM {table}")]

    def put(self, table: str, record: dict) -> dict:
        """Insert or update one record."""
        self.conn.execute(f"INSERT OR REPLACE INTO {table} (id, data) VALUES (?, ?)",
                          (record[RECORD_TABLES[table]], json.dumps(record)))
        return record

    def delete(self, table: str, record_id: str) -> None:
        self.conn.execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))

    def count(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ---- ledger ------------------------------------------------------------

    def add_transaction(self, row: dict) -> None:
        self.conn.execute("INSERT INTO transactions (data) VALUES (?)", (json.dumps(row),))

    def transactions(self) -> list[dict]:
        return [json.loads(d) for (d,) in self.conn.execute("SELECT data FROM transactions ORDER BY seq")]

    # ---- idempotency ---------------------------------------------------------

    def idempotent_response(self, method: str, path: str, request_id: str) -> tuple[int, bytes] | None:
        row = self.conn.execute("SELECT status, body FROM idempotency WHERE method = ? AND path = ? AND request_id = ?",
                                (method, path, request_id)).fetchone()
        return (row[0], bytes(row[1])) if row else None

    def save_idempotent_response(self, method: str, path: str, request_id: str, status: int, body: bytes) -> None:
        self.conn.execute("INSERT OR REPLACE INTO idempotency VALUES (?, ?, ?, ?, ?)", (method, path, request_id, status, body))

    # ---- settings and counters -------------------------------------------------

    def meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value)))
