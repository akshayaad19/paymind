"""Executor: turn a tool call into a real HTTP request and send it.

    refund_captured_payment(capture_id="ABC", amount={...})
        │  look up the tool card in tools.json
        │  each parameter's "x-in" says where it goes: path / query / body
        ▼
    POST {PAYPAL_BASE_URL}/v2/payments/captures/ABC/refund
         body {"amount": {...}}
         PayPal-Request-Id: pm-...   (writes only; the SAME id on every retry)

Retries only what can succeed on retry: timeouts, connection errors, 429 and
5xx, waiting 1s, 2s, 4s. A 4xx is never retried; its PayPal error is turned
into one plain sentence for the LLM. Reusing the request id on retries means a
refund that went through but timed out is not done twice.

The executor does no permission or safety checks; the validator (5c) runs
before it.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[3]
RETRY_STATUSES = {429, 500, 502, 503, 504}
DEFAULT_WAITS = (1.0, 2.0, 4.0)


class ToolRegistry:
    """All tool cards from tools.json, by name."""

    def __init__(self, path: Path = ROOT / "data/tools/tools.json"):
        self.tools: dict[str, dict] = {t["name"]: t for t in json.loads(Path(path).read_text())}

    def get(self, name: str) -> dict | None:
        return self.tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __len__(self) -> int:
        return len(self.tools)


@dataclass
class HttpRequest:
    method: str
    path: str
    query: dict[str, Any] = field(default_factory=dict)
    body: dict[str, Any] | None = None


@dataclass
class ExecutionResult:
    ok: bool
    tool: str
    method: str
    path: str
    status_code: int | None          # None if PayPal never answered
    body: Any                        # parsed JSON (or text) from PayPal
    error: str | None                # one plain sentence when ok is False
    attempts: int
    request_id: str | None           # PayPal-Request-Id sent (writes only)
    replayed: bool = False           # PayPal returned a remembered answer for this request id


def build_request(card: dict, params: dict[str, Any]) -> HttpRequest:
    """Place each parameter where the card says: path, query or body."""
    properties = card["parameters"].get("properties", {})
    path = card["path"]
    query: dict[str, Any] = {}
    body: dict[str, Any] = {}
    for name, value in params.items():
        where = properties.get(name, {}).get("x-in", "body")
        if where == "path":
            path = path.replace("{" + name + "}", quote(str(value), safe=""))
        elif where == "query":
            query[name] = value
        else:
            body[name] = value
    missing = [seg for seg in path.split("/") if seg.startswith("{") and seg.endswith("}")]
    if missing:
        raise ValueError(f"missing path parameter(s): {', '.join(m.strip('{}') for m in missing)}")
    send_body = card["method"] not in ("GET", "DELETE")
    return HttpRequest(card["method"], path, query, body if send_body else None)


def parse_body(response: httpx.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return response.text


def describe_error(status: int | None, body: Any) -> str:
    """PayPal error JSON → one sentence the LLM (and a person) can understand."""
    if status is None:
        return "PayPal did not respond (timed out or unreachable) after several tries. The action may or may not have happened."
    if isinstance(body, dict):
        detail = (body.get("details") or [{}])[0]
        issue = detail.get("issue") or body.get("name") or "ERROR"
        text = detail.get("description") or body.get("message") or ""
        where = f" (field: {detail['field']})" if detail.get("field") else ""
        return f"{status} {issue}: {text}{where}".strip()
    return f"{status}: {str(body)[:200]}"


class Executor:
    def __init__(
        self,
        registry: ToolRegistry,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
        waits: tuple[float, ...] = DEFAULT_WAITS,
        sleep: Callable[[float], None] = time.sleep,
        access_token: str | None = None,
    ):
        self.registry = registry
        self.base_url = (base_url or os.getenv("PAYPAL_BASE_URL") or "http://localhost:8000").rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.waits = waits
        self.sleep = sleep
        self.access_token = access_token or os.getenv("PAYPAL_ACCESS_TOKEN") or "mock-token"

    def execute(self, tool_name: str, params: dict[str, Any], request_id: str | None = None,
                extra_headers: dict[str, str] | None = None) -> ExecutionResult:
        card = self.registry.get(tool_name)
        if card is None:
            raise KeyError(f"unknown tool: {tool_name}")
        req = build_request(card, params)
        is_write = card["action_type"] == "write"
        if is_write:
            request_id = request_id or f"pm-{uuid.uuid4().hex}"  # one id for all retries of this action

        headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
        if is_write and request_id:
            headers["PayPal-Request-Id"] = request_id
        headers.update(extra_headers or {})

        status, body, attempts = None, None, 0
        replayed = False
        for wait in (0.0, *self.waits):
            if wait:
                self.sleep(wait)
            attempts += 1
            try:
                response = self.client.request(req.method, self.base_url + req.path, params=req.query or None,
                                               json=req.body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError):
                status, body = None, None
                continue  # no answer: try again with the same request id
            status, body = response.status_code, parse_body(response)
            replayed = response.headers.get("x-mock-idempotent-replay") == "true"
            if status not in RETRY_STATUSES:
                break

        ok = status is not None and 200 <= status < 300
        return ExecutionResult(
            ok=ok, tool=tool_name, method=req.method, path=req.path, status_code=status, body=body,
            error=None if ok else describe_error(status, body), attempts=attempts,
            request_id=request_id if is_write else None, replayed=replayed,
        )
