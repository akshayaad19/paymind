"""What the LLM sees: tool definitions for the offered PayPal tools plus the
always-available built-in tools.

Tool cards carry extra keys for our own code (x-in for the executor, roles,
action_type...). The LLM only gets name, description and a clean JSON Schema.
"""

from __future__ import annotations

import json
from typing import Any

FIND_TOOLS = "find_tools"
SYSTEM_SEARCH = "system_search"
CHECK_UPDATES = "check_updates"
ORDER_STATUS = "order_status"
RAG_SEARCH = "rag_search"
BUILTINS = (FIND_TOOLS, SYSTEM_SEARCH, CHECK_UPDATES, ORDER_STATUS, RAG_SEARCH)

BUILTIN_SCHEMAS = [
    {
        "name": RAG_SEARCH,
        "description": (
            "Search policy documents: the shop's own policies (returns, refunds, shipping and delivery, purchase "
            "orders, invoices and payment terms, disputes, support hours) and PayPal's (User Agreement, Purchase "
            "Protection, Seller Protection, fees, privacy, acceptable use, developer docs on disputes and invoicing). "
            "Use it for any question about rules, policies, time limits or fees. Returns the relevant passages with "
            "their source; answer only from them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question, in plain words."},
                "source": {"type": "string", "enum": ["shop", "paypal"],
                           "description": "Only the shop's own policies, or only PayPal's. Leave out to search both."},
            },
            "required": ["query"],
        },
    },
    {
        "name": ORDER_STATUS,
        "description": (
            "Purchase orders and where they are: submitted, accepted, invoiced (waiting for payment), paid "
            "(processing, with expected delivery date), shipped (carrier + tracking number), delivered, "
            "not received, or declined. Customers see their own; the shop sees all. Use for 'where is my "
            "order?', 'tracking number?', 'which orders do I need to ship?'."
        ),
        "parameters": {"type": "object", "properties": {"po_id": {"type": "string", "description": "Only this PO, e.g. PO-1001."}}},
    },
    {
        "name": CHECK_UPDATES,
        "description": (
            "What needs this user's attention on open disputes: new_message (the other side wrote and it's unread), "
            "needs_reply (they wrote, no answer yet from this user) and no_reply_yet (this user wrote days ago and "
            "is still waiting); plus purchase orders: new_po (shop: a customer sent a PO) and po_accepted / "
            "po_rejected (customer: the shop's decision, with expected delivery date). Each item has its text and time. Use it for 'anything new?', "
            "'any messages?', 'did the shop reply?'. Listing here doesn't mark messages as read; opening a "
            "dispute with show_dispute_details does."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": FIND_TOOLS,
        "description": (
            "Search the full catalog of PayPal tools when none of the tools you have fits the task, "
            "or when the next step needs a different tool. Returns tool names you can call next."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What you need to do, in plain words."}},
            "required": ["query"],
        },
    },
    {
        "name": SYSTEM_SEARCH,
        "description": (
            "Questions about this assistant itself. mode='capabilities': what can the assistant do "
            "(e.g. 'what can you do with invoices?'). mode='activity': this user's own past requests "
            "and their status (e.g. 'what's the status of my last request?')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["capabilities", "activity"]},
                "query": {"type": "string", "description": "For capabilities: the topic, e.g. 'invoices'."},
                "tool": {"type": "string", "description": "For activity: only this tool name."},
                "status": {"type": "string", "enum": ["success", "failed", "declined", "blocked"]},
                "limit": {"type": "integer", "description": "For activity: how many past requests (default 5)."},
            },
            "required": ["mode"],
        },
    },
]


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop our private keys and fill gaps some LLM APIs reject (e.g. arrays without item types)."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "x-in" or key.startswith("x-"):
            continue
        if key == "properties":
            out[key] = {name: clean_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = clean_schema(value) if value else {"type": "string"}
        else:
            out[key] = value
    if out.get("type") == "array" and "items" not in out:
        out["items"] = {"type": "string"}
    if "required" in out:
        out["required"] = [r for r in out["required"] if r in out.get("properties", {})]
        if not out["required"]:
            del out["required"]
    return out


def tool_schema(card: dict) -> dict[str, Any]:
    """Description + the card's required fields and a filled-in example call, so the LLM
    knows what a minimal correct call looks like without exploring."""
    description = card["description"]
    if card.get("action_type") == "write":
        description += " (Changes data; the user is asked to confirm automatically.)"
    if card.get("required_params"):
        description += f" Required: {', '.join(card['required_params'])}."
    if card.get("example_call"):
        example = json.dumps(card["example_call"], separators=(",", ":"))
        description += f" Example call (replace the values with the user's real details): {example}"
    return {"name": card["name"], "description": description, "parameters": clean_schema(card["parameters"])}
