"""Read a purchase order from an uploaded document (photo or scan, often handwritten).

Gemini gets the file plus a fixed form to fill in. Code then cleans the result:
quantities must be positive whole numbers, prices numbers with 2 decimals,
dates real dates. The customer always reviews and corrects the result before
submitting, because reading handwriting is never fully reliable.
"""

from __future__ import annotations

import base64
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Callable

from pydantic import BaseModel, Field

ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "application/pdf": ".pdf"}
MAX_BYTES = 10 * 1024 * 1024

PROMPT = """This is a customer's purchase order (PO) for a shop. It may be handwritten, photographed at an angle, or a scan.
Read it and fill in the form:
- customer_po_ref: the PO number written on it, if any.
- items: every line the customer wants to buy: name, quantity (a whole number), unit_price only if a price is written (a number like "49.50", no currency sign).
- requested_delivery_date: the date they need it by, as YYYY-MM-DD, if written. Today is {today}.
- notes: any other instructions (delivery address, contact, special requests), briefly.
- unclear: short notes on anything you could not read confidently (e.g. "quantity on line 2 could be 3 or 8").
Only use what is on the document. Don't invent items, prices or dates."""


class POLine(BaseModel):
    name: str
    quantity: int = Field(ge=1)
    unit_price: str | None = None


class POExtraction(BaseModel):
    customer_po_ref: str | None = None
    items: list[POLine] = Field(default_factory=list)
    requested_delivery_date: str | None = None
    notes: str | None = None
    unclear: list[str] = Field(default_factory=list)


# (file bytes, mime type) -> POExtraction. Swappable so tests and the page work without Gemini.
Reader = Callable[[bytes, str], POExtraction]


def clean_price(value) -> str | None:
    if value in (None, ""):
        return None
    try:
        price = Decimal(re.sub(r"[^\d.]", "", str(value)))
    except InvalidOperation:
        return None
    return f"{price:.2f}" if price > 0 else None


def clean_date(value) -> str | None:
    try:
        return date.fromisoformat(str(value)[:10]).isoformat() if value else None
    except ValueError:
        return None


def clean_items(items: list[dict]) -> list[dict]:
    """Normalise item lines from the AI or from the customer's edits."""
    out = []
    for item in items or []:
        name = str(item.get("name") or "").strip()
        try:
            qty = int(Decimal(str(item.get("quantity") or 0)))
        except (InvalidOperation, ValueError):
            qty = 0
        if name and qty > 0:
            out.append({"name": name[:200], "quantity": qty, "unit_price": clean_price(item.get("unit_price"))})
    return out


def gemini_reader(models: list[str]) -> Reader:
    """Read with Gemini (it accepts images and PDFs), skipping models out of daily quota."""
    from langchain_core.messages import HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    from ..agent.factory import ModelChain, gemini_timeout

    structured = [(m, ChatGoogleGenerativeAI(model=m, max_retries=0, timeout=gemini_timeout()).with_structured_output(POExtraction))
                  for m in models]
    exhausted: dict[str, str] = {}

    def read(data: bytes, mime: str) -> POExtraction:
        message = HumanMessage(content=[
            {"type": "text", "text": PROMPT.format(today=date.today().isoformat())},
            {"type": "media", "mime_type": mime, "data": base64.b64encode(data).decode()},
        ])
        return ModelChain(structured, exhausted).invoke([message])

    return read
