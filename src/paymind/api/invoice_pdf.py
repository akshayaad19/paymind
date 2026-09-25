"""Turn a PayPal invoice (JSON) into a downloadable PDF."""

from __future__ import annotations

from decimal import Decimal

from fpdf import FPDF

STATUS_WORDS = {
    "DRAFT": "Draft", "SENT": "Awaiting payment", "UNPAID": "Awaiting payment", "SCHEDULED": "Scheduled",
    "PARTIALLY_PAID": "Partly paid", "PAID": "Paid", "MARKED_AS_PAID": "Paid", "CANCELLED": "Cancelled",
}
INK, MUTED, LINE, ACCENT = (23, 26, 43), (107, 113, 137), (228, 231, 240), (79, 70, 229)


def _money(obj: dict | None) -> str:
    value = Decimal(str((obj or {}).get("value", "0")))
    return f"{(obj or {}).get('currency_code', 'USD')} {value:,.2f}"


def _latin(text: str) -> str:
    """The built-in PDF fonts only cover Latin-1; swap anything else for '?' rather than failing."""
    return str(text or "").encode("latin-1", "replace").decode("latin-1")


def invoice_pdf(invoice: dict) -> bytes:
    detail = invoice.get("detail") or {}
    invoicer = invoice.get("invoicer") or {}
    billing = ((invoice.get("primary_recipients") or [{}])[0].get("billing_info") or {})
    name = billing.get("name") or {}
    bill_to = " ".join(x for x in (name.get("given_name"), name.get("surname")) if x) or billing.get("email_address") or "-"

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)

    # header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, _latin(invoicer.get("business_name") or "Invoice"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5, _latin(invoicer.get("email_address") or ""), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_text_color(*INK)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 8, _latin(f"Invoice {detail.get('invoice_number', '')}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    meta = [("Status", STATUS_WORDS.get(invoice.get("status"), invoice.get("status", ""))),
            ("Issued", detail.get("invoice_date", "-")),
            ("Due", (detail.get("payment_term") or {}).get("due_date", "-")),
            ("Reference", invoice.get("id", ""))]
    for label, value in meta:
        pdf.cell(28, 6, label)
        pdf.set_text_color(*INK)
        pdf.cell(0, 6, _latin(value), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*MUTED)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "BILL TO", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*INK)
    pdf.cell(0, 6, _latin(bill_to), new_x="LMARGIN", new_y="NEXT")
    if billing.get("email_address") and billing.get("email_address") != bill_to:
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 6, _latin(billing["email_address"]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # items
    widths = (94, 18, 30, 32)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*MUTED)
    pdf.set_draw_color(*LINE)
    for label, w, align in zip(("ITEM", "QTY", "UNIT PRICE", "AMOUNT"), widths, "LRRR"):
        pdf.cell(w, 8, label, border="B", align=align)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*INK)
    for item in invoice.get("items") or []:
        qty = Decimal(str(item.get("quantity", "1")))
        unit = item.get("unit_amount") or {}
        total = {"currency_code": unit.get("currency_code", "USD"), "value": str(Decimal(str(unit.get("value", "0"))) * qty)}
        row = (_latin(item.get("name", "Item")), f"{qty.normalize()}", _money(unit), _money(total))
        for text, w, align in zip(row, widths, "LRRR"):
            pdf.cell(w, 8, text, border="B", align=align)
        pdf.ln()
    pdf.ln(4)

    # totals
    paid = (invoice.get("payments") or {}).get("paid_amount")
    totals = [("Total", _money(invoice.get("amount")), True)]
    if paid and Decimal(str(paid.get("value", "0"))) > 0:
        totals.append(("Paid", _money(paid), False))
    totals.append(("Amount due", _money(invoice.get("due_amount")), True))
    for label, value, bold in totals:
        pdf.set_font("Helvetica", "B" if bold else "", 11)
        pdf.cell(sum(widths[:3]), 7, label, align="R")
        pdf.cell(widths[3], 7, value, align="R", new_x="LMARGIN", new_y="NEXT")

    if detail.get("note"):
        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 6, "NOTE", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 5, _latin(detail["note"]))

    return bytes(pdf.output())
