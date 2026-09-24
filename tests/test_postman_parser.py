import json
from pathlib import Path

import pytest

from paymind.ingest.postman_parser import (
    build_path,
    clean_description,
    classify_action,
    infer_schema,
    parse_collection,
    parse_json_body,
    slugify,
)

COLLECTION = Path(__file__).resolve().parents[1] / "data/postman/paypal_collection.json"


def make_collection(*items):
    return {"info": {}, "item": list(items)}


def make_request(name, method="GET", path=("v1", "things"), **request):
    return {"name": name, "request": {"method": method, "url": {"path": list(path)}, **request}, "response": []}


def test_slugify():
    assert slugify("Show invoice details") == "show_invoice_details"
    assert slugify("Onboarding (Limited Release)") == "onboarding"
    assert slugify("Confirm Payment Source - 3DS") == "confirm_payment_source_3ds"


def test_clean_description_strips_html_and_markdown():
    raw = 'Creates a draft. <a href="#x">Send it</a>.<br/><br/>Use `items` and [docs](https://x.y).'
    assert clean_description(raw) == "Creates a draft. Send it. Use items and docs."


def test_build_path_converts_variables():
    url = {"path": ["v2", "invoicing", "invoices", ":invoice_id", "send"]}
    assert build_path(url) == "/v2/invoicing/invoices/{invoice_id}/send"
    assert build_path({"path": ["v2", "quotes", "{{fx_id}}"]}) == "/v2/quotes/{fx_id}"


def test_parse_json_body_handles_unquoted_placeholder():
    raw = '{"id": "{{webhook_id}}", "event": {{payload}}}'
    assert parse_json_body(raw) == {"id": "{{webhook_id}}", "event": "{{payload}}"}
    assert parse_json_body("") is None


def test_infer_schema_nested():
    schema = infer_schema({"amount": {"value": "10.00"}, "items": [{"qty": 1}], "draft": True})
    assert schema["properties"]["amount"]["properties"]["value"] == {"type": "string"}
    assert schema["properties"]["items"]["items"]["properties"]["qty"] == {"type": "integer"}
    assert schema["properties"]["draft"] == {"type": "boolean"}


@pytest.mark.parametrize("method,name,expected", [
    ("GET", "Show invoice details", "read"),
    ("POST", "Search for invoices", "read"),
    ("POST", "Refund captured payment", "write"),
    ("DELETE", "Delete invoice", "write"),
])
def test_classify_action(method, name, expected):
    assert classify_action(method, name) == expected


def test_parameters_from_path_query_and_body():
    item = make_request(
        "Refund captured payment",
        method="POST",
        path=("v2", "payments", "captures", ":capture_id", "refund"),
        header=[{"key": "PayPal-Request-Id"}],
        body={"mode": "raw", "raw": '{"amount": {"value": "10.00", "currency_code": "USD"}}'},
    )
    item["request"]["url"]["variable"] = [{"key": "capture_id", "description": "(Required) The capture ID."}]
    item["request"]["url"]["query"] = [
        {"key": "page", "value": "1"},
        {"key": "fields", "value": "all", "description": "(Required) Fields to return."},
    ]
    tools, _ = parse_collection(make_collection({"name": "Payments", "item": [item]}))
    params = tools[0]["parameters"]

    assert params["properties"]["capture_id"] == {"type": "string", "x-in": "path", "description": "The capture ID."}
    assert params["properties"]["page"] == {"type": "integer", "x-in": "query"}
    assert params["properties"]["amount"]["x-in"] == "body"
    assert params["required"] == ["capture_id", "fields"]
    assert tools[0]["supports_idempotency"] is True
    assert tools[0]["action_type"] == "write"


def test_skips_authorization_folder_and_flags_missing_description():
    collection = make_collection(
        {"name": "Authorization", "item": [make_request("Generate access_token", method="POST")]},
        {"name": "Invoices", "item": [{"name": "Templates", "item": [make_request("List templates")]}]},
    )
    tools, examples = parse_collection(collection)
    assert [t["name"] for t in tools] == ["list_templates"]
    assert tools[0]["category"] == "invoices"
    assert tools[0]["subcategory"] == "templates"
    assert tools[0]["needs_description"] is True
    assert tools[0]["description"] == "List templates"
    assert set(examples) == {"list_templates"}


def test_duplicate_names_get_category_prefix():
    collection = make_collection(
        {"name": "Orders", "item": [make_request("Show details")]},
        {"name": "Disputes", "item": [make_request("Show details")]},
    )
    tools, _ = parse_collection(collection)
    assert [t["name"] for t in tools] == ["show_details", "disputes_show_details"]


@pytest.mark.skipif(not COLLECTION.exists(), reason="PayPal collection not present")
def test_real_paypal_collection():
    tools, examples = parse_collection(json.loads(COLLECTION.read_text()))
    names = {t["name"] for t in tools}

    assert len(tools) == 112
    assert len(names) == len(tools)
    assert {"create_draft_invoice", "send_invoice", "refund_captured_payment", "list_disputes"} <= names
    assert all(t["path"].startswith("/") and "{{" not in t["path"] for t in tools)
    assert all(examples[t["name"]] for t in tools)
