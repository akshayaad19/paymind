"""Parse a Postman collection (v2.1) into raw tool cards.

Each request in the collection becomes one tool card with a name, category,
HTTP method, path, description and a JSON Schema for its parameters. The
cards are "raw": descriptions come straight from Postman and have no example
questions or roles yet. The LLM enricher fills those in.

Example responses are written to a separate file, keyed by tool name, so the
mock server can replay them without bloating the tool cards.

Usage:
    python -m paymind.ingest.postman_parser
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Iterator

# Folders that are plumbing (OAuth tokens), not something a user asks for.
SKIP_FOLDERS = {"Authorization"}

# Non-GET requests whose names show they only read data.
READ_NAME_PATTERN = re.compile(r"^(search|list|show|get|retrieve|calculate|verify|generate)\b", re.I)

PLACEHOLDER = re.compile(r"\{\{\s*([\w.-]+)\s*\}\}")
REQUIRED_MARK = re.compile(r"^\s*\(Required\)\s*", re.I)


def slugify(text: str) -> str:
    """'Show invoice details' -> 'show_invoice_details'."""
    text = re.sub(r"\(.*?\)", "", text)  # drop "(Limited Release)" etc.
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def clean_description(text: str | None) -> str:
    """Strip the HTML/markdown PayPal uses in descriptions down to plain text."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = text.replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def infer_schema(value: Any) -> dict[str, Any]:
    """Build a JSON Schema from an example value."""
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, list):
        return {"type": "array", "items": infer_schema(value[0]) if value else {}}
    if isinstance(value, dict):
        return {"type": "object", "properties": {k: infer_schema(v) for k, v in value.items()}}
    return {"type": "string"}


def infer_scalar_type(value: str | None) -> str:
    """Guess a type for a query-string example like 'true' or '10'."""
    if value is None or PLACEHOLDER.fullmatch(value.strip()):
        return "string"
    if value.lower() in ("true", "false"):
        return "boolean"
    if re.fullmatch(r"-?\d+", value):
        return "integer"
    return "string"


def parse_json_body(raw: str) -> Any | None:
    """Parse a raw JSON body, tolerating unquoted {{placeholders}}. Returns None if empty or invalid."""
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Quote placeholders used as bare values, e.g.  "webhook_event": {{payload}}
    quoted = re.sub(r"(?<!\")\{\{\s*([\w.-]+)\s*\}\}(?!\")", r'"{{\1}}"', raw)
    try:
        return json.loads(quoted)
    except json.JSONDecodeError:
        return None


def iter_requests(items: list[dict], folders: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict]]:
    """Yield (folder path, item) for every request, walking nested folders."""
    for item in items:
        if "item" in item:
            yield from iter_requests(item["item"], folders + (item["name"],))
        elif "request" in item:
            yield folders, item


def build_path(url: dict) -> str:
    """['v2', 'invoicing', 'invoices', ':invoice_id'] -> '/v2/invoicing/invoices/{invoice_id}'."""
    segments = []
    for seg in url.get("path", []):
        if seg.startswith(":"):
            seg = "{" + seg[1:] + "}"
        seg = PLACEHOLDER.sub(r"{\1}", seg)
        segments.append(seg)
    return "/" + "/".join(segments)


def build_parameters(request: dict, path: str) -> dict[str, Any]:
    """Collect path, query and body parameters into one JSON Schema object.

    Each property carries "x-in" (path / query / body) so the executor knows
    where to put it when building the HTTP request.
    """
    url = request["url"]
    properties: dict[str, Any] = {}
    required: list[str] = []

    var_docs = {
        v.get("key"): clean_description(REQUIRED_MARK.sub("", v.get("description") or ""))
        for v in url.get("variable", [])
    }
    for name in re.findall(r"\{(\w+)\}", path):
        prop = {"type": "string", "x-in": "path"}
        if var_docs.get(name):
            prop["description"] = var_docs[name]
        properties[name] = prop
        required.append(name)

    for q in url.get("query", []):
        name = q.get("key")
        if not name or name in properties:
            continue
        doc = q.get("description") or ""
        prop = {"type": infer_scalar_type(q.get("value")), "x-in": "query"}
        if clean_description(REQUIRED_MARK.sub("", doc)):
            prop["description"] = clean_description(REQUIRED_MARK.sub("", doc))
        properties[name] = prop
        if REQUIRED_MARK.match(doc):
            required.append(name)

    body = request.get("body") or {}
    if body.get("mode") == "raw":
        parsed = parse_json_body(body.get("raw", ""))
        if isinstance(parsed, dict):
            for name, value in parsed.items():
                properties.setdefault(name, {**infer_schema(value), "x-in": "body"})
        elif isinstance(parsed, list):
            properties["items"] = {**infer_schema(parsed), "x-in": "body"}
    elif body.get("mode") in ("urlencoded", "formdata"):
        for field in body.get(body["mode"], []):
            name = field.get("key")
            if name and name not in properties:
                prop = {"type": "string", "x-in": "body"}
                if clean_description(field.get("description")):
                    prop["description"] = clean_description(field.get("description"))
                properties[name] = prop

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def classify_action(method: str, name: str) -> str:
    """First guess at read vs write. The enricher reviews this."""
    if method == "GET" or READ_NAME_PATTERN.match(name):
        return "read"
    return "write"


def extract_example_body(request: dict) -> Any | None:
    body = request.get("body") or {}
    if body.get("mode") == "raw":
        return parse_json_body(body.get("raw", ""))
    return None


def extract_responses(item: dict) -> list[dict[str, Any]]:
    responses = []
    for resp in item.get("response", []):
        raw = resp.get("body") or ""
        try:
            body: Any = json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError:
            body = raw
        responses.append({"code": resp.get("code"), "status": resp.get("status"), "name": resp.get("name"), "body": body})
    return responses


def parse_collection(collection: dict, service: str = "paypal") -> tuple[list[dict], dict[str, list[dict]]]:
    """Return (tool cards, example responses keyed by tool name)."""
    tools: list[dict] = []
    examples: dict[str, list[dict]] = {}
    seen: set[str] = set()

    for folders, item in iter_requests(collection["item"]):
        if folders and folders[0] in SKIP_FOLDERS:
            continue
        request = item["request"]
        category = slugify(folders[0]) if folders else "general"
        name = slugify(item["name"])
        if name in seen:
            name = f"{category}_{name}"
        seen.add(name)

        method = request["method"].upper()
        path = build_path(request["url"])
        description = clean_description(request.get("description"))
        headers = {h.get("key") for h in request.get("header", [])}

        tools.append({
            "name": name,
            "title": item["name"],
            "service": service,
            "category": category,
            "subcategory": slugify(folders[1]) if len(folders) > 1 else None,
            "method": method,
            "path": path,
            "description": description or item["name"],
            "needs_description": not description,
            "parameters": build_parameters(request, path),
            "example_request_body": extract_example_body(request),
            "action_type": classify_action(method, item["name"]),
            "supports_idempotency": "PayPal-Request-Id" in headers,
        })
        examples[name] = extract_responses(item)

    return tools, examples


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", type=Path, default=root / "data/postman/paypal_collection.json")
    ap.add_argument("--tools-out", type=Path, default=root / "data/tools/raw_tools.json")
    ap.add_argument("--examples-out", type=Path, default=root / "data/mock/example_responses.json")
    args = ap.parse_args()

    collection = json.loads(args.collection.read_text())
    tools, examples = parse_collection(collection)

    args.tools_out.parent.mkdir(parents=True, exist_ok=True)
    args.examples_out.parent.mkdir(parents=True, exist_ok=True)
    args.tools_out.write_text(json.dumps(tools, indent=2) + "\n")
    args.examples_out.write_text(json.dumps(examples, indent=2) + "\n")

    reads = sum(t["action_type"] == "read" for t in tools)
    print(f"Parsed {len(tools)} tools ({reads} read, {len(tools) - reads} write) -> {args.tools_out}")
    print(f"Example responses for {len(examples)} tools -> {args.examples_out}")


if __name__ == "__main__":
    main()
