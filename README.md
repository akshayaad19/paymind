# PayMind

A scalable tool-calling agent for payment operations. Users chat in plain English ("Send an invoice for $50 to john@x.com", "Is there a dispute open from user_123?") and the agent picks the right API, fills in the parameters, validates the call and executes it.

The core problem: LLM accuracy drops as you attach more tools. PayMind never shows the LLM every tool. It searches a tool index first and passes only the top few matches, so the number of tools stops being a hard limit.

The demo uses PayPal's REST API collection (116 APIs) against a mock backend, plus ~450 synthetic tool cards to test tool selection at scale.

## Architecture

```
OFFLINE
  Postman collection ──► Parser ──► LLM enricher ──► tools.json ──► fastembed ──► Qdrant "tools"
  Policy / help docs ──► Heading + recursive splitter ──────────────► fastembed ──► Qdrant "docs"

RUNTIME (LangGraph)
  rewrite_query ──► search_tools (hybrid BM25 + dense, RRF, role filter → top 5)
               ──► agent (ReAct; always has rag_search, system_search, find_tools)
               ──► validator (schema, role, grounding, confirmation)
               ──► executor ──► mock payments API
```

Cross-cutting: audit log, checkpointed state, LangSmith tracing, evaluation harness.

## Stack

LangGraph · LangChain · Qdrant · fastembed · FastAPI · SQLite · LangSmith · Streamlit

## Getting started

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Postman collection → raw tool cards + example responses
.venv/bin/python -m paymind.ingest.postman_parser

.venv/bin/pytest
```

## Pipeline steps

### Step 1: Postman parser ✅

`src/paymind/ingest/postman_parser.py` turns the Postman collection into raw tool cards. It is a **deterministic** parser: plain Python rules, no LLM, no randomness. The same input always produces byte-identical output.

```
data/postman/paypal_collection.json   (116 APIs, Postman v2.1)
        │  parser (skips the 4 Authorization / OAuth APIs)
        ▼
data/tools/raw_tools.json             112 tool cards
data/mock/example_responses.json      PayPal's example responses per tool, keyed by tool name
```

**How one API is mapped** (`Refund captured payment`):

| Postman | Rule | Tool card |
|---|---|---|
| `"name": "Refund captured payment"` | lowercase, spaces → `_` | `"name": "refund_captured_payment"` |
| Folder `Payments` | top folder → category | `"category": "payments"` |
| `"method": "POST"` | copied | `"method": "POST"` |
| `path: [v2, payments, captures, :capture_id, refund]` | join, `:var` / `{{var}}` → `{var}` | `"/v2/payments/captures/{capture_id}/refund"` |
| Path variable `capture_id`, "(Required) The PayPal-generated ID…" | path variable → required param, `x-in: path` | `capture_id: {type: string, x-in: path}` |
| Query params | `x-in: query`; required if marked "(Required)"; type from example value | e.g. `page: {type: integer, x-in: query}` |
| Body `{"amount": {...}, "invoice_id": ..., "note_to_payer": ...}` | each top-level key → param, JSON Schema inferred from the example, `x-in: body` | `amount` (object), `invoice_id`, `note_to_payer` |
| Description with HTML / markdown | tags and links stripped | plain-text `description` |
| `PayPal-Request-Id` header present | → idempotency flag | `"supports_idempotency": true` |
| GET, or name starts with list / show / search / get… | → read; otherwise write | `"action_type": "write"` |
| Example responses (201, 401, 404, 422…) | copied to a separate file under the same name | `example_responses.json["refund_captured_payment"]` |

**Output:** 112 cards (40 read, 72 write), 73 support idempotency. Every parameter carries `x-in` (`path` / `query` / `body`) so the executor knows where to put it when building the HTTP request.

**Known gaps, left to the LLM enricher (step 2):**
- Body parameters are inferred from example bodies, so none are marked required yet.
- 6 APIs have no description in Postman (`needs_description: true`).
- Some names are clumsy (`creates_a_payment_resource_for_a_single_item_purchase`).
- No example user questions or roles yet.

**Why deterministic:** the parser owns the facts (paths, methods, parameters), where mistakes would break API calls. The enricher only adds language (descriptions, example questions, roles), so an LLM mistake can't break an API call.

### Step 2: LLM enricher ⏭️

Next: improve descriptions, add example questions, category, read/write review and roles → `data/tools/tools.json`.

## Status

Design complete; step 1 (parser) done; step 2 (enricher) next.
