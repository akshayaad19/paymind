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

## Who uses it and how access is controlled

PayMind is an AI assistant for **one shop's PayPal business account** (the mock shop is *PayMind Demo Store*). It is not a store and not a payment app.

```
Accountant (shop staff) / Customer (buyer)
        │  chat in plain English
        ▼
PayMind agent  ──calls PayPal APIs──►  the shop's PayPal account  (mock server + SQLite)
```

| Person | Works for | Can see | Can do |
|---|---|---|---|
| **Accountant** | The shop (its finance team) | Everything in the shop's account | Invoices, refunds, orders, disputes, reports |
| **Customer** | Nobody; buys from the shop | **Only their own** records | View their invoices/payments/refunds, manage their own disputes and subscriptions |

PayPal staff are not users of the assistant. Both roles are answered from the **same** PayPal data; what differs is how much each person may see and do.

**Two databases, two worlds**

| Database | Holds | Status |
|---|---|---|
| `data/mock/paypal_mock.db` | The shop's PayPal account: customers (buyers), payments, refunds, invoices, disputes, orders, ledger | ✅ Step 4 |
| `data/app.db` | Our app's own data: **users** (name, role, and for customers their `payer_id`), **audit log**, conversation state | ⏭️ Step 5 |

**Access control.** Nobody touches PayPal or the databases directly; every request goes through the agent:

| Layer | What it does | Status |
|---|---|---|
| 1. Login | Identifies the user and their role (`users` table). A customer user is linked to their PayPal `payer_id` | ⏭️ Step 5 |
| 2. Tool search role filter | A customer never even *sees* accountant tools such as refunds | ✅ Step 3 |
| 3. Validator | Checks the role again before anything runs | ⏭️ Step 5 |
| 4. Data scope | Customers can only fetch records with **their** `payer_id` (Rahul can't see another customer's invoice) | ⏭️ Step 5 |
| 5. Confirmation | Money-moving (write) actions need an explicit "yes" | ✅ cards (Step 2), enforced in Step 5 |
| 6. Audit log | Every action recorded: who, what, when, result | ⏭️ Step 5 |

The shop's PayPal API credentials stay on the server; users never see them, so the only way in is through the assistant and its checks.

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

### Step 2: LLM enricher ✅

`src/paymind/ingest/enricher.py` adds the language that search and the agent need. It is a **one-time, offline** step: the app reads the finished `tools.json` and never calls the enricher at runtime. Re-run it only when tools change; it skips tools that are already enriched (`--force` redoes them).

```
data/tools/raw_tools.json  ──► Gemini, one category per call ──► validate ──► data/tools/tools.json
```

**What it adds to each card:**

| Field | Source |
|---|---|
| `description` | LLM: 1–2 plain sentences in users' words, written to tell sibling tools apart |
| `example_questions` | LLM: 4 varied things a user might type |
| `action_type` | LLM reviews the parser's read/write guess |
| `allowed_roles` | LLM: `customer` and/or `accountant` |
| `requires_confirmation` | **Code**: `true` for every write |
| `enrichment_status`, `eval_only` | Code |

**Example** (`send_invoice`):
> *Deliver an invoice to the customer immediately or schedule it to be sent automatically on a future date. This changes the bill from a draft to an active, payable request.*
> • "Send out invoice INV-481 now" • "Mail the bill for $500 to the client" • "Schedule this draft to be sent next Monday" • "Deliver invoice #204 to the buyer"

**Design choices:**
- **One category per call.** The LLM sees sibling tools side by side (send invoice vs send reminder vs cancel vs delete) and writes descriptions that separate them, which is what search needs. It also cuts ~112 calls to ~14.
- **Facts are never touched.** Name, method, path and parameters are always copied from the raw card; the LLM only sees them as context.
- **Every LLM answer is validated** (Pydantic): 3–6 questions, roles from the allowed set, read/write only. A tool that fails keeps its raw values, is marked `failed`, and defaults to the safest role (accountant only).
- **Progress is saved after every call**, so an interrupted run resumes where it stopped.
- **Retry and fallback.** Overloaded (503) or rate-limited (429) calls are retried with backoff (5s → 10s → 20s → 40s), then the next model in `GEMINI_FALLBACK_MODELS` is tried. A model whose free-tier **daily** quota is used up is skipped for the rest of the run instead of being retried.

**Result:** 112/112 enriched, 4 questions each, facts unchanged (checked automatically). 40 read / 72 write (all writes require confirmation). Roles: 94 accountant-only, 16 both, 2 customer-only (accepting or denying a seller's dispute offer, which only the buyer can do).

**Setup:** copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (from https://aistudio.google.com/apikey). Never put the real key in `.env.example`; that file is committed.

```bash
.venv/bin/python -m paymind.ingest.enricher                 # enrich what's missing
.venv/bin/python -m paymind.ingest.enricher --only invoices # one category
.venv/bin/python -m paymind.ingest.enricher --force         # redo everything
```

### Step 3: Tool index (Qdrant) ✅

`src/paymind/retrieval/tool_index.py` stores every tool in Qdrant and finds the best tools for a user's question.

**In one line:** each tool is stored as two vectors (keywords + meaning). A question is turned into the same two vectors, Qdrant searches both, merges the results and returns the top 5 tools.

#### Who does what

| Piece | What it is | Its job |
|---|---|---|
| **Our code** | `tool_index.py` | Builds the text for each tool, picks the models and settings, passes the user's role |
| **fastembed** | Python library | **Creates** both vectors: sparse (BM25 rules) and dense (runs bge-small) |
| **bge-small** (`BAAI/bge-small-en-v1.5`) | Small transformer embedding model (~33M parameters) | Turns text into **384 numbers** that represent meaning |
| **ONNX Runtime** | Engine used by fastembed | Does the transformer's maths on the **CPU**. Only needed for the **dense** vector |
| **qdrant-client** | Python library | Calls fastembed for us and sends the vectors to Qdrant |
| **Qdrant** | Vector database (local folder or Qdrant Cloud) | **Stores and searches**: inverted index, IDF, HNSW, cosine, RRF, role filter |

---

#### Part A: Storing the tools (offline, once)

```
tools.json
   │
   ▼  OUR CODE builds one text per tool:
   │  name + category + description + example questions
   │  (roles, action_type etc. are NOT in the text; they go in the payload)
   │
   ▼  FASTEMBED makes two vectors from that text
   ├── SPARSE (BM25):  lowercase → drop filler words → stem → hash → count
   │                   → {hash: TF, hash: TF, ...}          no ONNX, just rules
   │
   └── DENSE:          bge-small transformer, run by ONNX Runtime on the CPU
                       → [0.12, -0.83, ... 384 numbers]
   │
   ▼  QDRANT stores ONE POINT per tool
      ├── "bm25":  {481716713: 2.03, 1647276219: 1.99, ...}
      ├── "dense": [0.12, -0.83, ...]
      └── payload: name, category, service, action_type, allowed_roles, eval_only
   │
   ▼  QDRANT builds the indexes automatically
      ├── sparse → INVERTED INDEX  (hash → tools that contain that word)
      └── dense  → HNSW            (network of nearest neighbours)
```

**Sparse vector, step by step** (real output for `send_invoice_reminder`):

| Step | What happens | Example |
|---|---|---|
| 1. Drop filler words | Remove words like a, an, to, the, about | "Send **an** email reminder **to a** client" → send, email, reminder, client |
| 2. Stem | Cut regular endings so versions match | invoice, invoices → `invoic` · sending, sends → `send` (irregular *sent* and *sender* stay as they are) |
| 3. Hash | Turn each stem into a fixed number (its ID / position). The same word always gets the same hash, in every tool | `invoic` → **481716713** |
| 4. TF | How much **this tool** uses the word. More uses → higher, with diminishing returns; longer texts count each word slightly less | `invoic` appears 4× → **2.03** · once → **1.54** |

→ The tool's sparse vector is just its **hash → TF** pairs, **one pair per unique word** (21 for this tool). A word used 4 times is still one hash, just with a higher TF.

**Dense vector:** the whole text goes through the bge-small transformer (self-attention, feed-forward layers, using its learned weights). That's millions of multiplications, done by ONNX Runtime on the CPU, giving 384 numbers. Similar meanings give similar numbers.

**What we set vs. what's automatic**

| We set (in `tool_index.py`) | Automatic |
|---|---|
| The text: `search_text()` | Filler removal, stemming, hashing, TF (fastembed) |
| Model names: `Qdrant/bm25`, `BAAI/bge-small-en-v1.5` | Downloading and running the models (fastembed + ONNX) |
| Dense: `size=384`, `distance=COSINE` | Storing vectors, building HNSW and the inverted index (Qdrant) |
| Sparse: `modifier=IDF` (tells Qdrant to apply IDF) | Calculating IDF at search time (Qdrant) |
| **Payload indexes** for every field we filter on (`name`, `allowed_roles`, `category`, `service`, `eval_only`) | |

> **Two kinds of index.** *Vector indexes* (HNSW, inverted index) are built by Qdrant automatically. *Payload indexes* (for filtering by role, name…) must be created by us; Qdrant Cloud refuses to filter on a field without one. Local mode doesn't enforce this.

---

#### Part B: Searching (every user question)

```
"Is there a dispute open from user_123?"   role = accountant
   │
   ▼  FASTEMBED makes the same two vectors for the QUESTION
   ├── sparse: drop fillers → stem → hash     {hash(disput): 1, hash(open): 1, ...}
   │           (each question word gets weight 1; TF only matters on the tools' side)
   └── dense:  bge-small via ONNX Runtime      [384 numbers]
   │
   ▼  QDRANT, in one query
   ├── SPARSE search
   │     1. INVERTED INDEX → only tools containing the question's words
   │     2. IDF per word   = log(total tools ÷ tools with the word)   ← rare words count more
   │     3. score          = Σ TF × IDF                                → top 20
   │
   ├── DENSE search
   │     HNSW walks to the tools closest to the question
   │     closeness         = COSINE similarity                        → top 20
   │
   │     (the role filter is applied inside both searches)
   │
   └── RRF merges the two lists by rank position                      → TOP 5 TOOLS
   │
   ▼
list_disputes, show_dispute_details, provide_evidence, ...
```

**Where each calculation happens**

| Calculation | Where | When |
|---|---|---|
| Filler removal, stemming, hashing | fastembed | Storing (tools) and searching (question) |
| **TF** | **fastembed** | Storing: saved as the sparse vector's values |
| **IDF** | **Qdrant** | Searching: from the whole collection, never stored |
| TF × IDF score | Qdrant | Searching |
| Dense vector (transformer maths) | fastembed + **ONNX Runtime** on the CPU | Storing and searching |
| **Cosine similarity** | Qdrant | Searching, **dense side only** |
| **HNSW** | Qdrant | Built while storing, used while searching |
| Inverted index | Qdrant | Built while storing, used while searching |
| **RRF** merge | Qdrant | Searching |
| Role filter | Qdrant (we pass the role) | Searching |

---

#### BM25 with 2 tools: a worked example

**The 2 tools**
```
Tool A: send_invoice    →  "send invoice to customer"
Tool B: list_invoices   →  "list all invoices"
```

**Step 1: Clean each tool** (drop filler words, stem)
```
Tool A: send, invoic, custom        ("to" dropped)
Tool B: list, invoic                ("all" dropped, invoices → invoic)
```

**Step 2: TF** (how many times each word is in *that* tool)
```
Tool A: send = 1,  invoic = 1,  custom = 1
Tool B: list = 1,  invoic = 1
```

**Step 3: Inverted index** (word → tools that have it)
```
send   → [A]
invoic → [A, B]
custom → [A]
list   → [B]
```

**Step 4: The user asks "send the invoice"** → drop "the" → `send`, `invoic`

**Step 5: IDF** for each question word = `log(total tools ÷ tools with the word)`, total tools = 2
```
send   → in 1 tool  → log(2 ÷ 1) = 0.69   ← rare, so it matters
invoic → in 2 tools → log(2 ÷ 2) = 0      ← in every tool, can't help choose
```

**Step 6: Score each tool** = Σ (TF × IDF) over the question words
```
Tool A:  send   → TF 1 × IDF 0.69 = 0.69
         invoic → TF 1 × IDF 0    = 0
         Total = 0.69   ✅ highest → selected

Tool B:  send   → not in B        = 0
         invoic → TF 1 × IDF 0    = 0
         Total = 0
```

**Result:** Tool A wins. "invoice" is in both tools, so it can't tell them apart; "send" is only in Tool A, so it decides. (Real BM25 smooths the formula so common words get a small value rather than exactly 0, but the idea is the same.)

---

#### Don't mix these up

| Pair | Difference |
|---|---|
| **Hash vs TF** | Hash = *which* word (same number in every tool). TF = *how much this tool uses it* (different per tool). A sparse vector is hash → TF pairs |
| **TF vs IDF** | TF is counted **inside one tool**. IDF = log(total tools ÷ tools with the word), counted **across all tools**: how rare the word is |
| **Inverted index vs IDF** | Inverted index **finds** the candidate tools (speed). IDF **weights** the words (scoring). IDF is calculated across all tools but only applied to the candidates. The count "tools with the word" is just the length of that word's list in the inverted index |
| **ONNX Runtime vs HNSW** | ONNX Runtime **creates** dense vectors (runs the model). HNSW **searches** stored dense vectors (inside Qdrant) |
| **Cosine vs BM25** | Cosine scores the **dense** side. BM25 (TF × IDF) scores the **sparse** side. RRF merges them |
| **Model vs engine vs hardware** | bge-small = *what* to compute (a transformer + its weights). ONNX Runtime = software that *does* the maths. CPU = hardware *where* it's computed |
| **Deterministic vs no calculation** | Both vectors are deterministic (same text → same vector). Sparse is simple rules, so no ONNX needed; dense is a neural network, so it needs ONNX Runtime |
| **Text vs payload** | Text (name, category, description, questions) is **scored**. Payload (roles, action_type…) is only used to **filter** |
| **Tools vs chunks** | Points in the `tools` collection are tools. Chunks are for the RAG `docs` collection later |

**Why fastembed (not PyTorch / sentence-transformers or an API)?** A model file can't run by itself; it needs an engine. fastembed runs models on ONNX Runtime, which is light and fast on a CPU. PyTorch would run the same model but is a multi-GB install, uses more memory and is slower on a CPU. An embedding API costs money per call, needs internet and has rate limits. fastembed also gives BM25 and a reranker in the same library and is built into qdrant-client. At high volume, embeddings would move to a GPU-backed embedding service.

**Is HNSW used with only 112 tools?** Not yet. Below a size threshold, Qdrant simply compares the question against every vector (exact search), which is fast at this size. HNSW is built automatically once the collection grows, and that's what keeps search fast at thousands or millions of tools.

---

#### Local or cloud: same code

| `QDRANT_URL` / `QDRANT_API_KEY` in `.env` | Index lives in |
|---|---|
| empty | local folder `./qdrant_data` (git-ignored) |
| set | Qdrant Cloud (free tier: 1 GB RAM, 4 GB disk; suspended after 1 week unused, deleted after 4) |

The index is always rebuilt from `tools.json`, so if the cluster is suspended or deleted, resume it or create a new one and run `build` again. Point IDs are a stable hash of `service/name`, so rebuilding overwrites instead of duplicating. Vectors are created locally; only the finished vectors are sent to Qdrant.

```bash
.venv/bin/python -m paymind.retrieval.tool_index build
.venv/bin/python -m paymind.retrieval.tool_index search "Is there a dispute open from user_123?" --role accountant
```

**Look inside Qdrant Cloud:** cluster → *Open Dashboard* (log in with `QDRANT_API_KEY`).
- *Collections → tools → Graph*: each dot is a tool, lines join nearest neighbours by meaning (dense vectors). Orange = expanded by you, teal = not expanded yet.
- *Console*: see a tool's sparse vector (`indices` = hashes, `values` = TF). Qdrant never stores the words themselves or IDF.
```
POST collections/tools/points/scroll
{ "filter": { "must": [ { "key": "name", "match": { "value": "send_invoice_reminder" } } ] },
  "with_payload": ["name"], "with_vector": ["bm25"], "limit": 1 }
```

#### Results on the 112 PayPal tools (identical on local and Qdrant Cloud)

| Question | Role | Top result |
|---|---|---|
| Is there a dispute open from user_123? | accountant | `list_disputes` |
| What was my total sales volume last month? | accountant | `list_transactions` |
| Send an invoice for $50 to john@x.com | accountant | `create_draft_invoice` (#1) and `send_invoice` (#3): both steps of the task in the top 5 |
| give the customer their money back | accountant | `refund_captured_payment` (meaning match, no shared keyword) |
| I want to cancel my subscription | customer | `cancel_subscription` |
| refund order 123 | customer | `show_refund_details` (the refund action is hidden from customers by the role filter) |

Indexing 112 tools: ~4 s local, ~10–13 s to Qdrant Cloud.

### Step 4: Mock PayPal server ✅

`src/paymind/mock_paypal/` is a small FastAPI server that **pretends to be PayPal**: same URLs, same JSON shapes, fake data we control.

**Why we need it.** `tools.json` is only a *description* of PayPal's API (the menu). When the agent calls `GET /v1/customer/disputes`, some server has to receive the request and answer (the kitchen). Real PayPal needs an account and keys, its sandbox starts empty (no dispute from user_123, no sales last month), and it can't fail on demand. The mock fixes all of that. Switching to real PayPal later means changing only the base URL.

```
Agent ── GET /v1/customer/disputes?dispute_state=REQUIRED_ACTION ──► mock server (FastAPI, port 8000)
      ◄── { "items": [ { dispute PP-D-88561 from user_123, $40 } ] } ──
```

**Two kinds of endpoint**

| Kind | Count | How it answers |
|---|---|---|
| **Stateful** | 27 | Real logic on the SQLite data: a refund changes the payment's status, creates a refund record and lowers the balance; a sent invoice can't be sent again |
| **Example replay** | 83 | Returns PayPal's own example response from `example_responses.json` (header `X-Mock-Source: example:<tool>`) |

All 112 tools are answered (two tools share a URL with another tool). Every stateful handler uses the exact path from its tool card.

Stateful areas: invoices (create, show, list, send, remind, cancel, delete, record payment, search, next number) · captures and refunds · orders (create, show, capture) · disputes (list, show, accept claim, make/accept/deny offer, message, evidence, escalate) · transactions and balance.

#### The data: SQLite is the source of truth

There is no generated or in-memory data. **All data lives in SQLite**, and every request reads and writes it directly.

```
data/mock/initial.db       starting data (committed to git, never changed by the server)
        │  copied on first start
        ▼
data/mock/paypal_mock.db   working database the server reads and writes (git-ignored)
        ▲
        └── POST /mock/reset copies initial.db over it again
```

| Table | Rows in `initial.db` | What |
|---|---|---|
| `customers` | 6 | including **user_123** (Rahul Sharma) and **john@x.com** |
| `captures` | 47 | payments from mid-July to 23 Sep 2026 |
| `refunds` | 2 | one partial, one full |
| `invoices` | 8 | 2 draft, 3 sent (one overdue), 2 paid, 1 cancelled |
| `disputes` | 4 | **user_123, $40, waiting for our response**; one under PayPal review; one waiting for the buyer to answer our offer; one resolved |
| `orders` | 2 | one created, one completed |
| `transactions` | 49 | the ledger behind transaction search and the balance |
| `meta` | | merchant details, starting balance, next invoice number |
| `idempotency` | | remembered answers for `PayPal-Request-Id` retries |

Each record is one row holding PayPal-shaped JSON. You can open either file in any SQLite viewer (e.g. the *SQLite Viewer* VS Code extension, or `sqlite3 data/mock/paypal_mock.db`), and edits made there are seen by the running server on the next request.

- **Changes survive restarts.** Refund a payment, restart the server: it's still refunded and the balance is still lower.
- **One transaction per request.** A refund updates the payment, adds the refund and adds the ledger row all together or not at all. A failed request (400/404/422) writes nothing.
- **New records use the real current time** and random PayPal-style IDs checked for uniqueness.
- **To change the starting data**, edit `initial.db` and run `POST /mock/reset`.
- The starting data covers July–September 2026. Questions like "sales last month" are answered from those dates.

**Realistic rules and errors** (PayPal's error format: `name`, `message`, `details[].issue`)

| Situation | Response |
|---|---|
| Unknown ID | 404 `INVALID_RESOURCE_ID` |
| Refund more than what's left | 422 `REFUND_AMOUNT_EXCEEDED` |
| Refund an already fully refunded payment | 422 `CAPTURE_FULLY_REFUNDED` |
| Send an invoice that isn't a draft / has no recipient | 422 `CANNOT_SEND_INVOICE` / `MISSING_RECIPIENT` |
| Act on a resolved dispute | 422 `DISPUTE_ALREADY_RESOLVED` |
| Transaction search over more than 31 days (PayPal's real limit) | 400 `INVALID_REQUEST` |

**Built-in support for testing error handling**
- **Idempotency:** a write sent again with the same `PayPal-Request-Id` header returns the first result (header `X-Mock-Idempotent-Replay: true`) instead of running twice. No double refunds.
- **Failures on demand:** header `X-Mock-Fail: 500` or `503` fails without doing anything; `X-Mock-Fail: timeout` does the work and then answers late (the "it worked but the client timed out" case that idempotency protects against). `MOCK_FAILURE_RATE=0.2` makes 20% of calls fail with 503.

**Deliberate difference from PayPal:** the dispute list includes each dispute's buyer, so "is there a dispute from user_123?" can be answered from one call (real PayPal needs a details call per dispute).

```bash
.venv/bin/uvicorn paymind.mock_paypal.app:create_app --factory --port 8000
# interactive API docs: http://localhost:8000/docs
# overview of the data:  http://localhost:8000/mock/summary
# back to the starting data: curl -X POST http://localhost:8000/mock/reset
```

### Step 5: The agent (in progress)

Built in parts: **5a** app database ✅ · **5b** executor ✅ · **5c** validator ✅ · **5d** agent loop (LangGraph + Gemini) ✅ · 5e chat screen.

#### 5a: App database ✅

`src/paymind/app/database.py`: our app's own data, separate from the mock PayPal database. PayPal only knows the shop's account; knowing who is chatting, their role, and what they did is the app's job.

```
data/app/initial.db   starting data: demo users (committed)
data/app/app.db       working copy (git-ignored), created from initial.db on first use
```

Unlike the mock (PayPal-shaped JSON), these tables use **real columns**, because System Search needs to filter them.

**`users`**: who can log in

| user_id | name | role | payer_id |
|---|---|---|---|
| `u_asha` | Asha Iyer | accountant | (none: sees everything) |
| `u_rahul` | Rahul Sharma | customer | `user_123` |
| `u_priya` | Priya Nair | customer | `user_456` |

A customer **must** be linked to their PayPal `payer_id` (checked in code and by a database constraint), so customers can only ever be scoped to their own records. Role is `customer` or `accountant`.

**`audit_log`**: one row per tool call the agent tries

`time, session_id, user_id, role, tool, method, path, params (JSON), status, http_status, result_summary, confirmed, request_id`

- `user_id` is the main link: every action belongs to a user. The payment / invoice / dispute ID the action was about is inside `params`.
- `status`: `success` · `failed` · `pending_confirmation` · `declined` · `blocked`.
- `request_id` is the `PayPal-Request-Id` used for idempotency, not a payment ID. `session_id` is the chat conversation.
- Reading the log is **always scoped to one user** with **fixed filters only** (tool, status, since, limit), no free-form SQL. Rahul can never read Asha's history.
- Records actions only; login/logout tracking is decided with the chat screen (5e).

#### 5b: Executor ✅

`src/paymind/agent/executor.py` turns a tool call into a real HTTP request and sends it to PayPal (the mock by default; `PAYPAL_BASE_URL` switches it).

```
refund_captured_payment(capture_id="ABC", amount={"currency_code": "USD", "value": "5.00"})
   │  look up the card in tools.json; each parameter's "x-in" says where it goes
   ▼
POST http://localhost:8000/v2/payments/captures/ABC/refund
     body: {"amount": {...}}
     PayPal-Request-Id: pm-...    (write tools only; the SAME id on every retry)
```

| Situation | What the executor does |
|---|---|
| Parameter with `x-in: path` / `query` / `body` | Puts it in the URL path (escaped), the query string or the JSON body. GET and DELETE send no body |
| Write tool | Adds a `PayPal-Request-Id`, reused on every retry of that action |
| Timeout, connection error, 429, 5xx | Retries up to 3 more times, waiting 1s → 2s → 4s |
| 4xx (bad request, not found, business rule) | **No retry**; PayPal's error becomes one sentence, e.g. `422 REFUND_AMOUNT_EXCEEDED: Refund of 99999 is more than the 32.00 left to refund. (field: amount.value)` |
| No answer after all retries | `ok=False`: "PayPal did not respond… The action may or may not have happened." |

It returns an `ExecutionResult`: `ok`, `status_code`, `body`, `error`, `attempts`, `request_id`, `replayed`.

**Tested live: the "it worked but timed out" case**
```
Attempt 1: refund sent (id pm-dad14a…) → the server DOES the refund but answers 10s late
           → executor stops waiting after 2s (timeout)
wait 1s
Attempt 2: same refund, SAME id → server: "already done, here's the same result" (replayed)
Result:    ok, and only 1 refund in the database. No double refund.
```

The executor does no permission or safety checks; the validator (5c) runs before it.

#### 5c: Validator ✅

`src/paymind/agent/validator.py` decides whether a tool call suggested by the LLM may run. **The LLM suggests; plain code decides**: no LLM involved, so the same input always gives the same decision.

```
Gemini suggests: refund_captured_payment(capture_id="CAP-999", amount=40)
                         ▼
                     VALIDATOR
   1. Tool allowed    exists? offered this turn? user's role may use it?
   2. Parameters      required present? right types? no made-up parameters?
   3. Business rules  amount > 0, max 2 decimals, USD only, valid emails
   4. Grounding       every ID appeared in the chat or an earlier tool result?
   5. Confirmation    write → ask the user; over $500 → stronger warning
                         ▼
   ok · needs_confirmation · invalid (LLM fixes and retries) · blocked (not allowed)
```

| Check | Example it catches | Outcome |
|---|---|---|
| 1. Role | A customer tries `refund_captured_payment` | `blocked` (second role check, after the search filter) |
| 1. Offered | The LLM calls a tool that search didn't offer this turn | `invalid`: "use find_tools first" |
| 2. Required | Refund without `capture_id` | `invalid`: "missing required parameter: capture_id" |
| 2. Made-up parameter | `list_disputes(buyer="user_123")` (PayPal has no such filter) | `invalid`: "unknown parameter 'buyer'" |
| 3. Business rules | Refund of −5, 5.001 or EUR | `invalid` with the reason |
| 4. **Grounding** | `capture_id="CAP-999"` that never appeared anywhere | `invalid`: "not mentioned… look it up first" |
| 5. Confirmation | Any write, e.g. a $40 refund | `needs_confirmation`: "Refund captured payment for 40 USD (capture_id=CAP123)?" |

- **Obvious type slips are tidied, not rejected:** `40` → `"40"`, `49.5` → `"49.50"`, `"5"` → `5`, `"true"` → `true`. Real mistakes are still rejected.
- Parameter checks come **automatically from each card's JSON Schema** (from the parser in step 1), so they work for all 112 tools; only the business rules are hand-written.
- **All errors are reported together**, so the LLM can fix everything in one retry.
- **Grounding** is what answers the brief's "hallucinate parameters": the LLM can't invent an ID, it has to look it up first.
- Not here: **customer data scope** (a customer may only see *their own* records). That needs the actual PayPal data, so it's checked around the executor in 5d.

#### 5d: Agent loop (LangGraph + Gemini) ✅

`src/paymind/agent/graph.py` connects everything: search, the LLM, the validator, customer scope, the executor and the audit log.

```
search_tools ─► agent (Gemini) ──tool calls?── no ──► reply
                   ▲                 │ yes
                   │                 ▼
                   │              gate   validate · customer scope · ⏸ ask yes/no   (changes no data)
                   │                 ▼
                   └── results ── tools  run approved calls · filter for customers · audit log
```

| Step | What it does |
|---|---|
| `search_tools` | Hybrid search (step 3) with the user's role → top 5 tools. Short follow-ups ("refund the first one") are searched together with the previous message |
| `agent` | Gemini sees the offered tools + `find_tools` + `system_search`, and either answers or calls tools. Max **6** steps per message |
| `gate` | Runs the validator (5c) and the customer scope check on every call; for writes, **pauses** with LangGraph `interrupt()` and asks yes/no |
| `tools` | Runs what the gate allowed through the executor (5b), filters list results for customers, writes the audit log (5a), and returns results to the agent |

**Why a separate gate:** when LangGraph resumes after a pause, it re-runs the paused step from the start. The gate only checks and asks, and execution happens in the next step, so an action is **never run twice**. Each write also carries a `PayPal-Request-Id` built from its tool-call ID, so even a repeat would be replayed by PayPal.

**Built-in tools** (always offered):
- `find_tools(query)`: search the whole catalog when the offered tools don't fit; found tools can be called in the next step.
- `system_search(mode)`: `capabilities` → what the assistant can do (tool index, role-filtered, fake tools hidden); `activity` → the user's own audit log.

**Customer data scope** (`scope.py`): before a customer touches a record named by ID (dispute, invoice, payment, order, refund), it's looked up and must belong to them (payer_id, or recipient email for invoices); list results are filtered to their own records.

**State** is checkpointed in `data/app/checkpoints.db` (per session), so follow-ups and pending confirmations survive between messages.

**LLM outage:** if every model is busy or out of quota, the agent tries 3 times (waiting 3s, then 8s) and then replies *"I can't reach the AI model right now… nothing was done. Please try again in a few minutes."*, not a crash.

**Example calls for tools** (`src/paymind/ingest/call_examples.py`, one-time, offline). A live test showed the agent **wandering**: for "send an invoice for $50", search offered the right tools, but `create_draft_invoice` had a 4,000-character schema with nothing marked required, so Gemini browsed old invoices to learn the format and ran out of steps. Fix: for the **91 tools with real inputs** (writes, and lookups with a body or filters; 21 simple ID lookups are skipped), Gemini writes `required_params` and a small `example_call`, in batches by category. Code checks every answer:
- fields exist in the schema, path parameters always required, types and business rules pass;
- for tools that create things, the example is **actually sent to a throwaway copy of the mock server** and must not be rejected;
- a rejected answer is sent back **once with the reason** so Gemini can correct it.

The agent then sees each offered tool's *"Required: …"* and *"Example call: {…}"*, plus the rule *"call the tools you were given directly; don't browse records to learn a format."* **Result: 91/91 tools have a verified example call.**

```bash
.venv/bin/python -m paymind.ingest.call_examples       # fill in missing examples
.venv/bin/python -m paymind.agent.factory --user u_asha   # chat in the terminal (mock server running)
```

**Live results so far** (Gemini + Qdrant Cloud + mock server):

| Request | Result |
|---|---|
| "Is there a dispute open from user_123?" | ✅ `list_disputes` → PP-D-88561, $40, item not received |
| "Send an invoice for $50 to john@x.com…" (before example calls) | ❌ Wandered and hit the 6-step limit; led to the example-call fix |
| Same request (after example calls) | ⏳ Not re-run yet: Gemini's free tier was out of quota or overloaded |

**Gemini free tier:** about 20 requests per model per day, and frequent 503 "high demand" errors. Each question takes 2–4 LLM calls, so replies can be slow or fail. Billing gives much higher limits.

## Status

Design complete; steps 1–4 done; step 5 (agent) in progress: 5a–5d done (app database, executor, validator, agent loop); 5e chat screen next, and a live re-test of the invoice flow once Gemini quota resets.
