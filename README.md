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

### Step 4: Mock PayPal server ⏭️

## Status

Design complete; steps 1–3 (parser, enricher, tool index) done.
