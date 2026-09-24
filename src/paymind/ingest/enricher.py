"""Enrich raw tool cards with an LLM.

Takes data/tools/raw_tools.json (from the Postman parser) and adds the
language that search and the agent need:

  - a clear description written in the words users actually use
  - example questions a user might ask
  - a reviewed read/write classification
  - which roles (customer / accountant) may use the tool

The LLM never touches facts: name, method, path and parameters are copied
from the raw card unchanged. Anything the LLM returns is validated, and a
tool whose output fails validation keeps its raw values and is flagged.

Tools are sent one category at a time, so the LLM sees sibling tools side by
side and can write descriptions that tell them apart (e.g. "send invoice" vs
"send invoice reminder"). Finished categories are saved as it goes, so a run
that hits a rate limit can be resumed.

Usage:
    python -m paymind.ingest.enricher              # enrich what's missing
    python -m paymind.ingest.enricher --force      # redo everything
    python -m paymind.ingest.enricher --only invoices disputes
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

ROLES = ("customer", "accountant")
MAX_BATCH = 12  # split large categories so each call stays focused

SYSTEM_PROMPT = """\
You write tool descriptions for an AI assistant that operates a payments account \
(PayPal-style APIs). A search engine will match user messages against your text, \
so write in the words real users use, not API jargon.

The assistant has two roles:
- customer: a buyer. May view things that concern them (an invoice sent to them, \
an order, a refund, a dispute they opened) and act on their own disputes \
(send a message, provide evidence, escalate, accept or deny an offer).
- accountant: the merchant's finance staff. May use every business tool: \
invoices, payments, refunds, orders, subscriptions, payouts, disputes, reports.
Developer or setup tools (webhooks, onboarding, payment tokens, simulations) \
are accountant-only.

For EACH tool you are given, return:
- name: the tool name exactly as given (do not change it)
- description: 1-2 plain sentences saying what it does and when to use it. \
Make it clearly different from the sibling tools in the same batch.
- example_questions: 4 different things a user might type that should trigger \
this tool. Vary the wording, mix casual and formal, include realistic IDs, \
amounts or names where they fit. Do not reuse the tool name as the question.
- action_type: "read" if it only looks up data, "write" if it creates, \
changes, sends, cancels or deletes anything.
- allowed_roles: a non-empty subset of ["customer", "accountant"].
"""


class ToolEnrichment(BaseModel):
    name: str
    description: str = Field(min_length=10)
    example_questions: list[str] = Field(min_length=3, max_length=6)
    action_type: Literal["read", "write"]
    allowed_roles: list[Literal["customer", "accountant"]] = Field(min_length=1)


class EnrichmentBatch(BaseModel):
    tools: list[ToolEnrichment]


# Something that takes (system prompt, user prompt) and returns a validated batch.
Enrich = Callable[[str, str], EnrichmentBatch]


def describe_for_prompt(tool: dict) -> dict:
    """The subset of a raw card the LLM needs to see."""
    params = tool["parameters"].get("properties", {})
    return {
        "name": tool["name"],
        "title": tool["title"],
        "method": tool["method"],
        "path": tool["path"],
        "current_description": tool["description"],
        "parameters": sorted(params),
        "guessed_action_type": tool["action_type"],
    }


def build_user_prompt(category: str, tools: list[dict]) -> str:
    payload = json.dumps([describe_for_prompt(t) for t in tools], indent=2)
    return f"Category: {category}\nTools ({len(tools)}):\n{payload}"


def batches(tools: list[dict], size: int = MAX_BATCH) -> list[tuple[str, list[dict]]]:
    """Group tools by category, splitting any category larger than `size`."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for tool in tools:
        by_category[tool["category"]].append(tool)
    result = []
    for category, group in by_category.items():
        for i in range(0, len(group), size):
            result.append((category, group[i : i + size]))
    return result


def merge(raw: dict, enrichment: ToolEnrichment | None) -> dict:
    """Combine a raw card with LLM output. Facts always come from the raw card."""
    card = {k: v for k, v in raw.items() if k != "needs_description"}
    if enrichment is None:
        card.update({
            "example_questions": [],
            "allowed_roles": ["accountant"],
            "enrichment_status": "failed",
        })
    else:
        questions = list(dict.fromkeys(q.strip() for q in enrichment.example_questions if q.strip()))
        card.update({
            "description": enrichment.description.strip(),
            "example_questions": questions,
            "action_type": enrichment.action_type,
            "allowed_roles": sorted(set(enrichment.allowed_roles), key=ROLES.index),
            "enrichment_status": "ok",
        })
    card["requires_confirmation"] = card["action_type"] == "write"
    card["eval_only"] = False
    return card


def enrich_batch(enrich: Enrich, category: str, tools: list[dict]) -> list[dict]:
    """Enrich one batch. Tools the LLM skipped or renamed fall back to raw values."""
    result = enrich(SYSTEM_PROMPT, build_user_prompt(category, tools))
    by_name = {e.name: e for e in result.tools}
    return [merge(tool, by_name.get(tool["name"])) for tool in tools]


TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand", "timed out")


def is_transient(exc: Exception) -> bool:
    """Overloaded or rate-limited: worth waiting and retrying."""
    return any(marker in str(exc) for marker in TRANSIENT_MARKERS)


def is_daily_quota(exc: Exception) -> bool:
    """Daily quota used up: retrying this model today is pointless, move to the next one."""
    return "PerDay" in str(exc)


def with_retry_and_fallback(
    enrichers: list[tuple[str, Enrich]],
    waits: tuple[float, ...] = (5, 10, 20, 40),
    sleep: Callable[[float], None] = time.sleep,
) -> Enrich:
    """Try each model in order; on transient errors, back off and retry before moving on."""

    exhausted: set[str] = set()  # models out of daily quota; skipped for the rest of the run

    def enrich(system: str, user: str) -> EnrichmentBatch:
        last_exc: Exception | None = None
        for model, call in enrichers:
            if model in exhausted:
                continue
            for wait in (0, *waits):
                if wait:
                    print(f"    {model} busy, retrying in {wait:.0f}s")
                    sleep(wait)
                try:
                    return call(system, user)
                except Exception as exc:
                    if not is_transient(exc):
                        raise
                    last_exc = exc
                    if is_daily_quota(exc):
                        print(f"    {model} daily quota used up, skipping it for this run")
                        exhausted.add(model)
                        break
            else:
                print(f"    {model} still unavailable, falling back")
        raise RuntimeError(f"all models unavailable: {last_exc}")

    return enrich


def make_gemini_enricher(model: str) -> Enrich:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=model, max_retries=0)  # retries handled by with_retry_and_fallback
    structured = llm.with_structured_output(EnrichmentBatch)

    def enrich(system: str, user: str) -> EnrichmentBatch:
        return structured.invoke([("system", system), ("human", user)])

    return enrich


def main() -> None:
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=root / "data/tools/raw_tools.json")
    ap.add_argument("--out", type=Path, default=root / "data/tools/tools.json")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    ap.add_argument(
        "--fallback-models", nargs="*",
        default=os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash gemini-3.5-flash gemini-3.5-flash-lite").split(),
        help="tried in order when the main model stays overloaded",
    )
    ap.add_argument("--only", nargs="*", help="only these categories")
    ap.add_argument("--force", action="store_true", help="re-enrich tools that are already done")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between calls (free-tier rate limits)")
    args = ap.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not set. Add it to .env (see .env.example).")

    raw_tools = json.loads(args.raw.read_text())
    done: dict[str, dict] = {}
    if args.out.exists() and not args.force:
        done = {t["name"]: t for t in json.loads(args.out.read_text()) if t.get("enrichment_status") == "ok"}

    todo = [t for t in raw_tools if t["name"] not in done and (not args.only or t["category"] in args.only)]
    work = batches(todo)
    print(f"{len(raw_tools)} tools, {len(done)} already enriched, {len(todo)} to do in {len(work)} calls ({args.model})")

    models = list(dict.fromkeys([args.model, *args.fallback_models]))
    enrich = with_retry_and_fallback([(m, make_gemini_enricher(m)) for m in models])
    for i, (category, group) in enumerate(work, 1):
        try:
            cards = enrich_batch(enrich, category, group)
        except Exception as exc:  # keep going; failed tools are retried on the next run
            print(f"  [{i}/{len(work)}] {category}: FAILED ({type(exc).__name__}: {exc})")
            continue
        for card in cards:
            if card["enrichment_status"] == "ok":
                done[card["name"]] = card
        ok = sum(c["enrichment_status"] == "ok" for c in cards)
        print(f"  [{i}/{len(work)}] {category}: {ok}/{len(cards)} ok")
        write_output(args.out, raw_tools, done)  # save progress after every call
        if i < len(work):
            time.sleep(args.delay)

    write_output(args.out, raw_tools, done)
    missing = [t["name"] for t in raw_tools if t["name"] not in done]
    print(f"Wrote {args.out}: {len(raw_tools) - len(missing)} enriched, {len(missing)} not enriched")
    failed_now = [t["name"] for t in todo if t["name"] not in done]
    if failed_now:
        print(f"  {len(failed_now)} failed this run; re-run to retry: " + ", ".join(failed_now))


def write_output(path: Path, raw_tools: list[dict], done: dict[str, dict]) -> None:
    """Write every tool in raw order; tools not yet enriched are written with raw values, flagged."""
    cards = [done.get(t["name"]) or merge(t, None) for t in raw_tools]
    path.write_text(json.dumps(cards, indent=2) + "\n")


if __name__ == "__main__":
    main()
