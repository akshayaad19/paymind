"""Add a filled-in example call and the required fields to tool cards (one-time, offline).

Why: the parser (step 1) can't tell which body fields are required, and big
write tools like create_draft_invoice have many nested fields. Without a
sample, the agent didn't know what a minimal call looks like.

For every tool with real inputs (write tools, and lookups with a body or query
filters; simple ID lookups are skipped), Gemini writes:
  required_params  top-level parameters that must be filled
  example_call     one small, realistic, filled-in call

Our code then checks each answer: required fields must exist, path parameters
are always required, the example must pass the same parameter and business-rule
checks the validator uses, and, for tools with no ID in the URL (e.g. create
an invoice), the example is actually sent to a throwaway copy of the mock
PayPal server and must not be rejected (4xx). An answer that fails is sent back
once with the reasons ("the server said: an invoice needs at least one item")
so the LLM can correct it; if it still fails it is left out and retried on the
next run.

Tools are sent in batches by category (about 8 calls for 91 tools). Progress is
saved after every batch, so a run that runs out of quota can be resumed.

Usage:
    python -m paymind.ingest.call_examples           # fill in what's missing
    python -m paymind.ingest.call_examples --force   # redo everything
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from ..agent.tools import clean_schema
from ..agent.validator import check_business_rules, check_parameters
from .enricher import make_gemini_enricher_for, with_retry_and_fallback

ROOT = Path(__file__).resolve().parents[3]
BATCH = 10
EXAMPLE_BODY_CHARS = 1500

SYSTEM_PROMPT = """\
You help an AI agent call PayPal REST APIs correctly. For EACH tool you are given, return:

- name: the tool name exactly as given.
- required_params: the top-level parameter names that must be provided for a typical, \
successful call. Your example will be sent to a test PayPal server, so it must really work. Always include path parameters, and for tools that send a body, the body fields PayPal \
needs (e.g. create_order needs intent and purchase_units). Keep it minimal: only what PayPal really needs.
- example_call_json: ONE small, realistic example of the arguments, as a JSON object string. \
Use ONLY parameter names from the tool's schema, with the same nesting and types. Fill only \
the required fields plus at most one or two common optional ones. Rules:
  * Money is always {"currency_code": "USD", "value": "50.00"} (value is a string, 2 decimals).
  * Date-times are ISO 8601 UTC like "2026-08-01T00:00:00Z"; dates are "2026-08-01".
  * Use plausible IDs in PayPal's style (e.g. capture "2GG279541U471931P", invoice \
"INV2-Z56S-5LLA-Q52L-CPZ5", dispute "PP-D-27803"), realistic emails like "john@x.com".
  * No placeholders like {{var}} or "<string>", no comments.
The PayPal example body is given only as a reference for field names; it is often far larger \
than needed.
"""


class CallExample(BaseModel):
    name: str
    required_params: list[str]
    example_call_json: str


class CallExampleBatch(BaseModel):
    tools: list[CallExample]


def needs_example(card: dict) -> bool:
    """Write tools, and lookups that take a body or query filters. Simple ID lookups don't need one."""
    places = {p.get("x-in") for p in card["parameters"].get("properties", {}).values()}
    return card["action_type"] == "write" or bool(places & {"body", "query"})


def prompt_for(category: str, cards: list[dict]) -> str:
    items = []
    for c in cards:
        body = json.dumps(c.get("example_request_body"), separators=(",", ":")) if c.get("example_request_body") else None
        items.append({
            "name": c["name"],
            "description": c["description"],
            "method": c["method"],
            "path": c["path"],
            "parameters": clean_schema(c["parameters"]),
            **({"paypal_example_body": body[:EXAMPLE_BODY_CHARS]} if body else {}),
        })
    return f"Category: {category}\nTools ({len(cards)}):\n{json.dumps(items, indent=1)}"


def sandbox_dry_run():
    """Send an example to a throwaway copy of the mock server; return an error sentence or None."""
    import tempfile

    from fastapi.testclient import TestClient

    from ..agent.executor import Executor, ToolRegistry
    from ..mock_paypal.app import create_app, stateful_routes

    stateful = stateful_routes()
    executor = Executor(ToolRegistry(), base_url="http://testserver",
                        client=TestClient(create_app(db_path=Path(tempfile.mkdtemp()) / "sandbox.db", slow_seconds=0)),
                        sleep=lambda s: None)

    def dry_run(card: dict, example: dict) -> str | None:
        has_path_ids = "{" in card["path"]
        if has_path_ids or (card["method"], card["path"]) not in stateful:
            return None  # needs real IDs, or the mock only replays examples: nothing meaningful to test
        result = executor.execute(card["name"], example)
        return None if result.ok else f"mock PayPal rejected the example: {result.error}"

    return dry_run


def check_example(card: dict, answer: CallExample, dry_run=None) -> tuple[list[str], dict, list[str]]:
    """Returns (required_params, example_call, problems). Code has the final say."""
    props = card["parameters"].get("properties", {})
    path_params = [n for n, p in props.items() if p.get("x-in") == "path"]
    problems = [f"required_params has unknown '{r}'" for r in answer.required_params if r not in props]
    required = list(dict.fromkeys(path_params + [r for r in answer.required_params if r in props]))
    try:
        example = json.loads(answer.example_call_json)
    except json.JSONDecodeError as exc:
        return required, {}, problems + [f"example is not valid JSON: {exc}"]
    if not isinstance(example, dict):
        return required, {}, problems + ["example must be a JSON object"]
    schema = {**card["parameters"], "required": required}
    cleaned, errors = check_parameters({**card, "parameters": schema}, example)
    rule_errors, _ = check_business_rules(cleaned)
    text = json.dumps(cleaned)
    if "{{" in text or "<" in text and ">" in text:
        errors.append("example contains placeholders")
    problems = problems + errors + rule_errors
    if not problems and dry_run:
        rejected = dry_run(card, cleaned)
        if rejected:
            problems.append(rejected)
    return required, cleaned, problems


def apply(card: dict, required: list[str], example: dict) -> dict:
    card = dict(card)
    card["parameters"] = {**card["parameters"], "required": required} if required else {
        k: v for k, v in card["parameters"].items() if k != "required"}
    card["required_params"] = required
    card["example_call"] = example
    card["call_example_status"] = "ok"
    return card


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", type=Path, default=ROOT / "data/tools/tools.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--delay", type=float, default=4.0)
    args = ap.parse_args()

    cards = json.loads(args.tools.read_text())
    todo = [c for c in cards if needs_example(c) and (args.force or c.get("call_example_status") != "ok")]
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in todo:
        groups[c["category"]].append(c)
    batches = [(cat, g[i:i + BATCH]) for cat, g in groups.items() for i in range(0, len(g), BATCH)]
    total = sum(needs_example(c) for c in cards)
    print(f"{total} tools need an example, {total - len(todo)} already done, {len(todo)} to do in {len(batches)} calls")

    models = list(dict.fromkeys([os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                                 *os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash gemini-3.5-flash gemini-3.5-flash-lite").split()]))
    enrich = with_retry_and_fallback([(m, make_gemini_enricher_for(m, CallExampleBatch)) for m in models])
    by_name = {c["name"]: i for i, c in enumerate(cards)}
    dry_run = sandbox_dry_run()

    for n, (category, group) in enumerate(batches, 1):
        try:
            result = enrich(SYSTEM_PROMPT, prompt_for(category, group))
        except Exception as exc:
            print(f"  [{n}/{len(batches)}] {category}: FAILED ({str(exc)[:160]})")
            if "all models unavailable" in str(exc):
                print("  All models are out of quota or unavailable. Progress is saved; re-run later to continue.")
                break
            continue
        answers = {a.name: a for a in result.tools}
        ok, rejected = 0, []
        for card in group:
            answer = answers.get(card["name"])
            if answer is None:
                print(f"      {card['name']}: no answer")
                continue
            required, example, problems = check_example(card, answer, dry_run)
            if problems:
                print(f"      {card['name']}: rejected ({'; '.join(problems)[:200]})")
                rejected.append((card, answer, problems))
                continue
            cards[by_name[card["name"]]] = apply(card, required, example)
            ok += 1

        if rejected:  # one correction round: show the LLM exactly why its answer was rejected
            feedback = "\n".join(
                f"- {card['name']}: your answer {a.model_dump_json()} was rejected because: {'; '.join(probs)}"
                for card, a, probs in rejected)
            try:
                retry = enrich(SYSTEM_PROMPT, prompt_for(category, [c for c, _, _ in rejected])
                               + "\n\nYour previous answers for these tools were rejected. Fix them:\n" + feedback)
                retry_answers = {a.name: a for a in retry.tools}
            except Exception as exc:
                print(f"      correction round failed ({str(exc)[:120]})")
                retry_answers = {}
            for card, _, _ in rejected:
                answer = retry_answers.get(card["name"])
                if answer is None:
                    continue
                required, example, problems = check_example(card, answer, dry_run)
                if problems:
                    print(f"      {card['name']}: still rejected after correction ({'; '.join(problems)[:200]})")
                    continue
                print(f"      {card['name']}: fixed after correction")
                cards[by_name[card["name"]]] = apply(card, required, example)
                ok += 1
        args.tools.write_text(json.dumps(cards, indent=2) + "\n")  # save after every batch
        print(f"  [{n}/{len(batches)}] {category}: {ok}/{len(group)} ok")
        if n < len(batches):
            time.sleep(args.delay)

    done = sum(c.get("call_example_status") == "ok" for c in cards)
    print(f"Done: {done}/{total} tools have an example call")


if __name__ == "__main__":
    main()
