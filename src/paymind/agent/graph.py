"""The PayMind agent: a LangGraph loop that connects search, the LLM, the
validator, customer scope, the executor and the audit log.

    search_tools ─► agent (LLM) ──tool calls?── no ──► reply + receipt
                       ▲               │ yes
                       │               ▼
                       │            gate   validate · customer scope · ⏸ ask yes/no
                       │               ▼
                       └── results ─ tools  run approved calls · filter · audit log · record for the receipt

The receipt under each reply is built by code from what the tools step actually
ran, not from the LLM's words: if the model says "refunded" but no refund ran,
the user still sees "No changes were made".

Why a separate gate: when LangGraph resumes after a pause it re-runs the
paused step from the start. The gate only checks and asks (nothing it does
changes data), and the tools step runs after the answer, so an action is never
executed twice. Each write also carries a PayPal-Request-Id built from its tool
call id, so even a repeat would be replayed by PayPal, not redone.
"""

from __future__ import annotations

import json
from decimal import ROUND_DOWN, Decimal
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from ..app.database import AppDatabase, User
from .executor import Executor, ToolRegistry
from .scope import check_access, filter_results
from .tools import (BUILTIN_SCHEMAS, BUILTINS, CHECK_UPDATES, CLOSE_CASE, CUSTOMER_ONLY, CUSTOMER_PAYMENTS, SHOP_ONLY, FIND_TOOLS, ORDER_STATUS, RAG_SEARCH,
                    MY_PURCHASES, PROBLEMS, REPORT_PROBLEM, REQUEST_RESOLUTION, SYSTEM_SEARCH, tool_schema)
from .validator import money_values, validate

MAX_STEPS = 8          # LLM turns per user message
IN_STORE_REFUND_SHARE = Decimal("0.5")  # shop policy: in-store purchases can be refunded at most 50%
LLM_WAITS = (3.0, 8.0) # extra tries when every model is busy (seconds to wait before each)
LLM_DOWN = ("I can't reach the AI model right now (the provider is busy or out of quota), so nothing was done. "
            "Please try again in a few minutes.")
TOP_K = 5              # tools offered from search
MAX_OFFERED = 10       # cap when tools carry over between turns
RESULT_CHARS = 20000   # tool result size sent back to the LLM
YES = {"yes", "y", "confirm", "approve", "approved", "ok", "okay", "sure", "go ahead", "do it"}
FOLLOW_UP = re.compile(r"\b(it|that|this|them|those|these|first|second|third|last|same|one)\b", re.I)

# search_fn(query, role, k, include_eval_only) -> [(tool_name, description)]
SearchFn = Callable[..., list[tuple[str, str]]]
# llm_for(tool_schemas) -> something with .invoke(messages) -> AIMessage
LLMFactory = Callable[[list[dict]], Any]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    offered: list[str]                 # PayPal tools the LLM may call this turn
    seen: str                          # user text + tool results: what grounding checks against
    steps: int
    decisions: dict[str, dict]         # tool_call_id -> gate decision
    actions: list[dict]                # PayPal calls that ran (or were stopped) for this message: the receipt


@dataclass
class Deps:
    llm_for: LLMFactory
    search: SearchFn
    executor: Executor
    registry: ToolRegistry
    appdb: AppDatabase
    docs_search: Callable[..., list] | None = None  # docs_search(query, k, source) -> [DocHit]; RAG knowledge base
    llm_waits: tuple[float, ...] = LLM_WAITS
    sleep: Callable[[float], None] = time.sleep
    llm_errors: list[str] = field(default_factory=list)  # last LLM failures, for logs and tests


def system_prompt(user: User) -> str:
    who = (f"The user is a CUSTOMER: {user.name} (PayPal payer_id {user.payer_id}, email {user.email}). "
           "They may only see their own records; never reveal anything about other customers. "
           "Their payer_id and other internal IDs are for your lookups only: don't show them in replies."
           if user.is_customer else
           f"The user is an ACCOUNTANT on the shop's finance team: {user.name}. They can see the whole account.")
    return f"""You are PayMind, an assistant that operates the PayPal business account of PayMind Demo Store.
Today is {date.today().isoformat()}. {who}

Rules:
- Get facts from tools. Never guess IDs, amounts, dates or statuses.
- An ID you haven't seen yet (payment, invoice, dispute...) must be looked up with a list or search tool first.
- If information only the user can give is missing (amount, recipient, which record), ask one short question instead of calling a tool.
- Money is always an object: {{"currency_code": "USD", "value": "50.00"}}. Date-times are ISO 8601 UTC, e.g. 2026-08-01T00:00:00Z. Transaction search covers at most 31 days per call.
- Tools that change data are confirmed with the user automatically. Just call them; don't ask "are you sure?" yourself.
- If a tool returns an error, read it, fix the call and try once more. If it still fails, explain plainly.
- Do every part of the user's request before answering. Never say an action happened (sent, refunded, paid, created) unless its tool succeeded; if something is left undone, say what and why.
- The tools you were given were already chosen for this request. Call them directly, following each tool's example call; don't browse existing records or templates just to learn a format.
- For 'anything new?' or 'any messages?', call check_updates. It returns new_message (they wrote, unread), needs_reply (they wrote, user hasn't answered), action_needed (PayPal says it's the user's turn, with a response deadline: always mention days left or overdue) and no_reply_yet (user wrote days ago, no answer; offer to send a reminder). Always say how long ago things happened (e.g. '2 days ago'), using today's date. When you show a dispute's messages, say who wrote each one.
- check_updates also lists overdue or soon-due invoices: always mention them, with the amount and how many days overdue or left.
- For purchase orders ('where is my order?', tracking, delivery date, orders to ship), call order_status. Customers confirm delivery or report a missing order on the Orders tab.
- Questions about rules, policies, time limits or fees: call rag_search and answer ONLY from the passages it returns, citing the source in brackets, e.g. (Refunds and returns › Return window). Say whether it's the shop's policy or PayPal's. If it finds nothing relevant, say you couldn't find it in the policies; never answer policy questions from memory.
- Shop acting for a customer named in words ("refund Rahul", "invoice Maria"): first call customer_payments. If several customers match, list them (name and email) and ask which one; never pick by name alone. If the customer has several payments and the user didn't say which, list them (date, items, amount) and ask. For a refund without a dispute, if the user gave no reason, ask for one: it's sent to the customer as the refund note. Only then call the tool.
- Shop refunding a customer who has an open dispute on that payment: first read the dispute (show_dispute_details). The customer's claim decides the amount: the disputed amount (e.g. one extra charger), not the whole payment; don't ask the user for a reason, the dispute has it. Then use accept_claim (refunds the disputed amount; add refund_amount only if the user asks for a different amount). After it runs, report the amount in the result's refund field, never the payment total. There are no offers: a refund just happens. The shop never closes a case: after a refund or replacement it waits for the customer to confirm and close it. Use refund_captured_payment only for payments with no dispute.
- Overdue invoices are the shop's to sort out: send_invoice_reminder to nudge the customer, record_payment_for_invoice if they paid another way (cash, bank transfer), or cancel_sent_invoice.
- On a customer's existing open dispute, when they want their money back or a replacement, use request_resolution (wants: refund or replacement) with a short polite message; not a plain message. The customer confirms before it's sent.
- When a customer says they're satisfied with a case (got the refund, parcel arrived, happy with the replacement), offer to close it and use close_case once they agree. Tell them the shop's refund can take a few days to show.
- When a customer reports a NEW problem with something they bought (not working, damaged, not as described, not received, charged wrongly): 1) find the purchase with my_purchases (ask which one if several match); 2) if they haven't said, ask briefly what's wrong and whether they'd like a refund or a replacement (one short question, both together); 3) then call report_problem. For in-store purchases the shop's policy allows only a partial refund (at most 50%) or a replacement at the store: say so before they choose (rag_search has the details). Afterwards, ask them to attach a photo of the problem in the case (PayMind data → Disputes → open the case → 📎 Attach photo), which helps the shop decide quickly.
- Replacements are free, and the shop checks photos first: when check_updates shows photo_needed, tell the customer what the shop asked for and to attach it in the case (PayMind data → Disputes → open it → 📎). The shop sends the replacement with a tracking ID once the photos are approved.
- When the shop has refunded a customer or sent a replacement (check_updates shows confirm_resolution), tell the customer plainly what the shop did, and that they can close the case (close_case, or ✅ Mark as resolved in the case) once they're happy; if something's still wrong, they can reply in the case.
- If none of your tools fits, call find_tools. For "what can you do" or "status of my last request", call system_search.
- For totals, add up the amounts yourself and state the number.
- Messages to the other side of a dispute: if asked to write, word or format one, write it clearly and politely in the user's name (greeting, the facts they gave, a friendly close) and send it with the messaging tool; the user sees the exact text and approves it before it's sent. If they only ask for a draft, show the draft and don't send.
- Reply briefly with the key facts (IDs, amounts, statuses)."""


def compact(value: Any) -> str:
    """JSON for the LLM: no links, no nulls, no spaces, capped in size."""
    def strip(v):
        if isinstance(v, dict):
            return {k: strip(x) for k, x in v.items() if k != "links" and x is not None}
        if isinstance(v, list):
            return [strip(x) for x in v]
        return v
    text = json.dumps(strip(value), separators=(",", ":"), default=str)
    return text if len(text) <= RESULT_CHARS else text[:RESULT_CHARS] + '..."(truncated)"'


def search_query(messages: list[AnyMessage]) -> str:
    """The latest user message; for short follow-ups ("refund the first one"), add the previous one."""
    human = [m.content for m in messages if isinstance(m, HumanMessage)]
    latest = human[-1] if human else ""
    if len(human) > 1 and (len(latest.split()) <= 6 or FOLLOW_UP.search(latest)):
        return f"{human[-2]} {latest}"
    return latest


def real_line_breaks(value: Any) -> Any:
    """Models sometimes write a message's line breaks as the two characters backslash-n (escaped
    twice in the tool call). Turn them back into real line breaks in every text parameter."""
    if isinstance(value, str):
        return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    if isinstance(value, dict):
        return {k: real_line_breaks(v) for k, v in value.items()}
    if isinstance(value, list):
        return [real_line_breaks(v) for v in value]
    return value


def user_of(config: dict, appdb: AppDatabase) -> User:
    user = appdb.get_user(config["configurable"]["user_id"])
    if user is None:
        raise ValueError("unknown user")
    return user


def build_graph(deps: Deps, checkpointer=None):
    # ---- search_tools: pick the tools for this turn ---------------------------------
    def search_tools(state: AgentState, config) -> dict:
        user = user_of(config, deps.appdb)
        latest = state["messages"][-1].content if state["messages"] else ""
        hits = [name for name, _ in deps.search(search_query(state["messages"]), role=user.role, k=TOP_K)]
        carried = [t for t in state.get("offered", []) if t not in hits]
        return {
            "offered": (hits + carried)[:MAX_OFFERED],
            "seen": state.get("seen", "") + "\nuser: " + str(latest),
            "steps": 0,
            "decisions": {},
            "actions": [],
        }

    # ---- agent: the LLM decides what to do next ---------------------------------------
    def agent(state: AgentState, config) -> dict:
        user = user_of(config, deps.appdb)
        steps = state.get("steps", 0) + 1
        if steps > MAX_STEPS:  # say what was really done (from the same record as the receipt), not a guess
            done = [f"{RECEIPT_MARKS[a['outcome']]}: {a['label']}" for a in state.get("actions", []) if a["write"]]
            summary = ("So far: " + "; ".join(done) + ".") if done else "Nothing was changed."
            return {"steps": steps, "messages": [AIMessage(
                f"I reached my step limit before finishing my answer. {summary} "
                "Tell me if anything else is needed.")]}
        schemas = [tool_schema(deps.registry.get(n)) for n in state.get("offered", []) if deps.registry.get(n)]
        builtins = [b for b in BUILTIN_SCHEMAS if b["name"] not in (SHOP_ONLY if user.is_customer else CUSTOMER_ONLY)]
        llm = deps.llm_for(schemas + builtins)
        prompt = [SystemMessage(system_prompt(user)), *state["messages"]]
        for wait in (0.0, *deps.llm_waits):
            if wait:
                deps.sleep(wait)
            try:
                reply = llm.invoke(prompt)
                deps.llm_errors.clear()  # healthy again
                return {"steps": steps, "messages": [reply]}
            except Exception as exc:  # every model failed (busy, out of quota, network)
                deps.llm_errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
        return {"steps": steps, "messages": [AIMessage(LLM_DOWN)]}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "gate" if isinstance(last, AIMessage) and last.tool_calls and state.get("steps", 0) <= MAX_STEPS else END

    # ---- gate: check every call, ask the user about writes. Changes no data. -------------
    def gate(state: AgentState, config) -> dict:
        user = user_of(config, deps.appdb)
        decisions: dict[str, dict] = {}
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], real_line_breaks(call.get("args") or {})
            if name in (SHOP_ONLY if user.is_customer else CUSTOMER_ONLY):
                who = "the shop" if user.is_customer else "customers"
                decisions[call["id"]] = {"outcome": "blocked", "params": args, "errors": [f"{name} is for {who} only"]}
                continue
            if name == REQUEST_RESOLUTION:
                decisions[call["id"]] = resolution_decision(user, args)
                continue
            if name == REPORT_PROBLEM:
                decisions[call["id"]] = problem_decision(user, args)
                continue
            if name == CLOSE_CASE:
                decisions[call["id"]] = close_decision(user, args)
                continue
            if name in BUILTINS:
                decisions[call["id"]] = {"outcome": "ok", "params": args}
                continue
            v = validate(deps.registry.get(name), name, args, user, set(state.get("offered", [])), state.get("seen", ""))
            decision = {"outcome": v.outcome, "params": v.params, "errors": v.errors}
            if v.outcome in ("ok", "needs_confirmation"):
                reason = check_access(user, v.params, deps.executor)
                if reason:
                    decision = {"outcome": "blocked", "params": v.params, "errors": [reason]}
            if decision["outcome"] in ("ok", "needs_confirmation") and v.params.get("capture_id"):
                problem = payment_problem(name, v.params["capture_id"], user)
                if problem:  # e.g. refunding a payment that has an open dispute: settle the dispute instead
                    decision = {"outcome": "invalid", "params": v.params, "errors": [problem]}
            if decision["outcome"] == "needs_confirmation":
                question = v.confirmation + who_and_what(v.params.get("capture_id"))
                if name == "accept_claim":
                    question = claim_confirmation(v.params, user) or question
                answer = interrupt({"question": question, "tool": name, "params": v.params, "large_amount": v.large_amount})
                decision["approved"] = str(answer).strip().lower() in YES
            decisions[call["id"]] = decision
        return {"decisions": decisions}

    def payment_problem(tool: str, capture_id: str, user: User) -> str | None:
        """Code rules on a payment before a write, whatever the model decided."""
        if tool != "refund_captured_payment":
            return None
        disputes = deps.executor.execute("list_disputes", {"page_size": 50}, caller=user)
        open_case = next((d["dispute_id"] for d in (disputes.body.get("items", []) if disputes.ok else [])
                          if d["disputed_transactions"][0]["seller_transaction_id"] == capture_id and d.get("status") != "RESOLVED"), None)
        if open_case:
            return (f"payment {capture_id} has an open dispute ({open_case}): don't refund the payment directly; settle the "
                    "dispute with accept_claim (refunds the disputed amount, or refund_amount for part of it)")
        return None

    def claim_confirmation(params: dict, user: User) -> str | None:
        """accept_claim, in words: how much goes back, to whom, and for what, looked up from PayPal."""
        from ..app.purchases import purchase_details

        found = deps.executor.execute("show_dispute_details", {"dispute_id": params.get("dispute_id", "")}, caller=user)
        if not found.ok:
            return None
        d = found.body
        amount = (params.get("refund_amount") or d["dispute_amount"])
        part = "part of the disputed amount" if params.get("refund_amount") else "the disputed amount"
        buyer = (d.get("disputed_transactions") or [{}])[0].get("buyer") or {}
        who = f"{buyer.get('name', 'the customer')}" + (f" ({buyer['email']})" if buyer.get("email") else "")
        q = f"Refund {amount['value']} {amount['currency_code']} to {who} for dispute {d['dispute_id']} ({part})?"
        p = purchase_details(d["disputed_transactions"][0]["seller_transaction_id"], deps.executor)
        if p:
            q += f"\n\nPayment: {', '.join(p['items']) or 'purchase'}, {p['amount']} USD on {p['date'][:10]}"
        if params.get("note"):
            q += f"\n\n“{params['note']}”"
        return q

    def who_and_what(capture_id: str | None) -> str:
        """Who gets the money and for what, written by code for the confirmation, so the user checks the
        real customer and purchase, not the model's description of them."""
        if not capture_id:
            return ""
        from ..app.purchases import purchase_details

        p = purchase_details(capture_id, deps.executor)
        if not p:
            return ""
        c = p.get("customer") or {}
        who = f"{c.get('name') or 'Unknown customer'}" + (f" ({c['email']})" if c.get("email") else "")
        what = ", ".join(p["items"]) or "a purchase"
        return f"\n\nTo: {who}\nFor: {what}, paid {p['amount']} USD on {p['date'][:10]} ({p['paid_how']})"

    def confirm(question: str, tool: str, params: dict) -> dict:
        answer = interrupt({"question": question, "tool": tool, "params": params, "large_amount": False})
        return {"outcome": "needs_confirmation", "params": params, "approved": str(answer).strip().lower() in YES}

    def resolution_decision(user: User, args: dict) -> dict:
        """request_resolution is a write: on the customer's own open dispute, refund or replacement, confirmed."""
        dispute_id, message = str(args.get("dispute_id") or "").strip(), str(args.get("message") or "").strip()
        wants = str(args.get("wants") or "").strip().lower()
        params = {"dispute_id": dispute_id, "wants": wants, "message": message}
        if not dispute_id or not message or wants not in ("refund", "replacement"):
            return {"outcome": "invalid", "params": params,
                    "errors": ["dispute_id, message and wants (refund or replacement) are all required; ask the customer which they want"]}
        found = deps.executor.execute("show_dispute_details", {"dispute_id": dispute_id}, caller=user)
        if not found.ok:
            return {"outcome": "invalid", "params": params, "errors": [f"no dispute {dispute_id}: look it up with list_disputes first"]}
        if check_access(user, params, deps.executor):
            return {"outcome": "blocked", "params": params, "errors": [f"dispute {dispute_id} does not belong to this customer"]}
        if found.body.get("status") == "RESOLVED":
            return {"outcome": "invalid", "params": params, "errors": ["this dispute is already resolved"]}
        amount = found.body["dispute_amount"]
        params["amount"] = amount
        ask = (f"refund {amount['value']} {amount['currency_code']}, the disputed amount" if wants == "refund"
               else "send a replacement")
        return confirm(f"Ask the shop to {ask} (dispute_id={dispute_id})?\n\n“{message[:600]}”", REQUEST_RESOLUTION, params)

    def close_decision(user: User, args: dict) -> dict:
        """close_case: only the customer's own open dispute, confirmed."""
        dispute_id, message = str(args.get("dispute_id") or "").strip(), str(args.get("message") or "").strip()
        params = {"dispute_id": dispute_id, "message": message}
        found = deps.executor.execute("show_dispute_details", {"dispute_id": dispute_id}, caller=user) if dispute_id else None
        if not found or not found.ok:
            return {"outcome": "invalid", "params": params, "errors": [f"no dispute {dispute_id or '(none given)'}: look it up with list_disputes first"]}
        if check_access(user, params, deps.executor):
            return {"outcome": "blocked", "params": params, "errors": [f"dispute {dispute_id} does not belong to this customer"]}
        if found.body.get("status") == "RESOLVED":
            return {"outcome": "invalid", "params": params, "errors": ["this dispute is already closed"]}
        note = f"\n\n“{message[:600]}”" if message else ""
        return confirm(f"Close case {dispute_id} as resolved? The shop will be told you're satisfied.{note}", CLOSE_CASE, params)

    def run_close_case(user: User, params: dict, session_id: str | None) -> tuple[str, bool]:
        r = deps.executor.client.post(f"{deps.executor.base_url}/mock/disputes/{params['dispute_id']}/close",
                                      json={"message": params.get("message") or ""})
        ok = r.status_code < 300
        deps.appdb.log_action(user, CLOSE_CASE, params, "success" if ok else "failed", session_id=session_id, method="POST",
                              path=f"/mock/disputes/{params['dispute_id']}/close", http_status=r.status_code,
                              result_summary=(f"case {params['dispute_id']} closed by the customer" if ok else r.text)[:300], confirmed=True)
        if not ok:
            return compact({"ok": False, "error": r.text[:300]}), False
        return compact({"ok": True, "result": "The case is closed as resolved; the shop has been told."}), True

    def problem_decision(user: User, args: dict) -> dict:
        """report_problem opens a new case: only on the customer's own purchase, with no case already open."""
        from ..app.purchases import customer_purchases

        payment_id, message = str(args.get("payment_id") or "").strip(), str(args.get("message") or "").strip()
        problem, wants = str(args.get("problem") or ""), str(args.get("wants") or "").strip().lower()
        params = {"payment_id": payment_id, "problem": problem, "wants": wants, "message": message}
        if not payment_id or not message or problem not in PROBLEMS or wants not in ("refund", "replacement"):
            return {"outcome": "invalid", "params": params, "errors": [
                "payment_id (from my_purchases), problem, wants (refund or replacement: ask the customer) and message are all required"]}
        purchase = next((p for p in customer_purchases(user, deps.executor) if p["payment_id"] == payment_id), None)
        if purchase is None:
            return {"outcome": "invalid", "params": params, "errors": [f"{payment_id} is not one of this customer's purchases: use my_purchases"]}
        if purchase["open_case"]:
            return {"outcome": "invalid", "params": params, "errors": [
                f"this purchase already has an open case ({purchase['open_case']}): use request_resolution on it"]}
        left = Decimal(purchase["left"])
        limit, why = left, "what's left of this payment"
        if purchase["paid_how"] == "in store" and wants == "refund":  # shop policy: in-store purchases get partial refunds only
            limit = min(left, (Decimal(purchase["amount"]) * IN_STORE_REFUND_SHARE).quantize(Decimal("0.01"), rounding=ROUND_DOWN))
            why = (f"the shop's policy: in-store purchases can get at most {IN_STORE_REFUND_SHARE:.0%} back; "
                   "tell the customer and offer the partial refund or a replacement at the store")
        try:
            amount = Decimal(str(args.get("amount") or limit)).quantize(Decimal("0.01"))
        except Exception:
            return {"outcome": "invalid", "params": params, "errors": ["amount must be a number like 29.99"]}
        if not Decimal("0.01") <= amount <= limit:
            return {"outcome": "invalid", "params": params, "errors": [f"amount must be between 0.01 and {limit} ({why})"]}
        params |= {"amount": {"currency_code": "USD", "value": f"{amount:.2f}"}, "items": ", ".join(purchase["items"]) or "your purchase"}
        ask = f"a refund of {amount:.2f} USD" if wants == "refund" else "a replacement"
        when = purchase["date"][:10]
        return confirm(f"Report a problem with {params['items']} (paid {purchase['amount']} USD on {when}) and ask the shop "
                       f"for {ask}?\n\n“{message[:600]}”", REPORT_PROBLEM, params)

    def run_resolution(user: User, params: dict, session_id: str | None, cid: str) -> tuple[str, bool]:
        """Send the message to the shop, then record what the customer asked for. Returns (result for the LLM, ok)."""
        send = {"dispute_id": params["dispute_id"], "message": params["message"]}
        result = deps.executor.execute("send_message_about_dispute_to_other_party", send,
                                       request_id=f"pm-{session_id}-{cid}", caller=user)
        card = deps.registry.get("send_message_about_dispute_to_other_party") or {}
        deps.appdb.log_action(user, REQUEST_RESOLUTION, params, "success" if result.ok else "failed", session_id=session_id,
                              method=card.get("method"), path=card.get("path"), http_status=result.status_code,
                              result_summary=(result.error or f"{params['wants']} requested on {params['dispute_id']}")[:300],
                              confirmed=True, request_id=result.request_id)
        if not result.ok:
            return compact({"ok": False, "error": result.error}), False
        deps.appdb.request_refund(user.user_id, params["dispute_id"], params["amount"], params["message"], wants=params["wants"])
        return compact({"ok": True, "result": f"Sent. The dispute now shows '{params['wants'].capitalize()} requested' until the "
                                              "shop answers. Tell the customer they can attach a photo in the case: PayMind data → "
                                              "Disputes → open it → 📎 Attach photo."}), True

    def run_report_problem(user: User, params: dict, session_id: str | None) -> tuple[str, bool]:
        """Open the case with the shop (mock: POST /mock/disputes), then record refund or replacement."""
        body = {"capture_id": params["payment_id"], "reason": PROBLEMS[params["problem"]], "amount": params["amount"],
                "message": params["message"]}
        r = deps.executor.client.post(f"{deps.executor.base_url}/mock/disputes", json=body)
        ok = r.status_code < 300
        dispute = r.json() if ok else {}
        deps.appdb.log_action(user, REPORT_PROBLEM, params, "success" if ok else "failed", session_id=session_id,
                              method="POST", path="/mock/disputes", http_status=r.status_code,
                              result_summary=(f"case {dispute.get('dispute_id')} opened" if ok else r.text)[:300], confirmed=True)
        if not ok:
            return compact({"ok": False, "error": r.text[:300]}), False
        deps.appdb.request_refund(user.user_id, dispute["dispute_id"], params["amount"], params["message"], wants=params["wants"])
        return compact({"ok": True, "dispute_id": dispute["dispute_id"], "result": (
            f"Case opened with the shop; it shows '{params['wants'].capitalize()} requested'. The shop has until "
            f"{dispute.get('seller_response_due_date', '')[:10]} to respond. Ask the customer to attach a photo of the problem: "
            "PayMind data → Disputes → open the case → 📎 Attach photo.")}), True

    # ---- tools: run what the gate allowed ----------------------------------------------------
    def run_builtin(name: str, args: dict, user: User, state: AgentState) -> tuple[str, list[str]]:
        if name == CHECK_UPDATES:
            from ..app.updates import whats_new

            items = whats_new(user, deps.executor, deps.appdb)
            return compact(items or "Nothing new: no new messages, no replies owed, nothing waiting."), []
        if name == RAG_SEARCH:
            if deps.docs_search is None:
                return "The policy documents aren't available right now.", []
            hits = [h for h in deps.docs_search(args.get("query", ""), k=5, source=args.get("source")) if h.relevant]
            if not hits:
                return "Nothing relevant found in the policy documents. Say you couldn't find it; don't guess.", []
            return compact([{"source": f"{h.title}{' › ' + h.section if h.section else ''}",
                             "from": "the shop's own policy" if h.source == "shop" else "PayPal",
                             **({"url": h.url} if h.url else {}), "text": h.text} for h in hits]), []
        if name == ORDER_STATUS:
            from ..app.updates import sync_po_payment

            pos = deps.appdb.list_pos(user_id=user.user_id) if user.is_customer else \
                [p for p in deps.appdb.list_pos() if p["status"] != "draft"]
            if args.get("po_id"):
                pos = [p for p in pos if p["po_id"] == args["po_id"]]
            keep = ("po_id", "status", "customer_po_ref", "items", "requested_date", "expected_date", "invoice_id",
                    "paid_at", "carrier", "tracking_number", "shipped_at", "delivered_at", "not_received_note", "reject_reason")
            rows = [{k: p.get(k) for k in keep} for p in (sync_po_payment(p, deps.executor, deps.appdb) for p in pos)]
            return compact(rows or "No purchase orders found."), []
        if name == CUSTOMER_PAYMENTS:
            from ..app.purchases import customer_payments

            return compact(customer_payments(args.get("query", ""), deps.executor, caller=user, item=args.get("item"))), []
        if name == MY_PURCHASES:
            from ..app.purchases import customer_purchases

            rows = customer_purchases(user, deps.executor, item=args.get("item"))
            return compact(rows or "No purchases in the last 90 days."), []
        if name == FIND_TOOLS:
            hits = deps.search(args.get("query", ""), role=user.role, k=TOP_K)
            return compact([{"tool": n, "description": d} for n, d in hits]), [n for n, _ in hits]
        if args.get("mode") == "activity":
            rows = deps.appdb.recent_actions(user.user_id, limit=int(args.get("limit") or 5),
                                             tool=args.get("tool"), status=args.get("status"))
            keep = ("time", "tool", "status", "http_status", "result_summary", "confirmed", "params")
            return compact([{k: r[k] for k in keep} for r in rows] or "No past requests found."), []
        hits = deps.search(args.get("query") or "what can you do", role=user.role, k=10, include_eval_only=False)
        return compact([{"tool": n, "description": d} for n, d in hits]), []

    def tools(state: AgentState, config) -> dict:
        user = user_of(config, deps.appdb)
        session_id = config["configurable"].get("thread_id")
        offered = list(state.get("offered", []))
        seen = state.get("seen", "")
        actions = list(state.get("actions", []))
        messages = []
        for call in state["messages"][-1].tool_calls:
            name, cid = call["name"], call["id"]
            decision = state["decisions"].get(cid, {"outcome": "invalid", "params": {}, "errors": ["not checked"]})
            params = decision["params"]
            card = deps.registry.get(name) or {}
            audit = dict(session_id=session_id, method=card.get("method"), path=card.get("path"))

            action = {"tool": name, "label": action_label(card, name, params), "write": card.get("action_type") == "write"}
            if name in (REQUEST_RESOLUTION, REPORT_PROBLEM, CLOSE_CASE):
                title = {"refund": "Request refund", "replacement": "Request replacement"}.get(params.get("wants"), "Request")
                if name == REPORT_PROBLEM:
                    title = f"Report problem · {title.split()[-1]}"
                if name == CLOSE_CASE:
                    title = f"Close case {params.get('dispute_id', '')}"
                action = {"tool": name, "label": action_label({"title": title}, name, params if params.get("wants") != "replacement" else {}), "write": True}
                if decision["outcome"] == "needs_confirmation" and decision.get("approved"):
                    content, ok = (run_resolution(user, params, session_id, cid) if name == REQUEST_RESOLUTION
                                   else run_close_case(user, params, session_id) if name == CLOSE_CASE
                                   else run_report_problem(user, params, session_id))
                    actions.append(action | ({"outcome": "done"} if ok else {"outcome": "failed", "error": "the shop's system refused it"}))
                elif decision["outcome"] == "needs_confirmation":
                    content = "The user declined, so nothing was done. Tell them it was cancelled; don't retry."
                    deps.appdb.log_action(user, name, params, "declined", **audit)
                    actions.append(action | {"outcome": "declined"})
                elif decision["outcome"] == "blocked":
                    content = "Not allowed: " + "; ".join(decision["errors"]) + ". Tell the user plainly; don't retry."
                    deps.appdb.log_action(user, name, params, "blocked", result_summary=content, **audit)
                    actions.append(action | {"outcome": "blocked"})
                else:
                    content = "Not run. Fix these problems and try again: " + "; ".join(decision["errors"])
            elif name in CUSTOMER_ONLY | SHOP_ONLY and decision["outcome"] == "blocked":
                content = "Not allowed: " + "; ".join(decision["errors"]) + ". Tell the user plainly; don't retry."
            elif name in BUILTINS:
                content, new_tools = run_builtin(name, params, user, state)
                offered += [t for t in new_tools if t not in offered]
                actions.append(action | {"write": False, "outcome": "done"})
            elif decision["outcome"] == "invalid":  # sent back to the LLM to fix; not an action yet
                content = "Not run. Fix these problems and try again: " + "; ".join(decision["errors"])
            elif decision["outcome"] == "blocked":
                content = "Not allowed: " + "; ".join(decision["errors"]) + ". Tell the user plainly; don't retry."
                deps.appdb.log_action(user, name, params, "blocked", result_summary=content, **audit)
                actions.append(action | {"outcome": "blocked"})
            elif decision["outcome"] == "needs_confirmation" and not decision.get("approved"):
                content = "The user declined, so nothing was done. Tell them it was cancelled; don't retry."
                deps.appdb.log_action(user, name, params, "declined", **audit)
                actions.append(action | {"outcome": "declined"})
            else:
                result = deps.executor.execute(name, params, request_id=f"pm-{session_id}-{cid}", caller=user)
                body = filter_results(user, name, result.body)
                if result.ok and name == "show_dispute_details" and isinstance(body, dict) and body.get("dispute_id"):
                    from ..app.updates import mark_seen

                    mark_seen(deps.appdb, user, body)  # the user sees the thread in chat: same as opening it
                content = compact({"ok": result.ok, "status": result.status_code, "result": body}
                                  if result.ok else {"ok": False, "status": result.status_code, "error": result.error})
                deps.appdb.log_action(
                    user, name, params, "success" if result.ok else "failed", http_status=result.status_code,
                    result_summary=(result.error or f"{result.method} {result.path} -> {result.status_code}")[:300],
                    confirmed=decision["outcome"] == "needs_confirmation", request_id=result.request_id, **audit)
                refunded = (body or {}).get("refund", {}).get("amount") if result.ok and isinstance(body, dict) else None
                if refunded:  # say what was actually refunded (the request may not name an amount)
                    action["label"] = f"Refund {refunded['value']} {refunded['currency_code']} (dispute {params.get('dispute_id', '')})"
                actions.append(action | ({"outcome": "done"} if result.ok else {"outcome": "failed", "error": result.error}))
            seen += f"\ntool {name}: {content}"
            messages.append(ToolMessage(content=content, tool_call_id=cid, name=name))
        return {"messages": messages, "offered": offered[:MAX_OFFERED + 5], "seen": seen, "decisions": {}, "actions": actions}

    graph = StateGraph(AgentState)
    graph.add_node("search_tools", search_tools)
    graph.add_node("agent", agent)
    graph.add_node("gate", gate)
    graph.add_node("tools", tools)
    graph.add_edge(START, "search_tools")
    graph.add_edge("search_tools", "agent")
    graph.add_conditional_edges("agent", after_agent, ["gate", END])
    graph.add_edge("gate", "tools")
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


# ---- the receipt: what really happened, from code ----------------------------------------------

def action_label(card: dict, name: str, params: dict) -> str:
    """Short plain name for an action, e.g. "Refund captured payment (79.99 USD)"."""
    label = card.get("title") or name.replace("_", " ").capitalize()
    amounts = ", ".join(f"{m['value']} {m['currency_code']}" for _, m in money_values(params or {}))
    return f"{label} ({amounts})" if amounts else label


RECEIPT_MARKS = {"done": "✅ Done", "failed": "❌ Failed", "declined": "🚫 Cancelled by you", "blocked": "⛔ Not allowed"}


def receipt(actions: list[dict]) -> dict:
    """What changed during one message. Reads only count as lookups; every write is listed with its outcome."""
    changes = [{"label": a["label"], "outcome": a["outcome"], **({"error": a["error"]} if a.get("error") else {})}
               for a in actions if a["write"]]
    lookups = sum(1 for a in actions if not a["write"] and a["outcome"] == "done")
    lines = [f"{RECEIPT_MARKS[c['outcome']]}: {c['label']}" for c in changes]
    if not any(c["outcome"] == "done" for c in changes):
        lines.append("No changes were made")
    if lookups:
        lines.append(f"{lookups} lookup{'s' if lookups != 1 else ''}")
    return {"changes": changes, "lookups": lookups, "summary": " · ".join(lines)}


# ---- a small wrapper for chat screens -----------------------------------------------------

@dataclass
class Reply:
    text: str | None                 # the assistant's answer (None while waiting for a yes/no)
    confirmation: dict | None        # {"question", "tool", "params", "large_amount"} when waiting
    receipt: dict | None = None      # what really ran for this message (see receipt()); None while waiting


class PayMindAgent:
    def __init__(self, deps: Deps, checkpointer):
        self.deps = deps
        self.graph = build_graph(deps, checkpointer)

    def _config(self, user_id: str, session_id: str) -> dict:
        user = self.deps.appdb.get_user(user_id)
        role = user.role if user else "unknown"
        return {"configurable": {"thread_id": session_id, "user_id": user_id}, "recursion_limit": 50,
                "run_name": "paymind_chat", "tags": [f"role:{role}"],
                "metadata": {"user_id": user_id, "role": role, "session_id": session_id}}

    def _reply(self, result: dict) -> Reply:
        pending = result.get("__interrupt__")
        if pending:
            return Reply(None, pending[0].value)
        last = result["messages"][-1]
        text = last.content if isinstance(last.content, str) else " ".join(
            p.get("text", "") for p in last.content if isinstance(p, dict))
        return Reply(text, None, receipt(result.get("actions", [])))

    def send(self, user_id: str, session_id: str, text: str) -> Reply:
        return self._reply(self.graph.invoke({"messages": [HumanMessage(text)]}, self._config(user_id, session_id)))

    def answer(self, user_id: str, session_id: str, approve: bool) -> Reply:
        """Answer a pending yes/no confirmation."""
        return self._reply(self.graph.invoke(Command(resume="yes" if approve else "no"), self._config(user_id, session_id)))

    # ---- streaming: the answer's words as Gemini writes them ----------------------------------

    def stream(self, user_id: str, session_id: str, text: str | None = None, approve: bool | None = None):
        """Same run as send()/answer(), but yields events while it runs:

          {"type": "token", "text": ...}   a piece of the answer, as Gemini writes it
          {"type": "restart"}              a new LLM answer began (e.g. after a tool ran, or a fallback
                                           model took over): the page clears what it showed so far
          {"type": "done", "reply", "confirmation", "receipt"}   the final result, exactly as send() returns it

        Only text written by the agent node is streamed; tool calls, tool results and the pause for a
        yes/no are not. The final "done" event is the source of truth (the page replaces the streamed
        text with it).
        """
        config = self._config(user_id, session_id)
        run_input = {"messages": [HumanMessage(text)]} if approve is None else Command(resume="yes" if approve else "no")
        current = None
        for chunk, meta in self.graph.stream(run_input, config, stream_mode="messages"):
            if meta.get("langgraph_node") != "agent" or not isinstance(chunk, (AIMessageChunk, AIMessage)):
                continue
            piece = chunk.content if isinstance(chunk.content, str) else "".join(
                p.get("text", "") for p in chunk.content if isinstance(p, dict) and p.get("type", "text") == "text")
            if not piece:
                continue
            if chunk.id != current:
                current = chunk.id
                yield {"type": "restart"}
            yield {"type": "token", "text": piece}
        snapshot = self.graph.get_state(config)
        pending = [i for task in snapshot.tasks for i in task.interrupts]
        if pending:
            reply = Reply(None, pending[0].value)
        else:
            reply = self._reply(snapshot.values)
        yield {"type": "done", "reply": reply.text, "confirmation": reply.confirmation, "receipt": reply.receipt}
