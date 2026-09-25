"""The PayMind agent: a LangGraph loop that connects search, the LLM, the
validator, customer scope, the executor and the audit log.

    search_tools ─► agent (LLM) ──tool calls?── no ──► reply
                       ▲               │ yes
                       │               ▼
                       │            gate   validate · customer scope · ⏸ ask yes/no
                       │               ▼
                       └── results ─ tools  run approved calls · filter · audit log

Why a separate gate: when LangGraph resumes after a pause it re-runs the
paused step from the start. The gate only checks and asks (nothing it does
changes data), and the tools step runs after the answer, so an action is never
executed twice. Each write also carries a PayPal-Request-Id built from its tool
call id, so even a repeat would be replayed by PayPal, not redone.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from ..app.database import AppDatabase, User
from .executor import Executor, ToolRegistry
from .scope import check_access, filter_results
from .tools import BUILTIN_SCHEMAS, BUILTINS, CHECK_UPDATES, FIND_TOOLS, ORDER_STATUS, SYSTEM_SEARCH, tool_schema
from .validator import validate

MAX_STEPS = 6          # LLM turns per user message
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


@dataclass
class Deps:
    llm_for: LLMFactory
    search: SearchFn
    executor: Executor
    registry: ToolRegistry
    appdb: AppDatabase
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
- The tools you were given were already chosen for this request. Call them directly, following each tool's example call; don't browse existing records or templates just to learn a format.
- For 'anything new?' or 'any messages?', call check_updates. It returns new_message (they wrote, unread), needs_reply (they wrote, user hasn't answered), action_needed (PayPal says it's the user's turn, with a response deadline: always mention days left or overdue) and no_reply_yet (user wrote days ago, no answer; offer to send a reminder). Always say how long ago things happened (e.g. '2 days ago'), using today's date. When you show a dispute's messages, say who wrote each one.
- For purchase orders ('where is my order?', tracking, delivery date, orders to ship), call order_status. Customers confirm delivery or report a missing order on the Orders tab.
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
        }

    # ---- agent: the LLM decides what to do next ---------------------------------------
    def agent(state: AgentState, config) -> dict:
        user = user_of(config, deps.appdb)
        steps = state.get("steps", 0) + 1
        if steps > MAX_STEPS:
            return {"steps": steps, "messages": [AIMessage(
                "I stopped after several steps without finishing. Here's where things stand above; "
                "please tell me how you'd like to continue.")]}
        schemas = [tool_schema(deps.registry.get(n)) for n in state.get("offered", []) if deps.registry.get(n)]
        llm = deps.llm_for(schemas + BUILTIN_SCHEMAS)
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
            name, args = call["name"], call.get("args") or {}
            if name in BUILTINS:
                decisions[call["id"]] = {"outcome": "ok", "params": args}
                continue
            v = validate(deps.registry.get(name), name, args, user, set(state.get("offered", [])), state.get("seen", ""))
            decision = {"outcome": v.outcome, "params": v.params, "errors": v.errors}
            if v.outcome in ("ok", "needs_confirmation"):
                reason = check_access(user, v.params, deps.executor)
                if reason:
                    decision = {"outcome": "blocked", "params": v.params, "errors": [reason]}
            if decision["outcome"] == "needs_confirmation":
                answer = interrupt({"question": v.confirmation, "tool": name, "params": v.params, "large_amount": v.large_amount})
                decision["approved"] = str(answer).strip().lower() in YES
            decisions[call["id"]] = decision
        return {"decisions": decisions}

    # ---- tools: run what the gate allowed ----------------------------------------------------
    def run_builtin(name: str, args: dict, user: User, state: AgentState) -> tuple[str, list[str]]:
        if name == CHECK_UPDATES:
            from ..app.updates import whats_new

            items = whats_new(user, deps.executor, deps.appdb)
            return compact(items or "Nothing new: no new messages, no replies owed, nothing waiting."), []
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
        messages = []
        for call in state["messages"][-1].tool_calls:
            name, cid = call["name"], call["id"]
            decision = state["decisions"].get(cid, {"outcome": "invalid", "params": {}, "errors": ["not checked"]})
            params = decision["params"]
            card = deps.registry.get(name) or {}
            audit = dict(session_id=session_id, method=card.get("method"), path=card.get("path"))

            if name in BUILTINS:
                content, new_tools = run_builtin(name, params, user, state)
                offered += [t for t in new_tools if t not in offered]
            elif decision["outcome"] == "invalid":
                content = "Not run. Fix these problems and try again: " + "; ".join(decision["errors"])
            elif decision["outcome"] == "blocked":
                content = "Not allowed: " + "; ".join(decision["errors"]) + ". Tell the user plainly; don't retry."
                deps.appdb.log_action(user, name, params, "blocked", result_summary=content, **audit)
            elif decision["outcome"] == "needs_confirmation" and not decision.get("approved"):
                content = "The user declined, so nothing was done. Tell them it was cancelled; don't retry."
                deps.appdb.log_action(user, name, params, "declined", **audit)
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
            seen += f"\ntool {name}: {content}"
            messages.append(ToolMessage(content=content, tool_call_id=cid, name=name))
        return {"messages": messages, "offered": offered[:MAX_OFFERED + 5], "seen": seen, "decisions": {}}

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


# ---- a small wrapper for chat screens -----------------------------------------------------

@dataclass
class Reply:
    text: str | None                 # the assistant's answer (None while waiting for a yes/no)
    confirmation: dict | None        # {"question", "tool", "params", "large_amount"} when waiting


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
        return Reply(text, None)

    def send(self, user_id: str, session_id: str, text: str) -> Reply:
        return self._reply(self.graph.invoke({"messages": [HumanMessage(text)]}, self._config(user_id, session_id)))

    def answer(self, user_id: str, session_id: str, approve: bool) -> Reply:
        """Answer a pending yes/no confirmation."""
        return self._reply(self.graph.invoke(Command(resume="yes" if approve else "no"), self._config(user_id, session_id)))
