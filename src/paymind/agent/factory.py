"""Wire the agent to the real pieces: Qdrant tool search, Gemini, the mock
PayPal server, the app database and a SQLite checkpointer.

Try it in the terminal (mock server must be running):
    python -m paymind.agent.factory --user u_asha
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import uuid
from pathlib import Path

from .executor import Executor, ToolRegistry
from .graph import Deps, PayMindAgent

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ROOT / "data/app/checkpoints.db"


def qdrant_search(registry: ToolRegistry):
    from ..retrieval.tool_index import get_client, search_tools

    from langsmith import traceable

    client = get_client()

    @traceable(name="tool_search", run_type="chain")
    def traced_search(query: str, role: str | None, k: int, include_eval_only: bool) -> list[dict]:
        """What the trace shows: each tool with its score and who may use it."""
        return [{"rank": i, "tool": h.name, "score": round(h.score, 3), "type": h.action_type,
                 "allowed_roles": (registry.get(h.name) or {}).get("allowed_roles", []),
                 "description": (registry.get(h.name) or {}).get("description", "")}
                for i, h in enumerate(search_tools(client, query, role=role, k=k, include_eval_only=include_eval_only), 1)]

    def search(query: str, role: str | None = None, k: int = 5, include_eval_only: bool = True):
        return [(h["tool"], h["description"]) for h in traced_search(query, role, k, include_eval_only)]

    return search


class ModelChain:
    """Try models in order. A model whose DAILY quota is used up is skipped until the next day,
    so calls don't waste time on models that can't answer today."""

    def __init__(self, bound: list[tuple[str, object]], exhausted: dict[str, str]):
        self.bound, self.exhausted = bound, exhausted

    def invoke(self, messages):
        from datetime import date

        today, last_error = date.today().isoformat(), None
        for model, llm in self.bound:
            if self.exhausted.get(model) == today:
                continue
            try:
                return llm.invoke(messages)
            except Exception as exc:  # busy, rate-limited, out of quota: try the next model
                last_error = exc
                if "PerDay" in str(exc):
                    self.exhausted[model] = today
        raise last_error or RuntimeError("every model is out of daily quota")


def gemini_llm_factory(models: list[str]):
    """Bind tools per call; if a model fails (overloaded, out of quota), try the next."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    clients = [(m, ChatGoogleGenerativeAI(model=m, max_retries=0)) for m in models]
    exhausted: dict[str, str] = {}  # model -> day its daily quota ran out (shared by every call)

    def llm_for(schemas: list[dict]):
        return ModelChain([(m, c.bind_tools(schemas)) for m, c in clients], exhausted)

    return llm_for


def agent_models() -> list[str]:
    main = os.getenv("GEMINI_AGENT_MODEL") or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash gemini-3.5-flash gemini-3.5-flash-lite").split()
    return list(dict.fromkeys([main, *fallbacks]))


def build_agent() -> PayMindAgent:
    from dotenv import load_dotenv
    from langgraph.checkpoint.sqlite import SqliteSaver

    from ..app.database import AppDatabase

    load_dotenv(ROOT / ".env")
    registry = ToolRegistry()
    CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
    deps = Deps(
        llm_for=gemini_llm_factory(agent_models()),
        search=qdrant_search(registry),
        executor=Executor(registry),
        registry=registry,
        appdb=AppDatabase(),
    )
    return PayMindAgent(deps, SqliteSaver(sqlite3.connect(CHECKPOINTS, check_same_thread=False)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with PayMind in the terminal.")
    ap.add_argument("--user", default="u_asha", help="u_asha (accountant), u_rahul or u_priya (customers)")
    ap.add_argument("--session", default=None, help="reuse a session id to continue a conversation")
    args = ap.parse_args()

    agent = build_agent()
    session = args.session or f"cli-{uuid.uuid4().hex[:8]}"
    print(f"PayMind · user {args.user} · session {session} · type 'quit' to exit\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        reply = agent.send(args.user, session, text)
        while reply.confirmation:
            answer = input(f"confirm> {reply.confirmation['question']} (yes/no) ").strip().lower()
            reply = agent.answer(args.user, session, answer in ("y", "yes"))
        print(f"paymind> {reply.text}\n")


if __name__ == "__main__":
    main()
