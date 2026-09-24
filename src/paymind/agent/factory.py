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

    client = get_client()

    def search(query: str, role: str | None = None, k: int = 5, include_eval_only: bool = True):
        hits = search_tools(client, query, role=role, k=k, include_eval_only=include_eval_only)
        return [(h.name, (registry.get(h.name) or {}).get("description", "")) for h in hits]

    return search


def gemini_llm_factory(models: list[str]):
    """Bind tools per call; if the first model fails (overloaded, out of quota), try the next."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    clients = [ChatGoogleGenerativeAI(model=m, max_retries=1) for m in models]

    def llm_for(schemas: list[dict]):
        bound = [c.bind_tools(schemas) for c in clients]
        return bound[0].with_fallbacks(bound[1:]) if len(bound) > 1 else bound[0]

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
