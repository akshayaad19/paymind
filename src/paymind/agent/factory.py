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


def gemini_timeout() -> float:
    """Seconds to wait for one model before moving on to the next. Without a limit, a model that
    stops answering makes the request hang forever."""
    return float(os.getenv("GEMINI_TIMEOUT", "45"))


COOLDOWN_SECONDS = 300  # a model that timed out or was overloaded is skipped for this long


class ModelChain:
    """Try models in order, skipping ones that can't answer right now:
      - daily quota used up → skipped until the next day
      - timed out or overloaded (503) → skipped for 5 minutes, then tried again
    so calls don't keep waiting on a model that's down. If every model is being skipped,
    they're all tried anyway rather than failing without asking."""

    def __init__(self, bound: list[tuple[str, object]], exhausted: dict[str, str], clock=None):
        import time

        self.bound, self.exhausted = bound, exhausted  # exhausted: model → day, or "until:<epoch seconds>"
        self.clock = clock or time.time

    def _skipped(self, model: str, today: str) -> bool:
        mark = self.exhausted.get(model, "")
        return mark == today or (mark.startswith("until:") and float(mark[6:]) > self.clock())

    def invoke(self, messages):
        from datetime import date

        today, last_error = date.today().isoformat(), None
        available = [(m, llm) for m, llm in self.bound if not self._skipped(m, today)]
        for model, llm in available or [(m, llm) for m, llm in self.bound if self.exhausted.get(m) != today]:
            try:
                return llm.invoke(messages)
            except Exception as exc:  # busy, rate-limited, timed out, out of quota: try the next model
                last_error = exc
                text = f"{type(exc).__name__} {exc}"
                if "PerDay" in text:
                    self.exhausted[model] = today
                elif any(s in text for s in ("503", "UNAVAILABLE", "Timeout", "timed out", "DeadlineExceeded")):
                    self.exhausted[model] = f"until:{self.clock() + COOLDOWN_SECONDS}"
        raise last_error or RuntimeError("every model is out of daily quota")


def qdrant_docs_search():
    """The RAG knowledge base (Qdrant "docs" collection), traced in LangSmith."""
    from langsmith import traceable

    from ..retrieval.doc_index import search_docs
    from ..retrieval.tool_index import get_client

    client = get_client()

    @traceable(name="rag_search", run_type="chain")
    def search(query: str, k: int = 5, source: str | None = None):
        return search_docs(client, query, k=k, source=source)

    return search


def gemini_llm_factory(models: list[str]):
    """Bind tools per call; if a model fails (overloaded, out of quota), try the next."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    clients = [(m, ChatGoogleGenerativeAI(model=m, max_retries=0, timeout=gemini_timeout())) for m in models]
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
        docs_search=qdrant_docs_search(),
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
