"""RAG knowledge base: loading, chunking, hybrid search + reranker, and the assistant's rag_search tool."""

import json

import pytest
from qdrant_client import QdrantClient

from paymind.ingest.docs import chunk_documents, load_documents
from paymind.retrieval.doc_index import build_doc_index, indexed_text, search_docs

SHOP_DOCS = [d for d in load_documents() if d["source"] == "shop"]


def test_shop_policies_load_with_titles_and_topics():
    titles = {d["doc"]: (d["title"], d["topic"]) for d in SHOP_DOCS}
    assert titles["refunds-and-returns"] == ("Refunds and returns (PayMind Demo Store)", "refunds")
    assert {d["topic"] for d in SHOP_DOCS} == {"refunds", "shipping", "purchase_orders", "invoices", "disputes"}


def test_chunks_keep_their_section_heading():
    chunks = chunk_documents(SHOP_DOCS)
    window = next(c for c in chunks if c["section"] == "Return window")
    assert "30 days of delivery" in window["text"] and window["source"] == "shop"
    assert all(len(c["text"]) <= 2000 for c in chunks)
    assert len({(c["doc"], c["index"]) for c in chunks}) == len(chunks)  # stable, unique ids
    assert indexed_text(window).startswith("Refunds and returns (PayMind Demo Store) — Return window\n")


def test_long_sections_are_split_with_overlap():
    long = {"doc": "long", "source": "shop", "title": "Long", "topic": "general", "url": None,
            "text": "# Long\n## Part\n" + " ".join(f"sentence {i} about refunds." for i in range(800))}
    pieces = chunk_documents([long])
    assert len(pieces) > 3 and all(p["section"] == "Part" for p in pieces)
    assert pieces[0]["text"][-100:].split()[-1] in pieces[1]["text"]  # overlap: the cut is repeated


@pytest.fixture(scope="module")
def client():
    c = QdrantClient(":memory:")
    assert build_doc_index(c, chunk_documents(SHOP_DOCS)) > 10
    return c


@pytest.mark.parametrize("question,section", [
    ("How long do I have to return an item?", "Return window"),
    ("What happens if my order never arrives?", "If your order doesn't arrive"),   # the heading carries the words
    ("When are purchase order invoices due?", "Payment terms"),
    ("What are your support hours?", "Support hours"),
])
def test_questions_find_the_right_section(client, question, section):
    top = search_docs(client, question, k=3)[0]
    assert top.section == section and top.relevant


def test_off_topic_question_is_flagged_not_relevant(client):
    assert not any(h.relevant for h in search_docs(client, "What's the weather in Chennai tomorrow?", k=3))


def test_source_filter(client):
    assert search_docs(client, "refund", k=3, source="paypal") == []  # only shop docs in this index


# ---- the assistant's rag_search tool (fake knowledge base, no Gemini) -------------------------------------

def test_rag_search_tool_returns_only_relevant_passages_with_sources(tmp_path):
    from langchain_core.messages import AIMessage
    from test_agent_graph import ScriptedLLM, call, fake_search, tool_results, REGISTRY
    from paymind.agent.executor import Executor
    from paymind.agent.graph import Deps, PayMindAgent
    from paymind.app.database import AppDatabase
    from paymind.retrieval.doc_index import DocHit
    from langgraph.checkpoint.memory import InMemorySaver

    def docs(query, k=5, source=None):
        return [DocHit("You can return most items within 30 days of delivery.", "Refunds and returns", "Return window", "shop", None, 7.4, True),
                DocHit("Unrelated text.", "PayPal Privacy Statement", None, "paypal", "https://paypal.com/privacy", -6.0, False)]

    def no_docs(query, k=5, source=None):
        return [DocHit("Unrelated.", "PayPal User Agreement", None, "paypal", None, -9.0, False)]

    for search, expect in [(docs, "relevant"), (no_docs, "none")]:
        llm = ScriptedLLM([AIMessage("", tool_calls=[call("rag_search", {"query": "return window?"})]),
                           lambda msgs: AIMessage(json.dumps(tool_results(msgs)[-1]))])
        deps = Deps(llm_for=llm, search=fake_search([]), executor=Executor(REGISTRY, base_url="http://unused"),
                    registry=REGISTRY, appdb=AppDatabase(tmp_path / f"{expect}.db"), docs_search=search)
        out = json.loads(PayMindAgent(deps, InMemorySaver()).send("u_rahul", "s1", "what's the return window?").text)
        if expect == "relevant":
            assert out == [{"source": "Refunds and returns › Return window", "from": "the shop's own policy",
                            "text": "You can return most items within 30 days of delivery."}]
        else:
            assert out.startswith("Nothing relevant found")
