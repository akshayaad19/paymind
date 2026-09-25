"""The knowledge base for RAG: policy documents, chunked and indexed in Qdrant.

Same hybrid search as the tool index (BM25 sparse + bge-small dense, merged
with RRF), on a separate collection "docs". On top of that, a cross-encoder
reranker reads the question and each candidate chunk TOGETHER and re-sorts the
top 20; its score also tells us when nothing relevant was found, so the
assistant can say "I couldn't find that" instead of guessing.

Each point: dense + sparse vectors of the chunk, and a payload with the text,
the document title, the section it came from, its source ("shop" or "paypal"),
topic and URL.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from .tool_index import DENSE_MODEL, DENSE_SIZE, PREFETCH_LIMIT, SPARSE_MODEL

COLLECTION = "docs"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"
# Reranker scores are relevance logits: above this, a chunk is treated as actually answering the question.
RELEVANT_SCORE = float(os.getenv("RAG_RELEVANT_SCORE", "0.0"))


@dataclass
class DocHit:
    text: str
    title: str
    section: str | None
    source: str
    url: str | None
    score: float                 # reranker score (or RRF score when reranking is off)
    relevant: bool


def indexed_text(c: dict) -> str:
    """What gets embedded (and reranked) for a chunk: document title, section heading, then the text."""
    return f"{c['title']} — {c['section']}\n{c['text']}" if c.get("section") else f"{c['title']}\n{c['text']}"


def chunk_id(source: str, doc: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"paymind/docs/{source}/{doc}/{index}"))


def build_doc_index(client: QdrantClient, chunks: list[dict], name: str = COLLECTION, batch_size: int = 64) -> int:
    """Recreate the collection and index every chunk. chunks: {text, title, section, source, topic, url, doc, index}."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in ("source", "topic"):
        client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
    for i in range(0, len(chunks), batch_size):
        points = []
        for c in chunks[i: i + batch_size]:
            # the heading path is embedded with the text, so "refund window" also matches a section titled that way
            text = indexed_text(c)
            points.append(models.PointStruct(
                id=chunk_id(c["source"], c["doc"], c["index"]),
                vector={"dense": models.Document(text=text, model=DENSE_MODEL), "bm25": models.Document(text=text, model=SPARSE_MODEL)},
                payload={k: c.get(k) for k in ("text", "title", "section", "source", "topic", "url", "doc", "index")},
            ))
        client.upsert(name, points=points)
    return client.count(name).count


_reranker = None


def reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def search_docs(client: QdrantClient, query: str, k: int = 5, source: str | None = None, rerank: bool = True,
                name: str = COLLECTION) -> list[DocHit]:
    """Hybrid search (BM25 + dense, RRF) → top 20 → optional cross-encoder rerank → top k."""
    query_filter = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]) if source else None
    result = client.query_points(
        name,
        prefetch=[
            models.Prefetch(query=models.Document(text=query, model=SPARSE_MODEL), using="bm25", filter=query_filter, limit=PREFETCH_LIMIT),
            models.Prefetch(query=models.Document(text=query, model=DENSE_MODEL), using="dense", filter=query_filter, limit=PREFETCH_LIMIT),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=PREFETCH_LIMIT if rerank else k,
    )
    points = result.points
    if not points:
        return []
    if rerank:
        # the reranker reads the same heading + text that was embedded; a chunk's heading often
        # carries the key words ("If your order doesn't arrive") that its body doesn't repeat
        scores = list(reranker().rerank(query, [indexed_text(p.payload) for p in points]))
        ranked = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)[:k]
        return [DocHit(p.payload["text"], p.payload["title"], p.payload.get("section"), p.payload["source"], p.payload.get("url"),
                       round(float(s), 3), float(s) > RELEVANT_SCORE) for p, s in ranked]
    return [DocHit(p.payload["text"], p.payload["title"], p.payload.get("section"), p.payload["source"], p.payload.get("url"),
                   round(p.score, 3), True) for p in points[:k]]
