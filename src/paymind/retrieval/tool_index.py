"""Build and search the tool index in Qdrant.

Every tool card becomes one Qdrant point with two vectors and a payload:

  - "bm25":  sparse keyword vector (fastembed Qdrant/bm25: stop words removed,
             stemmed, hashed, counted; Qdrant applies IDF)
  - "dense": meaning vector (fastembed BAAI/bge-small-en-v1.5, 384 numbers)
  - payload: name, category, service, action_type, allowed_roles, eval_only

Vectors are made locally by fastembed; only the finished vectors go to Qdrant.
Search runs both vector searches, merges them with Reciprocal Rank Fusion and
filters by the user's role.

Where the index lives is set in .env:
  QDRANT_URL / QDRANT_API_KEY set  -> Qdrant Cloud (or any Qdrant server)
  not set                          -> local folder ./qdrant_data

The index is rebuilt from tools.json, so it can always be recreated.

Usage:
    python -m paymind.retrieval.tool_index build
    python -m paymind.retrieval.tool_index search "refund order 123" --role accountant
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[3]
COLLECTION = "tools"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_SIZE = 384
SPARSE_MODEL = "Qdrant/bm25"
PREFETCH_LIMIT = 20  # candidates from each search before fusion

# Payload fields that get an index so filtering stays fast on a real server.
PAYLOAD_INDEXES = {
    "name": models.PayloadSchemaType.KEYWORD,
    "allowed_roles": models.PayloadSchemaType.KEYWORD,
    "category": models.PayloadSchemaType.KEYWORD,
    "service": models.PayloadSchemaType.KEYWORD,
    "eval_only": models.PayloadSchemaType.BOOL,
}


@dataclass
class ToolHit:
    name: str
    score: float
    category: str
    action_type: str


def search_text(tool: dict) -> str:
    """The text that gets embedded: only the fields that help match a user's words."""
    return " ".join([
        tool["name"].replace("_", " "),
        tool["category"].replace("_", " "),
        tool["description"],
        *tool.get("example_questions", []),
    ])


def point_id(tool_name: str, service: str) -> str:
    """Stable ID, so re-indexing the same tool overwrites it instead of duplicating it."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"paymind/{service}/{tool_name}"))


def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=60)
    return QdrantClient(path=str(ROOT / "qdrant_data"))


def create_collection(client: QdrantClient, name: str = COLLECTION) -> None:
    """(Re)create the collection with a dense and a sparse (BM25, IDF) vector."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field, field_schema=schema)


def to_point(tool: dict) -> models.PointStruct:
    text = search_text(tool)
    return models.PointStruct(
        id=point_id(tool["name"], tool["service"]),
        vector={
            "dense": models.Document(text=text, model=DENSE_MODEL),
            "bm25": models.Document(text=text, model=SPARSE_MODEL),
        },
        payload={
            "name": tool["name"],
            "service": tool["service"],
            "category": tool["category"],
            "action_type": tool["action_type"],
            "allowed_roles": tool["allowed_roles"],
            "eval_only": tool.get("eval_only", False),
        },
    )


def build_index(client: QdrantClient, tools: list[dict], name: str = COLLECTION, batch_size: int = 64) -> int:
    """Recreate the collection and index every tool. Returns the number of points stored."""
    create_collection(client, name)
    for i in range(0, len(tools), batch_size):
        client.upsert(name, points=[to_point(t) for t in tools[i : i + batch_size]])
    return client.count(name).count


def search_tools(
    client: QdrantClient,
    query: str,
    role: str | None = None,
    k: int = 5,
    include_eval_only: bool = True,
    name: str = COLLECTION,
) -> list[ToolHit]:
    """Hybrid search: BM25 + dense, merged with RRF, filtered by role.

    include_eval_only=False hides the synthetic scaling-test tools (used by
    System Search, so users never see fake tools listed).
    """
    conditions: list[models.Condition] = []
    if role:
        conditions.append(models.FieldCondition(key="allowed_roles", match=models.MatchValue(value=role)))
    if not include_eval_only:
        conditions.append(models.FieldCondition(key="eval_only", match=models.MatchValue(value=False)))
    query_filter = models.Filter(must=conditions) if conditions else None

    result = client.query_points(
        name,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=SPARSE_MODEL), using="bm25",
                filter=query_filter, limit=PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=DENSE_MODEL), using="dense",
                filter=query_filter, limit=PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
    )
    return [
        ToolHit(p.payload["name"], p.score, p.payload["category"], p.payload["action_type"])
        for p in result.points
    ]


def load_tools(path: Path = ROOT / "data/tools/tools.json") -> list[dict]:
    return json.loads(path.read_text())


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="(re)build the tool index from tools.json")
    b.add_argument("--tools", type=Path, default=ROOT / "data/tools/tools.json")
    s = sub.add_parser("search", help="try a query")
    s.add_argument("query")
    s.add_argument("--role", choices=["customer", "accountant"])
    s.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    client = get_client()
    where = os.getenv("QDRANT_URL") or "local ./qdrant_data"
    if args.cmd == "build":
        tools = load_tools(args.tools)
        start = time.time()
        count = build_index(client, tools)
        print(f"Indexed {count} tools into '{COLLECTION}' at {where} in {time.time() - start:.1f}s")
    else:
        start = time.time()
        hits = search_tools(client, args.query, role=args.role, k=args.k)
        print(f"{args.query!r} (role={args.role or 'any'}, {(time.time() - start) * 1000:.0f} ms, {where})")
        for i, h in enumerate(hits, 1):
            print(f"  {i}. {h.name:45} {h.score:.3f}  [{h.category}, {h.action_type}]")


if __name__ == "__main__":
    main()
