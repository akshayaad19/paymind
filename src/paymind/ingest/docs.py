"""Build the RAG knowledge base (offline).

    python -m paymind.ingest.docs fetch     # download PayPal's pages → data/docs/paypal/ (git-ignored)
    python -m paymind.ingest.docs build     # chunk every document → Qdrant collection "docs"
    python -m paymind.ingest.docs search "what's the refund window?"

Two sources:
  data/docs/shop/*.md     the shop's own policies (written for this project, committed)
  data/docs/paypal/*.md   PayPal's public pages, downloaded locally from data/docs/paypal_sources.json.
                          They're PayPal's copyrighted text, so they're fetched, never committed.

Chunking: split by headings first (so a chunk stays about one topic, and remembers
its section), then long sections by size: about 500 tokens (~2,000 characters)
with ~50 tokens (~200 characters) of overlap, so a sentence cut in two still
appears whole in one chunk.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "data/docs"
SOURCES = DOCS / "paypal_sources.json"
CHUNK_CHARS, OVERLAP_CHARS = 2000, 200
MIN_WORDS = 80  # a download with less text than this is a JavaScript shell, not the page
SHOP_TOPICS = {"refunds-and-returns": "refunds", "shipping-and-delivery": "shipping", "purchase-orders": "purchase_orders",
               "invoices-and-payments": "invoices", "disputes-and-contact": "disputes"}
HEADER = re.compile(r"^<!-- (.*) -->\n")


def fetch(sources: Path = SOURCES, out: Path = DOCS / "paypal") -> None:
    import httpx
    import trafilatura

    out.mkdir(parents=True, exist_ok=True)
    for src in json.loads(sources.read_text()):
        try:
            html = httpx.get(src["url"], headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True, timeout=30).text
        except httpx.HTTPError as exc:
            print(f"  skip {src['slug']}: {type(exc).__name__}")
            continue
        text = trafilatura.extract(html, output_format="markdown", include_links=False, include_tables=True, favor_recall=True) or ""
        if len(text.split()) < MIN_WORDS:
            print(f"  skip {src['slug']}: only {len(text.split())} words (page needs JavaScript)")
            continue
        meta = json.dumps({"title": src["title"], "url": src["url"], "topic": src["topic"], "fetched": time.strftime("%Y-%m-%d")})
        (out / f"{src['slug']}.md").write_text(f"<!-- {meta} -->\n{text}\n")
        print(f"  {src['slug']}: {len(text.split()):,} words")
        time.sleep(1)  # be polite to paypal.com


def load_documents(root: Path = DOCS) -> list[dict]:
    """[{doc, source, title, topic, url, text}] for every shop and downloaded PayPal document."""
    docs = []
    for path in sorted((root / "shop").glob("*.md")):
        text = path.read_text()
        title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
        docs.append({"doc": path.stem, "source": "shop", "title": title, "topic": SHOP_TOPICS.get(path.stem, "general"),
                     "url": None, "text": text})
    for path in sorted((root / "paypal").glob("*.md")):
        raw = path.read_text()
        match = HEADER.match(raw)
        meta = json.loads(match.group(1)) if match else {"title": path.stem, "topic": "general", "url": None}
        docs.append({"doc": path.stem, "source": "paypal", "title": meta["title"], "topic": meta.get("topic", "general"),
                     "url": meta.get("url"), "text": raw[match.end():] if match else raw})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    by_heading = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=True)
    by_size = RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARS, chunk_overlap=OVERLAP_CHARS)
    chunks = []
    for doc in docs:
        index = 0
        for section in by_heading.split_text(doc["text"]):
            heading = section.metadata.get("h3") or section.metadata.get("h2")  # h1 is the document title
            for piece in by_size.split_text(section.page_content):
                piece = re.sub(r"\n{3,}", "\n\n", piece).strip()
                if len(piece.split()) < 8:  # stray headings, empty list markers
                    continue
                chunks.append({**{k: doc[k] for k in ("doc", "source", "title", "topic", "url")},
                               "section": heading, "text": piece, "index": index})
                index += 1
    return chunks


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download PayPal's pages")
    sub.add_parser("build", help="chunk and index every document")
    s = sub.add_parser("search", help="try a question")
    s.add_argument("query")
    s.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    if args.cmd == "fetch":
        fetch()
        return
    from ..retrieval.doc_index import build_doc_index, search_docs
    from ..retrieval.tool_index import get_client

    client = get_client()
    if args.cmd == "build":
        docs = load_documents()
        chunks = chunk_documents(docs)
        start = time.time()
        count = build_doc_index(client, chunks)
        per = {}
        for c in chunks:
            per[c["title"]] = per.get(c["title"], 0) + 1
        print(f"Indexed {count} chunks from {len(docs)} documents in {time.time() - start:.0f}s")
        for title, n in per.items():
            print(f"  {n:4}  {title}")
    else:
        for i, h in enumerate(search_docs(client, args.query, rerank=not args.no_rerank), 1):
            print(f"{i}. [{h.score:+.2f} {'relevant' if h.relevant else 'weak'}] {h.title}{' › ' + h.section if h.section else ''} ({h.source})")
            print("   " + h.text[:220].replace("\n", " ") + ("…" if len(h.text) > 220 else ""))


if __name__ == "__main__":
    main()
