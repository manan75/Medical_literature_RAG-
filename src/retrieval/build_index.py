"""Phase 3+4 entry point: embed every chunk and load it into Chroma.

    python -m src.retrieval.build_index            # rebuild from chunks.jsonl
    python -m src.retrieval.build_index --check    # nearest-neighbour sanity check
    python -m src.retrieval.build_index --append   # embed only chunks not yet indexed

Rebuild is destructive by design: the collection is dropped first, so a changed
chunking strategy can never leave stale vectors behind that still answer queries.
"""

from __future__ import annotations

import argparse
import time

from src import config
from src.embeddings import encoder
from src.retrieval.vector_store import VectorStore
from src.schema import load_chunks

# Hand-written probes covering the query shapes the system must handle: a named
# adverse-effect lookup, a section-targeted lookup, an interaction lookup, and a
# disease-level question. Printing their neighbours is the Phase 3 "definition of
# done" -- a qualitative check that the biomedical encoder puts sensible chunks first.
CHECK_QUERIES = [
    "What are the common side effects of metformin?",
    "warfarin contraindications",
    "Can aspirin interact with warfarin?",
    "first line treatment for type 2 diabetes",
    "serotonin syndrome symptoms",
]


def build(batch_size: int = 16) -> VectorStore:
    if not config.CHUNKS_FILE.exists():
        raise SystemExit(
            f"{config.CHUNKS_FILE} not found. Run: python -m src.chunking.pipeline"
        )

    chunks = load_chunks(config.CHUNKS_FILE)
    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL}")

    start = time.monotonic()
    embeddings = encoder.embed_chunks(chunks, batch_size=batch_size)
    elapsed = time.monotonic() - start
    print(f"  embedded in {elapsed:.1f}s "
          f"({len(chunks) / max(elapsed, 1e-9):.1f} chunks/s, "
          f"dim={len(embeddings[0])})")

    store = VectorStore()
    if store.count():
        print(f"  dropping existing collection ({store.count()} vectors)")
        store.reset()

    store.add(chunks, embeddings)
    print(f"  indexed {store.count()} vectors -> "
          f"{config.VECTORSTORE_DIR.relative_to(config.ROOT)}")
    return store


def append(batch_size: int = 16, chunks=None,
           store: VectorStore | None = None) -> VectorStore:
    """Embed only chunks missing from the collection and add them.

    A full rebuild re-embeds everything (5-20 min on CPU). When a new source is
    added and the existing chunks are unchanged, only the new ones need vectors.
    Chunk ids are deterministic ("<doc_id>::<section>::<part>"), so "missing from
    the collection" is exactly "new". BM25 needs nothing: it is rebuilt from
    chunks.jsonl on every load.
    """
    chunks = chunks if chunks is not None else load_chunks(config.CHUNKS_FILE)
    store = store or VectorStore()
    have = store.ids()
    before = len(have)

    stale = have - {c.chunk_id for c in chunks}
    if stale:
        # Not deleted automatically: stale vectors mean the chunking changed, and
        # the safe fix is a full rebuild, not a partial patch.
        print(f"  WARNING: {len(stale)} indexed chunk(s) no longer exist in "
              f"chunks.jsonl -- run a full rebuild (no --append)")

    new = [c for c in chunks if c.chunk_id not in have]
    by_source: dict[str, int] = {}
    for c in new:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    print(f"Appending {len(new)} new chunk(s) to {before} existing vectors "
          f"{by_source or ''}")
    if not new:
        return store

    start = time.monotonic()
    store.add(new, encoder.embed_chunks(new, batch_size=batch_size))
    print(f"  embedded in {time.monotonic() - start:.1f}s")

    after = store.count()
    if after != before + len(new):
        raise SystemExit(f"Count mismatch: {before} + {len(new)} != {after}")
    print(f"  collection: {before} + {len(new)} = {after} vectors")
    return store


def check(store: VectorStore | None = None, top_k: int = 3) -> None:
    store = store or VectorStore()
    if store.count() == 0:
        raise SystemExit("Index is empty. Run: python -m src.retrieval.build_index")

    print("\n" + "=" * 74)
    print("NEAREST-NEIGHBOUR CHECK (dense retrieval only)")
    print("=" * 74)
    for q in CHECK_QUERIES:
        print(f"\nQ: {q}")
        for r in store.search(q, top_k=top_k):
            print(f"  {r.rank}. {r.score:.3f}  {r.chunk.title[:44]:44s} "
                  f"[{r.chunk.section[:26]}]")
            print(f"       {r.chunk.text[:150].strip()}...")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the vector index.")
    ap.add_argument("--check", action="store_true",
                    help="Run nearest-neighbour probes after building.")
    ap.add_argument("--check-only", action="store_true",
                    help="Skip the rebuild and only run the probes.")
    ap.add_argument("--append", action="store_true",
                    help="Keep the existing vectors; embed and add only new chunks.")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config.ensure_dirs()
    if args.check_only:
        store = VectorStore()
    elif args.append:
        store = append(batch_size=args.batch_size)
    else:
        store = build(batch_size=args.batch_size)
    if args.check or args.check_only:
        check(store)


if __name__ == "__main__":
    main()
