"""Phase 2 entry point: documents.jsonl -> chunks.jsonl.

    python -m src.chunking.pipeline            # chunk the corpus
    python -m src.chunking.pipeline --inspect  # also print sample chunks to eyeball
"""

from __future__ import annotations

import argparse
import statistics

from src import config
from src.chunking.splitter import chunk_documents
from src.schema import Chunk, load_documents, write_jsonl


def _report(chunks: list[Chunk]) -> None:
    sizes = [c.token_estimate for c in chunks]
    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1

    print(f"\n{len(chunks)} chunks")
    print(f"  tokens: min {min(sizes)} | median {int(statistics.median(sizes))} "
          f"| mean {int(statistics.mean(sizes))} | max {max(sizes)}")
    oversized = sum(1 for s in sizes if s > config.CHUNK_TARGET_TOKENS * 1.5)
    if oversized:
        print(f"  {oversized} chunk(s) exceed 1.5x target (expected: intact tables)")
    for source, n in sorted(by_source.items()):
        print(f"  {source:12s} {n}")

    sections = {c.section for c in chunks}
    print(f"  {len(sections)} distinct section headings preserved")


def _inspect(chunks: list[Chunk], n: int = 3) -> None:
    """Print a few chunks in full -- the Phase 2 definition of done is that a chunk
    read in isolation still makes sense, and that can only be checked by reading."""
    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS (read these in isolation -- do they stand alone?)")
    print("=" * 70)
    seen: set[str] = set()
    shown = 0
    for c in chunks:
        if c.source in seen and shown >= n:
            break
        if c.source in seen:
            continue
        seen.add(c.source)
        shown += 1
        print(f"\n[{c.chunk_id}]  ~{c.token_estimate} tokens")
        print(f"  cite: {c.citation()}")
        print(f"  {c.text[:500]}{'...' if len(c.text) > 500 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk the processed corpus.")
    ap.add_argument("--inspect", action="store_true",
                    help="Print sample chunks for manual spot-checking.")
    args = ap.parse_args()

    config.ensure_dirs()
    if not config.DOCUMENTS_FILE.exists():
        raise SystemExit(
            f"{config.DOCUMENTS_FILE} not found. Run: python -m src.ingestion.pipeline"
        )

    docs = load_documents(config.DOCUMENTS_FILE)
    print(f"Chunking {len(docs)} documents "
          f"(target ~{config.CHUNK_TARGET_TOKENS} tokens, "
          f"overlap ~{config.CHUNK_OVERLAP_TOKENS})...")

    chunks = chunk_documents(docs)
    write_jsonl(config.CHUNKS_FILE, chunks)
    _report(chunks)
    if args.inspect:
        _inspect(chunks)

    print(f"\nWrote {config.CHUNKS_FILE.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
