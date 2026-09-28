"""Retrieval CLI -- query the corpus and inspect what comes back.

    python -m src.retrieval.search "side effects of metformin"
    python -m src.retrieval.search "metformin contraindications" --compare
    python -m src.retrieval.search --eval

`--compare` prints dense, BM25 and hybrid side by side, which is how the Phase 4
definition of done ("hybrid outperforms either method alone on manual inspection")
is actually checked.
"""

from __future__ import annotations

import argparse

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.types import RetrievalResult

# A fixed probe set, deliberately mixing query shapes that favour different
# retrievers: exact drug names and lab thresholds (BM25's strength), paraphrased
# clinical questions (dense's strength), and interaction lookups (needs both).
EVAL_QUERIES = [
    "What are the common side effects of metformin?",
    "metformin contraindications renal impairment",
    "Can aspirin interact with warfarin?",
    "CYP3A4 inhibitors and simvastatin",
    "first line treatment for type 2 diabetes",
    "serotonin syndrome symptoms",
    "levothyroxine dosing in hypothyroidism",
    "What monitoring is needed for patients on warfarin?",
]


def _print_results(results: list[RetrievalResult], limit: int = 5,
                   indent: str = "  ") -> None:
    if not results:
        print(f"{indent}(no results)")
        return
    for r in results[:limit]:
        detail = ""
        if r.method == "hybrid":
            parts = []
            if "dense_rank" in r.components:
                parts.append(f"d#{int(r.components['dense_rank'])}")
            if "bm25_rank" in r.components:
                parts.append(f"b#{int(r.components['bm25_rank'])}")
            detail = f"  ({'+'.join(parts) if parts else 'single'})"
        print(f"{indent}{r.rank}. {r.score:.4f}{detail}  "
              f"{r.chunk.title[:42]:42s} [{r.chunk.section[:24]}]")


def run_query(retriever: HybridRetriever, query: str, top_k: int,
              compare: bool) -> None:
    print(f"\n{'=' * 78}\nQ: {query}\n{'=' * 78}")

    if not compare:
        results = retriever.search(query, top_k=top_k)
        _print_results(results, limit=top_k)
        if results:
            top = results[0]
            print(f"\n  top chunk ({top.chunk.citation()}):")
            print(f"  {top.chunk.text[:400].strip()}...")
        return

    by_method = retriever.search_all_methods(query, top_k=top_k)
    for method in ("dense", "bm25", "hybrid"):
        print(f"\n  --- {method.upper()} ---")
        _print_results(by_method[method], limit=top_k, indent="  ")

    # Where the two signals disagree is where fusion actually earns its keep.
    dense_ids = {r.chunk_id for r in by_method["dense"][:top_k]}
    bm25_ids = {r.chunk_id for r in by_method["bm25"][:top_k]}
    hybrid_ids = {r.chunk_id for r in by_method["hybrid"][:top_k]}
    print(f"\n  overlap: dense&bm25 {len(dense_ids & bm25_ids)}/{top_k}"
          f" | hybrid contributes {len(hybrid_ids - dense_ids)} chunk(s)"
          f" that dense alone missed")


def run_rerank_query(reranker, query: str, top_k: int) -> None:
    """Print the candidate list before and after reranking, side by side."""
    after, before = reranker.search_with_candidates(query, top_k=top_k)
    print(f"\n{'=' * 78}\nQ: {query}\n{'=' * 78}")

    print("\n  --- BEFORE (hybrid) ---")
    _print_results(before, limit=top_k)

    print("\n  --- AFTER (cross-encoder rerank) ---")
    for r in after:
        moved = int(r.components.get("retrieval_rank", r.rank)) - r.rank
        arrow = f"  (was #{int(r.components['retrieval_rank'])}, {moved:+d})" if moved else ""
        print(f"  {r.rank}. {r.score:+8.3f}{arrow}  "
              f"{r.chunk.title[:40]:40s} [{r.chunk.section[:24]}]")

    promoted = [r for r in after
                if r.components.get("retrieval_rank", 0) > top_k]
    if promoted:
        print(f"\n  reranking promoted {len(promoted)} chunk(s) from outside the "
              f"original top {top_k}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the medical corpus.")
    ap.add_argument("query", nargs="*", help="The query text.")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--compare", action="store_true",
                    help="Show dense vs BM25 vs hybrid side by side.")
    ap.add_argument("--rerank", action="store_true",
                    help="Show the candidate list before and after reranking.")
    ap.add_argument("--eval", action="store_true",
                    help="Run the fixed evaluation query set.")
    args = ap.parse_args()

    queries = EVAL_QUERIES if args.eval else [" ".join(args.query)]
    if not queries[0]:
        ap.error("give a query, or use --eval")

    if args.rerank:
        from src.retrieval.rerank import RerankingRetriever
        reranker = RerankingRetriever()
        print(f"Corpus: {reranker.hybrid.vector_store.count()} vectors, "
              f"{len(reranker.hybrid.bm25)} BM25 documents")
        for q in queries:
            run_rerank_query(reranker, q, args.top_k)
        return

    retriever = HybridRetriever()
    print(f"Corpus: {retriever.vector_store.count()} vectors, "
          f"{len(retriever.bm25)} BM25 documents")

    for q in queries:
        run_query(retriever, q, args.top_k, args.compare or args.eval)


if __name__ == "__main__":
    main()
