"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion.

Why RRF rather than a weighted sum of scores: cosine similarity lives in [0, 1]
and BM25 scores are unbounded and corpus-dependent, so any weighted blend of the
two needs normalisation constants that have to be retuned whenever the corpus
changes. RRF throws the magnitudes away and fuses on *rank* alone:

    score(d) = sum over retrievers of  1 / (k + rank_r(d))

with k = 60 (the value from Cormack et al., 2009, and the usual default). It needs
no tuning, and it lets a document that both retrievers rank moderately well beat
one that a single retriever loves — which is the behaviour we want, because
agreement between a semantic and a lexical signal is decent evidence of relevance.
"""

from __future__ import annotations

from src import config
from src.retrieval.sparse import BM25Index
from src.retrieval.types import RetrievalResult, renumber
from src.retrieval.vector_store import VectorStore


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievalResult]],
    k: int | None = None,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse several ranked lists into one. Keys of `ranked_lists` name the method."""
    k = config.RRF_K if k is None else k

    fused: dict[str, float] = {}
    best: dict[str, RetrievalResult] = {}
    components: dict[str, dict[str, float]] = {}

    for method, results in ranked_lists.items():
        for r in results:
            cid = r.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r.rank)
            components.setdefault(cid, {})[method] = r.score
            components[cid][f"{method}_rank"] = float(r.rank)
            # Keep one copy of the chunk; identical across retrievers.
            best.setdefault(cid, r)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if top_k:
        ordered = ordered[:top_k]

    out = [
        RetrievalResult(
            chunk=best[cid].chunk, score=score, method="hybrid",
            components=components[cid],
        )
        for cid, score in ordered
    ]
    return renumber(out)


class HybridRetriever:
    """Dense + sparse retrieval over the same chunk set, fused with RRF."""

    def __init__(self, vector_store: VectorStore | None = None,
                 bm25: BM25Index | None = None):
        self.vector_store = vector_store or VectorStore()
        self.bm25 = bm25 or BM25Index.from_file()

    def search(self, query: str, top_k: int | None = None,
               dense_k: int | None = None, sparse_k: int | None = None,
               ) -> list[RetrievalResult]:
        # Each retriever contributes a deeper candidate list than we return, so
        # fusion has room to promote documents the other one ranked poorly.
        dense = self.vector_store.search(query, top_k=dense_k or config.DENSE_TOP_K)
        sparse = self.bm25.search(query, top_k=sparse_k or config.SPARSE_TOP_K)
        return reciprocal_rank_fusion(
            {"dense": dense, "bm25": sparse},
            top_k=top_k or config.DENSE_TOP_K,
        )

    def search_all_methods(self, query: str, top_k: int = 10,
                           ) -> dict[str, list[RetrievalResult]]:
        """Return dense, sparse and hybrid side by side, for comparison in Phase 4
        evaluation and for the --compare mode of the search CLI."""
        dense = self.vector_store.search(query, top_k=config.DENSE_TOP_K)
        sparse = self.bm25.search(query, top_k=config.SPARSE_TOP_K)
        hybrid = reciprocal_rank_fusion({"dense": dense, "bm25": sparse}, top_k=top_k)
        return {"dense": dense[:top_k], "bm25": sparse[:top_k], "hybrid": hybrid}
