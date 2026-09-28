"""Cross-encoder reranking: precision pass over the hybrid candidate list.

Why a second model at all. The bi-encoder in Phase 3 embeds the query and the
chunk *independently*, so it can only ever compare two summaries of meaning; that
is what makes it fast enough to search 1,300 chunks, and also what makes it blur
distinctions like "metformin" vs "metformin and sitagliptin". A cross-encoder
reads the query and one chunk *together* in a single forward pass and scores the
pair directly, which is far more accurate and far too slow to run over a whole
corpus. So the pipeline uses each where it is strong: hybrid retrieval casts a
wide net (~20 candidates), the cross-encoder reorders them, and only the top few
reach the LLM.

The reranker is general-domain (MS MARCO), not biomedical. It is judging
query-chunk *relevance*, not clinical semantics, and the biomedical signal is
already carried by the retrieval stage that produced the candidates.
"""

from __future__ import annotations

from src import config
from src.retrieval.types import RetrievalResult, renumber

_model = None  # lazy: importing this module must not pull the cross-encoder


def get_model(model_name: str | None = None):
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        name = model_name or config.RERANKER_MODEL
        print(f"Loading reranker: {name} (first run downloads ~90 MB)")
        _model = CrossEncoder(name)
    return _model


def rerank(query: str, results: list[RetrievalResult],
           top_k: int | None = None, batch_size: int = 16,
           ) -> list[RetrievalResult]:
    """Reorder `results` by cross-encoder relevance, keeping the top `top_k`.

    The original retrieval score is preserved in `components` so that the effect
    of reranking stays inspectable rather than being silently overwritten.
    """
    top_k = top_k or config.RERANK_TOP_K
    if not results:
        return []
    if len(results) == 1:
        results[0].method = "reranked"
        return renumber(results)

    model = get_model()
    # Pair the query with the same contextualised text the retriever indexed, so
    # the section heading is visible to the reranker too.
    pairs = [
        (query, f"{r.chunk.title} | {r.chunk.section}\n{r.chunk.text}")
        for r in results
    ]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    for r, score in zip(results, scores):
        r.components["retrieval_score"] = r.score
        r.components["retrieval_rank"] = float(r.rank)
        r.components["rerank"] = float(score)
        r.score = float(score)
        r.method = "reranked"

    results.sort(key=lambda r: r.score, reverse=True)
    return renumber(results[:top_k])


class RerankingRetriever:
    """Hybrid retrieval followed by cross-encoder reranking.

    This is the full Phase 1-5 retrieval stack and the object the generation layer
    should depend on.
    """

    def __init__(self, hybrid=None, candidate_k: int | None = None,
                 top_k: int | None = None):
        from src.retrieval.hybrid import HybridRetriever
        self.hybrid = hybrid or HybridRetriever()
        # Rerank a deeper list than we return: the reranker can only promote what
        # retrieval handed it, so a too-shallow candidate list caps its usefulness.
        self.candidate_k = candidate_k or config.DENSE_TOP_K
        self.top_k = top_k or config.RERANK_TOP_K

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        candidates = self.hybrid.search(query, top_k=self.candidate_k)
        return rerank(query, candidates, top_k=top_k or self.top_k)

    def search_with_candidates(self, query: str, top_k: int | None = None,
                               ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """Return (reranked, pre-rerank candidates) for before/after comparison."""
        candidates = self.hybrid.search(query, top_k=self.candidate_k)
        before = [
            RetrievalResult(chunk=c.chunk, score=c.score, rank=c.rank,
                            method=c.method, components=dict(c.components))
            for c in candidates
        ]
        return rerank(query, candidates, top_k=top_k or self.top_k), before
