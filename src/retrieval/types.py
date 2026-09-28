"""The single result type every retriever returns.

Dense, sparse, fused and reranked retrieval all hand back `list[RetrievalResult]`,
so they are interchangeable and directly comparable. Each result carries the whole
`Chunk`, not just an id, which is what guarantees that anything reaching the
generation layer already has its citation attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.schema import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int = 0
    method: str = ""
    # Per-method detail kept for explainability, e.g. {"dense": 0.81, "bm25": 12.4}.
    components: dict[str, float] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def __repr__(self) -> str:
        return (f"<{self.rank}. {self.score:.4f} [{self.method}] "
                f"{self.chunk.title[:40]} | {self.chunk.section}>")


def renumber(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Assign 1-based ranks in list order."""
    for i, r in enumerate(results, start=1):
        r.rank = i
    return results
