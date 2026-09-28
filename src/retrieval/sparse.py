"""BM25 keyword retrieval.

Why keep a sparse retriever at all when we have biomedical embeddings: dense
retrieval generalises but blurs exact tokens, and in this domain exact tokens are
often the whole query. "eGFR 30", "CYP3A4", "500 mg", "INR" and brand names are
rare strings that BM25 matches precisely and an embedding may smear across
near-synonyms. Hybrid retrieval (src/retrieval/hybrid.py) exists to get both.

The index is rebuilt in memory from chunks.jsonl rather than pickled: it takes
well under a second at this corpus size, and it can never go stale against the
chunk file the way a serialised index can.
"""

from __future__ import annotations

import re

from src import config
from src.retrieval.types import RetrievalResult, renumber
from src.schema import Chunk, load_chunks

# Deliberately minimal. Aggressive stopword removal would strip "no", "not" and
# "without" -- negation words that flip the clinical meaning of a sentence.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "as",
    "at", "by", "from", "is", "are", "was", "were", "be", "been", "that", "this",
    "these", "those", "it", "its", "which", "who", "whom", "what", "when", "how",
}

# Keep digits and internal punctuation: "cyp3a4", "500mg", "eGFR" must survive.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """In-memory BM25 index over chunk text plus its section heading.

    The heading is indexed alongside the text so that a query naming a section
    ("metformin contraindications") can match on the heading even when the body
    never repeats that word.
    """

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        corpus = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        # BM25Okapi divides by average document length, so an empty corpus is fatal.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def from_file(cls, path=None) -> "BM25Index":
        path = path or config.CHUNKS_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m src.chunking.pipeline"
            )
        return cls(load_chunks(path))

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or config.SPARSE_TOP_K
        if self._bm25 is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        out: list[RetrievalResult] = []
        for i in order[:top_k]:
            # A zero score means no query term appeared; that is not a result.
            if scores[i] <= 0:
                break
            out.append(RetrievalResult(
                chunk=self.chunks[i], score=float(scores[i]), method="bm25",
                components={"bm25": float(scores[i])},
            ))
        return renumber(out)
