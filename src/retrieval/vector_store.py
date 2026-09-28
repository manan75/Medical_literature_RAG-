"""Chroma-backed dense retrieval.

Embeddings are computed by us (src/embeddings/encoder.py) and handed to Chroma
explicitly rather than letting Chroma pick its own default encoder -- the whole
point of Phase 3 is that the encoder is a biomedical one.

Chroma metadata values must be scalars, so `drug_names` is stored as a
comma-joined string and split back out on read. The full chunk is reconstructed
on the way out so that every result is citable without a second lookup.
"""

from __future__ import annotations

from src import config
from src.embeddings import encoder
from src.retrieval.types import RetrievalResult, renumber
from src.schema import Chunk


def _client():
    import chromadb
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))


def _to_metadata(c: Chunk) -> dict:
    return {
        "doc_id": c.doc_id,
        "section": c.section,
        "source": c.source,
        "title": c.title,
        "url": c.url,
        "date": c.date,
        "drug_names": ",".join(c.drug_names),
        "token_estimate": c.token_estimate,
    }


def _from_record(chunk_id: str, text: str, meta: dict) -> Chunk:
    drugs = meta.get("drug_names") or ""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=meta.get("doc_id", ""),
        text=text,
        section=meta.get("section", ""),
        source=meta.get("source", ""),
        title=meta.get("title", ""),
        url=meta.get("url", ""),
        date=meta.get("date", ""),
        drug_names=[d for d in drugs.split(",") if d],
        token_estimate=int(meta.get("token_estimate") or 0),
    )


class VectorStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self, collection_name: str | None = None, client=None):
        self.collection_name = collection_name or config.COLLECTION_NAME
        self._client = client or _client()
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            # Vectors are L2-normalised, so cosine is the matching space.
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection -- an index rebuild must not leave
        chunks from a previous chunking strategy lying around."""
        self._client.delete_collection(self.collection_name)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]],
            batch_size: int = 256) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings[i:i + batch_size],
                metadatas=[_to_metadata(c) for c in batch],
            )

    def search(self, query: str, top_k: int | None = None,
               where: dict | None = None) -> list[RetrievalResult]:
        """Dense semantic search. Returns results scored in [0, 1], higher better."""
        top_k = top_k or config.DENSE_TOP_K
        if self.count() == 0:
            return []

        res = self.collection.query(
            query_embeddings=[encoder.embed_query(query)],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: list[RetrievalResult] = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            similarity = 1.0 - float(dist)  # cosine distance -> cosine similarity
            chunk = _from_record(cid, text, meta)
            out.append(RetrievalResult(
                chunk=chunk, score=similarity, method="dense",
                components={"dense": similarity},
            ))
        return renumber(out)
