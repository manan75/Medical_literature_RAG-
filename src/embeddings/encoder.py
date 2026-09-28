"""Biomedical embedding generation.

Model: PubMedBERT-based sentence embeddings (see CLAUDE.md §3 for why a
domain-specific model matters here -- a general-purpose encoder scores
"metformin" and "metoprolol" as near neighbours on surface form alone, which is
exactly the confusion a drug-safety system cannot afford).

Contextualised embedding text
-----------------------------
A chunk is not embedded as bare text. It is embedded as

    <document title> | <section heading>
    <chunk text>

because a chunk from "Drug Interactions" and a chunk from "Adverse Reactions" can
be near-identical in wording while answering completely different questions. The
heading is the disambiguator, so it has to be inside the vector, not only in the
metadata beside it. The stored `text` stays clean -- only the embedding input is
decorated, so citations and LLM context never show the prefix.
"""

from __future__ import annotations

from src import config
from src.schema import Chunk

_model = None  # loaded lazily: importing this module must not pull ~420 MB


def get_model(model_name: str | None = None):
    """Load (and cache) the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        name = model_name or config.EMBEDDING_MODEL
        print(f"Loading embedding model: {name} (first run downloads ~420 MB)")
        _model = SentenceTransformer(name)
    return _model


def embedding_text(chunk: Chunk) -> str:
    """The contextualised string actually handed to the encoder."""
    header = " | ".join(p for p in (chunk.title, chunk.section) if p)
    return f"{header}\n{chunk.text}" if header else chunk.text


def embed_texts(texts: list[str], batch_size: int = 16,
                show_progress: bool = True) -> list[list[float]]:
    """Encode raw strings. Vectors are L2-normalised, so cosine similarity is a
    plain dot product and Chroma's distance is directly comparable across queries."""
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


def embed_chunks(chunks: list[Chunk], batch_size: int = 16,
                 show_progress: bool = True) -> list[list[float]]:
    return embed_texts([embedding_text(c) for c in chunks],
                       batch_size=batch_size, show_progress=show_progress)


def embed_query(query: str) -> list[float]:
    """Encode a single user query. Symmetric with chunk encoding -- the model is a
    bi-encoder trained for symmetric similarity, so no query prefix is applied."""
    return embed_texts([query], show_progress=False)[0]


def embedding_dimension() -> int:
    return get_model().get_sentence_embedding_dimension()
