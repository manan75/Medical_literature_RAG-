"""Central configuration. Every path and model name used by the pipeline lives here.

Reads from a .env file at the repo root (see .env.example). Nothing in this module
requires a key to be present -- modules that actually need one check at call time,
so the retrieval half of the pipeline runs fully offline.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---- Paths ----
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PUBMED_DIR = RAW_DIR / "pubmed"
RAW_FDA_DIR = RAW_DIR / "fda"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Canonical artifact files passed between pipeline stages.
DOCUMENTS_FILE = PROCESSED_DIR / "documents.jsonl"
CHUNKS_FILE = CHUNKS_DIR / "chunks.jsonl"

# ---- Models ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---- NCBI E-utilities ----
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_TOOL = os.getenv("NCBI_TOOL", "medical-literature-rag")

# ---- Chunking ----
CHUNK_TARGET_TOKENS = 350   # approximate; see src/chunking for the word-based proxy
CHUNK_OVERLAP_TOKENS = 60
CHUNK_MIN_TOKENS = 40       # below this a chunk is merged into its neighbour

# ---- Retrieval ----
COLLECTION_NAME = "medical_literature"
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RRF_K = 60                  # reciprocal-rank-fusion damping constant
RERANK_TOP_K = 5            # chunks handed to the LLM after reranking


def ensure_dirs() -> None:
    """Create every data directory the pipeline writes to."""
    for d in (
        RAW_PUBMED_DIR, RAW_FDA_DIR, PROCESSED_DIR, CHUNKS_DIR, VECTORSTORE_DIR
    ):
        d.mkdir(parents=True, exist_ok=True)
