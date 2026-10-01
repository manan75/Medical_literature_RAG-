"""Central configuration. Every path and model name used by the pipeline lives here.

Reads from a .env file at the repo root (see .env.example). Nothing in this module
requires a key to be present -- modules that actually need one check at call time,
so the retrieval half of the pipeline runs fully offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Windows pipes default to cp1252, which crashes on medical text ("≥", "µg").
# Every CLI imports this module, so fix it once here.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ---- Paths ----
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PUBMED_DIR = RAW_DIR / "pubmed"
RAW_FDA_DIR = RAW_DIR / "fda"
RAW_MEDLINEPLUS_DIR = RAW_DIR / "medlineplus"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Canonical artifact files passed between pipeline stages.
DOCUMENTS_FILE = PROCESSED_DIR / "documents.jsonl"
CHUNKS_FILE = CHUNKS_DIR / "chunks.jsonl"

# ---- Models ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ---- NCBI E-utilities ----
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
NCBI_TOOL = os.getenv("NCBI_TOOL", "medical-literature-rag")

# ---- Chunking ----
CHUNK_TARGET_TOKENS = 350   # approximate; see src/chunking for the word-based proxy
CHUNK_OVERLAP_TOKENS = 60
CHUNK_MIN_TOKENS = 40       # below this a trailing chunk is merged into its neighbour
CHUNK_NOISE_FLOOR = 10      # a lone chunk this short carries no retrievable meaning

# ---- Retrieval ----
COLLECTION_NAME = "medical_literature"
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RRF_K = 60                  # reciprocal-rank-fusion damping constant
RERANK_TOP_K = 5            # chunks handed to the LLM after reranking

# ---- Generation (Phase 6) ----
# Thresholds are on the cross-encoder's raw logit scale (ms-marco MiniLM), chosen
# from observed --eval scores; see SESSION_LOG.md 2026-09-29 for the numbers.
NOT_FOUND_THRESHOLD = 0.0   # chunks below this are dropped; none left -> "not found"
HIGH_SCORE = 5.0            # best chunk at/above this counts as strong evidence
CONFIDENCE_SPREAD = 3.0     # docs scoring within this of the best one "agree"


def ensure_dirs() -> None:
    """Create every data directory the pipeline writes to."""
    for d in (
        RAW_PUBMED_DIR, RAW_FDA_DIR, RAW_MEDLINEPLUS_DIR, PROCESSED_DIR, CHUNKS_DIR, VECTORSTORE_DIR
    ):
        d.mkdir(parents=True, exist_ok=True)
