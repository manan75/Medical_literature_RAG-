# AI-Powered Medical Literature Retrieval and Drug Interaction Analysis (RAG)

A Retrieval-Augmented Generation system that answers questions about diseases,
symptoms, treatments and medications by retrieving grounded evidence from trusted
medical literature and drug-label databases — and that analyses potential
drug–drug interactions with cited evidence.

> **Medical disclaimer.** This system is for **medical information retrieval and
> education only**. It does not diagnose, prescribe, or provide medical advice.
> Every clinical answer it produces carries citations to the retrieved source
> material. Always consult a qualified healthcare professional.

---

## Architecture

```
Medical Documents (PubMed · PMC · FDA labels · PDF · HTML)
    ↓  src/data_sources/     harvest raw files
    ↓  src/ingestion/        extract + normalise → Documents
    ↓  src/chunking/         section-aware chunking → Chunks
    ↓  src/embeddings/       biomedical embeddings (PubMedBERT)
    ↓  src/retrieval/        Chroma vector DB + BM25 → hybrid (RRF)
    ↓  src/retrieval/        cross-encoder reranking
    ↓  src/generation/       grounded LLM answer (Gemini)
    ↓
Grounded Medical Response + Sources + Evidence
```

## Setup

```bash
git clone https://github.com/manan75/Medical_literature_RAG-.git
cd Medical_literature_RAG-

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

cp .env.example .env            # then fill in the keys you have
```

**Keys.** None are required for the retrieval half of the pipeline.

| Variable | Needed for | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | answer generation (Phase 6+) | https://aistudio.google.com/apikey — free tier |
| `NCBI_API_KEY` | faster PubMed harvesting (optional) | https://account.ncbi.nlm.nih.gov/settings/ |

## Usage

```bash
# 1. Build the raw corpus (hits NCBI + openFDA)
python -m src.data_sources.harvest            # add --small for a quick subset

# 2. Extract and normalise into data/processed/documents.jsonl
python -m src.ingestion.pipeline

# 3. Chunk into retrieval units → data/chunks/chunks.jsonl
python -m src.chunking.pipeline

# 4. Embed every chunk and load Chroma (~7 min on CPU for ~1300 chunks)
python -m src.retrieval.build_index --check

# 5. Query it
python -m src.retrieval.search "What are the common side effects of metformin?"
python -m src.retrieval.search "metformin contraindications" --compare   # dense vs BM25 vs hybrid
python -m src.retrieval.search "warfarin aspirin interaction" --rerank   # before vs after reranking
python -m src.retrieval.search --eval                                    # fixed probe set

# Tests
python -m pytest
```

Data sources: **PubMed / PMC** via the NCBI E-utilities API (public domain
metadata; PMC full text restricted to the open-access subset) and **FDA drug
labels** via openFDA (US Government public domain). No licensed or proprietary
datasets are used.

## Project documentation

- **[CLAUDE.md](CLAUDE.md)** — architecture, working rules, and the phased roadmap
  with live progress checkboxes.
- **[SESSION_LOG.md](SESSION_LOG.md)** — dated record of every working session:
  what was built, what was decided, and what comes next.

## Status

See the roadmap in [CLAUDE.md §4](CLAUDE.md). **Phases 0–5 complete** (ingestion → chunking → embeddings → hybrid retrieval → reranking). Phase 6 (grounded generation) is next and needs `GEMINI_API_KEY`.
