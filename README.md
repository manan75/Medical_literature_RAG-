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
Medical Documents (MedlinePlus · PubMed · PMC · FDA labels · PDF · HTML)
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

python -m venv .venv             # Python 3.12 or 3.13
.venv\Scripts\activate           # Windows (cmd / PowerShell)
# source .venv/Scripts/activate  # Windows Git Bash
# source .venv/bin/activate      # macOS / Linux

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
# 1. Build the raw corpus (hits NCBI + openFDA + MedlinePlus)
python -m src.data_sources.harvest            # add --small for a quick subset
python -m src.data_sources.harvest --medlineplus-only   # only refresh MedlinePlus

# 2. Extract and normalise into data/processed/documents.jsonl
python -m src.ingestion.pipeline

# 3. Chunk into retrieval units → data/chunks/chunks.jsonl
python -m src.chunking.pipeline

# 4. Embed every chunk and load Chroma (~15-30 min on CPU for ~3000 chunks)
python -m src.retrieval.build_index --check
#    ...or, after adding a new source with existing chunks unchanged,
#    embed only the chunks not yet in the index:
python -m src.retrieval.build_index --append

# 5. Query it
python -m src.retrieval.search "What are the common side effects of metformin?"
python -m src.retrieval.search "metformin contraindications" --compare   # dense vs BM25 vs hybrid
python -m src.retrieval.search "warfarin aspirin interaction" --rerank   # before vs after reranking
python -m src.retrieval.search --eval --rerank                           # fixed probe set

# 6. Grounded answers with citations (needs GEMINI_API_KEY)
python -m src.generation.answer "What are the common side effects of metformin?"

# 7. Drug interaction evidence check (needs GEMINI_API_KEY)
python -m src.interactions.check warfarin aspirin

# 8. Web UI: "Ask a Question" and "Drug Interaction Check" tabs
streamlit run app.py

# Tests (offline; no key or network needed)
python -m pytest
```

Data sources:

| Source | What it adds | Access | Licence |
|---|---|---|---|
| **MedlinePlus Health Topics** (NLM) | plain-language disease summaries (1,014 English topics) — answers lay questions like "what are the symptoms of malaria?" | daily Health Topic XML, [medlineplus.gov/xml.html](https://medlineplus.gov/xml.html) | health topic summaries are public domain. *Source: MedlinePlus, National Library of Medicine.* |
| **PubMed / PMC** | research abstracts and open-access full text | NCBI E-utilities | public domain metadata; PMC limited to the open-access subset |
| **FDA drug labels** | dosing, contraindications, adverse reactions, drug interactions for 20 drugs | openFDA | US Government public domain |

Only MedlinePlus's own topic summaries are ingested. Its A.D.A.M. Medical
Encyclopedia and ASHP drug monographs are copyrighted, so they are excluded.
Mayo Clinic and Cleveland Clinic were rejected because their terms prohibit
scraping and reuse, and DrugBank because it requires a paid licence. No licensed
or proprietary datasets are used.

## Project documentation

- **[CLAUDE.md](CLAUDE.md)** — architecture, working rules, and the phased roadmap
  with live progress checkboxes.
- **[SESSION_LOG.md](SESSION_LOG.md)** — dated record of every working session:
  what was built, what was decided, and what comes next.

## Status

See the roadmap in [CLAUDE.md §4](CLAUDE.md). **Phases 0–7 complete** (ingestion →
chunking → embeddings → hybrid retrieval → reranking → grounded generation →
drug-interaction evidence), plus a Streamlit demo UI pulled forward from Phase 10.

How an answer is produced: hybrid retrieval (PubMedBERT + BM25, fused with RRF) →
cross-encoder rerank → passages scoring below 0 are dropped, and if none remain the
system says "not found" without calling the LLM → Gemini answers only from the
numbered passages, citing each claim. Confidence (High / Medium / Low) combines
the best rerank score (≥ 5 is strong) with how many distinct documents score
within 3 of it.

Each source in an answer is tagged MedlinePlus, PubMed, PMC or FDA label.

Known limits: the corpus covers ~1,000 MedlinePlus topics, 20 drug labels and
~150 papers; brand names (e.g.
Coumadin) are not recognised unless they appear in the corpus; "no interaction
evidence found" never means a combination is safe.
