# AI-Powered Medical Literature Retrieval and Drug Interaction Analysis (RAG)

A retrieval-augmented generation (RAG) system that answers questions about diseases,
symptoms, treatments and medications using only passages retrieved from trusted public
sources, and that checks two drugs for interaction evidence. Every answer cites its
sources and carries a confidence rating.

> **Medical disclaimer.** This system is for **medical information retrieval and
> education only**. It does not diagnose, prescribe, or provide medical advice.
> Answers can be incomplete or wrong. "No interaction evidence found" never means a
> combination is safe. Always consult a qualified healthcare professional.

---

## How it works

```
OFFLINE (build the knowledge base)
  PubMed / PMC (NCBI E-utilities) · FDA drug labels (openFDA) · MedlinePlus health topics (NLM)
      ↓  src/data_sources/   harvest raw files into data/raw/
      ↓  src/ingestion/      parse XML / JSON (PDF and HTML also supported) → Documents with sections
      ↓  src/chunking/       section-aware chunks, ~350 tokens, ~60 overlap, never crossing a section
      ↓  src/embeddings/     NeuML/pubmedbert-base-embeddings (768-dim, runs on CPU)
      ↓  src/retrieval/      Chroma vector store (data/vectorstore/) + BM25 index (in memory)

ONLINE (answer a question)
  question
      ↓  dense top 20 (Chroma, cosine) + BM25 top 20
      ↓  Reciprocal Rank Fusion (k = 60) → 20 candidates
      ↓  cross-encoder rerank (cross-encoder/ms-marco-MiniLM-L-6-v2) → top 5
      ↓  gate: drop passages scoring below 0.0; none left → "not found", LLM not called
      ↓  Gemini 3.5 Flash-Lite (gemini-3.5-flash-lite), one call, answers only from numbered passages
  answer + numbered citations + High / Medium / Low confidence + every passage used
```

The drug interaction check normalises two drug names against the corpus, scans both FDA
labels for mentions of the other drug, searches the literature, keeps only passages
naming both drugs, and explains them with citations. With no evidence it says "No
interaction evidence was found in our corpus" and does not call the LLM.

## Data sources

| Source | What it adds | Access | Licence | In corpus |
|---|---|---|---|---|
| **MedlinePlus Health Topics** (NLM) | plain-language disease summaries | daily Health Topic XML, [medlineplus.gov/xml.html](https://medlineplus.gov/xml.html) | health topic summaries are public domain. *Source: MedlinePlus, National Library of Medicine.* | 1,014 topics, 1,820 passages |
| **FDA drug labels** | dosing, contraindications, warnings, adverse reactions, drug interactions for 20 drugs | openFDA label API | US Government public domain | 20 labels, 569 passages |
| **PubMed** | research abstracts | NCBI E-utilities | freely available abstracts | 140 papers, 318 passages |
| **PMC** (PubMed Central) | open-access full-text papers | NCBI E-utilities | open-access subset only | 9 papers, 301 passages |

Total: 1,183 documents, 3,008 passages. What gets harvested is defined in
`src/data_sources/corpus_spec.py`.

Only MedlinePlus's own topic summaries are used. Its A.D.A.M. Medical Encyclopedia and
ASHP drug monographs are copyrighted and are not ingested. **Rejected sources:** DrugBank
(paid licence), Mayo Clinic and Cleveland Clinic (terms prohibit scraping and reuse).

## Setup

Requires Python 3.12 or 3.13.

```bash
git clone https://github.com/manan75/Medical_literature_RAG-.git
cd Medical_literature_RAG-

python -m venv .venv
.venv\Scripts\activate            # Windows (cmd / PowerShell)
# source .venv/Scripts/activate   # Windows Git Bash
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
cp .env.example .env              # then add your keys
```

| Variable | Needed for | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | answer generation and interaction explanations | https://aistudio.google.com/apikey (free tier) |
| `NCBI_API_KEY` | faster PubMed harvesting (optional) | https://account.ncbi.nlm.nih.gov/settings/ |

No key is needed for harvesting, indexing, retrieval or the tests. Model names can be
overridden in `.env` (`GEMINI_MODEL`, `EMBEDDING_MODEL`, `RERANKER_MODEL`).

## Build the data

`data/` is gitignored and regenerated from the sources.

```bash
# 1. Download raw files (NCBI + openFDA + MedlinePlus)
python -m src.data_sources.harvest
python -m src.data_sources.harvest --medlineplus-only   # refresh only MedlinePlus

# 2. Parse into data/processed/documents.jsonl
python -m src.ingestion.pipeline

# 3. Chunk into data/chunks/chunks.jsonl
python -m src.chunking.pipeline

# 4. Embed and index into Chroma (full rebuild), then run sample nearest-neighbour checks
python -m src.retrieval.build_index --check

#    Or, after adding a new source when existing chunks are unchanged,
#    embed only the chunks not yet in the index:
python -m src.retrieval.build_index --append
```

Indexing runs PubMedBERT on every chunk on the CPU. Measured on a laptop: 2.4 to 4.6
chunks per second, so a full rebuild of 3,008 chunks takes about 11 to 21 minutes.
`--append` warns and asks for a full rebuild if indexed chunks no longer exist in
`chunks.jsonl`. Note that PubMed and openFDA results change over time, so a fresh
harvest produces a slightly different corpus.

## Use it

```bash
# Web app: "Ask a question" and "Check a drug interaction" tabs
streamlit run app.py
#   open straight onto an answer: http://localhost:8501/?q=What+are+the+symptoms+of+malaria

# Grounded answer with citations (needs GEMINI_API_KEY)
python -m src.generation.answer "What are the common side effects of metformin?"

# Drug interaction evidence (needs GEMINI_API_KEY)
python -m src.interactions.check warfarin aspirin

# Retrieval only, no LLM
python -m src.retrieval.search "What are the common side effects of metformin?"
python -m src.retrieval.search "metformin contraindications" --compare   # dense vs BM25 vs hybrid
python -m src.retrieval.search "warfarin aspirin interaction" --rerank   # before vs after reranking
python -m src.retrieval.search --eval --rerank                           # fixed probe set

# Tests: 110, all offline (no network or API key)
python -m pytest
```

The first question after starting the app loads both models (about 40 seconds on a
laptop). After that a question takes about 9 seconds, mostly reranking and the Gemini call.

## Known limitations

- **Small, US-centric, English-only corpus:** 20 drug labels, 149 papers and 1,014
  MedlinePlus topics. Many questions will correctly return "not found".
- **Brand names are not recognised** unless they appear in the corpus labels (for
  example "Coumadin" is not). RxNorm normalisation is planned.
- **Thin not-found margin:** answerable questions have scored from +1.3 and
  unanswerable ones up to -1.1 against the 0.0 threshold.
- **Confidence measures evidence, not correctness.** Correct answers backed by a
  single source (for example one MedlinePlus page) are rated Low or Medium.
- **Citations are checked for presence, not support.** An answer with no valid
  citation is marked Low, but cited passages are not automatically verified against
  each claim. Every passage is shown in the app for manual checking.
- **Interaction evidence** must name both drugs, so class-level statements (such as
  "NSAIDs") are not matched to a specific drug.
- **No formal evaluation yet.** Retrieval has been checked by hand on probe questions;
  labelled metrics are planned (Phase 9).
- Questions and retrieved public passages are sent to the Gemini API.

## Project documentation

- **[docs/PRESENTATION_PREP.md](docs/PRESENTATION_PREP.md):** end-to-end explanation of
  every component, worked traces, design trade-offs and a question bank.
- **[CLAUDE.md](CLAUDE.md):** architecture, working rules and the phased roadmap.
- **[SESSION_LOG.md](SESSION_LOG.md):** dated record of every working session.

## Status

Phases 0 to 7 are complete: ingestion, chunking, embeddings, hybrid retrieval,
reranking, grounded generation and the drug interaction module, plus MedlinePlus as a
plain-language source and a Streamlit demo app pulled forward from Phase 10. Next:
advanced features (Phase 8) and formal evaluation (Phase 9). See
[CLAUDE.md section 4](CLAUDE.md).
