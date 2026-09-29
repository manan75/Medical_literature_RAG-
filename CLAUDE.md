# CLAUDE.md

Guidance for Claude (and any collaborator) working in this repository. This file defines how the project is structured, how work should be sequenced, and how every session must be documented so that progress is traceable across time.

---

## 1. Project Overview

**Name:** AI-Powered Medical Literature Retrieval and Drug Interaction Analysis using RAG

**Purpose:** Build a Retrieval-Augmented Generation (RAG) system that answers questions about diseases, symptoms, treatments, and medications by retrieving grounded evidence from trusted medical literature and drug databases, and that additionally analyzes potential drug–drug interactions with cited evidence.

**Explicit scope boundary:** This system is for **medical information retrieval and education only**. It must never present itself as capable of diagnosis or prescription, and every response involving clinical content must carry source citations and, ideally, a confidence/evidence indicator.

**Core pipeline (fixed reference architecture):**

```
Medical Documents
    ↓
PDF / HTML Extraction
    ↓
Medical Text Chunking
    ↓
Biomedical Embeddings
    ↓
Vector Database
    ↓
Hybrid Retrieval
    ↓
Reranking
    ↓
LLM
    ↓
Grounded Medical Response
    ↓
Sources + Evidence
```

**Feature set:**

- Core: document ingestion, disease info retrieval, drug info retrieval, semantic search, RAG answers, source citations
- Advanced: drug–drug interaction retrieval, disease→symptom→treatment extraction, medication comparison, paper summarization, conflicting-source detection, confidence/evidence indicator, multilingual queries

---

## 2. Working Rules for Claude

These rules govern how work in this repo proceeds, regardless of which phase is active.

1. **One phase at a time.** Do not start work on a later phase until the current phase's checklist (Section 4) is fully checked off and its session log entry (Section 5) is written, unless the user explicitly says to jump ahead.
2. **One step at a time within a phase.** Implement a single checklist item, verify it works (run it, test it, inspect output), then move to the next item. Do not batch multiple unverified steps together.
3. **Stop and report at phase boundaries.** At the end of each phase, summarize what was built, what was verified, and ask before proceeding — unless the user has told you to run ahead autonomously for a given session.
4. **No silent scope changes.** If a step reveals that the architecture, library choice, or plan needs to change, say so explicitly and log it as a decision (Section 5) rather than quietly diverging.
5. **Every coding session gets a log entry.** Before ending a session (or when the user pauses/stops), append an entry to `SESSION_LOG.md` using the template in Section 5. This is not optional — it is how continuity across sessions is maintained.
6. **Grounding is non-negotiable.** Any code path that produces a user-facing medical answer must carry retrieved source references alongside it. Do not implement an LLM-answer path that skips citation of retrieved chunks, even in a prototype.
7. **Flag medical-safety edges.** If a task would cause the system to give diagnostic or prescriptive advice instead of retrieval-grounded information, pause and flag it rather than implementing it silently.

---

## 3. Repository Conventions

*(Decided in Phase 0 — 2026-09-28.)*

- **Language/runtime:** Python 3.12–3.13 (verified on 3.13.1 and 3.12.10, Windows)
- **Package manager:** `pip` + `venv` (`.venv/`, `requirements.txt`). Chosen over
  poetry/uv because the marking environment is a plain college machine — one less
  tool to install before the project runs.
- **Vector database:** **Chroma** (`chromadb`), persisted to `data/vectorstore/`.
  Embedded and file-backed: no server to start, no account, no network at demo time.
  FAISS was the alternative but stores no metadata alongside vectors, and citations
  depend on that metadata travelling with the vector.
- **Embedding model:** **`NeuML/pubmedbert-base-embeddings`** (768-dim, ~420 MB),
  a PubMedBERT variant tuned for sentence similarity. Domain-specific matters here:
  a general model treats "metformin" and "metoprolol" as near-neighbours on
  surface form, which is exactly the confusion a drug-safety system cannot make.
  Runs on CPU. Configurable via `EMBEDDING_MODEL`.
- **Reranker:** **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (~90 MB). General-domain
  but small and fast; Phase 5 measures whether it earns its place.
- **LLM for generation:** **Gemini** (`gemini-3.5-flash-lite`) via `google-genai`, on the
  free tier. Accessed behind a provider interface so it can be swapped later.
  (`gemini-2.0-flash`, the original choice, was shut down on 2026-06-01.)
- **Data sources:** PubMed + PMC open-access (NCBI E-utilities) and FDA drug labels
  (openFDA). Both are public domain / open access with no licensing restriction on
  academic use. **DrugBank was rejected** — its full interaction dataset requires a
  paid licence. Drug-interaction evidence therefore comes from the
  `drug_interactions` section of FDA labels plus PubMed literature.
- **Plain-language disease content: MedlinePlus Health Topics** (NLM), added
  2026-09-29 (licensing checked against medlineplus.gov that day).
  PubMed and FDA labels are technical; lay questions ("what are the symptoms of
  malaria?") had nothing to retrieve. Source: the daily compressed Health Topic XML
  (`medlineplus.gov/xml.html`, ~4.7 MB zip, regenerated Tuesday–Saturday).
  **Only the Health Topic summaries are used** — NLM lists "Summaries on health topic
  pages" as public domain. The same XML also carries third-party link records,
  which are not ingested. **Not used:** the A.D.A.M. Medical Encyclopedia and the
  ASHP drug monographs on MedlinePlus, which are copyrighted and licensed to NLM
  only. NLM asks for the credit "Source: MedlinePlus, National Library of Medicine".
  **Mayo Clinic and Cleveland Clinic were rejected**: their terms of use prohibit
  scraping and reuse of their content.
- **Folder structure (actual):**
  ```
  /data/raw/pubmed/       # harvested E-utilities XML + provenance sidecars
  /data/raw/fda/          # harvested openFDA label JSON
  /data/raw/medlineplus/  # MedlinePlus Health Topic XML zip + provenance sidecar
  /data/processed/        # documents.jsonl  (extracted, normalised)
  /data/chunks/           # chunks.jsonl     (retrieval units)
  /data/vectorstore/      # Chroma persistent store
  /src/config.py          # all paths, model names, retrieval constants
  /src/schema.py          # Document / Chunk dataclasses + JSONL I/O
  /src/data_sources/      # network harvesting (pubmed, openfda, corpus_spec)
  /src/ingestion/         # PDF / HTML / XML / JSON extraction + normalisation
  /src/chunking/          # section-aware chunking
  /src/embeddings/        # embedding generation
  /src/retrieval/         # vector store, BM25, hybrid fusion, reranking
  /src/generation/        # LLM prompting + grounded response assembly
  /src/interactions/      # drug-drug interaction module
  /src/api/               # API / CLI layer
  /tests/                 # pytest suite
  /notebooks/             # exploration, not production code
  SESSION_LOG.md          # dated session-by-session log (see Section 5)
  CLAUDE.md               # this file
  ```
- **Testing:** `pytest`, run with `python -m pytest`. Every module in `/src` should
  have a corresponding test before being marked complete in a phase checklist.
  Tests use synthetic in-process fixtures and never hit the network.
- **Secrets/config:** never commit API keys; use `.env` + `.env.example`.
  `.env` and all harvested/derived data are gitignored — the corpus is regenerable
  from `src/data_sources/corpus_spec.py`.

## 4. Phased Roadmap

Each phase has a goal, a step-by-step checklist, and an explicit "definition of done." Check items off as completed (`- [x]`) directly in this file so the roadmap doubles as a live progress tracker.

### Phase 0 — Project Setup & Scoping
**Goal:** Environment, repo skeleton, and data-source decisions in place before any pipeline code is written.
- [x] Decide language/runtime, package manager, and fill in Section 3
- [x] Initialize repo structure (folders above)
- [x] Decide trusted data sources (e.g., PubMed/PMC, DrugBank, FDA labels, clinical guideline sets) and note licensing constraints
- [x] Decide vector database choice (e.g., Chroma, FAISS, Qdrant, Pinecone) and justify briefly
- [x] Decide embedding model (general vs. biomedical-specific, e.g., PubMedBERT/BioBERT-family or an API embedding model) and justify briefly
- [x] Decide LLM provider/model for generation
- [x] Create `.env.example`, `requirements`/`pyproject`, and a minimal `README.md` pointing to this file
- [x] Create `SESSION_LOG.md` with the template from Section 5
**Definition of done:** repo runs a `hello world`-level smoke test; all major tooling choices recorded here with rationale.

### Phase 1 — Document Ingestion
**Goal:** Raw medical documents become clean, structured text.
- [x] PDF extraction (text + basic structure/section headers)
- [x] HTML extraction (for web-sourced guidelines/drug pages)
- [x] Normalize extracted text (strip boilerplate, headers/footers, references sections handled deliberately)
- [x] Store raw + processed text with document-level metadata (title, source, date, section)
- [x] Unit tests on extraction with at least 2–3 sample documents of different formats
**Definition of done:** a small corpus (e.g., 10–20 docs) is ingested end-to-end into `/data/processed/` with metadata intact.

### Phase 2 — Medical Text Chunking
**Goal:** Processed text is split into retrieval-ready chunks without breaking clinical meaning.
- [x] Choose chunking strategy (fixed-size vs. semantic/section-aware) and justify
- [x] Preserve section context (e.g., "Dosage," "Contraindications," "Side Effects") as chunk metadata
- [x] Handle tables (e.g., dosage tables) as a special case if present
- [x] Tests verifying no chunk crosses a hard semantic boundary inappropriately
**Definition of done:** sample corpus chunked with metadata; spot-checked chunks make sense read in isolation.

### Phase 3 — Biomedical Embeddings
**Goal:** Chunks are embedded with a model suited to medical text.
- [x] Implement embedding generation over chunked corpus
- [x] Benchmark chosen embedding model on a handful of hand-written medical queries (qualitative check)
- [x] Store embeddings with linkage back to source chunk + document metadata
**Definition of done:** embeddings generated and persisted; a manual nearest-neighbor check returns sensible results.

### Phase 4 — Vector Database & Hybrid Retrieval
**Goal:** Fast, relevant retrieval combining semantic + keyword signals.
- [x] Stand up vector DB and load embeddings
- [x] Implement semantic (dense) retrieval
- [x] Implement keyword/BM25 (sparse) retrieval
- [x] Combine into hybrid retrieval with a fusion strategy (e.g., reciprocal rank fusion)
- [x] Tests on a fixed query set with expected-relevant-doc checks
**Definition of done:** given a test query set, hybrid retrieval outperforms either method alone on manual inspection.

### Phase 5 — Reranking
**Goal:** Improve precision of top-k retrieved chunks before they reach the LLM.
- [x] Choose reranker (cross-encoder or LLM-based reranking)
- [x] Integrate reranking after hybrid retrieval, before generation
- [x] Evaluate top-k precision before/after reranking on test query set
**Definition of done:** measurable or clearly observable improvement in top-k relevance on the test set.

### Phase 6 — Grounded Generation (Core RAG Loop)
**Goal:** LLM produces answers strictly grounded in retrieved chunks, with citations.
- [x] Prompt design enforcing "answer only from provided context, cite sources"
- [x] Response assembly: answer text + explicit source list (document + section)
- [x] Handle "not found in corpus" gracefully instead of hallucinating
- [x] Add a basic confidence/evidence indicator (e.g., based on retrieval score spread or source agreement)
- [x] End-to-end test: ask sample questions (e.g., "common side effects of metformin") and verify grounded, cited answers
**Definition of done:** the example from the project brief ("side effects of metformin") works end-to-end with correct citations.

### Phase 7 — Drug Interaction Analysis Module
**Goal:** Given two (or more) medications, retrieve and explain interaction evidence.
- [x] Ingest/structure a drug-interaction data source (dedicated dataset or literature-derived)
- [x] Implement interaction lookup + retrieval of supporting evidence
- [x] Generate an explanation grounded in retrieved evidence, with citations
- [x] Test with known interacting pairs and known non-interacting pairs (including a "no known interaction found" path)
**Definition of done:** the "Can Drug A interact with Drug B?" example from the brief works with cited evidence.

### Phase 8 — Advanced Features (incremental, pick order with user)
Each sub-feature is its own mini-checklist; do not start the next until the previous is done and logged.
- [ ] Disease → symptoms → treatment relationship extraction
- [ ] Compare two medications side by side
- [ ] Summarize a given medical paper
- [ ] Identify conflicting information across sources
- [ ] Multilingual query support
**Definition of done (per sub-feature):** working end-to-end demo + test cases, logged individually in `SESSION_LOG.md`.

### Phase 9 — Evaluation & Hardening
**Goal:** Systematic quality check, not just spot checks.
- [ ] Build a small labeled evaluation set (questions + expected-relevant sources)
- [ ] Measure retrieval quality (e.g., recall@k) and answer groundedness
- [ ] Adversarial checks: ambiguous queries, out-of-corpus queries, attempts to elicit diagnosis/prescription behavior
- [ ] Fix issues found, re-run evaluation
**Definition of done:** evaluation report checked into repo; known failure modes documented.

### Phase 10 — Packaging & Documentation
**Goal:** The system is usable and understandable by someone else.
- [ ] API or CLI/UI entry point finalized
- [ ] `README.md` fully written (setup, usage, architecture diagram, limitations, medical-use disclaimer)
- [ ] Final pass over `SESSION_LOG.md` for consistency
**Definition of done:** a new developer could clone the repo and run the system using only the README.

---

## 5. Session Documentation (Mandatory)

Every working session — regardless of how small — gets an entry appended to `SESSION_LOG.md` (create this file in Phase 0). Use this exact template:

```markdown
## Session: YYYY-MM-DD

**Phase(s) worked on:** e.g., Phase 2 — Chunking

**Goal for this session:**
- What was intended to be accomplished

**What was done:**
- Bullet list of concrete changes (files touched, functions added, decisions implemented)

**Decisions made / deviations from plan:**
- Any choice that changed course from what CLAUDE.md originally specified, with reasoning

**Verification:**
- What was tested/run, and the result (pass/fail/partial)

**Open issues / blockers:**
- Anything unresolved

**Next session should start with:**
- The single next concrete step
```

This log is the source of truth for "where did we leave off" — always read the latest entry at the start of a new session before doing anything else.

---

## 6. Non-Negotiables (Recap)

- No diagnostic or prescriptive framing — retrieval and education only.
- No ungrounded medical claims in generated answers — always cite retrieved sources.
- No skipping the phase checklist or the session log.
- Flag, don't silently resolve, any medical-safety-relevant ambiguity.
