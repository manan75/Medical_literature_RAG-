# Session Log

Dated, session-by-session record of work on this project. Newest entries at the
bottom. Template is defined in [CLAUDE.md §5](CLAUDE.md).

**Always read the latest entry before starting new work.**

---

## Session: 2026-09-28

**Phase(s) worked on:** Phase 0 — Setup & Scoping · Phase 1 — Document Ingestion

**Goal for this session:**
- Take the repo from empty (CLAUDE.md only) to a working ingestion pipeline
- Lock in and justify every Phase 0 tooling decision
- Get a real medical corpus on disk, extracted and normalised

**What was discussed:**
- First evaluation is in a few days, so the session targets several phases rather
  than one, front-loading the parts a demo depends on.
- Two decisions were put to the user because they change what gets built:
  - **LLM provider** → *Gemini free tier*. Free key, no cost, adequate for a minor
    project; coded behind a provider interface so it can be swapped.
  - **Corpus** → *PubMed/PMC + FDA labels*. Both free, keyless (or free-key), and
    public domain, and FDA labels are what make the Phase 7 interaction module
    possible without a paid DrugBank licence.

**What was done:**
- **Environment:** `.venv` on Python 3.13.1; `requirements.txt`; all dependencies
  installed and import-verified (torch 2.14.0+cpu, chromadb 1.5.9,
  sentence-transformers, pymupdf, chromadb, rank-bm25, google-genai, pytest).
- **Repo skeleton:** full `src/` package tree, `data/` tree, `tests/`, `notebooks/`,
  `docs/`; `.gitignore` (excludes `.env`, all harvested and derived data),
  `.env.example`, `pytest.ini`.
- **`src/config.py`** — single source of truth for paths, model names and retrieval
  constants; loads `.env` but never requires a key at import time, so the retrieval
  half of the pipeline runs fully offline.
- **`src/schema.py`** — `Document` / `Section` / `Chunk` dataclasses plus JSONL I/O.
  `Chunk.citation()` is defined here, at the data layer, so no downstream code can
  construct a chunk that cannot be cited.
- **`src/data_sources/`** — `http.py` (per-host rate limiting + backoff),
  `pubmed.py` (E-utilities esearch/efetch for PubMed and PMC), `openfda.py`
  (drug-label fetch, generic-name with brand-name fallback), `corpus_spec.py`
  (the 20 drugs, 6 interaction pairs and 15 queries that define the demo corpus),
  `harvest.py` (CLI entry point).
- **`src/ingestion/`** — `normalize.py`, `pubmed_parser.py` (PubMed + PMC JATS),
  `fda_parser.py`, `pdf_extract.py` (PyMuPDF, font-size heading detection),
  `html_extract.py` (BeautifulSoup, document-order heading splits),
  `pipeline.py` (entry point → `data/processed/documents.jsonl`).
- **`tests/test_ingestion.py`** — 17 tests across all four extractors plus
  normalisation, all on synthetic in-process fixtures (offline, deterministic).

**Decisions made / deviations from plan:**
- **CLAUDE.md §3 filled in** with every Phase 0 choice and its rationale (Chroma,
  PubMedBERT embeddings, Gemini, pip+venv). See that section for the reasoning.
- **DrugBank rejected** as a data source — the full interaction dataset is behind a
  paid licence. Drug-interaction evidence will instead come from the
  `drug_interactions` section of FDA labels plus PubMed literature. This is a
  deviation from the CLAUDE.md Phase 0 wording, which listed DrugBank as a
  candidate, and it shapes Phase 7.
- **Harvesting split from ingestion** (`data_sources/` vs `ingestion/`), which
  CLAUDE.md did not specify. Ingestion is a pure filesystem transform, so parser
  changes can be re-run without re-hitting NCBI.
- **Dependency pins loosened** for the ML stack: `torch==2.5.1` has no Python 3.13
  wheel. ML libraries are now floor-pinned (`>=`), data libraries stay exact.
- **Structured abstracts are preserved, not flattened.** PubMed labels abstract
  sections (BACKGROUND / METHODS / RESULTS / CONCLUSIONS) and openFDA keys label
  sections by name. Both are kept as `Section` headings, because a chunk that
  arrives at the LLM tagged "Contraindications" is interpretable in isolation and
  one tagged nothing is not.
- **PDF body-font detection changed from `statistics.mode` to character-weighted
  frequency.** Found by a failing test: with as many heading lines as body lines,
  `mode` elected the heading size as body size and collapsed every section into
  one. Weighting by character count is robust because body text always dominates
  by volume even when it does not dominate by line count.

**Verification:**
- `python -m src.data_sources.harvest --small` — **pass**, live. 4 FDA labels
  (metformin, warfarin, aspirin, ibuprofen) + 9 PubMed abstracts fetched.
- `python -m src.ingestion.pipeline` — **pass**. 13 documents / 55 sections written
  to `data/processed/documents.jsonl`.
- Manual spot-check of output — **pass**. FDA label sections correctly named
  (`Indications and Usage`, `Dosage and Administration`, `Warnings`); PubMed
  structured-abstract labels preserved (`Importance`, `Observations`, `Conclusion`);
  authors, journal, date and URL all intact.
- `python -m pytest` — **17 passed**.

**Open issues / blockers:**
- `GEMINI_API_KEY` is not yet set. Nothing before Phase 6 needs it, but it must be
  in `.env` before generation can be demoed. *(User action.)*
- `NCBI_API_KEY` unset — harvesting is limited to 3 req/sec. Works, just slower.
- Only the `--small` corpus has been harvested (13 docs). The full corpus
  (20 drugs, 15 queries, ~150+ docs) still needs one full harvest run.
- The embedding model (~420 MB) has not been downloaded yet — first run of the
  embedding step will pull it.

**Next session should start with:**
- Phase 2 — section-aware chunking (`src/chunking/`), over the ingested corpus.
