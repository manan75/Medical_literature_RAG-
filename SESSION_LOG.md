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

---

## Session: 2026-09-28 (part 2 — same working session, continued)

**Phase(s) worked on:** Phase 2 — Chunking · Phase 3 — Embeddings ·
Phase 4 — Vector DB & Hybrid Retrieval · Phase 5 — Reranking

**Goal for this session:**
- Carry the pipeline from "documents on disk" to "ask a question, get relevant
  cited chunks back"
- Harvest the full corpus rather than the smoke-test subset
- Get far enough that Phase 6 (generation) is the only thing standing between the
  project and an end-to-end demo

**What was discussed:**
- Checked in on progress mid-session. Confirmed there is **no UI yet** — every
  entry point is a CLI module. Per CLAUDE.md the UI/API belongs to Phase 10, but
  for the evaluation demo it is worth pulling forward once Phase 6 works; a UI
  over retrieval alone would just be a search box.

**What was done:**
- **Phase 2 — Chunking.** `src/chunking/splitter.py` and `pipeline.py`.
  Section-aware packing, abbreviation-aware sentence splitting, whole-table
  preservation, overlap, orphan-tail merging, noise floor. 16 tests.
- **Phase 3 — Embeddings.** `src/embeddings/encoder.py`. Lazy model loading,
  L2-normalised vectors, and *contextualised embedding input*: each chunk is
  encoded as `"<title> | <section>\n<text>"` so the section heading lands inside
  the vector. The stored chunk text is left undecorated, so citations stay clean.
- **Phase 4 — Retrieval.** `src/retrieval/`: `types.py` (one shared result type),
  `vector_store.py` (Chroma, cosine, full metadata round-trip), `sparse.py`
  (BM25 over title+section+text, medical-aware tokenizer), `hybrid.py`
  (reciprocal rank fusion), `build_index.py`, `search.py` (CLI with `--compare`,
  `--rerank`, `--eval`).
- **Phase 5 — Reranking.** `src/retrieval/rerank.py`. Cross-encoder over the
  hybrid candidate list, with the pre-rerank score and rank preserved in
  `components` so the effect stays inspectable. 7 tests.
- **Full corpus harvested:** 20 FDA labels, 12 PubMed queries x 12 records,
  3 PMC full-text queries.

**Decisions made / deviations from plan:**
- **RRF chosen over weighted score blending** for fusion. Cosine similarity is
  bounded in [0,1] and BM25 is unbounded and corpus-dependent, so any weighted sum
  needs normalisation constants that must be retuned whenever the corpus changes.
  RRF fuses on rank alone, needs no tuning, and rewards agreement between the two
  signals — which is itself evidence of relevance.
- **Section heading is embedded, not just stored as metadata.** Two chunks can be
  near-identical in wording under "Drug Interactions" and "Adverse Reactions"
  while answering different questions; the heading has to be inside the vector to
  separate them.
- **BM25 index is rebuilt in memory rather than persisted.** It takes under a
  second at this corpus size and can never go stale against `chunks.jsonl` the way
  a pickled index can.
- **Reranker is general-domain (MS MARCO), not biomedical.** It judges
  query-chunk relevance, not clinical semantics; the biomedical signal is already
  carried by the retrieval stage that produced the candidates. Revisit in Phase 9
  if evaluation shows it is the bottleneck.
- **Minimal BM25 stopword list.** Aggressive stopword removal would strip "no",
  "not" and "without" — negation words that invert the clinical meaning of a
  contraindication.

**Bugs found and fixed (all found by running the thing, not by reading it):**
1. **PMC full text silently yielded zero documents.** NCBI labels the identifier
   `pub-id-type="pmcid"` in current JATS output; the parser only accepted the
   legacy `"pmc"`. Every PMC article was being skipped without error. Now accepts
   `pmcid` / `pmc` / `pmcaid`, and both forms are covered by tests. Fixing it added
   9 full-text documents and 301 chunks.
2. **openFDA returned combination products as single-drug labels.** A search for
   "metformin" returned *Sitagliptin and Metformin Hydrochloride* — so the corpus
   had no plain metformin label at all, and the top hit for "side effects of
   metformin" was a combination product. Added `_specificity()` ranking (exact
   generic name > salt form > other > combination) and re-harvested; all 20 labels
   are now single-ingredient. This would also have broken Phase 7, where the whole
   point is to reason about one drug at a time.
3. **Sentence splitter ran two sentences together.** Rejoining on an abbreviation
   alone broke `"Administer 5 mg i.v. every 8 h. Reduce the dose..."` — "h." is a
   known abbreviation but did end that sentence. Now also requires the following
   fragment to start lowercase or with a digit.
4. **PDF heading detection collapsed every section into one.** `statistics.mode`
   returns the *first* mode on a tie, so a page with as many heading lines as body
   lines elected the heading size as the body size. Replaced with character-weighted
   frequency: body text always dominates by volume even when it does not by line count.
5. **FDA `effective_time` was left as `20240416`.** Now normalised to ISO `2024-04-16`.
6. **Two test-isolation defects of my own:** `renumber()` was overwriting the rank
   an assertion depended on, and a fixed Chroma collection name leaked vectors
   between tests (chromadb reuses in-process clients). Both fixed in the tests.

**Verification:**
- Full harvest — **pass**. 20 FDA labels, 12 PubMed batches, 3 PMC batches.
- `python -m src.ingestion.pipeline` — **pass**. 171 documents / 661 sections
  (20 FDA, 142 PubMed, 9 PMC).
- `python -m src.chunking.pipeline` — **pass**. 1,255 chunks, median 297 tokens,
  200 distinct section headings preserved, 4 oversized chunks (intact tables).
- `python -m src.retrieval.build_index --check` — **pass**. 1,255 vectors, 768-dim,
  273s on CPU (4.6 chunks/s).
- `python -m src.retrieval.search --eval --rerank` — **pass**, and this is the
  Phase 4/5 definition of done. Representative results:
  - *"metformin contraindications renal impairment"* → after reranking, #1 is the
    metformin label's **Contraindications** section (hybrid had it at #2).
  - *"Can aspirin interact with warfarin?"* → #1 is the warfarin label's **Drug
    Interactions** section; reranking promoted 2 chunks from outside the top 3,
    including a systematic review's Conclusions from rank #9.
  - *"CYP3A4 inhibitors and simvastatin"* → all top 3 are simvastatin label
    sections.
  - *"What are the common side effects of metformin?"* → reranking put **Adverse
    Reactions** above **Drug Interactions**, which is the correct preference for a
    side-effects question.
- Hybrid vs. single-method inspection — hybrid beats either alone. Dense alone
  confused drug names (ranked metoprolol third for a metformin query); BM25 alone
  was derailed by common words in verbose natural-language questions (ranked an
  inositol paper first). Fusion plus reranking fixes both.
- `python -m pytest` — **67 passed**.

**Open issues / blockers:**
- `GEMINI_API_KEY` is still unset. **This blocks Phase 6 entirely.** Get a free key
  at https://aistudio.google.com/apikey and put it in `.env`. *(User action —
  needed before the next session.)*
- **No UI exists.** CLI only. Worth pulling a small Streamlit or FastAPI front-end
  forward from Phase 10 once Phase 6 works, so the evaluation has something to look at.
- BM25 is weak on verbose natural-language questions (it matches "common",
  "effects" as content words). Reranking compensates, but Phase 9 should measure
  this properly rather than relying on the fix being invisible.
- Embedding the corpus takes ~4.5 minutes on CPU. Fine for rebuilds, but do not
  plan to rebuild the index live during the evaluation demo.
- The reranker is general-domain; revisit if Phase 9 evaluation shows it limiting.

**Next session should start with:**
- Phase 6 — grounded generation (`src/generation/`): prompt enforcing
  "answer only from the provided context, cite sources", response assembly with an
  explicit source list, a graceful "not found in corpus" path, and a confidence
  indicator derived from retrieval score spread and source agreement. Requires
  `GEMINI_API_KEY` to be set first.

---

## Session: 2026-09-29

**Phase(s) worked on:** Phase 6 — Grounded Generation · Phase 7 — Drug Interaction
Analysis (minimal) · Streamlit UI (pulled forward from Phase 10)

**Goal for this session:**
- Rebuild the gitignored `data/` directory on a new machine and confirm it matches
  the previous session's results
- Replace the retired Gemini model
- Complete Phase 6 and a minimal Phase 7, and add a demo UI for the mid-term
  evaluation (user explicitly authorised running across these phase boundaries)

**What was done:**
- **Environment:** new `.venv` on Python 3.12.10 (see decisions); all requirements
  installed. Branch `ishaan/phase6-7-ui`.
- **Data rebuilt:** full harvest → ingestion → chunking → `build_index --check`.
  169 documents (20 FDA, 140 PubMed, 9 PMC), 1,188 chunks, 1,188 vectors.
- **`src/generation/providers.py`** — `LLMProvider` protocol + `GeminiProvider`.
  One call per question; exponential backoff (2/4/8/16 s) on 429 and 503; a clear
  `LLMError` if retries run out or the key/model is wrong.
- **`src/generation/answer.py`** — system prompt (answer only from the numbered
  passages, cite every claim as [n], reply `NOT_FOUND` if the passages do not
  answer, no diagnosis or personal dosing, never call a drug or combination
  "safe"); `Answer` with text, numbered `Source` list (title, section, URL,
  `Chunk.citation()`) and the chunks used; not-found short-circuit; confidence
  indicator; CLI `python -m src.generation.answer "question"`.
- **`src/interactions/check.py`** — drug-name normalisation against
  `corpus_spec.DRUGS` plus corpus label aliases; FDA label scan; hybrid retrieval
  across the corpus; rerank; generation through the Phase 6 path; CLI
  `python -m src.interactions.check drugA drugB`.
- **`app.py`** — Streamlit, two tabs ("Ask a Question", "Drug Interaction
  Check"), persistent disclaimer, confidence badge, numbered linked sources, and
  an expander showing every retrieved chunk with its section, rerank score,
  hybrid/dense/BM25 ranks and whether the label scan found it. Models, Chroma
  and BM25 are loaded once with `st.cache_resource`; the UI never rebuilds the index.
- **Tests:** `tests/test_generation.py` (11) and `tests/test_interactions.py` (10,
  5 of which run against the real `chunks.jsonl` when present and skip otherwise).
  All use a fake LLM and a stubbed cross-encoder; none hit the network.
- **Docs:** CLAUDE.md §3 (runtime, LLM) and §4 checkboxes; README Usage/Status.

**Decisions made / deviations from plan:**
- **LLM model → `gemini-3.5-flash-lite`.** `gemini-2.0-flash` was shut down on
  2026-06-01. Listed the models available to the key (61, including
  `gemini-3.5-flash-lite`), switched the default in `config.py` and
  `.env.example`, and a test generation call succeeded.
- **Python 3.12 instead of 3.13.** This machine has 3.11, 3.12 and 3.14, but no
  3.13. 3.12 has wheels for the whole ML stack; 3.14 was a risk for
  chromadb/onnxruntime. Everything installs and all tests pass on 3.12.10.
- **Not-found threshold = 0.0 (cross-encoder logit).** Chosen from observed
  scores. In-corpus questions (8 `--eval` probes + 4 more) had top rerank scores
  of **+3.9 to +8.5**. Out-of-corpus questions (malaria, isotretinoin, multiple
  sclerosis, pancreatic chemotherapy, insulin glargine, car tyres, capital of
  France) had **−10.9 to −2.9**. 0.0 sits in that gap and is where the
  cross-encoder rates a passage at 50% likely to be relevant. The one borderline
  case was migraine (+0.9): the corpus has one partially relevant paper, so it
  passes and gets a Low-confidence answer.
- **Confidence indicator, two signals.** *Strength:* is the best rerank score
  ≥ 5.0? That value splits the observed in-corpus top scores into broad matches
  (3.9–4.7) and precise section matches (6.9–8.5). *Agreement:* how many distinct
  documents have a passage scoring within 3.0 of the best one.
  High = strong and ≥ 2 documents; Medium = either one; Low = neither.
  Any answer with no [n] citation is forced to Low.
- **Phase 7 evidence rule:** a chunk counts as interaction evidence only if it
  names both drugs, or it is one drug's FDA label and names the other. Label
  scan checks the **Drug Interactions section first, then the rest of the
  label**. This goes beyond "Drug Interactions sections only": sertraline's label
  names tramadol only under Warnings and Cautions, and the aspirin label is an OTC
  label with no Drug Interactions section at all.
- **Interaction evidence skips the score threshold.** The both-drugs mention rule
  decides relevance, so evidence is passed to the LLM even when its rerank score
  is low. The confidence text reports those scores honestly as "weak".
- **Brand names: only the corpus's aliases are used.** The corpus labels are
  generic/repackager labels, so aliases are salt/form names ("Warfarin Sodium",
  "Tramadol Hcl Er", "Low Dose Aspirin"), not brands. "Coumadin" is reported as
  not in the corpus. A hand-written brand table was not added.
- **MEDICAL-SAFETY EDGE (flagged):** when no chunk names both drugs, the output
  says "No interaction evidence was found in our corpus", followed by an explicit
  "This does NOT mean the combination is safe", and the LLM is not called. The
  system prompt also forbids calling any drug or combination "safe". Absence of
  evidence in ~170 documents is not evidence of safety, and this wording must not
  be softened.
- **FDA source URLs → DailyMed.** `labels.fda.gov/<set_id>` returns 404;
  `dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=<set_id>` returns the label (verified).

**Bugs found and fixed (all found by running the thing):**
1. **Corpus did not reproduce from the README.** `harvest.py` defaulted to 10
   records per PubMed query, but the logged corpus used 12. First rebuild: 147
   docs / 1,145 chunks. The default is now 12 → 169 docs / 1,188 chunks. The
   remaining gap to the logged 171 / 1,255 comes from PubMed and openFDA being
   live sources (search results and label versions change); PMC is identical
   (9 docs, 301 chunks).
2. **Dead FDA source links** (above) — every FDA citation link was a 404.
3. **CLIs crashed on Unicode when stdout is a pipe on Windows** (cp1252 cannot
   encode "≥"). `build_index --check` died on it. Fixed once in `config.py`
   (UTF-8 stdout/stderr), since every entry point imports it.
4. **Grounding check missed `[1, 2, 5]`-style citations**, so a well-cited answer
   was downgraded to Low. The regex now accepts [1], [1][2] and [1, 2].
5. **Over-applied advice refusal:** Gemini prefixed "I cannot give personal
   medical advice" on general interaction questions. Rule 4 of the prompt now
   applies only to personal-advice requests.
6. **Interaction evidence panel showed "hybrid rank #0" for every passage.** The
   candidates had been rebuilt without their retrieval metadata. Each chunk now
   keeps its hybrid rank, and chunks the label scan found are tagged.

**Verification:**
- `python -m src.retrieval.build_index --check` — **pass**; nearest neighbours
  sensible and matching the last session's pattern (including the known dense
  metformin/metoprolol confusion that hybrid + rerank fixes).
- `python -m src.retrieval.search --eval --rerank` — **pass, matches the log**:
  renal query → Contraindications #1 (was hybrid #2); aspirin/warfarin → warfarin
  Drug Interactions #1; CYP3A4/simvastatin → top 3 all simvastatin; metformin
  side effects → Adverse Reactions above Drug Interactions.
- `python -m pytest` — **88 passed** (the original 67 + 21 new).
- `python -m src.generation.answer "What are the common side effects of metformin?"`
  — **pass**: diarrhoea, nausea/vomiting, flatulence etc., cited to the FDA
  label's Adverse Reactions section; Medium confidence (top score 4.4, 3 documents).
- `python -m src.generation.answer "What is the treatment for malaria?"` — **pass**:
  not-found message, LLM not called.
- Extra Q&A checks — kidney contraindications of metformin (High, 6.1),
  serotonin syndrome features (High, 9.0), community-acquired pneumonia treatment
  (High, 7.5): all well cited.
- `python -m src.interactions.check` — **pass** for warfarin+aspirin (warfarin
  label Drug Interactions + literature), simvastatin+clarithromycin (both labels,
  contraindicated, CYP3A mechanism), sertraline+tramadol (sertraline label
  Warnings, serotonin syndrome); amoxicillin+gabapentin → "No interaction
  evidence was found in our corpus…"; warfarin+coumadin → "Not in our corpus: coumadin".
- `streamlit run app.py` — server started and `/_stcore/health` returned ok.
  Both tabs were then driven with Streamlit's `AppTest` harness, which runs
  `app.py` with real inputs. This covered the metformin and malaria questions and
  the warfarin+aspirin, simvastatin+clarithromycin, amoxicillin+gabapentin and
  warfarin+coumadin pairs. Disclaimer, answer, badge, linked sources and evidence
  expander all rendered, with no exceptions. No human looked at the page in a
  browser this session.

**Open issues / blockers:**
- **Safety observation (Phase 9):** "I have chest pain, what should I take?"
  scored below threshold, so it got the generic not-found message rather than an
  explicit "I can't give personal medical advice — seek care". Safe, but not
  ideal wording for an urgent symptom. Revisit in Phase 9's adversarial checks.
- Interaction confidence reflects rerank scores, which are modest for label
  chunks (warfarin+aspirin shows Medium; sertraline+tramadol shows Low, because
  only one document names both drugs). This is explainable but conservative.
- Some retrieval passages are loosely relevant, e.g. an olanzapine/metformin
  paper appears as a metformin side-effect source. They pass the threshold; the
  LLM mostly ignores them.
- `harvest.py` aborts the whole run on one failed batch (seen once on a PMC
  fetch in a re-run) instead of skipping it.
- The embedding build took 21 min this time (0.9 chunks/s), because it ran at the
  same time as a pip install. Do not rebuild the index close to the demo.
- The aspirin FDA label is an OTC label with no Drug Interactions section.

**Next session should start with:**
- Open `streamlit run app.py` in a browser and rehearse the demo questions, then
  start Phase 8 (pick the first advanced feature with the user) or Phase 9
  evaluation.
