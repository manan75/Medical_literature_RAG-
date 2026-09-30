# Presentation Prep: Medical Literature RAG

One study document for both presenters. It teaches the whole system from the ground up, so either of us can explain any part and answer follow-up questions.

**How to trust the numbers in this file.** Every number specific to our system was read from the code (`src/config.py` and the modules in `src/`), from `SESSION_LOG.md`, or from a command run on 2026-10-01 against the current index. Where a number came from a run on this laptop, the section says so. General explanations of techniques (BM25, embeddings, HNSW, cross-encoders) are standard background, kept simple.

**Current system at a glance (verified 2026-10-01)**

| Item | Value | Where it comes from |
|---|---|---|
| Documents | 1,183 (MedlinePlus 1,014, PubMed 140, FDA labels 20, PMC 9) | `data/processed/documents.jsonl` |
| Passages (chunks) | 3,008 (MedlinePlus 1,820, FDA 569, PubMed 318, PMC 301) | `data/chunks/chunks.jsonl` |
| Vectors in Chroma | 3,008, 768 dimensions each | Chroma collection count |
| Embedding model | `NeuML/pubmedbert-base-embeddings` | `config.EMBEDDING_MODEL` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `config.RERANKER_MODEL` |
| LLM | `gemini-3.5-flash-lite` | `config.GEMINI_MODEL` |
| Dense / sparse candidates | 20 / 20 | `DENSE_TOP_K`, `SPARSE_TOP_K` |
| RRF constant k | 60 | `RRF_K` |
| Passages sent to the LLM | up to 5 | `RERANK_TOP_K` |
| Not-found threshold | 0.0 (rerank score) | `NOT_FOUND_THRESHOLD` |
| Confidence: strong score / agreement window | 5.0 / 3.0 | `HIGH_SCORE`, `CONFIDENCE_SPREAD` |
| Chunk size / overlap | about 350 / about 60 tokens | `CHUNK_TARGET_TOKENS`, `CHUNK_OVERLAP_TOKENS` |
| Tests | 110 passing, all offline | `python -m pytest` |

---

## 0. Quick-start

### 0.1 The 60-second pitch

> Large language models write fluent medical answers, but they invent facts, they cannot tell you where an answer came from, and their knowledge stops at a training date. For medicine that is not acceptable.
>
> We built a retrieval-augmented system that answers only from trusted public sources: 1,014 MedlinePlus health topics from the US National Library of Medicine, 20 FDA drug labels, and 149 PubMed and PubMed Central papers. That is 3,008 searchable passages.
>
> When you ask a question, we search those passages two ways at once, by meaning and by exact words, merge the results, and re-score the best 20 with a second, more precise model. If nothing scores as relevant, we say "not found" and never call the language model. Otherwise Gemini writes an answer using only the top passages, with a numbered citation on every claim, and we show a confidence rating and every passage it used.
>
> A second tool checks two drugs for interaction evidence in the FDA labels and the literature. If it finds nothing, it says so, and it never claims a combination is safe. It is for information and education only, not medical advice.

### 0.2 The 3-minute architecture walkthrough (script)

Point at the architecture slide while speaking.

1. **Two halves.** "The system has an offline half that builds the knowledge base once, and an online half that answers each question."
2. **Offline, left to right.** "We harvest three public sources: PubMed and PubMed Central through NCBI's E-utilities API, FDA drug labels through openFDA, and MedlinePlus health topics as a daily XML file from the National Library of Medicine. Parsers turn each format into one common Document record with sections like 'Contraindications' or 'Summary'. A section-aware chunker splits each section into passages of about 350 tokens with about 60 tokens of overlap, and never lets a passage cross a section boundary. PubMedBERT, a biomedical language model, turns each passage into a list of 768 numbers that captures its meaning. We store those vectors in Chroma, a local vector database. We also build a BM25 keyword index over the same passages."
3. **Online, left to right.** "A question is embedded with the same PubMedBERT model. We take the 20 nearest passages by meaning from Chroma and the 20 best keyword matches from BM25, and merge the two lists with Reciprocal Rank Fusion, which combines them by rank position. A cross-encoder then reads the question together with each of the 20 candidates and gives a precise relevance score. We keep the top 5."
4. **The gate.** "If the best score is below zero, nothing in our corpus is relevant, so we return 'not found' and the language model is never called. That is the main defence against made-up answers."
5. **Generation.** "Otherwise we number the passages 1 to 5, put them in a prompt that says 'answer only from these passages, cite every claim, do not diagnose or prescribe', and send it to Gemini 3.5 Flash-Lite once."
6. **Output.** "We return the answer, a numbered source list with links, a confidence rating from the scores, and a panel showing every passage and its score, so anyone can check the answer."
7. **Interactions.** "The drug interaction tab reuses the same pipeline, but only accepts passages that name both drugs, and says clearly when no evidence exists."

---

## 1. The big picture

### 1.1 What an LLM is, and why a plain chatbot is unsafe for medicine

**Large language model (LLM):** a neural network trained on a huge amount of text to predict the next word, which lets it write fluent answers to questions.

How it works: during training, the model reads billions of sentences and adjusts its internal numbers (its **parameters**, the learned weights inside the network) so that it gets better at guessing the next word. After training, it produces text one word at a time, each time choosing a likely next word. It has no database of facts. Everything it "knows" is compressed into those parameters.

Everyday analogy: a student who read the whole library once, cannot open any book during the exam, and must answer from memory. Usually right, sometimes confidently wrong, and unable to point to the page.

Three problems make a plain chatbot unsafe for medical questions:

1. **Hallucination:** the model produces a confident, fluent statement that is false. It predicts plausible text, not true text. A made-up dose or interaction reads exactly like a real one.
2. **Knowledge cutoff:** the model only knows what was in its training data up to a fixed date. It cannot know about a label change or a new guideline published after that.
3. **No citations:** it cannot reliably tell you where a statement came from, so a reader cannot check it.

### 1.2 What RAG is

**Retrieval-augmented generation (RAG):** a design where, before the LLM answers, a search system finds relevant passages from a trusted collection of documents, and the LLM is told to answer only from those passages.

Analogy: the same student now takes an open-book exam, is handed the five most relevant pages, and must quote the page number for every sentence. If the pages do not contain the answer, the student must say so.

What RAG fixes: the answer is grounded in text we chose and can show; the knowledge is as fresh as the last time we rebuilt the collection; every claim can carry a citation. What it does not fix by itself: the LLM can still misread a passage. That is why we add the relevance gate, citation checks and the evidence panel.

**Corpus:** the collection of documents a search system can search. Ours has 1,183 documents.

**Chunk (passage):** a short piece of a document, a few paragraphs long, that is the unit we search and cite. Ours has 3,008.

### 1.3 Offline and online pipelines

**Offline pipeline:** the steps run ahead of time to build the searchable knowledge base. Run once, or when data changes.

**Online pipeline:** the steps run every time a user asks a question.

```
OFFLINE: build the knowledge base (run once)

 PubMed/PMC (NCBI)   openFDA labels   MedlinePlus XML
        \                 |                 /
         src/data_sources/harvest.py  (+ pubmed.py, openfda.py, medlineplus.py)
                          |  raw files in data/raw/
                          v
         src/ingestion/pipeline.py  (pubmed_parser, fda_parser,
                          |          medlineplus_parser, pdf_extract, html_extract)
                          |  data/processed/documents.jsonl  (1,183 Documents)
                          v
         src/chunking/pipeline.py  (splitter.py)
                          |  data/chunks/chunks.jsonl        (3,008 Chunks)
                          v
         src/retrieval/build_index.py  (embeddings/encoder.py -> PubMedBERT)
                          |
            +-------------+--------------+
            v                            v
   Chroma vector store             BM25 keyword index
   data/vectorstore/               rebuilt in memory from chunks.jsonl
   (retrieval/vector_store.py)     on every start (retrieval/sparse.py)


ONLINE: answer one question (every request)

 question
    |
    v
 src/retrieval/hybrid.py
    |-- dense:  embed question (PubMedBERT) -> Chroma top 20
    |-- sparse: tokenize question -> BM25 top 20
    +-- Reciprocal Rank Fusion (k = 60) -> 20 candidates
    |
    v
 src/retrieval/rerank.py  cross-encoder scores 20 -> keep top 5
    |
    v
 src/generation/answer.py  gate: drop passages scoring < 0.0
    |                               none left? -> "not found", LLM NOT called
    v
 build numbered prompt -> src/generation/providers.py -> Gemini (1 call)
    |
    v
 answer + numbered sources + confidence + evidence panel  (app.py / CLI)

 Drug interactions (src/interactions/check.py): normalise 2 names ->
 FDA label scan + hybrid search -> keep passages naming BOTH drugs ->
 rerank -> same generation step; none -> "No interaction evidence was
 found in our corpus", LLM NOT called.
```

**Say-it-out-loud answers**

- *"Why not just use ChatGPT?"* A general chatbot answers from memory, so it can invent facts, has a training cutoff, and cannot show sources. Our system answers only from passages we retrieved from public medical sources, cites every claim, and refuses when nothing relevant is found. You can click every citation and read the exact passage.
- *"What is the difference between the offline and online parts?"* Offline we download, clean, split and index the documents once. That is the slow part, minutes of CPU time. Online we only search the prepared index and make one LLM call, which takes seconds.

---

## 2. Offline pipeline, stage by stage

### 2.1 Data sources

**a) The concept**

**Data source:** where our documents come from. For a medical system the key questions are "is it trustworthy" and "are we allowed to use it".

**API (application programming interface):** a web address a program can call to get data in a structured format instead of a web page.

**Public domain:** content with no copyright, which anyone may copy and reuse. Works produced by the US federal government are public domain under US law.

**Open access:** content the copyright holder has licensed for free reading and, usually, reuse.

**b) Our choice**

| Source | What it is | How we fetch it | Licence | In corpus |
|---|---|---|---|---|
| **PubMed** | NCBI's index of biomedical research papers. We take titles and abstracts. | NCBI **E-utilities** API: `esearch` (find IDs for a query, sorted by relevance) then `efetch` (download the records as XML). `src/data_sources/pubmed.py` | Abstract metadata is freely available; we use it for academic retrieval | 140 docs, 318 passages |
| **PMC (PubMed Central)** | NCBI's archive of full-text papers. We only request the open-access subset. | Same E-utilities, `db=pmc`, queries include `open access[filter]` | Open-access subset only | 9 docs, 301 passages |
| **FDA drug labels** | The official prescribing information for a drug, split into sections such as Indications, Dosage, Contraindications, Warnings, Adverse Reactions, Drug Interactions. | **openFDA** label API (`api.fda.gov/drug/label.json`). `src/data_sources/openfda.py` | US Government work, public domain | 20 docs, 569 passages |
| **MedlinePlus Health Topics** | The US National Library of Medicine's plain-language pages on diseases and conditions. | One daily compressed XML file listed on `medlineplus.gov/xml.html`. `src/data_sources/medlineplus.py` | The health topic summaries are public domain. NLM asks for the credit "Source: MedlinePlus, National Library of Medicine". | 1,014 docs, 1,820 passages |

How `src/data_sources/corpus_spec.py` decides what gets harvested. It is plain data, so the corpus is reproducible:

- `DRUGS`: 20 generic drug names (metformin, warfarin, aspirin, ibuprofen, lisinopril, atorvastatin, simvastatin, clarithromycin, amoxicillin, omeprazole, levothyroxine, metoprolol, sertraline, tramadol, amlodipine, clopidogrel, prednisone, furosemide, gabapentin, ciprofloxacin). One FDA label is fetched per drug.
- `INTERACTION_PAIRS`: 6 pairs used by the interaction tests. 4 are expected to interact (warfarin+aspirin, simvastatin+clarithromycin, sertraline+tramadol, warfarin+ciprofloxacin). 2 are controls with no major interaction expected (metformin+levothyroxine, amoxicillin+gabapentin).
- `PUBMED_QUERIES`: 12 topic queries (for example "metformin adverse effects review", "serotonin syndrome clinical features"). `harvest.py` fetches 12 records per query by default.
- `PMC_QUERIES`: 3 full-text queries, 3 records each.
- `MEDLINEPLUS_LANGUAGE = "English"` and `MEDLINEPLUS_GROUPS = None`, meaning every English topic is kept.

Details worth knowing:

- **Politeness and rate limits.** All requests go through `src/data_sources/http.py`, which waits between calls to the same host and retries up to 3 times with exponential backoff (waits of 1, 2 then 4 seconds). NCBI allows 3 requests per second without an API key and 10 with one, so the code waits 0.34 s or 0.11 s between NCBI calls.
- **Picking the right FDA label.** A search for "metformin" also returns combination products such as "sitagliptin and metformin". `openfda._specificity()` ranks candidates: exact generic name first, then salt forms like "metformin hydrochloride", then others, with combination products last. This bug was found and fixed in session 2.
- **MedlinePlus file.** The file is regenerated Tuesday to Saturday under a dated name (for example `mplus_topics_compressed_2026-09-29.zip`, 4,665 KB zipped, 29,433 KB unzipped). The harvester reads the index page, picks the newest date, and saves it under a fixed name with a provenance sidecar: source URL, generation date, size and SHA-256 checksum.
- **Provenance sidecars.** Every PubMed batch is saved with a `.meta.json` file recording which query produced which IDs.

**c) Why this and not the alternatives**

- **DrugBank was rejected.** Its full drug-interaction dataset needs a paid licence. That is why interaction evidence comes from the `drug_interactions` section of FDA labels plus literature (CLAUDE.md section 3).
- **Mayo Clinic and Cleveland Clinic were rejected.** Their terms of use prohibit scraping and reuse of their content.
- **MedlinePlus was added on 2026-09-29** because PubMed and FDA labels are written for clinicians. Before it, all six lay test questions ("what are the symptoms of malaria?") returned "not found". Only the NLM-written summaries are used. MedlinePlus also hosts the A.D.A.M. Medical Encyclopedia and ASHP drug monographs, which are copyrighted and licensed to NLM only, so we never fetch them. The same XML file also contains links to third-party pages, which we ignore.
- **Why harvesting is separate from parsing.** Harvesting hits the network; parsing only reads files on disk. Keeping them apart means a parser fix never needs a re-download.

**d) Say-it-out-loud answers**

- *"Are you allowed to use this data?"* Yes. FDA labels are US Government works and public domain. MedlinePlus health topic summaries are listed by NLM as public domain; we exclude the encyclopedia and drug monographs they license from others. PMC is limited to the open-access subset. We rejected DrugBank, Mayo Clinic and Cleveland Clinic specifically because of licensing.
- *"Why only 20 drugs?"* It is a demonstration corpus defined in one file, `corpus_spec.py`, chosen to include well-known interaction pairs and two control pairs, so we can test both "interaction found" and "no evidence found". Adding a drug is one line plus a re-harvest.
- *"How does the data stay current?"* Every source is re-downloadable with one command. MedlinePlus publishes a new file five days a week and the harvester always picks the newest. After re-harvesting, we rebuild the index. There is no automatic schedule yet.
- *"Why is the corpus so small?"* For a mid-term prototype we chose depth of engineering over size: every stage is tested and inspectable. The pipeline itself does not change with size, only the build time does.

### 2.2 Ingestion and parsing

**a) The concept**

**Ingestion:** turning raw downloaded files in different formats into one clean, consistent record format.

**Parsing:** reading a structured file format (XML, JSON, HTML) and pulling out the parts we need.

**XML:** a text format that wraps data in named tags, like `<AbstractText Label="RESULTS">...</AbstractText>`. **JSON:** a text format of key-value pairs, like `{"drug_interactions": ["..."]}`. **HTML:** the tag-based format of web pages.

**Normalisation:** cleaning text so it is consistent, without changing its meaning.

**Metadata:** data about the text: its title, source, section, date and link.

Analogy: a librarian receiving books, journals and leaflets in different languages and bindings, and re-filing every item onto the same kind of index card: title, section, date, where it came from, and the text.

**b) Our choice**

`src/ingestion/pipeline.py` walks `data/raw/` and sends each file to the right parser by its pattern:

| Format | Parser | What it keeps |
|---|---|---|
| PubMed XML | `pubmed_parser.parse_pubmed_xml` | Title, journal, date, up to 12 authors, and each labelled abstract part (BACKGROUND, METHODS, RESULTS, CONCLUSIONS) as its own section. Records with no abstract are skipped. |
| PMC XML (JATS, the standard XML format for journal articles) | `pubmed_parser.parse_pmc_xml` | Abstract plus each leaf `<sec>` of the body as a section. Accepts `pmcid`, `pmc` or `pmcaid` ID labels. |
| openFDA JSON | `fda_parser.parse_label` | 11 label sections (Indications and Usage, Dosage and Administration, Contraindications, Warnings and Cautions, Warnings, Boxed Warning, Drug Interactions, Adverse Reactions, Use in Specific Populations, Clinical Pharmacology, Description), with field names turned into headings. Date converted to ISO format (YYYY-MM-DD). Link points to the label on DailyMed. |
| MedlinePlus zip of XML | `medlineplus_parser.parse_file` | English topics only: title (as "MedlinePlus: Malaria"), URL, date created, one "Summary" section, "also called" names, groups, and the source agency. |
| PDF | `pdf_extract.extract_pdf` (PyMuPDF) | Detects headings by font size and boldness, and stops at a "References" heading. |
| HTML | `html_extract.extract_html` (BeautifulSoup) | Drops scripts, navigation and footers, and splits on h1 to h4 headings in document order. |

**Important honesty point:** the PDF and HTML extractors are implemented and unit-tested, but **the current corpus contains no PDF or HTML documents**. All live data is XML (PubMed, PMC, MedlinePlus) or JSON (FDA).

Normalisation (`src/ingestion/normalize.py`), applied to every extractor's output:

- Unicode normalisation (NFKC), which folds ligatures such as the single character "ﬁ" into "fi".
- Repairs words broken across lines ("hyper-" newline "tension" becomes "hypertension").
- Removes boilerplate lines such as "Downloaded from", "All rights reserved", "Page 3 of 10".
- Collapses repeated spaces and blank lines.
- Deliberately does **not** strip numbers, units or punctuation, so "2.5 mg/kg q12h" survives intact.
- MedlinePlus-specific: strips the summary HTML, keeps list items as "- item" lines, moves the agency credit line (for example "Centers for Disease Control and Prevention", found on 634 topics) into metadata, and writes "Also called: ..." into the text.

Duplicate documents (the same paper found by two queries) are removed by document ID.

**The schema (`src/schema.py`).** A **schema** is the fixed list of fields every record must have.

- `Document`: `doc_id` (for example `pubmed:12345`, `fda:<set_id>`, `medlineplus:315`), `source`, `title`, ordered `sections` (each a heading plus text), `url`, `date`, `authors`, `journal`, `drug_names`, `extra`.
- `Chunk`: `chunk_id` (`<doc_id>::<section index>::<part index>`, for example `medlineplus:315::0::0`), `doc_id`, `text`, `section`, `source`, `title`, `url`, `date`, `drug_names`, `token_estimate`.
- `Chunk.citation()` builds the human-readable label, for example `MedlinePlus: Malaria [Summary] (1999-10-01)`.

**Why metadata travels with every chunk.** The citation is built from the chunk itself. Every retrieval result carries the whole `Chunk`, not just an ID, so by the time a passage reaches the LLM it already has its title, section, date and link attached. It is impossible to retrieve a passage we cannot cite. The section name also helps retrieval: it is embedded with the text (section 2.4) and indexed by BM25 (section 2.6).

**c) Why this and not the alternatives**

- **Section-preserving parsing instead of flat text.** A chunk tagged "Contraindications" can be understood on its own; a chunk tagged with nothing cannot. PubMed abstract labels and FDA section names are kept for that reason.
- **Dataclasses instead of loose dictionaries.** A typo in a field name fails loudly at the boundary instead of producing a chunk that cannot be cited.
- **Bugs found by running the parsers (logged):** PMC files silently produced zero documents because NCBI now labels the ID `pmcid` (fixed, which added 9 documents and 301 chunks); PDF heading detection collapsed every section into one because it picked the most common font size by line count (fixed by weighting by character count); MedlinePlus summaries with nested lists would have duplicated text in 140 topics (fixed).

**d) Say-it-out-loud answers**

- *"Which formats can you ingest?"* XML, JSON, HTML and PDF are all supported and tested. The live corpus uses XML from PubMed, PMC and MedlinePlus and JSON from openFDA. PDF and HTML support is there for guideline documents, but we have not added any yet.
- *"Why keep section names?"* Because in medicine the same sentence means different things in different sections. "Take 500 mg" under Dosage is information; under Contraindications it would be dangerous. The section name travels with the text into the index, the prompt and the citation.
- *"What do you do with references and boilerplate?"* The normaliser drops boilerplate lines like copyright notices, and the PDF extractor stops at a References heading, because bibliographies match many keywords but answer nothing.

### 2.3 Chunking

**a) The concept**

**Chunking:** splitting long documents into short passages that can be searched and cited one at a time.

Why chunk at all: a search result should be small enough to be specific, and the LLM prompt has limited room. A whole FDA label is tens of thousands of words; a question about side effects needs one section.

**Token:** a unit of text a language model reads, roughly a word or a piece of a word. "metformin" might be split into pieces such as "met", "##form", "##in".

**Fixed-size chunking:** cutting every N tokens regardless of content. Simple, but it can cut mid-sentence and mix two sections in one chunk.

**Section-aware chunking:** splitting only inside a section, at sentence boundaries, so a chunk never contains the end of one section and the start of the next.

**Overlap:** repeating the last sentences of one chunk at the start of the next, so a fact and its qualifier are not separated.

Analogy: cutting a recipe book into index cards. Fixed-size cutting puts the end of "Ingredients" and the start of "Allergy warnings" on one card. Section-aware cutting starts a new card at every heading and never splits a sentence.

**b) Our choice** (`src/chunking/splitter.py`, settings in `src/config.py`)

| Setting | Value | Meaning |
|---|---|---|
| `CHUNK_TARGET_TOKENS` | 350 | aim for about 350 tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | 60 | carry about 60 tokens of the previous chunk forward |
| `CHUNK_MIN_TOKENS` | 40 | a last chunk smaller than this is merged into the one before |
| `CHUNK_NOISE_FLOOR` | 10 | a chunk under 10 tokens that is a whole section (for example a bare URL) is dropped |

Step by step, for each section of each document:

1. **Split into blocks** at blank lines.
2. **Detect tables.** A block is a table if at least half its lines (and at least 2) look like columns: two or more spaces, a pipe `|`, or a tab between words. A table is kept as one unbreakable unit, because half a dosage table is worse than none.
3. **Split other blocks into sentences**, with an abbreviation guard. "Administer 5 mg i.v. every 8 h" must not be cut after "i.v.". The splitter rejoins a split only if the previous piece ends in a known abbreviation (mg, i.v., b.i.d., e.g., and others) **and** the next piece starts with a lowercase letter or a digit.
4. **Pack sentences greedily** into a chunk until adding the next one would pass 350 tokens. Then start a new chunk that begins with the last sentences of the previous one, up to 60 tokens, as overlap.
5. A single unit larger than 350 tokens (a big table) becomes its own chunk rather than being broken.
6. A trailing chunk under 40 tokens is merged back into the previous chunk.
7. Chunk ID is `<doc_id>::<section index>::<part index>`.

**Token counting is an estimate.** `estimate_tokens()` counts whitespace words and multiplies by 1.3, to avoid loading a tokenizer during chunking. We checked it against PubMedBERT's real tokenizer on all 3,008 chunks: the median real ratio is 1.30 tokens per word, so the estimate is accurate on average.

Results on the current corpus (logged and re-measured): 3,008 chunks; estimated size minimum 10, median 291, maximum 1,054 tokens; 4 chunks exceed 1.5 times the target (intact tables); 200 distinct section headings preserved. For MedlinePlus, 615 topics are 1 chunk, 137 are 2, 144 are 3, 91 are 4 and 27 are 5.

**c) Why this and not the alternatives**

- **Section-aware over fixed-size.** From the chunker's own design note: a chunk that merges the tail of "Dosage and Administration" with the head of "Contraindications" reads as though a dose were being recommended where it is contraindicated. The section boundary is treated as a hard stop.
- **Overlap.** A fact and its exception often sit in adjacent sentences ("may be used in renal impairment" / "except when eGFR is below 30"). Overlap means retrieving one also retrieves the other.
- **The cost.** Chunks vary in size, and a few table chunks are too long for the embedding model (section 2.4 gives the exact count).

**d) Say-it-out-loud answers**

- *"How big are your chunks and why?"* About 350 tokens with 60 tokens of overlap. That is a few paragraphs: large enough to hold a fact with its context, small enough that five of them fit comfortably in the prompt and the embedding model can read each one whole.
- *"Why not fixed-size chunks?"* Fixed-size chunks can glue two sections together. In a drug label that could put a dose next to a contraindication. We never cross a section boundary, never cut a sentence, and keep tables whole.
- *"How do you handle tables?"* We detect column-shaped blocks and keep each table as one chunk, even if it is larger than the target size. 4 chunks in the corpus are larger than 1.5 times the target for this reason.

### 2.4 Embeddings

**a) The concept**

**Embedding (vector):** a list of numbers that represents the meaning of a piece of text, produced by a neural network, such that texts with similar meaning get similar lists.

**Dimensions:** how many numbers are in the list. Ours has 768. You can think of each text as a point in a 768-dimensional space. Texts about the same thing sit close together.

**Cosine similarity:** a score from -1 to 1 for how closely two vectors point in the same direction. 1 means the same direction (same meaning), 0 means unrelated. If vectors are **normalised** (scaled to length 1), cosine similarity is simply the sum of the products of matching numbers, which is fast.

Analogy: a map where every document is a pin. Documents about diabetes cluster in one neighbourhood and documents about antibiotics in another. A question becomes a pin too, and search means "find the pins nearest to this one".

**BERT:** a transformer model from Google (2018) that reads a whole sentence at once and produces a vector for every token. A **transformer** is a network built from layers of **attention**, a mechanism that lets each word look at every other word in the sentence to work out its meaning in context. "Base" size BERT has 12 layers.

**Pretraining:** the first, expensive training stage, where a model learns language by filling in hidden words in a huge amount of text.

**Fine-tuning:** a second, smaller training stage that adapts a pretrained model to one task.

**Bi-encoder:** a model that turns the question and each document into vectors *separately*. Because documents are encoded independently, all of them can be embedded ahead of time, and search is just comparing vectors. That is what makes searching thousands of passages in milliseconds possible.

**Pooling:** combining the per-token vectors into one vector for the whole text. **Mean pooling** takes the average of all token vectors.

**b) Our choice: `NeuML/pubmedbert-base-embeddings`**

Read the name in parts:

- **NeuML:** the company that published it on Hugging Face (the public model hub).
- **PubMedBERT:** a BERT model that Microsoft pretrained from scratch on PubMed text, with its own vocabulary built from biomedical text. Its model config names the base model as `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`: "uncased" means it lowercases input, and "abstract-fulltext" means it was pretrained on PubMed abstracts and PMC full-text articles. Because its vocabulary was learned from biomedical text, drug and disease names are split into fewer, more meaningful pieces than a general vocabulary would produce.
- **base:** the 12-layer size.
- **embeddings:** NeuML fine-tuned it with the sentence-transformers library to produce one vector per text for similarity search. Its model card says the training data was PubMed title-abstract pairs plus similar-title pairs, trained with MultipleNegativesRankingLoss (a loss that pulls each title towards its own abstract and away from the other abstracts in the same batch) for 1 epoch.

Verified from the model files in our local cache:

| Property | Value |
|---|---|
| Architecture | BERT, 12 layers, 12 attention heads |
| Output size | 768 dimensions |
| Vocabulary | 30,522 word pieces |
| Maximum input | 512 tokens (longer input is cut off) |
| Pooling | mean pooling |
| Weights file | about 418 MB |

How our code uses it (`src/embeddings/encoder.py`):

- Loaded lazily, the first time it is needed, through `sentence_transformers.SentenceTransformer`.
- **Contextualised embedding input:** each chunk is embedded as `"<title> | <section>"` on the first line, then the chunk text. For example `MedlinePlus: Malaria | Summary` followed by the text. The heading goes *inside* the vector, because a chunk in "Drug Interactions" and one in "Adverse Reactions" can use almost the same words while answering different questions. The stored chunk text stays clean, so citations never show this prefix.
- Vectors are normalised (`normalize_embeddings=True`), so cosine similarity is a plain dot product.
- Batch size 16. Runs on CPU.
- A question is embedded with the same model and no special prefix. The model was trained for symmetric similarity.

**Measured truncation.** We tokenized every chunk's embedding input with the model's own tokenizer: the median length is 290 tokens, and **14 of 3,008 chunks (0.5%) exceed 512 tokens**, so their ends are cut off when embedded. They are 8 PMC and 6 FDA chunks, mostly the intact tables. BM25 still indexes their full text.

**The metformin vs metoprolol example.** Both names start with "met" and share letters. CLAUDE.md's reason for choosing a biomedical model is that a general model tends to treat such names as near neighbours on surface form.

We tested this directly on 2026-10-01, comparing our model with `BAAI/bge-small-en-v1.5`, a popular general-purpose embedding model (384 dimensions), which was already in the local model cache.

| Test | PubMedBERT (ours) | bge-small (general) |
|---|---|---|
| Cosine similarity of the words "metformin" and "metoprolol" | **0.416** | 0.768 |
| Cosine similarity of "metformin" and "glucophage" (its brand name) | 0.516 | 0.675 |
| Search all 569 FDA label chunks for "What are the common side effects of metformin?": rank of the first metoprolol chunk | 3 | 7 |
| Metformin chunks in that top 5 | 4 of 5 | 5 of 5 |

What this shows, honestly:

- **At the level of drug names, the claim holds.** The general model places metformin closer to metoprolol (0.768) than to its own brand name Glucophage (0.675). PubMedBERT keeps metformin and metoprolol apart (0.416) and closer to Glucophage (0.516).
- **In this one retrieval test, the general model did better.** Dense search with PubMedBERT still put a metoprolol chunk at rank 3; the session 2 log recorded the same. This is one question on one slice of the corpus, so it proves neither model better overall, but it means we must not claim that PubMedBERT alone solves look-alike names.
- **This is exactly why the pipeline does not rely on dense search alone.** BM25 matches "metformin" exactly, and the reranker reads the question with each passage. In the final reranked top 5 for this question, the first two are the metformin label's Adverse Reactions sections and no metoprolol passage appears.

Say it this way: "A general model scores metformin as more similar to metoprolol than to its own brand name; PubMedBERT separates them. But no embedding model is perfect on look-alike names, so we pair it with exact keyword search and a reranker."

**c) Why this and not the alternatives**

- **Domain-specific over general (CLAUDE.md section 3).** The model has a biomedical vocabulary and was trained on PubMed text, so it represents medical terms better. NeuML's model card reports it scoring higher than general models such as `all-MiniLM-L6-v2`, `bge-base-en-v1.5` and `gte-base` on three PubMed similarity benchmarks (average 95.62 vs 93.46 to 95.37). Those are the publisher's numbers, not ours.
- **Runs on CPU, no API key, no network at demo time.** An embedding API would add cost, a network dependency and a second place where medical text leaves the machine.
- **Rejected alternatives:** a general sentence model (weaker on medical terms), and larger biomedical models (slower on a laptop CPU).
- **Known cost:** 512-token limit (14 truncated chunks), and dense search alone still confuses look-alike drug names in practice, which is why we add BM25 and a reranker.

**d) Say-it-out-loud answers**

- *"What is an embedding?"* It is a list of 768 numbers that captures what a passage means. Passages about similar things get similar lists, so searching becomes "find the passages whose numbers are closest to the question's numbers", measured by cosine similarity.
- *"Why PubMedBERT and not a general model?"* It was pretrained on PubMed abstracts and full-text papers with a biomedical vocabulary, then fine-tuned for similarity on PubMed title-abstract pairs, so it handles drug and disease names better. It also runs locally on a CPU with no API key. We do not rely on it alone: BM25 and the reranker cover its mistakes with look-alike names.
- *"What happens to long chunks?"* The model reads at most 512 tokens. We measured that 14 of our 3,008 chunks are longer, mostly tables, and those are cut off in the embedding. BM25 still sees their full text, so they can still be found by keywords.
- *"Why put the section heading into the embedding?"* Because the same words can answer different questions depending on the section. Putting "Contraindications" inside the vector lets a question about contraindications match that section rather than a similar sentence under Dosage.

### 2.5 Vector database

**a) The concept**

**Vector database:** a store for vectors that can quickly return the vectors closest to a query vector, together with any data attached to them.

**Nearest neighbour search:** finding the stored vectors most similar to the query. Exact search compares the query with every vector; with millions of vectors that is slow.

**Approximate nearest neighbour (ANN) search:** a faster search that finds almost all of the true nearest neighbours by skipping most comparisons, trading a tiny amount of accuracy for a large speed-up.

**HNSW (Hierarchical Navigable Small World graph):** a common ANN method. Each vector becomes a node linked to a handful of its nearest neighbours. There are several layers: the top layers have few nodes and long links, like motorways; the bottom layer has every node and short links, like local streets. A search starts at the top, greedily moves to whichever neighbour is closer to the query, drops down a layer, and repeats until it reaches the closest nodes at the bottom.

Analogy: finding a house in a new city. Take the motorway to the right region, then main roads to the right district, then local streets to the right house, without visiting every street.

HNSW settings, in plain words:

- **Neighbours per node (often called M):** how many links each node keeps. More links, better accuracy, more memory.
- **ef_construction:** how many candidates are considered when building the links. Higher means a better graph and a slower build.
- **ef_search:** how many candidates are kept while searching. Higher means more accurate and slower search.

**b) Our choice: Chroma** (`chromadb` 1.5.9, `src/retrieval/vector_store.py`)

- `chromadb.PersistentClient` stores everything in files under `data/vectorstore/`. No server, no account, no network.
- Collection name `medical_literature`, distance metric **cosine** (`hnsw:space = cosine`).
- HNSW settings read from the collection on 2026-10-01: 16 neighbours per node, ef_construction 100, ef_search 100. These are Chroma's defaults; we did not tune them.
- We compute embeddings ourselves and pass them in. Chroma's own default embedding function is never used.
- Each vector is stored with its chunk text and metadata: `doc_id`, `section`, `source`, `title`, `url`, `date`, `drug_names` (joined with commas, because Chroma metadata values must be single values), `token_estimate`.
- Search returns a distance, and we convert it to similarity: `similarity = 1 - distance`. Results are rebuilt into full `Chunk` objects, so every result can be cited without a second lookup.
- Vectors are added in batches of 256.
- Current count: 3,008 vectors.

**c) Why Chroma and not FAISS**

**FAISS** is Facebook's vector search library, very fast, but it stores only vectors and numeric IDs. We would need a separate store to keep each vector's title, section, date and URL in sync. CLAUDE.md section 3: "FAISS was the alternative but stores no metadata alongside vectors, and citations depend on that metadata travelling with the vector." Chroma keeps the metadata with the vector, and it is embedded and file-backed, so the demo machine needs nothing extra. The cost: Chroma is slower and heavier than FAISS at very large scale, which does not matter at 3,008 vectors.

**d) Say-it-out-loud answers**

- *"What is a vector database for?"* It stores the 3,008 passage vectors and, given the question's vector, quickly returns the closest ones along with their text and citation data.
- *"Why Chroma?"* It runs inside our Python process from local files, so there is no server to install, and it stores metadata next to each vector, which our citations depend on. FAISS would have needed a separate metadata store.
- *"Is the search exact?"* No. It uses HNSW, an approximate method that walks a layered graph of neighbours. At our size it is effectively exact, and a dense search takes about 0.13 seconds including embedding the question.

### 2.6 Keyword search (BM25)

**a) The concept**

**Sparse retrieval (keyword search):** ranking documents by the words they share with the query, rather than by meaning. "Sparse" because each document is represented by the few words it contains out of a huge vocabulary.

**Term frequency (TF):** how many times a query word appears in a document. More mentions suggests more relevance.

**Inverse document frequency (IDF):** how rare a word is across the whole collection. A word in every document ("patient") tells you little; a word in three documents ("clarithromycin") tells you a lot. IDF gives rare words a high weight.

**TF-IDF:** score = sum over query words of TF times IDF. Its weakness: a document that repeats a word 50 times gets 50 times the credit, and long documents win just because they contain more words.

**BM25 (Best Matching 25):** an improved TF-IDF with two fixes:

1. **Saturation, controlled by k1.** The credit for repeating a word grows quickly at first and then levels off. The fifth mention adds much less than the first. Higher k1 means slower levelling off.
2. **Length normalisation, controlled by b.** A match in a short document counts for more than the same match in a long one. b = 0 means no length adjustment; b = 1 means full adjustment.

For each query word the score is roughly:

```
IDF(word) * tf * (k1 + 1) / ( tf + k1 * (1 - b + b * doc_length / average_doc_length) )
```

and a document's score is the sum over the query's words.

Analogy: grading essays by keywords. TF-IDF rewards an essay that repeats "insulin" 40 times. BM25 says "after a few mentions, more repetition does not help, and a short essay that mentions it is more focused than a long one".

**b) Our choice** (`rank_bm25` 0.2.2, `BM25Okapi`, in `src/retrieval/sparse.py`)

- `BM25Okapi` with library defaults: **k1 = 1.5, b = 0.75**. IDF is `ln((N - n + 0.5) / (n + 0.5))`, where N is the number of chunks and n is the number of chunks containing the word. Words that appear in more than half the chunks would get a negative IDF, so the library raises those to 0.25 times the average IDF.
- **What is indexed:** title, section heading and chunk text together, so a query naming a section ("metformin contraindications") matches the heading even if the body never repeats the word.
- **Tokenisation (our own):** lowercase, then take runs of letters, digits and hyphens with the pattern `[a-z0-9][a-z0-9\-]*`. This keeps "cyp3a4", "500mg" and hyphenated terms intact.
- **Minimal stopword list.** **Stopwords** are common words removed before indexing. Ours removes only 33 words like "the", "of", "is", "what", "how", and deliberately keeps "no", "not" and "without", which reverse the clinical meaning of a sentence.
- **Top 20** (`SPARSE_TOP_K = 20`). A document with score 0 (no query word present) is never returned.
- **Rebuilt in memory** from `chunks.jsonl` every time the app starts (0.74 seconds for 3,008 chunks), so it can never go out of date relative to the chunk file. A search takes about 8 milliseconds.

**c) Why keyword search at all**

Dense embeddings generalise, but they blur exact tokens, and in medicine the exact token is often the whole question: "CYP3A4", "eGFR 30", "500 mg", "INR", a specific drug name. BM25 matches those exactly. The logs show both failure modes: dense search alone ranked metoprolol third for a metformin query, and BM25 alone was thrown off by common words in long natural-language questions (it ranked an inositol paper first for "What are the common side effects of metformin?", matching "common" and "effects"). Combining the two (section 3.2) fixes both.

The alternative was dense-only retrieval (simpler, but it misses exact names and codes) or a learned sparse model such as SPLADE (better, but another model to download and run).

**d) Say-it-out-loud answers**

- *"What is BM25?"* A keyword ranking formula. It rewards rare words that appear in a passage, gives diminishing credit for repeating the same word, and favours shorter passages. We use the standard settings, k1 = 1.5 and b = 0.75.
- *"Why do you need keyword search if you have embeddings?"* Embeddings capture meaning but can blur exact names and codes. Our logs show dense search alone ranking metoprolol third for a metformin question. BM25 matches "metformin" exactly. Using both catches what either one misses.
- *"Why not remove more stopwords?"* Because words like "not" and "without" flip the meaning of a clinical sentence, so we keep them.

### 2.7 Building the index, and the incremental `--append`

**a) The concept**

**Indexing:** computing every chunk's embedding and storing it in the vector database. It is the slowest offline step, because the embedding model must run on every chunk.

**Incremental update:** adding only the new items to an existing index instead of rebuilding everything.

**b) Our choice** (`src/retrieval/build_index.py`)

- `python -m src.retrieval.build_index` rebuilds from scratch: it **drops the collection first**, so an old chunking scheme can never leave stale vectors behind.
- `--check` then runs five sample questions and prints their nearest neighbours as a sanity check.
- `--append` embeds **only chunks whose IDs are not already in Chroma**. Chunk IDs are deterministic (`<doc_id>::<section>::<part>`), so "not in the collection" means exactly "new". It verifies that the count equals old count plus new chunks. If the collection holds chunk IDs that no longer exist in `chunks.jsonl`, it prints a warning and tells you to do a full rebuild. It does not delete them silently. BM25 needs no step at all, because it is rebuilt from `chunks.jsonl` on every start.

How MedlinePlus was added (session log, 2026-09-29):

1. Snapshot `chunks.jsonl`, re-run ingestion and chunking with MedlinePlus included.
2. Confirm the 1,188 existing chunks are **byte-identical** to the snapshot, which proves the old vectors are still valid.
3. `build_index --append`: "Appending 1820 new chunk(s) to 1188 existing vectors", embedded in 748 seconds, "1188 + 1820 = 3008 vectors".

**Measured build times on this laptop's CPU (from the session log):**

| Run | Chunks | Time | Rate |
|---|---|---|---|
| Session 2, full build | 1,255 | 273 s | 4.6 chunks/s |
| Session 3, full build (a pip install was running at the same time) | 1,188 | 1,282 s (about 21 min) | 0.9 chunks/s |
| MedlinePlus `--append` | 1,820 | 748 s | 2.4 chunks/s |

So "about 21 minutes" was a full build slowed by another heavy process. A full rebuild of today's 3,008 chunks would take roughly 11 minutes at the best measured rate and longer on a busy machine. The time goes into running a 12-layer, 768-wide BERT model on each chunk on a CPU with no GPU.

**c) Why this and not the alternatives**

A full rebuild is always correct but slow. `--append` saved re-embedding 1,188 chunks. It is only safe when existing chunks are unchanged, which is why it refuses silently mixing old and new and asks for a full rebuild when it detects stale IDs.

**d) Say-it-out-loud answers**

- *"How long does indexing take?"* On this laptop's CPU, between 2.4 and 4.6 chunks per second depending on load, so about 11 to 21 minutes for all 3,008 chunks. Nearly all of that time is the BERT model running on each chunk. We never rebuild during a demo.
- *"How did you add MedlinePlus without rebuilding?"* We checked that the existing 1,188 chunks came out byte-identical after re-chunking, then embedded only the 1,820 new ones and verified 1,188 + 1,820 = 3,008 vectors.

---

## 3. Online pipeline, stage by stage

### 3.1 Query embedding, dense retrieval and sparse retrieval

**a) The concept**

**Dense retrieval:** search by meaning. Embed the question into a vector and return the chunks whose vectors are closest. "Dense" because every one of the 768 numbers carries some information.

**Top-k:** keep only the k best results.

**Recall:** the share of the truly relevant passages that the search manages to find. **Precision:** the share of returned passages that are truly relevant. A first-stage search should favour recall (do not miss anything), and a later stage should favour precision (keep only the best).

**b) Our choice** (`src/retrieval/hybrid.py`, `vector_store.py`, `sparse.py`)

- **Dense:** the question is embedded by the same PubMedBERT model (`encoder.embed_query`, no prefix, normalised), then Chroma returns the **top 20** by cosine similarity (`DENSE_TOP_K = 20`).
- **Sparse:** the question is tokenised with the BM25 tokenizer, and BM25 returns the **top 20** (`SPARSE_TOP_K = 20`).
- Both lists are deeper than the 5 we finally keep, so later stages have room to promote a passage one method ranked low.

Measured on this laptop (median of 10 runs, 2026-10-01): embedding the question about 0.11 s; Chroma search including that embedding about 0.13 s; BM25 search about 0.008 s.

**c) Why two retrievers** (see 2.6): dense search understands paraphrases ("high blood pressure" vs "hypertension"), BM25 catches exact names and codes. Each covers the other's blind spot.

**d) Say-it-out-loud answers**

- *"How many results do you retrieve?"* 20 from dense search and 20 from BM25. They are merged into 20 candidates, the reranker scores all 20, and at most 5 go to the LLM.
- *"Why retrieve 20 if you only use 5?"* The first stage is cheap and aims to not miss anything. The reranker can only promote what retrieval hands it, so a deeper list gives it room. Our logs show it regularly promoting passages from outside the hybrid top 5.

### 3.2 Hybrid fusion: Reciprocal Rank Fusion (RRF)

**a) The concept**

**Hybrid retrieval:** combining dense and sparse results into one ranking.

**Reciprocal Rank Fusion (RRF):** a way to merge ranked lists using only each item's position, not its score. For each passage:

```
RRF score(passage) = sum over each list the passage appears in of  1 / (k + rank in that list)
```

A passage near the top of both lists gets two large contributions. A passage in only one list gets one. The constant k damps the difference between rank 1 and rank 2, so one list cannot dominate just by ranking something first.

Analogy: two judges rank the same contestants. You do not average their raw marks, because one judge marks out of 10 and the other out of 100. You look at placements instead: someone placed 1st and 3rd beats someone placed 1st by one judge and absent from the other judge's list.

**b) Our choice** (`reciprocal_rank_fusion` in `src/retrieval/hybrid.py`)

- **k = 60** (`RRF_K`), the value from the original RRF paper (Cormack and others, 2009) and the usual default.
- Output: the top 20 fused candidates (`DENSE_TOP_K`), each remembering its dense rank and BM25 rank for display.

**Worked example with real numbers** from the question "What causes diabetes?" (full trace in section 6.1):

| Passage | Dense rank | BM25 rank | RRF score |
|---|---|---|---|
| MedlinePlus: Diabetes [Summary] | 1 | 3 | 1/(60+1) + 1/(60+3) = 0.016393 + 0.015873 = **0.032266** |
| MedlinePlus: How to Prevent Diabetes [Summary] | 5 | 2 | 1/(60+5) + 1/(60+2) = 0.015385 + 0.016129 = **0.031514** |
| A passage found only by dense search at rank 1 | 1 | absent | 1/61 = **0.016393** |

The code printed 0.03227 and 0.03151 for the first two, matching the arithmetic. Notice that a passage ranked first by only one method scores about half of one ranked well by both. Agreement between two independent signals is treated as evidence of relevance.

**c) Why fuse by rank instead of by raw scores**

Cosine similarity lives between -1 and 1 (our results sit around 0.6). BM25 scores have no fixed range and depend on the corpus (6.58 for the top diabetes hit). Adding them directly lets BM25 dominate; combining them fairly needs normalisation weights that must be re-tuned whenever the corpus changes. RRF ignores the magnitudes, needs no tuning, and rewards agreement (session log, "RRF chosen over weighted score blending"). The alternative, a weighted sum of normalised scores, can be slightly better when tuned on labelled data, which we do not have yet.

**d) Say-it-out-loud answers**

- *"What is RRF?"* Each passage gets 1 divided by (60 plus its rank) from each list it appears in, and we add those up. It merges the meaning-based and keyword-based rankings using only positions, so we never have to compare a cosine similarity with a BM25 score.
- *"Why k = 60?"* It is the standard value from the paper that introduced RRF. It flattens the gap between rank 1 and rank 2, so one list cannot dominate by itself. We did not tune it, because without a labelled evaluation set any tuning would be guesswork.
- *"Show me a calculation."* "MedlinePlus: Diabetes" was dense rank 1 and BM25 rank 3: 1/61 + 1/63 = 0.0323. "How to Prevent Diabetes" was dense 5 and BM25 2: 1/65 + 1/62 = 0.0315. A passage only one list found at rank 1 would get 0.0164, about half.

### 3.3 Reranking

**a) The concept**

**Why two stages.** Stage one (retrieval) must be cheap enough to search every passage, so it can only compare precomputed vectors and keywords. Stage two (reranking) runs an expensive, more accurate model on just the 20 survivors. This is a recall step followed by a precision step.

**Cross-encoder:** a model that reads the question and one passage *together*, as a single input, and outputs one relevance score for that pair. Because every word of the question can attend to every word of the passage, it judges relevance much more accurately than comparing two separately computed vectors. It cannot be precomputed, since the score depends on the question, so it is only affordable on a short list.

```
BI-ENCODER (retrieval)                  CROSS-ENCODER (reranking)

question --> [encoder] --> vector q     [CLS] question [SEP] passage [SEP]
passage  --> [encoder] --> vector p                  |
                                           [one encoder reads both]
score = cosine(q, p)                                 |
passages encoded ahead of time,           score = one number for this pair
compared in milliseconds                  computed fresh for each question
```

Analogy: a recruiter first filters 3,000 CVs by keywords and a quick skim (bi-encoder), then an interviewer reads each of the 20 shortlisted CVs carefully against the job description (cross-encoder).

**Logit:** the raw output number of a model before it is squashed into a probability. It can be any real number. Positive means the model leans "relevant", negative means "not relevant", and 0 corresponds to 50% if you apply the sigmoid function (`1 / (1 + e^(-x))`).

**b) Our choice: `cross-encoder/ms-marco-MiniLM-L-6-v2`** (`src/retrieval/rerank.py`)

Read the name in parts:

- **cross-encoder:** the Hugging Face organisation of the sentence-transformers project, and the model type.
- **MS MARCO:** Microsoft MAchine Reading COmprehension, a large dataset of real Bing search queries with human-marked relevant passages. The model learned "does this passage answer this search query" from it.
- **MiniLM:** a family of small BERT-style models made by distillation. **Distillation** means training a small "student" model to copy the outputs of a large "teacher" model, keeping most of the accuracy at a fraction of the size.
- **L-6:** 6 transformer layers (BERT-base has 12). Its config also lists `ms-marco-MiniLM-L-12-v2` as its origin, so it is the 6-layer sibling of a 12-layer version.
- **v2:** the second version of these models.

Verified from the model files: 6 layers, hidden size 384 (half of BERT-base's 768), 12 attention heads, maximum input 512 tokens, weights file about 87 MB. The output activation is the identity function, so **scores are raw logits, not probabilities**. That is why our scores range from about -11 to +9.

How our code uses it:

- `RerankingRetriever.search`: hybrid returns 20 candidates (`candidate_k = DENSE_TOP_K = 20`), the cross-encoder scores all 20, and we keep the **top 5** (`RERANK_TOP_K = 5`).
- Each pair is `(question, "<title> | <section>\n<text>")`, the same contextualised text used for embeddings, so the reranker also sees the heading.
- Batch size 16. The original hybrid score and rank are kept in `components` (`retrieval_score`, `retrieval_rank`), so the UI can show how far the reranker moved each passage.
- Measured cost: scoring 20 candidates takes about 3.4 s on this laptop's CPU (median of 10 warm runs; 8.5 s on a cold first run). **This is the slowest part of retrieval.**

**Measured effect** (from `python -m src.retrieval.search --eval --rerank` on the current index, 8 probe questions):

- In **7 of 8** questions, the passage the reranker put first was **not** hybrid's first.
- **18 of the 40** final top-5 slots went to passages hybrid had ranked outside its top 5.
- Examples: "metformin contraindications renal impairment": the metformin label's **Contraindications** section went from hybrid #2 to #1 (score +8.21). "first line treatment for type 2 diabetes": "Metformin: clinical use in type 2 diabetes" went from hybrid #12 to #1 (+7.19). "What are the common side effects of metformin?": the label's **Adverse Reactions** went from #3 to #1, above its Drug Interactions section.

**Honest limit:** we have not measured precision at 5 against a labelled answer key. These are observed reorderings that we inspected by hand and judged correct. Formal measurement is Phase 9.

**c) Why this and not the alternatives**

- **General-domain, small and fast.** It judges question-passage relevance, not clinical meaning; the biomedical signal is already carried by the retrieval stage that produced the candidates (session log). At 6 layers it keeps reranking to a few seconds on a CPU.
- **Rejected:** LLM-based reranking (asking Gemini to rank passages), which would add a second API call per question on a rate-limited free tier; and larger or biomedical cross-encoders, which would be slower. CLAUDE.md marks the reranker as something to revisit if Phase 9 shows it limiting.

**d) Say-it-out-loud answers**

- *"Why rerank at all?"* Retrieval compares separately computed vectors, which is fast but blurry. The cross-encoder reads the question and each passage together and scores them precisely. On our 8 probe questions it changed the top result in 7, for example moving the Contraindications section to first place for a contraindications question.
- *"What is the difference between a bi-encoder and a cross-encoder?"* A bi-encoder embeds question and passage separately, so passages can be pre-computed and compared in milliseconds. A cross-encoder reads both together in one pass, which is far more accurate but must run fresh for every pair, so we only use it on 20 candidates.
- *"What do the scores mean?"* They are raw logits from the model, not probabilities. Positive means likely relevant, negative means likely not, and 0 is the 50-50 point. Our observed range is roughly -11 to +9.
- *"Isn't an MS MARCO model trained on web search, not medicine?"* Yes. It judges whether a passage answers a query, which transfers well. The medical knowledge comes from PubMedBERT in the retrieval stage. Whether a biomedical reranker would do better is on our evaluation list.

### 3.4 The not-found gate

**a) The concept**

**Relevance threshold (gate):** a minimum score a passage must reach to be used. If no passage reaches it, the system says it cannot answer instead of asking the LLM to answer from weak evidence.

Why it is needed: search always returns *something*. Asked about a drug we do not have, it returns the nearest passages about other drugs. Without a gate, the LLM would receive irrelevant passages and might stretch them into an answer.

**b) Our choice** (`generate_answer` in `src/generation/answer.py`)

- `NOT_FOUND_THRESHOLD = 0.0` on the reranker's logit scale.
- Every reranked passage scoring below 0.0 is dropped. If none remain, the answer is the fixed not-found message and **the LLM is not called**. The confidence reason says so.
- A second safety net: the prompt tells the model to reply exactly `NOT_FOUND` if the passages do not answer the question. That reply is also mapped to the not-found message.
- The drug interaction path does not use this gate (section 4).

How 0.0 was chosen (session log, 2026-09-29):

| Corpus | Out-of-corpus top scores | In-corpus top scores | Gap |
|---|---|---|---|
| Before MedlinePlus | -10.9 to -2.9 (malaria, isotretinoin, multiple sclerosis, chemotherapy, insulin glargine, car tyres, capital of France) | +3.9 to +8.5 (8 probe questions plus 4 more) | about 6.8 |
| After MedlinePlus | -10.9 to **-1.1** ("What is the half-life of adalimumab?" is the closest) | **+1.3** ("is tuberculosis contagious?") to +8.5 | about 2.4 |

0.0 sat in the gap both times and is the model's own 50% point, which is easy to explain. After MedlinePlus the gap got thinner for two reasons. Some former out-of-corpus questions, such as malaria, became in-corpus. And plainly worded lay questions score lower than technical ones: "is tuberculosis contagious?" found the right MedlinePlus page but scored only +1.3. Our likely explanation (not measured) is that short, casual questions give the cross-encoder less to match on. The threshold was re-checked and left at 0.0, with the thinner margin logged as a Phase 9 watch item.

**c) Why this and not the alternatives**

- **Fixed threshold on the reranker score vs asking the LLM to decide.** Asking the LLM "is this relevant?" costs a call and trusts the component we are trying to guard. The gate is cheap, deterministic and explainable.
- **Why not higher?** A higher value would reject real lay questions like the tuberculosis one (+1.3). **Why not lower?** The closest out-of-corpus question already scores -1.1.
- **Cost:** a single fixed number cannot be perfect. Borderline questions in the -1 to +1.5 band are where mistakes will happen.

**d) Say-it-out-loud answers**

- *"Why is the threshold exactly 0?"* We looked at the best reranker score for questions we know the corpus can answer and for questions it cannot. Answerable ones scored from +1.3 upwards and unanswerable ones -1.1 or lower. Zero sits in that gap, and it is also where the reranker's output means 50% relevant, so it is not an arbitrary number.
- *"What happens when nothing is relevant?"* The system says it could not find the information and suggests rephrasing, and the LLM is never called. The panel shows the reason: no passage scored above 0.
- *"Is the margin safe?"* It is thinner than before: 2.4 points instead of 6.8. It still separated every question we tested, but it is the first thing Phase 9 will test with more borderline questions.

### 3.5 Prompt construction

**a) The concept**

**Prompt:** the text we send to the LLM. **System prompt (system instruction):** a separate, higher-priority instruction that sets the model's role and rules for the whole conversation. **Grounding:** making the model's answer depend only on supplied evidence.

**b) Our choice** (`SYSTEM_PROMPT` and `build_prompt` in `src/generation/answer.py`)

The exact system prompt:

```
You are a medical literature retrieval assistant for education and information only.

Rules you must follow:
1. Answer ONLY from the numbered context passages supplied. Do not use outside
   knowledge, even if you are confident.
2. Cite every factual sentence with the passage number(s) it came from, like [1] or
   [2][3]. Never cite a number that is not in the context.
3. If the passages do not contain the answer, reply with exactly NOT_FOUND and
   nothing else.
4. Do not diagnose, do not recommend or adjust doses for a specific person, and do
   not tell anyone to start, stop or combine medications. Only if the question asks
   for that kind of personal advice, say you cannot give it, then summarise what the
   sources say in general terms and suggest consulting a healthcare professional.
   General questions about a drug's effects, side effects or interactions are not
   personal advice: answer them directly without a disclaimer.
5. Never state that a drug or drug combination is "safe". If the passages do not
   describe an interaction, say that no interaction evidence was found in the
   provided sources.
6. If passages disagree, say so and cite both sides.
7. Be concise: a short paragraph or a few bullet points.
```

The user prompt template, where each passage is numbered and labelled with `Chunk.citation()`:

```
Context passages:

[1] MedlinePlus: How to Prevent Diabetes [Summary] (2017-10-25)
<passage text>

---

[2] ...

Question: What causes diabetes?

Answer (with [n] citations):
```

What each rule is for:

| Rule | Purpose |
|---|---|
| 1. Only from the passages | Stops the model from filling gaps from memory, which is where hallucination comes from. |
| 2. Cite every sentence, never an absent number | Makes every claim checkable; the numbers match the source list we return. |
| 3. Reply NOT_FOUND | A second not-found path for when passages passed the gate but do not actually answer the question. |
| 4. No diagnosis or personal dosing | Keeps the system inside its scope: information and education, not medical advice. The second half was added after the model put a "cannot give personal advice" disclaimer on ordinary drug questions (logged bug). |
| 5. Never say "safe" | Absence of evidence in a small corpus is not evidence of safety. |
| 6. Disagreement | Asks the model to surface conflicts instead of picking a side. |
| 7. Concise | Short answers are easier to check against the citations. |

In the "What causes diabetes?" trace, the system prompt is 1,206 characters and the user prompt with five passages is 7,116 characters.

**c) Why this and not the alternatives**

Numbered passages with bracket citations are the simplest scheme a reader can verify. The alternative, asking for free-text references such as "according to the FDA label", cannot be checked automatically. Putting the rules in the system instruction rather than the user message gives them higher priority with the model.

**d) Say-it-out-loud answers**

- *"How do you stop the model from making things up?"* Three layers. The gate stops the call entirely when nothing relevant is found. The prompt says to answer only from the numbered passages, cite every sentence, and reply NOT_FOUND if they do not answer. And the answer is shown next to the exact passages, so a reader can check every claim. We cannot guarantee zero errors, but every claim is traceable.
- *"Is this medical advice?"* No. The prompt forbids diagnosis, personal dosing and telling anyone to start, stop or combine medicines. The UI carries a permanent disclaimer, and the system never calls a combination safe.

### 3.6 The LLM: Gemini 3.5 Flash-Lite

**a) The concept**

**Provider interface:** a small common definition of "something that can generate text", so the rest of the code does not depend on one company's API.

**Rate limit:** the maximum number of requests an API accepts per minute or per day. Going over returns **HTTP 429 "Too Many Requests"**. **HTTP 503** means the service is temporarily overloaded.

**Exponential backoff:** after a failure, wait and retry, doubling the wait each time (2, 4, 8, 16 seconds), so a busy service gets time to recover.

**Temperature:** a setting that controls randomness. Low temperature makes the model pick the most likely words, giving more literal, repeatable output.

**b) Our choice** (`src/generation/providers.py`)

- Model: `gemini-3.5-flash-lite` (`config.GEMINI_MODEL`, overridable in `.env`), Google's small, fast Gemini tier, used through the `google-genai` library (version 2.25.0) on the free tier.
- `LLMProvider` is a Python `Protocol`: anything with a `generate(prompt, system) -> str` method counts as a provider. `GeminiProvider` is the real one; the tests use a fake that returns canned text, so no test ever calls the network. Swapping to another LLM means writing one class.
- Settings: `temperature = 0.1` (faithful restatement, not creativity); automatic function calling disabled (we use no tools).
- **Exactly one LLM call per question.** Retries repeat that same call.
- Retries: up to 4 on HTTP 429 or 503, waiting 2, 4, 8 then 16 seconds. Any other error (bad key, bad model name) fails at once, because waiting will not fix it. After the last retry the user gets a clear `LLMError` message ("Gemini is rate-limiting or overloaded ... Wait a minute and try again") instead of a crash.
- The key is read from `.env` and never printed or logged. The retrieval half runs without any key.
- Measured: the Gemini call for "What causes diabetes?" took 5.5 s on 2026-10-01.

**Why `gemini-2.0-flash` had to be replaced.** The original choice was shut down by Google on 2026-06-01, so the configured default would fail. On 2026-09-29 we listed the models available to our key (61, including `gemini-3.5-flash-lite`), switched the default in `config.py` and `.env.example`, and a test call succeeded.

We do not quote exact free-tier request limits here, because Google sets and changes them per model and per account. The design assumption is simply "limits are tight": one call per question, and no LLM calls for reranking or for not-found questions.

**c) Why this and not the alternatives**

- **Gemini free tier:** no cost for a student project, adequate quality, and a free key in minutes (session 1 decision, made by the user).
- **Behind an interface:** we can switch to another hosted model or a local model later without touching retrieval, prompts or UI.
- **Rejected:** a local open-source LLM (no API limits, but slow on a laptop CPU and generally weaker at following citation rules), and paid APIs (cost).

**d) Say-it-out-loud answers**

- *"Why Gemini?"* It has a free tier, it follows citation instructions well, and it is fast. It sits behind a one-method interface, so replacing it is one class; the tests already use a fake provider.
- *"What if Gemini is down or rate-limited?"* We retry 429 and 503 errors up to four times with waits of 2, 4, 8 and 16 seconds, then show a clear "try again in a minute" message instead of crashing. Retrieval, the gate and the evidence panel do not need Gemini, and not-found questions never call it.
- *"Why is it one call per question?"* The free tier has tight limits and each call takes seconds. We do all the ranking with local models and use the LLM only to write the final answer.

### 3.7 Citation assembly and validation

**a) The concept**

**Citation validation:** checking automatically that the model's answer actually refers to the passages it was given.

**b) Our choice** (`generate_answer` in `src/generation/answer.py`)

- The source list is built from the passages sent to the model, numbered in the same order as in the prompt: `Source(number, title, section, url, citation, source_type)`. The source type is one of MedlinePlus, PubMed, PMC or FDA label.
- The answer is scanned for citation groups with the pattern `\[([\d,\s]+)\]`, which accepts `[1]`, `[1][2]` and `[1, 2]`.
- If the answer cites **no** valid passage number at all, confidence is forced to **Low** with the reason "the answer contains no citations".
- In the UI, each `[n]` becomes a superscript link to reference n. A number outside the range is left as plain text rather than linked.

**The `[1, 2]` bug (session log, 2026-09-29).** The first version only recognised `[1]`. Gemini wrote citations as `[1, 2, 5]`, so a well-cited simvastatin and clarithromycin answer was treated as uncited and downgraded to Low. The regular expression now accepts all three styles, and a test covers it.

**Honest limits**

- The check is **"at least one valid citation"**, not "every sentence is cited".
- It does **not** verify that a cited passage actually supports the claim. There is no **entailment check** (a second model that tests whether a claim logically follows from a passage). The evidence panel exists so a human can do that check.
- The source list shows **every passage the model saw**, not only the ones it cited. In the diabetes trace, source [2] is listed but never cited in the answer.

**c) Why this and not the alternatives**

A simple pattern check catches the most common failure, an answer with no grounding at all, at zero cost. Claim-level verification with a second model would be more rigorous but costs another model call per question. It is a clear next step.

**d) Say-it-out-loud answers**

- *"How do you know the citations are right?"* We check automatically that the answer cites at least one passage it was given, and an answer with no citations is marked Low. We do not yet check automatically that each cited passage supports its sentence. That is why every passage is shown in full in the evidence panel. Automatic claim checking is on our future work list.
- *"What was the [1, 2] bug?"* Our citation check only understood single numbers like [1]. Gemini writes [1, 2, 5], so good answers were being marked Low. We found it in an end-to-end run and fixed the pattern.

### 3.8 The confidence indicator

**a) The concept**

**Confidence indicator:** a simple label that tells the reader how strong the retrieved evidence is. It is not a probability that the answer is correct.

**b) Our choice** (`confidence_signals` and `confidence` in `src/generation/answer.py`)

Two signals, computed from the passages sent to the model:

1. **Strength:** the best rerank score. Strong if it is at least **5.0** (`HIGH_SCORE`).
2. **Agreement:** the number of **distinct documents** that have a passage scoring within **3.0** of the best one (`CONFIDENCE_SPREAD`).

Rules, exactly as coded:

| Level | Rule |
|---|---|
| **High** | strong (best score at least 5.0) **and** at least 2 agreeing documents |
| **Medium** | strong **or** at least 2 agreeing documents, but not both |
| **Low** | neither; **or** the answer has no valid citation (overrides everything) |
| **None** | nothing passed the gate, or the model said NOT_FOUND |

The explanation text also labels the best score "strong" (5.0 or more), "moderate" (above 0.0) or "weak" (0.0 or below; possible on the interaction path, which skips the gate).

Where 5.0 and 3.0 came from: before MedlinePlus, in-corpus top scores fell into two clusters: 3.9 to 4.7 for broad matches and 6.9 to 8.5 for precise section matches. 5.0 splits them. 3.0 is the agreement window we chose; the log records it but gives no measured basis, so present it as a design choice.

**Why a correct malaria answer can be rated Low.** "malaria can occur because of what?" found the right page (MedlinePlus: Malaria) with a best score of 2.0. That is not strong (below 5.0), and only 1 document agreed, because MedlinePlus has exactly one page per disease and the other sources (FDA labels, a small set of PubMed papers) say nothing about malaria. Neither signal holds, so the rule says Low. The answer is correct; the label says "one source, moderate match". "what are the symptoms of malaria?" scored +6.6 on the same page, which is strong, so it is Medium.

How we would improve it:

- Measure agreement at the **passage** level, or treat a single authoritative source (one MedlinePlus page or one FDA label section) as sufficient for a narrow factual question.
- Calibrate the thresholds on a labelled evaluation set (Phase 9) so each level corresponds to a measured accuracy.
- Add a claim-support check (section 3.7) as a third signal.

**c) Why this and not the alternatives**

Two signals anyone can check on screen, instead of a learned or LLM-produced confidence that no one can explain. The cost: it measures evidence strength and cross-source agreement, not correctness. It systematically rates single-source answers lower, which is conservative for medicine but can undersell correct lay answers.

**d) Say-it-out-loud answers**

- *"How is confidence calculated?"* Two numbers you can see on screen: how strong the best passage's rerank score is (5 or more counts as strong), and how many different documents have a passage within 3 points of it. Both hold: High. One holds: Medium. Neither, or the answer cites nothing: Low.
- *"Why is a correct malaria answer rated Low?"* Confidence measures evidence, not correctness. For "malaria can occur because of what?" the best passage scored 2.0, and only one document, the MedlinePlus malaria page, supports it, because MedlinePlus has one page per disease. The rule rates single-source, moderate-score evidence as Low. It is conservative on purpose, and we plan to handle authoritative single sources better.
- *"Is High confidence a guarantee?"* No. It means strong, agreeing evidence was retrieved. It is not a measured probability of correctness. Calibrating it is part of Phase 9.

---

## 4. Drug interaction module, end to end

**a) The concept**

**Drug-drug interaction (DDI):** when one drug changes the effect or level of another, for example by adding to its effect (aspirin plus warfarin, both raising bleeding risk) or by blocking the enzyme that clears it (clarithromycin raising simvastatin levels by inhibiting the liver enzyme CYP3A4).

**Normalisation (of drug names):** mapping what a user types ("Warfarin Sodium", "Low Dose Aspirin") to one standard name.

**b) Our choice** (`src/interactions/check.py`, `InteractionChecker`)

Step by step, for input drug A and drug B:

1. **Normalise both names** against the 20 drugs in `corpus_spec.DRUGS`:
   - exact match on an alias built from the FDA labels' own names (for example "warfarin sodium", "low dose aspirin", "metformin er 500 mg");
   - otherwise, a generic name contained in the input as a whole word ("Tramadol HCl 50 mg" becomes tramadol);
   - otherwise the drug is not covered, and the tool replies "Not in our corpus: <name>" with the list of covered drugs. No retrieval or LLM call happens.
   - Verified: "Warfarin Sodium" gives warfarin, "Low Dose Aspirin" gives aspirin, "coumadin" gives nothing.
   - The same drug twice gives "Please enter two different drugs."
2. **FDA label cross-mention scan.** For each direction (A's label naming B, and B's label naming A), find every chunk of that drug's label whose text contains the other drug's name as a whole word. Drug Interactions section chunks are listed first, then the rest of the label.
3. **Literature and label retrieval.** Run hybrid search (dense plus BM25, RRF, 20 candidates) for `"<A> <B> interaction"`, and keep only passages whose title or text names **both** drugs.
4. **Evidence rule.** A passage counts as evidence only if it names both drugs, or it is one drug's FDA label naming the other. Candidates from steps 2 and 3 are merged without duplicates. Label-scan passages are tagged so the UI can show where each came from.
5. **No evidence:** if nothing qualifies, the answer is: "No interaction evidence was found in our corpus for A and B. This does NOT mean the combination is safe: the corpus covers only 20 FDA labels and a small set of papers. Check a full interaction reference or ask a pharmacist." **The LLM is not called.**
6. **Evidence found:** rerank the candidates against the question "Can A interact with B? Describe any interaction, its mechanism and consequences as stated in the sources." and keep up to 6 (`MAX_EVIDENCE`). Then call the normal Phase 6 generation with `apply_threshold=False`, because the both-drugs rule already decided relevance. If the model replies NOT_FOUND, the no-evidence message is shown instead.

**Why the scan covers the whole label, not just Drug Interactions.** Session log, 2026-09-29: sertraline's label names tramadol only under **Warnings and Cautions** (serotonin syndrome risk), not under Drug Interactions. And the aspirin label is an over-the-counter label with no Drug Interactions section at all. A Drug-Interactions-only scan would have missed a known interacting pair.

**Verified results** (label scan run on 2026-10-01 and tested in `tests/test_interactions.py`):

| Pair | What the labels say | Result |
|---|---|---|
| warfarin + aspirin | Warfarin label names aspirin in Drug Interactions and Dosage and Administration; aspirin label does not mention warfarin | interaction explained, cited |
| simvastatin + clarithromycin | both labels name each other (Drug Interactions, Contraindications, Warnings and Cautions, Clinical Pharmacology) | contraindicated; CYP3A mechanism; myopathy and rhabdomyolysis risk |
| sertraline + tramadol | sertraline label names tramadol in Warnings and Cautions | serotonin syndrome risk |
| warfarin + ciprofloxacin | labels name each other | covered by a real-corpus test |
| amoxicillin + gabapentin | neither label mentions the other | "No interaction evidence was found in our corpus", LLM not called |

**The rule that "no evidence" is never reported as safe** is enforced in two places: the fixed no-evidence message (which says "does NOT mean the combination is safe"), and system prompt rule 5 ("Never state that a drug or drug combination is safe"). It was flagged as a medical-safety edge in the session log.

**Brand-name limitation.** The corpus labels are generic or repackager labels, so the aliases are salt and form names ("Warfarin Sodium", "Tramadol Hcl Er"), not brands. "Coumadin" (a warfarin brand) is reported as not in the corpus. We deliberately did not hand-write a brand table. The proper fix is to normalise names through **RxNorm**, the US National Library of Medicine's standard drug vocabulary, which links brand names, generic names and ingredients.

**c) Why this and not the alternatives**

- **Mention rule instead of the score gate.** For an interaction question, a passage about warfarin alone can score well with the reranker but says nothing about the pair. Requiring both names is a stricter and more explainable test. The cost: evidence can be low-scoring (the warfarin + aspirin trace in section 6.2 passes passages scoring down to -5.49 to the model), and class-level statements ("NSAIDs increase bleeding risk with warfarin") are not matched to a specific drug name like ibuprofen.
- **FDA labels plus literature instead of a curated DDI database,** because DrugBank needs a paid licence.

**d) Say-it-out-loud answers**

- *"How does the interaction check work?"* We map both names to our 20 drugs, scan each drug's FDA label for the other drug's name, and search the whole corpus for passages naming both. Only passages that name both drugs count as evidence. Those go to Gemini with the same cite-everything prompt. If there are none, we say no evidence was found, without calling the LLM.
- *"If you find nothing, does that mean the drugs are safe together?"* No, and the system says so in those words. Our corpus has 20 labels and a small set of papers. Absence of evidence here is not evidence of safety, so we point the user to a full interaction reference or a pharmacist.
- *"Why scan the whole label?"* Because important interactions are not always in the Drug Interactions section. Sertraline's label names tramadol only under Warnings and Cautions, and the aspirin label has no Drug Interactions section at all.
- *"What about brand names?"* Currently only names in our labels are recognised, so "Coumadin" is reported as not covered. The fix is RxNorm normalisation, which maps brands to generics.

---

## 5. Streamlit UI

**a) The concept**

**Streamlit:** a Python library that turns a Python script into a web app. The script re-runs from top to bottom every time the user interacts with it.

**Caching (`st.cache_resource`):** keeping a loaded object, such as a model, in memory across those re-runs so it is loaded only once per server process.

**b) Our choice** (`app.py`, markup in `src/api/ui_render.py`, styles in `assets/app.css`, theme in `.streamlit/config.toml`)

- **Two tabs:** "Ask a question" and "Check a drug interaction".
- **Persistent disclaimer** under the title: "Information and education only. This tool does not diagnose, prescribe or give medical advice."
- **Sidebar:** documents and passages per source, the three model names, and the MedlinePlus credit line.
- **Each answer shows:** the answer with superscript citation links; a confidence meter with both signals (strength and agreement) spelled out; a numbered References list with a source-type badge (MedlinePlus, PubMed, PMC, FDA label), a link and the section.
- **"How this answer was found" panel:** the six pipeline steps, then every passage the model saw with its source badge, a rerank score bar with the 0 threshold marked, its hybrid, dense and BM25 ranks, whether the FDA label scan found it, and the full text. This is there so an evaluator, or any user, can check each claim against its evidence and see why each passage was chosen.
- **Clickable examples:** five example questions and four known drug pairs. Drug pickers list the 20 covered drugs but also accept typed names, so the "not in corpus" path can be shown.
- **`?q=` links:** `http://localhost:8501/?q=What+are+the+symptoms+of+malaria` opens straight onto an answer.
- **Loaded once:** `load_pipeline()` is wrapped in `st.cache_resource`. It builds the retriever (Chroma plus BM25) and warms both models. Measured cold start: loading the embedding model took about 34 s and the reranker about 7 s on 2026-10-01. The UI never rebuilds the index; if the index is empty it shows the build command.
- **Safety in the page itself:** all corpus and LLM text is HTML-escaped before display, so text inside a passage or an answer cannot inject page code. There are tests for this.
- **`.streamlit/config.toml`:** `fileWatcherType = "none"`. Streamlit's file watcher walks every submodule of the transformers library and floods the console with torchvision import errors. The app is not edited during a demo, so the watcher is off. The same file sets the theme colours and fonts.

**c) Why Streamlit**

It turns our Python pipeline into a web app with no separate front-end project. The alternative, a FastAPI backend with a JavaScript front end, would be more flexible but is far more work for a demo. The cost: Streamlit re-runs the script on every interaction, so session state is needed to keep results on screen, and each browser tab shares the cached models.

**d) Say-it-out-loud answers**

- *"What does the evidence panel show?"* Every passage the model saw, where it came from, its rerank score against the 0 threshold, where dense search and BM25 ranked it, and its full text. You can check any sentence of the answer against its source.
- *"Why does the first question take longer?"* The first request loads PubMedBERT and the reranker, about 40 seconds on this laptop. `st.cache_resource` keeps them in memory, so later questions only pay for search, reranking (about 3.4 s) and the Gemini call (about 5 s).

---

## 6. Worked traces (real output, run 2026-10-01)

These were run through the real code on the current index. Every score below was printed by the code; nothing is invented. Two Gemini calls were used in total.

### 6.1 "What causes diabetes?" (lay question, answered from MedlinePlus)

**Dense top 5** (cosine similarity from Chroma):

```
d#1 0.637  MedlinePlus: Diabetes [Summary]
d#2 0.629  MedlinePlus: Diabetes [Summary]
d#3 0.626  MedlinePlus: Diabetes Type 1 [Summary]
d#4 0.622  MedlinePlus: Diabetes Complications [Summary]
d#5 0.620  MedlinePlus: How to Prevent Diabetes [Summary]
```

**BM25 top 5** (BM25 score):

```
b#1 6.58  Comparative Analysis of Clinical Practice Guidelines fo... [Purpose Of Review]
b#2 6.31  MedlinePlus: How to Prevent Diabetes [Summary]
b#3 6.20  MedlinePlus: Diabetes [Summary]
b#4 6.13  MedlinePlus: Diabetes Complications [Summary]
b#5 6.11  MedlinePlus: Diabetes [Summary]
```

Note that BM25's first hit is a guidelines paper that matched the words, which dense search did not rank highly.

**RRF fused** (20 candidates, top 8 shown, with where each came from):

```
h#1 0.03227 (d#1 + b#3)   MedlinePlus: Diabetes [Summary]
h#2 0.03151 (d#2 + b#5)   MedlinePlus: Diabetes [Summary]
h#3 0.03151 (d#5 + b#2)   MedlinePlus: How to Prevent Diabetes [Summary]
h#4 0.03125 (d#4 + b#4)   MedlinePlus: Diabetes Complications [Summary]
h#5 0.03008 (d#6 + b#7)   MedlinePlus: Diabetic Heart Disease [Summary]
h#6 0.02753 (d#18 + b#8)  MedlinePlus: Diabetic Diet [Summary]
h#7 0.02667 (d#15 + b#15) MedlinePlus: Blood Glucose [Summary]
h#8 0.02598 (d#16 + b#18) MedlinePlus: Diabetic Foot [Summary]
```

The guidelines paper that BM25 ranked first does not make the fused top 8, because dense search did not agree.

**Reranked top 5** (cross-encoder logit; took 8.5 s on this cold run):

```
1. +3.54 (was hybrid #3)   MedlinePlus: How to Prevent Diabetes [Summary]
2. +3.18 (was hybrid #4)   MedlinePlus: Diabetes Complications [Summary]
3. +2.51 (was hybrid #5)   MedlinePlus: Diabetic Heart Disease [Summary]
4. +2.34 (was hybrid #1)   MedlinePlus: Diabetes [Summary]
5. +2.26 (was hybrid #13)  MedlinePlus: Diabetes Complications [Summary]
```

**Gate:** 5 of 5 passages score at least 0.0, so the LLM is called.

**Prompt sent** (abbreviated): system prompt 1,206 characters; user prompt 7,116 characters:

```
Context passages:

[1] MedlinePlus: How to Prevent Diabetes [Summary] (2017-10-25)
Diabetes, also called diabetes mellitus, is a disease that causes your blood
glucose, or blood sugar, levels to be too high. There are three types of
diabetes: type 1 diabetes, gestational diabetes, and type 2 diabetes. ...
  ...
Question: What causes diabetes?

Answer (with [n] citations):
```

**Answer** (Gemini, 5.5 s):

> The causes vary depending on the type of diabetes:
> * **Type 1 diabetes:** Researchers believe it is caused by genes and environmental factors that might trigger the disease [4]. In this type, the body does not make insulin [3, 5].
> * **Type 2 diabetes:** Caused by several factors, including genes and lifestyle factors such as being overweight, having obesity, or not being physically active [1, 4]. With type 2 diabetes, the body does not make enough insulin or does not use it well (insulin resistance), which leads to glucose building up in the blood [1, 3, 5].
> * **Gestational diabetes:** Develops during pregnancy and is thought to be caused by hormonal changes during pregnancy combined with genetic and lifestyle factors, occurring when the body cannot make the extra insulin it needs [4].

**Citations:**

```
[1] (MedlinePlus) MedlinePlus: How to Prevent Diabetes [Summary]  https://medlineplus.gov/howtopreventdiabetes.html
[2] (MedlinePlus) MedlinePlus: Diabetes Complications [Summary]    https://medlineplus.gov/diabetescomplications.html
[3] (MedlinePlus) MedlinePlus: Diabetic Heart Disease [Summary]    https://medlineplus.gov/diabeticheartdisease.html
[4] (MedlinePlus) MedlinePlus: Diabetes [Summary]                  https://medlineplus.gov/diabetes.html
[5] (MedlinePlus) MedlinePlus: Diabetes Complications [Summary]    https://medlineplus.gov/diabetescomplications.html
```

**Confidence:** Medium. Best score 3.54 (moderate, below 5.0), and 4 distinct documents score within 3.0 of it. One signal holds, so Medium.

What to point out: source [2] is listed but never cited in the answer, because the list shows every passage the model saw. Before MedlinePlus was added, this question returned "not found" with a best score of -1.8.

### 6.2 warfarin + aspirin (interaction module)

**Normalisation:**

```
normalize('Warfarin')          -> 'warfarin'
normalize('aspirin')           -> 'aspirin'
normalize('Warfarin Sodium')   -> 'warfarin'
normalize('Low Dose Aspirin')  -> 'aspirin'
normalize('coumadin')          -> None      (brand name not in the corpus)
```

**Label scan** (the warfarin label has 33 chunks, the aspirin label 4):

```
Warfarin label names aspirin in: Drug Interactions, Dosage and Administration
Aspirin label does not mention warfarin
  hit: fda:2cbcc99d...::5::2 [Drug Interactions]
  hit: fda:2cbcc99d...::1::1 [Dosage and Administration]
  hit: fda:2cbcc99d...::1::2 [Dosage and Administration]
```

**Hybrid retrieval** for "warfarin aspirin interaction": 20 candidates, 6 name both drugs:

```
h#6  Novel oral anticoagulants for stroke prevention in atrial fi... [Abstract]
h#8  FDA Label: Warfarin [Dosage and Administration]
h#12 Novel oral anticoagulants for stroke prevention in atrial fi... [Abstract]
h#14 FDA Label: Warfarin [Dosage and Administration]
h#16 FDA Label: Warfarin [Drug Interactions]
h#17 Contemporary Antiplatelet and Anticoagulant Therapies for Se... [Recent Findings]
```

**Reranked evidence sent to the model** (no 0.0 gate on this path):

```
rerank +1.64  hybrid#16  label scan  FDA Label: Warfarin [Drug Interactions]
rerank -0.75  hybrid#17              Contemporary Antiplatelet and Anticoagulant Therapies ... [Recent Findings]
rerank -2.91  hybrid#12              Novel oral anticoagulants for stroke prevention ... [Abstract]
rerank -3.25  hybrid#6               Novel oral anticoagulants for stroke prevention ... [Abstract]
rerank -4.41  hybrid#8   label scan  FDA Label: Warfarin [Dosage and Administration]
rerank -5.49  hybrid#14  label scan  FDA Label: Warfarin [Dosage and Administration]
```

**Answer** (9.3 s including reranking and the Gemini call):

> Yes, warfarin can interact with aspirin [1].
> * **Interaction and Consequences:** Concomitant use of aspirin with warfarin increases the risk of bleeding [1].
> * **Mechanism:** Aspirin acts as an antiplatelet agent, and its properties are additive to the anticoagulant effects of warfarin [1].

**Confidence:** Medium. Best score 1.64 (moderate), 2 distinct documents within 3.0.

**Control pair, amoxicillin + gabapentin** (0.1 s, no LLM call):

> No interaction evidence was found in our corpus for amoxicillin and gabapentin. This does NOT mean the combination is safe: the corpus covers only 20 FDA labels and a small set of papers. Check a full interaction reference or ask a pharmacist.

Label scan: neither label mentions the other. Reason shown: "No chunk in the corpus mentions both drugs; the LLM was not called."

What to point out: the evidence comes from the warfarin label's own Drug Interactions section, found by both the label scan and retrieval. Several evidence passages score below 0 but are still sent, because on this path the rule "names both drugs" decides relevance, not the score.

### 6.3 Out-of-corpus: "What is the half-life of adalimumab?" (not-found path)

Adalimumab is not in our 20 drugs, but the question looks medical, which makes it the hardest kind of out-of-corpus question.

**Hybrid top 5:**

```
h#1 (d#16 + b#1)   FDA Label: Warfarin [Clinical Pharmacology]
h#2 (d#10 + b#6)   FDA Label: Clarithromycin [Clinical Pharmacology]
h#3 (d#4 + b#16)   Clinical Pharmacokinetics of the Novel HIV-1 Non-Nucleoside ... [Doravirine Pharmacokinetics]
h#4 (d#20 + b#19)  Pharmacokinetic, Pharmacodynamic, and Drug-Interaction Profi... [Absorption]
h#5 (d#1)          FDA Label: Levothyroxine Sodium [Clinical Pharmacology]
```

Retrieval found passages about the half-lives of *other* drugs. Search always returns something.

**Reranked top 5:**

```
1. -1.07  FDA Label: Furosemide [Clinical Pharmacology]
2. -1.15  FDA Label: Warfarin [Clinical Pharmacology]
3. -1.41  FDA Label: Metoprolol [Clinical Pharmacology]
4. -3.13  FDA Label: Gabapentin [Clinical Pharmacology]
5. -3.47  FDA Label: Gabapentin [Clinical Pharmacology]
```

**Gate:** the best is -1.07, below 0.0, so the answer is "not found". The run used a stand-in provider that fails if called, which proves the LLM was not called.

**Answer:** "I could not find information about this in the indexed medical literature and drug labels, so I will not attempt an answer. Try rephrasing, or ask about a drug or condition covered by the corpus."

**Confidence:** None: "No passage scored above the relevance threshold (0); the LLM was not called."

What to point out: without the gate, the LLM would have been handed five half-life passages about other drugs. This is the closest out-of-corpus question we have seen (-1.07), which is the thin margin discussed in section 3.4.

Other not-found checks run the same day: "What is the treatment for kuru?" best -5.59; "What is the capital of France?" best -10.82.

**Where the time goes** (medians on this laptop's CPU, 2026-10-01):

| Stage | Time |
|---|---|
| Load embedding model (once per server start) | about 34 s |
| Load reranker (once) | about 7 s |
| Build BM25 over 3,008 chunks (once) | 0.74 s |
| Embed question plus Chroma search | about 0.13 s |
| BM25 search | about 0.008 s |
| RRF | about 0.0002 s |
| Rerank 20 candidates | about 3.4 s warm, 8.5 s on a cold first run |
| Gemini call | 5.5 s in the diabetes trace |

So a warm question takes roughly 9 seconds, and about 90% of that is reranking plus the Gemini call.

---

## 7. Design decisions and trade-offs

| Decision | Alternative | Why we chose ours | Cost of our choice |
|---|---|---|---|
| Public sources only (PubMed/PMC, openFDA, MedlinePlus) | DrugBank; Mayo Clinic or Cleveland Clinic pages | Licensing: DrugBank needs a paid licence; Mayo and Cleveland prohibit scraping and reuse | No curated interaction database; small, US-centric corpus |
| Harvesting separate from parsing | One combined script | Parser fixes never need a re-download | Two steps to run |
| Section-aware chunking, about 350 tokens, 60 overlap | Fixed-size chunks | Never mixes Dosage with Contraindications; never cuts a sentence; tables kept whole | Uneven chunk sizes; 14 chunks over the 512-token embedding limit |
| Word count times 1.3 as the token estimate | Real tokenizer during chunking | No model needed at ingestion; measured median ratio is 1.30 | Individual chunks can be off |
| Section heading embedded with the text | Heading only as metadata | Same words can answer different questions in different sections | Heading words take up some of the 512-token budget |
| PubMedBERT embeddings | General model; embedding API | Biomedical vocabulary, separates drug names at the word level, local and free | 418 MB model, slow on CPU; still slipped on metformin vs metoprolol in retrieval |
| Chroma | FAISS | Metadata stored with each vector; no server | Heavier than FAISS at very large scale |
| BM25 rebuilt in memory on each start | Saved index file | Cannot go out of date; 0.74 s | Small start-up cost |
| Hybrid retrieval with RRF (k = 60) | Dense only; weighted score blend | Catches exact names and paraphrases; no tuning; rewards agreement | Not tuned on labelled data |
| Cross-encoder reranking (MiniLM-L-6) | No reranker; LLM reranking; larger or biomedical reranker | Changed the top result in 7 of 8 probes; no extra API call | About 3.4 s per question; general-domain model |
| Not-found gate at 0.0 | No gate; LLM decides relevance | Stops the LLM being fed irrelevant passages; cheap and explainable | One fixed number; margin now only about 2.4 |
| Gemini 3.5 Flash-Lite behind an interface | Local LLM; paid API | Free, good at following citation rules, swappable | Rate limits; needs internet; text sent to Google |
| One LLM call per question, backoff on 429 and 503 | Multi-step LLM pipelines | Fits free-tier limits; predictable latency | No self-checking pass |
| Two-signal confidence (5.0 and 3.0) | LLM self-rated or learned confidence | Explainable on screen | Rates single-source answers low; not calibrated |
| Interaction evidence must name both drugs | Use the score gate | Stricter, explainable relevance test | Misses class-level statements; passes low-scoring passages |
| Scan the whole FDA label | Drug Interactions section only | Sertraline names tramadol only under Warnings; aspirin label has no DI section | More, noisier matches (Dosage sections) |
| Incremental `--append` indexing | Always full rebuild | Added MedlinePlus in 12.5 min instead of re-embedding everything | Only valid if old chunks are unchanged; warns otherwise |
| Streamlit | FastAPI plus a JavaScript front end | One Python file; fast to build | Less control; script re-runs on every click |

---

## 8. Honest limitations and future work

1. **Brand names.** Only names that appear in our FDA labels are recognised, so "Coumadin" is reported as not covered. Fix: normalise names through **RxNorm**, NLM's drug vocabulary that links brands, generics and ingredients.
2. **Thin threshold margin.** Answerable questions start at +1.3 and unanswerable ones reach -1.1, a margin of about 2.4. Borderline wording could fall on the wrong side. Fix: test many more borderline questions in Phase 9, and consider a second check.
3. **Single-source confidence.** Correct answers from one MedlinePlus page are rated Low or Medium by design (malaria causes: Low). Fix: passage-level agreement, treating one authoritative source as enough for narrow facts, and calibration on labelled data.
4. **US-centric content.** MedlinePlus and FDA labels reflect US practice, drug names and approvals. Plan: add WHO fact sheets for international disease information, after checking their licence.
5. **English only.** No multilingual support yet (Phase 8). PubMedBERT and the reranker are English models.
6. **No formal evaluation yet.** Retrieval quality has been checked by hand on 8 probe questions and 6 lay questions, not measured as recall at k or precision on a labelled set. That is Phase 9.
7. **No claim-level verification.** We check that an answer cites at least one passage, not that each cited passage supports each sentence. Fix: an automatic entailment check, plus physician-reviewed answer validation of a sample set.
8. **Small corpus.** 20 drugs, 149 papers and 1,014 health topics. Many medical questions will correctly return "not found".
9. **Truncation.** 14 chunks (0.5%) are longer than PubMedBERT's 512-token limit and are cut off when embedded.
10. **Live sources drift.** PubMed search results and FDA labels change over time, so rebuilding from scratch gives a slightly different corpus (169 documents vs the originally logged 171). MedlinePlus changes daily; re-harvesting it requires a full rebuild.
11. **Minor retrieval regression after MedlinePlus.** For "first line treatment for type 2 diabetes", a relevant review (+5.30) fell out of the 20 candidates. Widening the candidate pool from 20 to 30 would likely recover it but costs about 50% more reranking time. Not yet decided.
12. **Interaction evidence misses drug classes.** A label saying "NSAIDs" does not match "ibuprofen".
13. **Urgent symptoms.** "I have chest pain, what should I take?" returns the generic not-found message rather than an explicit "seek care now". Safe, but the wording should be better (Phase 9 safety checks).
14. **Text leaves the machine.** Questions and retrieved passages are sent to Google's API. Fine for public-domain text and anonymous questions, not for patient data.

---

## 9. Evaluator question bank

The 10 most likely questions are marked **[TOP 10]**.

### Concept and value

1. **[TOP 10] How is this different from ChatGPT?**
   ChatGPT answers from memory, so it can invent facts, has a training cutoff, and cannot show where an answer came from. Our system first retrieves passages from public medical sources, lets the model use only those, and cites every claim. If nothing relevant is found, it says so without calling the model at all.

2. **[TOP 10] How do you know it doesn't hallucinate?**
   We cannot guarantee zero errors, but we reduce and expose them. The gate stops the LLM call when no passage is relevant. The prompt forbids outside knowledge and requires a citation on every sentence. An uncited answer is marked Low. And every passage the model saw is shown in full, so each claim can be checked. We do not yet verify claims automatically; that is future work.

3. **[TOP 10] Is this medical advice?**
   No. It is for information and education only. The prompt forbids diagnosis, personal dosing and telling anyone to start, stop or combine medicines. A disclaimer stays on screen, and the system never calls a drug combination safe.

4. **Why not fine-tune a model instead?**
   Fine-tuning bakes knowledge into the model's weights, where it cannot be cited, cannot be updated without retraining, and can still be hallucinated. Retrieval keeps the knowledge in documents we can show, update by rebuilding the index, and license-check. It also needs no GPU training.

5. **What is RAG, in one sentence?**
   Before the language model answers, a search system finds relevant passages from trusted documents, and the model is told to answer only from those passages and cite them.

6. **Who is this for?**
   Students and anyone who wants cited, plain-language or technical information about diseases, symptoms, treatments and drug interactions. It is not for making clinical decisions.

### Data

7. **Where does the data come from, and are you allowed to use it?**
   PubMed and PubMed Central through NCBI's E-utilities API, FDA drug labels through openFDA, and MedlinePlus health topics from the National Library of Medicine. FDA labels are public domain, MedlinePlus topic summaries are listed by NLM as public domain, and PMC is limited to open access. We rejected DrugBank, Mayo Clinic and Cleveland Clinic because of licensing.

8. **How does the data stay current?**
   Each source can be re-downloaded with one command, and MedlinePlus publishes a new file five days a week. After re-harvesting we rebuild the index, or append if only new documents were added. There is no automatic schedule yet.

9. **Why did you add MedlinePlus?**
   PubMed and FDA labels are written for clinicians, so plain questions like "what are the symptoms of malaria?" had nothing to match. All six lay test questions returned "not found" before, and all six are answered with citations after.

10. **Why only 20 drugs?**
    It is a demonstration corpus defined in one file, chosen to include known interacting pairs and control pairs. Adding a drug is one line plus a re-harvest.

### Retrieval

11. **[TOP 10] Why use both keyword search and embeddings?**
    Embeddings capture meaning but can blur look-alike names: with PubMedBERT alone, a metoprolol passage ranked third for a metformin question. BM25 matches exact words and codes like "CYP3A4". Combining them catches what either misses.

12. **What is Reciprocal Rank Fusion?**
    Each passage gets 1 divided by (60 plus its rank) from each list it appears in, and those are added. It merges rankings by position only, so we never compare incompatible scores, and passages both methods like rise to the top.

13. **Why k = 60?**
    It is the standard value from the paper that introduced RRF. It stops rank 1 in one list from dominating. We did not tune it because we have no labelled set to tune against yet.

14. **Why PubMedBERT?**
    It was pretrained on PubMed abstracts and full-text papers with a biomedical vocabulary, then fine-tuned for sentence similarity on PubMed title-abstract pairs. In our test it kept "metformin" and "metoprolol" far apart (similarity 0.42), while a general model put them closer together than metformin and its own brand name. It also runs locally without an API key.

15. **Why Chroma instead of FAISS?**
    Chroma stores each vector's metadata (title, section, date, link) next to it, and our citations depend on that. It is also file-based, with no server to run. FAISS stores only vectors and IDs, so we would need a separate metadata store.

16. **What does the reranker do, and why is it needed?**
    It reads the question and each of the 20 candidates together and scores relevance precisely. Retrieval is fast but compares separately made vectors. On our 8 probe questions, reranking changed the top result in 7, for example moving the Contraindications section to first for a contraindications question.

17. **What are the rerank scores?**
    Raw logits from the cross-encoder, not probabilities. Positive means relevant, negative means not, and 0 is the 50% point. Our observed range is about -11 to +9.

### Generation, safety and confidence

18. **[TOP 10] Why is the threshold exactly 0?**
    We compared the best rerank score for questions the corpus can answer and questions it cannot. Answerable questions scored +1.3 and above; unanswerable ones -1.1 and below. Zero is in that gap, and it is where the reranker's output means 50% relevant.

19. **[TOP 10] Why is a correct malaria answer rated Low?**
    Confidence measures evidence, not correctness. For "malaria can occur because of what?" the best passage scored 2.0 and only one document, the MedlinePlus malaria page, supports it, because MedlinePlus has one page per disease. Our rule rates single-source, moderate-score evidence as Low. It is deliberately conservative, and we plan to treat a single authoritative source as sufficient for narrow facts.

20. **How is confidence calculated?**
    Two signals on screen: is the best rerank score at least 5, and how many distinct documents score within 3 of it. Both: High. One: Medium. Neither, or no citation in the answer: Low.

21. **What happens when sources conflict?**
    The prompt tells the model to say so and cite both sides. We do not yet detect conflicts automatically; conflicting-source detection is a Phase 8 feature.

22. **What happens if the question asks for a diagnosis or a dose for a specific person?**
    The prompt tells the model to say it cannot give personal advice, summarise what the sources say in general, and suggest a healthcare professional. If nothing relevant is retrieved, the not-found message is shown instead. Better handling of urgent symptoms is on the Phase 9 safety list.

23. **Why does the drug check say "no evidence" instead of "safe"?**
    Because our corpus has 20 labels and a small set of papers. Not finding evidence here does not mean an interaction does not exist. Saying "safe" would be a dangerous overclaim, so the wording is fixed in code and forbidden in the prompt.

24. **What was the citation bug?**
    Our citation check only understood single numbers like [1], while Gemini writes [1, 2, 5], so good answers were marked Low. We caught it in an end-to-end run, fixed the pattern to accept all three styles, and added a test.

### LLM and operations

25. **[TOP 10] Why Gemini?**
    Free tier, fast, and good at following citation instructions. It is behind a one-method interface, so it can be replaced by writing one class. We moved from Gemini 2.0 Flash to 3.5 Flash-Lite because Google shut down 2.0 Flash on 2026-06-01.

26. **[TOP 10] What happens when Gemini is down or rate-limited?**
    We retry rate-limit (429) and overload (503) errors up to four times, waiting 2, 4, 8 and 16 seconds, then show a clear "try again in a minute" message. Retrieval, the gate and the evidence panel still work without Gemini, and not-found questions never call it.

27. **[TOP 10] What is the latency, and where does the time go?**
    On this laptop, a warm question takes about 9 seconds: about 3.4 seconds reranking 20 candidates on the CPU and about 5.5 seconds for the Gemini call. Embedding and both searches together take under 0.2 seconds. The first question after start-up also loads the models, about 40 seconds.

28. **How many LLM calls per question?**
    Exactly one, or zero when nothing relevant is found. Retries repeat that same call.

29. **How long does it take to build the index?**
    Between 2.4 and 4.6 chunks per second on this CPU, so about 11 to 21 minutes for all 3,008 chunks. When we added MedlinePlus we used an append mode that embedded only the 1,820 new chunks, in about 12.5 minutes.

### Evaluation and testing

30. **[TOP 10] How would you measure accuracy?**
    Phase 9: build a labelled set of questions with the passages that should be retrieved, measure recall at 5 and 20 for dense, BM25, hybrid and reranked retrieval, and measure answer groundedness by checking each cited sentence against its passage, first automatically and then on a sample reviewed by a physician. Right now we have hand-checked probe sets, not these metrics.

31. **How is the code tested?**
    110 automated tests, all offline. They use small synthetic documents, a fake LLM and a stubbed reranker, so no test needs the network or an API key. They cover parsing, chunking boundaries, fusion, reranking, the not-found path, citations, confidence rules, interactions and HTML escaping in the UI.

32. **Did anything go wrong while building it?**
    Yes, and running the pipeline found it. Examples: PMC papers were silently skipped because NCBI renamed an ID field; openFDA returned combination products instead of single drugs; FDA source links were all dead; citations like [1, 2] were misread; and the UI crashed when both tabs had results. Each fix is logged in SESSION_LOG.md.

### Design and limits

33. **Why chunk by section?**
    In a drug label, a passage that merged Dosage with Contraindications could read as a dose recommendation where the drug is contraindicated. We never cross a section boundary, never cut a sentence, and keep tables whole.

34. **What are the biggest limitations?**
    A small, US-centric, English-only corpus; brand names not recognised; a thin margin around the not-found threshold; confidence that under-rates single-source answers; and no formal accuracy measurement yet. All are in our plan.

35. **What would you do with more time?**
    The Phase 9 evaluation set and metrics; RxNorm for drug names; claim-level citation checking; WHO fact sheets for international coverage; multilingual questions; and a wider reranker candidate pool.

36. **Does any patient data leave the machine?**
    The system stores no patient data. The question and the retrieved public passages are sent to Google's Gemini API to write the answer. For real patient data, the provider interface would let us switch to a locally hosted model.

37. **Can you show the evidence for an answer?**
    Yes. Open "How this answer was found": it shows each passage, its source, its rerank score against the threshold, its ranks in dense and keyword search, and its full text.

38. **What did each of you build?**
    Fill this in yourselves. For reference, the git history shows Manan committing Phases 0 to 5 (project setup, data sources, ingestion, chunking, embeddings, hybrid retrieval and reranking) on 2026-09-28, and Ishaan committing Phases 6 and 7 (generation, interactions), the MedlinePlus source and the Streamlit UI afterwards. Commit authorship is not necessarily who designed each part, so adjust to reflect reality.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| API | A web address a program calls to get structured data. |
| Approximate nearest neighbour (ANN) | Fast search that finds almost all of the closest vectors by skipping most comparisons. |
| Attention | The mechanism in a transformer that lets each word look at every other word to understand context. |
| b (BM25) | Setting for how much document length reduces a match's credit; ours is 0.75. |
| BERT | A transformer language model from Google (2018) that reads a whole sentence at once. |
| Bi-encoder | A model that turns the question and each document into vectors separately, so documents can be encoded ahead of time. |
| BM25 | A keyword ranking formula that improves TF-IDF with saturation (k1) and length normalisation (b). |
| Chroma | The local, file-based vector database we use. |
| Chunk (passage) | A short piece of a document, the unit we search and cite. |
| Chunking | Splitting documents into chunks. |
| Citation | A numbered reference [n] from an answer to a retrieved passage. |
| Confidence indicator | Our High, Medium or Low label for how strong the retrieved evidence is. |
| Corpus | The collection of documents the system can search. |
| Cosine similarity | A score from -1 to 1 for how similar two vectors are in direction. |
| Cross-encoder | A model that reads the question and a passage together and outputs one relevance score. |
| Dense retrieval | Search by meaning, using embedding vectors. |
| Dimensions | How many numbers are in a vector; ours have 768. |
| Distillation | Training a small model to imitate a large one. |
| Drug-drug interaction (DDI) | When one drug changes the effect or level of another. |
| E-utilities | NCBI's web API for searching and downloading PubMed and PMC records. |
| ef_construction, ef_search | HNSW settings for how many candidates are considered when building and when searching; both 100 in our collection. |
| Embedding (vector) | A list of numbers representing the meaning of a text. |
| Entailment check | Using a model to test whether a claim logically follows from a passage. |
| Exponential backoff | Retrying after a failure with waits that double each time. |
| FAISS | Facebook's vector search library; stores vectors without metadata. |
| Fine-tuning | A second, smaller training stage that adapts a pretrained model to a task. |
| Grounding | Making an answer depend only on supplied evidence. |
| Hallucination | A fluent, confident statement from an LLM that is false. |
| HNSW | A layered graph used for fast approximate nearest neighbour search. |
| HTTP 429 / 503 | "Too many requests" (rate limited) / "service temporarily unavailable". |
| Hybrid retrieval | Combining dense and keyword search results. |
| IDF | Inverse document frequency: how rare a word is across the collection. |
| Incremental update (`--append`) | Adding only new chunks to an existing index. |
| Ingestion | Turning raw files into clean, consistent records. |
| JATS | The standard XML format for journal articles, used by PMC. |
| JSON | A text format of key-value pairs. |
| k (RRF) | The damping constant in RRF; ours is 60. |
| k1 (BM25) | Setting for how quickly repeated words stop adding credit; ours is 1.5. |
| Knowledge cutoff | The date after which an LLM has no training data. |
| LLM | Large language model: a neural network trained to predict the next word. |
| Logit | A model's raw output score before conversion to a probability. |
| Mean pooling | Averaging the vectors of all tokens into one vector. |
| Metadata | Data about a text: title, source, section, date, link. |
| MedlinePlus | The US National Library of Medicine's plain-language health information site. |
| MiniLM | A family of small, distilled BERT-style models. |
| MS MARCO | Microsoft's dataset of real search queries with relevant passages. |
| Normalisation (text) | Cleaning text for consistency without changing its meaning. |
| Normalisation (drug names) | Mapping typed drug names to one standard name. |
| Not-found gate | Our rule that drops passages scoring below 0.0 and skips the LLM if none remain. |
| openFDA | The FDA's public API, including drug labels. |
| Overlap | Repeating the end of one chunk at the start of the next. |
| Parameters | The learned weights inside a neural network. |
| Parsing | Reading a structured file and extracting what we need. |
| PMC | PubMed Central, NCBI's archive of full-text papers. |
| Precision | The share of returned results that are relevant. |
| Pretraining | The first, large-scale training stage of a language model. |
| Prompt | The text sent to an LLM. |
| Provider interface | Our one-method definition of "something that generates text", so the LLM can be swapped. |
| Public domain | Content with no copyright, free to reuse. |
| PubMed | NCBI's index of biomedical research papers. |
| PubMedBERT | A BERT model pretrained on PubMed text with a biomedical vocabulary. |
| RAG | Retrieval-augmented generation: retrieve evidence, then generate an answer from it. |
| Rate limit | The maximum number of API requests allowed in a time window. |
| Recall | The share of relevant items that a search finds. |
| Reciprocal Rank Fusion (RRF) | Merging ranked lists by summing 1 / (k + rank). |
| Reranking | Re-scoring a short list of candidates with a more accurate model. |
| RxNorm | NLM's standard drug vocabulary linking brands, generics and ingredients. |
| Schema | The fixed list of fields every record must have. |
| Section-aware chunking | Splitting only inside a section, at sentence boundaries. |
| Sparse retrieval | Keyword search, such as BM25. |
| `st.cache_resource` | Streamlit feature that keeps loaded objects, such as models, in memory across reruns. |
| Stopwords | Very common words removed before keyword indexing. |
| Streamlit | A Python library that turns a script into a web app. |
| System prompt | A higher-priority instruction that sets an LLM's role and rules. |
| Temperature | A setting for how random an LLM's word choices are; ours is 0.1. |
| TF | Term frequency: how often a word appears in a document. |
| TF-IDF | A keyword score of term frequency times inverse document frequency. |
| Token | A unit of text a model reads, roughly a word or a piece of a word. |
| Top-k | Keeping only the k best results. |
| Transformer | A neural network architecture built from layers of attention. |
| Vector database | A store that returns the vectors closest to a query vector, with their data. |
| XML | A text format that wraps data in named tags. |

---

## 11. Deck consistency check

Source: `Medical_Literature_RAG_Project.pptx`. The brief said it was at `docs/`, but it was found at `C:\Users\ishaa\Downloads\` and read from there. It was not copied into the repository and not edited. Each of its 20 slides is four image tiles; they were stitched together and every slide was read, along with the speaker notes.

**Fix these (claims that do not match the code or logs):**

| Slide | What the slide says | Problem | Correct value or wording |
|---|---|---|---|
| 8 (Key Design Decisions) | "PubMedBERT keeps 'metformin' and 'metoprolol' apart; a general model treats them as neighbours." | Half true. Our own logs and today's test show dense search with PubMedBERT ranked a metoprolol chunk 3rd for a metformin question, and a general model did better on that retrieval test. It also contradicts slide 14, which says dense search ranked metoprolol third. | "PubMedBERT separates the drug names (similarity 0.42 vs 0.77 for a general model), but dense search alone still slips, so we add BM25 and a reranker." |
| 16 (Testing) | "99 tests passing"; test_retrieval.py 28; test_ingestion.py 17 | Out of date, and two per-file counts were wrong even for 99 | **110** tests: test_retrieval.py **26**, test_ingestion.py **19**, test_chunking.py 16, test_generation.py 11, **test_ui_render.py 11** (new), test_interactions.py 10, test_medlineplus.py 10, test_rerank.py 7 |
| 6 (System Architecture) | Extract box: "XML · JSON · PDF · HTML → documents" | The corpus contains no PDF or HTML documents; those extractors exist but are unused | "XML · JSON → documents" (optionally add "PDF and HTML supported") |
| 6 (System Architecture) | Online lane: "Question or two drug names" flows through "Relevance gate score ≥ 0" | The interaction path skips the score gate; it uses the "names both drugs" rule instead | Label the gate "question path", or add a note: "interaction path: must name both drugs" |
| 15 (Effect of Adding MedlinePlus) | Row: "What is high blood pressure?" (-1.4 / +7.3 High) | Those scores were logged for the full question "what is high blood pressure and why is it dangerous?" | Use the full question text, or re-run the short one before quoting numbers |
| 11 (Drug Interaction Module) | Order: gather evidence (label scan + hybrid search + rerank), then "names both drugs?" | In code the "names both drugs" filter comes before reranking; reranking only runs when evidence exists. The "drug not in corpus" exit is also missing | Order: normalise, then (not in corpus? stop), then label scan + hybrid search, then "names both drugs?", then rerank, then Gemini |

**Consistency notes (not wrong, but pick one phrasing and use it everywhere):**

| Slide | Note |
|---|---|
| 2 (Project Overview) | "Trusted sources: 4" counts PubMed and PMC separately. The README and this brief say three sources (NCBI, openFDA, MedlinePlus). Both are defensible: the code tags four source types. Say "three providers, four source types" if asked. |
| 8 (Key Design Decisions) | "the best 5 reach the LLM": correct as a maximum. Passages under 0.0 are dropped first, so it can be fewer than 5. |
| 10 (Grounding and Confidence) | Ranges -10.9 to -1.1 and +1.3 to +9.0 match the logs. The +9.0 came from "What are the clinical features of serotonin syndrome?" (session 1). Rules for High, Medium and Low match the code. |

**Checked and correct:** slide 1 (names, title); slide 3 (RAG steps); slide 4 (problem statement); slide 5 (objectives); slide 7 (all counts: 1,014 / 1,820, 20 / 569, 140 / 318, 9 / 301, totals 1,183 / 3,008, rejected sources); slide 9 (all tools and `gemini-3.5-flash-lite`); slide 12 (screenshot of the current app); slide 13 (malaria: +6.6, Medium, 1 source); slide 14 (all four probe results match the eval run); slide 17 (every demo question verified: kuru gives not found at -5.59); slides 18 to 20.

---

## 12. Demo runbook

**Before the evaluation (at least 15 minutes early):**

```bash
cd "Medical_literature_RAG-"
source .venv/Scripts/activate          # Git Bash; use .venv\Scripts\activate in cmd
python -m pytest -q                    # expect: 110 passed
streamlit run app.py
```

Then open http://localhost:8501 and **ask one warm-up question** (for example the malaria question). The first question loads both models (about 40 s); after that answers take about 9 s. Keep the tab open.

Never rebuild the index before the demo. If `data/` is missing, the app shows the build command, but a rebuild takes 11 to 21 minutes.

**Question order and what to point at:**

| # | Tab | Input | What to point at |
|---|---|---|---|
| 1 | Ask | "What are the common side effects of metformin?" (example pill) | Citations link to the FDA label's **Adverse Reactions** section. Open "How this answer was found": the six steps and the score bars. The project brief's own example. |
| 2 | Ask | "What are the symptoms of malaria?" | A **MedlinePlus** badge; answered in plain language; score +6.6, Medium. Say: "Before we added MedlinePlus this returned not found." |
| 3 | Ask | "What is high blood pressure and why is it dangerous?" | **High** confidence: strong best score (+7.3) and several agreeing MedlinePlus topics (sources come from 4 topics). Point at the two signals in the meter. |
| 4 | Ask | "What is the treatment for kuru?" | **Not found**, and the reason line: "the LLM was not called". The gate in action. |
| 5 | Interaction | warfarin + aspirin (pill) | FDA label scan cards: the warfarin label names aspirin in Drug Interactions. Evidence panel: passages tagged "FDA label scan". |
| 6 | Interaction | simvastatin + clarithromycin (pill) | Both labels name each other; contraindicated; CYP3A mechanism. |
| 7 | Interaction | amoxicillin + gabapentin (pill) | "No interaction evidence was found in our corpus ... does NOT mean the combination is safe." |
| 8 (optional) | Interaction | type "warfarin" and "coumadin" | "Not in our corpus: coumadin": the honest brand-name limitation. |

**Fallback plan if Gemini fails live:**

1. The app shows a clear error ("Gemini is rate-limiting or overloaded ... Wait a minute and try again") instead of crashing. Say: "This is the rate-limit handling we built: it retries four times with backoff, then fails gracefully."
2. Show a **not-found question** (kuru) and the **amoxicillin + gabapentin** check: both work with no LLM call.
3. Show retrieval without the LLM from the terminal:
   ```bash
   python -m src.retrieval.search "What are the common side effects of metformin?" --rerank
   python -m src.retrieval.search "metformin contraindications" --compare
   ```
   The first prints the candidate list before and after reranking. The second prints dense vs BM25 vs hybrid side by side.
4. Fall back to slides 13 to 15, which show real results.

**If the app will not start:** check that the virtual environment is active; that port 8501 is free (otherwise run `streamlit run app.py --server.port 8502`); and that `data/vectorstore/` exists.

---

## 13. Presenter split (suggestion)

Balanced so each presenter owns one half of the pipeline end to end, and can take follow-up questions on it. Adjust freely.

| Part | Slides | Presenter A (offline and retrieval) | Presenter B (generation, safety and demo) |
|---|---|---|---|
| Opening | 1 to 5 | Title, overview, RAG explanation (1 to 3) | Problem statement and objectives (4 and 5) |
| Architecture | 6 | Offline lane | Online lane, the gate and the not-found branch |
| Data and design | 7 and 8 | Data sources, licensing, design decisions | |
| Implementation | 9 to 11 | Technology stack (9) | Grounding and confidence (10), interaction module (11) |
| Results | 12 to 16 | Retrieval results (14), testing (16) | Application (12), answer walkthrough (13), effect of MedlinePlus (15) |
| Live demo | 17 | Drives the keyboard | Narrates each step using the runbook |
| Close | 18 to 20 | Progress so far (18) | Next steps (19), questions (20) |

Suggested question ownership: Presenter A takes data, licensing, chunking, embeddings, Chroma, BM25, RRF, reranking and indexing (sections 2, 3.1 to 3.3, and questions 7 to 17 and 29). Presenter B takes the gate, prompt, Gemini, citations, confidence, interactions, the UI, safety and evaluation (sections 3.4 to 5, and questions 1 to 6, 18 to 28, and 30 to 37).

The git history suggests the split: Manan committed the offline and retrieval phases (0 to 5) and Ishaan the generation, interaction, MedlinePlus and UI work. Presenter A as Manan and Presenter B as Ishaan would match it, but confirm this against who actually built each part.
