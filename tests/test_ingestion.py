"""Phase 1 tests: extraction and normalisation.

All fixtures are synthetic and built in-process, so the suite runs offline and
stays stable even when the live corpus is re-harvested.
"""

from __future__ import annotations

import json

import pytest

from src.ingestion import fda_parser, html_extract, pdf_extract, pubmed_parser
from src.ingestion.normalize import estimate_tokens, normalize_text, prettify_heading


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------

def test_repairs_hyphenated_line_break():
    assert normalize_text("hyper-\ntension is common") == "hypertension is common"


def test_preserves_dosage_notation():
    """The single most important normalisation guarantee: dosages survive intact."""
    dosage = "Administer 2.5 mg/kg q12h; max 500 mg/day (do not exceed 2 g)."
    assert normalize_text(dosage) == dosage


def test_strips_boilerplate_lines():
    text = "Downloaded from example.org\nReal clinical content here.\n(c) 2024 Publisher"
    out = normalize_text(text)
    assert "Real clinical content here." in out
    assert "Downloaded from" not in out


def test_collapses_whitespace_but_keeps_paragraphs():
    out = normalize_text("Para one.\n\n\n\nPara   two.")
    assert out == "Para one.\n\nPara two."


def test_prettify_heading_lowercases_small_words():
    assert prettify_heading("drug_interactions") == "Drug Interactions"
    assert prettify_heading("use_in_specific_populations") == "Use in Specific Populations"


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three four five") > 5


# --------------------------------------------------------------------------
# PubMed / PMC
# --------------------------------------------------------------------------

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID>12345678</PMID>
   <Article>
    <Journal><Title>Journal of Test Medicine</Title>
      <JournalIssue><PubDate><Year>2023</Year><Month>Jun</Month></PubDate></JournalIssue>
    </Journal>
    <ArticleTitle>Metformin in type 2 diabetes: a review.</ArticleTitle>
    <Abstract>
      <AbstractText Label="BACKGROUND">Metformin is first-line therapy.</AbstractText>
      <AbstractText Label="CONCLUSIONS">It remains the preferred initial agent.</AbstractText>
    </Abstract>
    <AuthorList>
      <Author><LastName>Sharma</LastName><Initials>R</Initials></Author>
    </AuthorList>
   </Article>
  </MedlineCitation>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation>
   <PMID>99999999</PMID>
   <Article><ArticleTitle>An editorial with no abstract.</ArticleTitle></Article>
  </MedlineCitation>
 </PubmedArticle>
</PubmedArticleSet>
"""


def test_pubmed_preserves_structured_abstract_labels():
    docs = pubmed_parser.parse_pubmed_xml(PUBMED_XML)
    assert len(docs) == 1, "abstract-less records carry no retrievable content"

    doc = docs[0]
    assert doc.doc_id == "pubmed:12345678"
    assert doc.source == "pubmed"
    assert doc.journal == "Journal of Test Medicine"
    assert doc.date == "2023-Jun"
    assert doc.authors == ["Sharma R"]
    assert doc.url.endswith("/12345678/")
    assert [s.heading for s in doc.sections] == ["Background", "Conclusions"]
    assert "first-line therapy" in doc.sections[0].text


PMC_XML = """<?xml version="1.0"?>
<pmc-articleset>
 <article>
  <front>
   <journal-meta><journal-title>Open Med Reviews</journal-title></journal-meta>
   <article-meta>
    <article-id pub-id-type="pmc">PMC7654321</article-id>
    <title-group><article-title>Mechanisms of drug interaction</article-title></title-group>
    <pub-date><year>2022</year></pub-date>
    <contrib-group>
      <contrib><name><surname>Iyer</surname><given-names>A</given-names></name></contrib>
    </contrib-group>
    <abstract><p>We review CYP450-mediated interactions.</p></abstract>
   </article-meta>
  </front>
  <body>
   <sec>
     <title>Introduction</title>
     <p>Interactions arise via enzyme inhibition.</p>
   </sec>
   <sec>
     <title>Outer</title>
     <sec><title>Inner</title><p>Nested leaf content.</p></sec>
   </sec>
  </body>
 </article>
</pmc-articleset>
"""


def test_pmc_accepts_current_pmcid_label():
    """Live NCBI output labels the id 'pmcid'; older records use 'pmc'. Both must work."""
    current = PMC_XML.replace('pub-id-type="pmc"', 'pub-id-type="pmcid"')
    docs = pubmed_parser.parse_pmc_xml(current)
    assert len(docs) == 1 and docs[0].doc_id == "pmc:PMC7654321"


def test_pmc_bare_numeric_id_is_prefixed():
    bare = PMC_XML.replace('pub-id-type="pmc">PMC7654321', 'pub-id-type="pmcaid">7654321')
    docs = pubmed_parser.parse_pmc_xml(bare)
    assert len(docs) == 1 and docs[0].doc_id == "pmc:PMC7654321"


def test_pmc_extracts_abstract_and_leaf_sections_only():
    docs = pubmed_parser.parse_pmc_xml(PMC_XML)
    assert len(docs) == 1

    doc = docs[0]
    assert doc.doc_id == "pmc:PMC7654321"
    assert doc.date == "2022"
    headings = [s.heading for s in doc.sections]
    assert headings == ["Abstract", "Introduction", "Inner"], (
        "parent sections must not be emitted, or their text is duplicated"
    )


# --------------------------------------------------------------------------
# FDA labels
# --------------------------------------------------------------------------

FDA_RECORD = {
    "set_id": "abc-123",
    "effective_time": "20240416",
    "openfda": {
        "generic_name": ["METFORMIN HYDROCHLORIDE"],
        "brand_name": ["GLUCOPHAGE"],
        "route": ["ORAL"],
    },
    "indications_and_usage": ["Indicated as an adjunct to diet and exercise."],
    "drug_interactions": ["Carbonic anhydrase inhibitors may increase lactic acidosis risk."],
    "dosage_and_administration": ["Start 500 mg twice daily with meals."],
}


def test_fda_label_sections_become_clinical_headings():
    doc = fda_parser.parse_label(FDA_RECORD)
    assert doc is not None
    assert doc.doc_id == "fda:abc-123"
    assert doc.date == "2024-04-16", "effective_time must be normalised to ISO"
    assert doc.drug_names == ["Metformin Hydrochloride", "Glucophage"]

    headings = [s.heading for s in doc.sections]
    assert "Drug Interactions" in headings
    assert "Dosage and Administration" in headings
    # Order follows LABEL_SECTIONS (label order), not dict insertion order.
    assert headings.index("Indications and Usage") < headings.index("Drug Interactions")


def test_fda_label_without_content_sections_is_dropped():
    assert fda_parser.parse_label({"set_id": "x", "openfda": {}}) is None


def test_fda_label_without_set_id_is_dropped():
    assert fda_parser.parse_label({"indications_and_usage": ["text"]}) is None


def test_fda_parse_file_roundtrip(tmp_path):
    path = tmp_path / "fda__metformin.json"
    path.write_text(json.dumps(FDA_RECORD), encoding="utf-8")
    docs = fda_parser.parse_file(path)
    assert len(docs) == 1 and docs[0].doc_id == "fda:abc-123"


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

HTML_PAGE = """<html><head><title>Hypertension Guideline</title></head>
<body>
  <nav>Home | Search | Login</nav>
  <script>trackUser();</script>
  <main>
    <h1>Hypertension</h1>
    <p>Blood pressure above 140/90 mmHg.</p>
    <h2>Treatment</h2>
    <p>Lifestyle modification is first line.</p>
    <ul><li>Reduce sodium intake.</li></ul>
  </main>
  <footer>Copyright notice</footer>
</body></html>"""


def test_html_splits_on_headings_and_drops_chrome():
    doc = html_extract.extract_html(HTML_PAGE, doc_id="html:test", url="http://x/y")
    assert doc is not None
    assert doc.title == "Hypertension Guideline"

    by_heading = {s.heading: s.text for s in doc.sections}
    assert "140/90 mmHg" in by_heading["Hypertension"]
    assert "first line" in by_heading["Treatment"]
    assert "Reduce sodium intake." in by_heading["Treatment"], "list items are content"

    joined = doc.full_text
    assert "trackUser" not in joined
    assert "Login" not in joined
    assert "Copyright notice" not in joined


def test_html_with_no_content_returns_none():
    assert html_extract.extract_html("<html><body></body></html>", doc_id="x") is None


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

@pytest.fixture
def sample_pdf(tmp_path):
    """Build a two-section PDF with a larger heading font and a references tail."""
    pymupdf = pytest.importorskip("pymupdf")

    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for text, size in [
        ("Clinical Overview", 18),
        ("Metformin lowers hepatic glucose production in adults.", 10),
        ("Adverse Effects", 18),
        ("Gastrointestinal upset is the most common adverse effect.", 10),
        ("References", 18),
        ("Sharma R. Journal of Test Medicine. 2023;1:1-10.", 10),
    ]:
        page.insert_text((72, y), text, fontsize=size)
        y += size + 10

    path = tmp_path / "overview.pdf"
    doc.save(path)
    doc.close()
    return path


def test_pdf_detects_headings_by_font_size(sample_pdf):
    doc = pdf_extract.extract_pdf(sample_pdf)
    assert doc is not None
    assert doc.source == "pdf"

    headings = [s.heading for s in doc.sections]
    assert "Clinical Overview" in headings
    assert "Adverse Effects" in headings

    by_heading = {s.heading: s.text for s in doc.sections}
    assert "hepatic glucose" in by_heading["Clinical Overview"]
    assert "Gastrointestinal" in by_heading["Adverse Effects"]


def test_pdf_drops_references_section(sample_pdf):
    doc = pdf_extract.extract_pdf(sample_pdf)
    assert "References" not in [s.heading for s in doc.sections]
    assert "Journal of Test Medicine" not in doc.full_text


def test_pdf_keeps_references_when_asked(sample_pdf):
    doc = pdf_extract.extract_pdf(sample_pdf, drop_references=False)
    assert "Journal of Test Medicine" in doc.full_text
