"""Phase 2 tests: section-aware chunking.

The load-bearing guarantee is the first test in this file: no chunk may ever span
two sections. Everything else is sizing and tidiness.
"""

from __future__ import annotations

from src.chunking.splitter import (
    chunk_document,
    chunk_documents,
    split_sentences,
    _is_table_block,
)
from src.schema import Document, Section


def _doc(*sections: tuple[str, str], doc_id: str = "test:1") -> Document:
    return Document(
        doc_id=doc_id,
        source="fda_label",
        title="FDA Label: Testdrug",
        sections=[Section(heading=h, text=t) for h, t in sections],
        url="http://example.test/label",
        date="2024-01-01",
        drug_names=["Testdrug"],
    )


def _sentences(word: str, n: int) -> str:
    """Build filler prose. Sentences start capitalised (real text does), and each
    carries a lowercase marker word so a chunk can be traced back to its section."""
    return " ".join(
        f"Line {i} concerning {word} with several padding words here." for i in range(n)
    )


# --------------------------------------------------------------------------
# The core guarantee
# --------------------------------------------------------------------------

def test_no_chunk_ever_spans_two_sections():
    """A dosage sentence must never end up glued to a contraindication."""
    doc = _doc(
        ("Dosage and Administration", _sentences("dosage", 60)),
        ("Contraindications", _sentences("contraindication", 60)),
    )
    chunks = chunk_document(doc)
    assert len(chunks) > 2, "sections this long must split into several chunks"

    for c in chunks:
        has_dosage = "concerning dosage" in c.text
        has_contra = "concerning contraindication" in c.text
        assert not (has_dosage and has_contra), f"chunk crossed a section: {c.chunk_id}"
        # And the metadata must agree with the content it actually holds.
        assert c.section == ("Dosage and Administration" if has_dosage
                            else "Contraindications")


def test_short_sections_stay_separate_rather_than_merging():
    doc = _doc(
        ("Boxed Warning", _sentences("warning", 4)),
        ("Description", _sentences("description", 4)),
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    assert {c.section for c in chunks} == {"Boxed Warning", "Description"}


# --------------------------------------------------------------------------
# Chunk metadata / citability
# --------------------------------------------------------------------------

def test_every_chunk_carries_what_it_needs_to_be_cited():
    doc = _doc(("Drug Interactions", _sentences("interaction", 30)))
    for c in chunk_document(doc):
        assert c.doc_id == "test:1"
        assert c.section == "Drug Interactions"
        assert c.title and c.url and c.source
        assert c.token_estimate > 0
        assert "FDA Label: Testdrug" in c.citation()
        assert "Drug Interactions" in c.citation()


def test_chunk_ids_are_unique_and_traceable():
    doc = _doc(
        ("Warnings", _sentences("warn", 60)),
        ("Adverse Reactions", _sentences("adverse", 60)),
    )
    ids = [c.chunk_id for c in chunk_document(doc)]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("test:1::") for i in ids)


# --------------------------------------------------------------------------
# Sizing and overlap
# --------------------------------------------------------------------------

def test_chunks_respect_the_target_size():
    doc = _doc(("Warnings", _sentences("warn", 200)))
    chunks = chunk_document(doc, target_tokens=100, overlap_tokens=20)
    assert len(chunks) > 3
    # Greedy packing overshoots by at most the final sentence it admitted.
    assert all(c.token_estimate <= 130 for c in chunks)


def test_consecutive_chunks_overlap():
    doc = _doc(("Warnings", _sentences("warn", 100)))
    chunks = chunk_document(doc, target_tokens=100, overlap_tokens=30)
    assert len(chunks) >= 2

    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, "adjacent chunks must share overlap text"


def test_no_chunk_starts_mid_sentence():
    doc = _doc(("Warnings", _sentences("warn", 80)))
    for c in chunk_document(doc, target_tokens=100, overlap_tokens=20):
        assert c.text[0].isupper() or c.text[0].isdigit()


def test_orphan_tail_is_folded_back():
    """A section ending in a stub should not produce a standalone stub chunk."""
    text = _sentences("body", 80) + " See full prescribing information."
    doc = _doc(("Warnings", text))
    chunks = chunk_document(doc, target_tokens=100, overlap_tokens=20, min_tokens=40)
    assert chunks[-1].text != "See full prescribing information."
    assert "See full prescribing information." in chunks[-1].text


def test_sub_floor_stub_section_is_dropped():
    doc = _doc(
        ("Systematic Review Registration", "https://example.test/record?ID=CRD42021"),
        ("Objective", _sentences("objective", 20)),
    )
    sections = {c.section for c in chunk_document(doc)}
    assert sections == {"Objective"}


# --------------------------------------------------------------------------
# Sentence splitting and tables
# --------------------------------------------------------------------------

def test_dosage_abbreviations_do_not_split_sentences():
    text = "Administer 5 mg i.v. every 8 h. Reduce the dose in renal impairment."
    assert split_sentences(text) == [
        "Administer 5 mg i.v. every 8 h.",
        "Reduce the dose in renal impairment.",
    ]


def test_author_initials_do_not_split_sentences():
    assert len(split_sentences("Reported by Sharma R. The effect was modest.")) == 2


def test_eg_and_ie_do_not_split_sentences():
    text = "Avoid strong inhibitors, e.g. clarithromycin. Monitor INR closely."
    parts = split_sentences(text)
    assert parts[0] == "Avoid strong inhibitors, e.g. clarithromycin."
    assert len(parts) == 2


def test_table_blocks_are_detected():
    table = "eGFR      Dose\n>=45      500 mg\n30-44     250 mg"
    assert _is_table_block(table)
    assert not _is_table_block("A normal sentence of prose.\nAnd another one.")


def test_dosage_table_is_kept_whole():
    table = "eGFR      Dose\n>=45      1000 mg twice daily\n30-44     500 mg twice daily"
    doc = _doc(("Dosage and Administration", f"{_sentences('lead', 40)}\n\n{table}"))
    chunks = chunk_document(doc, target_tokens=60, overlap_tokens=10)

    holders = [c for c in chunks if ">=45" in c.text]
    assert len(holders) >= 1
    for c in holders:
        assert "30-44" in c.text, "a dosage table must not be cut in half"


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------

def test_chunk_documents_keeps_documents_distinct():
    docs = [
        _doc(("Warnings", _sentences("a", 20)), doc_id="test:1"),
        _doc(("Warnings", _sentences("b", 20)), doc_id="test:2"),
    ]
    chunks = chunk_documents(docs)
    assert {c.doc_id for c in chunks} == {"test:1", "test:2"}


def test_empty_document_produces_no_chunks():
    assert chunk_document(_doc(("Empty", "   "))) == []
