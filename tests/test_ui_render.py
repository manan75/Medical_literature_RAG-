"""UI markup tests: escaping of untrusted text, citation links, score bars."""

from __future__ import annotations

from src import config
from src.api import ui_render as ui
from src.generation.answer import Answer, Source
from src.retrieval.types import RetrievalResult
from src.schema import Chunk


def _r(cid, score, source="medlineplus", text="Malaria is caused by a parasite.",
       doc="d1", components=None):
    return RetrievalResult(
        Chunk(chunk_id=cid, doc_id=doc, text=text, section="Summary", source=source,
              title="MedlinePlus: Malaria", url="https://medlineplus.gov/malaria.html"),
        score, components=components or {})


def _ans(text="Caused by a parasite [1].", level="Medium", chunks=None, n_sources=1):
    chunks = chunks if chunks is not None else [_r("a", 6.6)]
    sources = [Source(i, "MedlinePlus: Malaria", "Summary",
                      "https://medlineplus.gov/malaria.html", "cit", "MedlinePlus")
               for i in range(1, n_sources + 1)]
    return Answer("q", text, True, level, "reason", sources, chunks)


def test_citations_become_links_in_all_three_formats():
    out = ui.link_citations("A [1]. B [1][2]. C [1, 2].", n_sources=2)
    assert out.count('href="#ref-1"') == 3 and out.count('href="#ref-2"') == 2
    assert "[" not in out


def test_out_of_range_citation_is_not_linked():
    assert ui.link_citations("See [7].", n_sources=2) == "See [7]."


def test_llm_output_cannot_inject_markup():
    md = ui.answer_markdown(_ans('<img src=x onerror="alert(1)"> fever [1] costs $5'))
    assert "<img" not in md and "&lt;img" in md
    assert '<sup class="cite">' in md          # our own markup is still added
    assert r"\$5" in md                         # not rendered as LaTeX


def test_markdown_syntax_survives_escaping():
    md = ui.answer_markdown(_ans("**Symptoms:**\n* fever [1]"))
    assert "**Symptoms:**" in md and "\n* fever" in md


def test_ledger_escapes_chunk_text_and_marks_threshold():
    evil = _r("x", -2.0, text="<script>alert(1)</script> text",
              components={"retrieval_rank": 3.0, "dense_rank": 1.0, "label_scan": 1.0})
    html = ui.ledger_html([_r("a", 6.6), evil])
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert 'class="fill below"' in html        # score under the threshold
    assert "hybrid #3" in html and "dense #1" in html and "FDA label scan" in html
    assert f'left:{ui._bar_pct(config.NOT_FOUND_THRESHOLD)}%' in html


def test_bar_pct_is_clamped_to_scale():
    assert ui._bar_pct(-100) == 0 and ui._bar_pct(100) == 100
    assert 0 < ui._bar_pct(0) < 100


def test_confidence_meter_fills_segments_by_level():
    html = ui.confidence_html(_ans(level="High", chunks=[_r("a", 7.0, doc="d1"),
                                                         _r("b", 6.0, doc="d2")]))
    assert html.count('class="seg on"') == 3
    assert "+7.0" in html and ">2<" in html    # strength and agreement signals


def test_references_link_out_and_anchor_citations():
    html = ui.references_html(_ans(n_sources=2).sources)
    assert 'id="ref-1"' in html and 'id="ref-2"' in html
    assert 'href="https://medlineplus.gov/malaria.html"' in html
    assert 'class="badge src-medlineplus"' in html


def test_badge_class_is_the_same_for_chunk_source_and_display_name():
    assert ui.type_badge("fda_label") == ui.type_badge("FDA label")


def test_label_scan_states():
    html = ui.label_scan_html(["Warfarin label names aspirin in: Drug Interactions",
                               "Aspirin label does not mention warfarin",
                               "Foo: no FDA label in corpus"])
    assert "scan-found" in html and "scan-absent" in html and "scan-missing" in html


def test_corpus_stats_counts_docs_and_passages():
    chunks = [_r("a", 0, doc="m1").chunk, _r("b", 0, doc="m1").chunk,
              _r("c", 0, source="pubmed", doc="p1").chunk]
    assert ui.corpus_stats(chunks) == [("MedlinePlus", 1, 2), ("PubMed", 1, 1)]
