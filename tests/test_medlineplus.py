"""MedlinePlus harvester + parser tests on a synthetic XML fixture (no network)."""

from __future__ import annotations

import zipfile

from src.chunking.splitter import chunk_documents
from src.data_sources.medlineplus import latest_zip_url
from src.ingestion.medlineplus_parser import parse_file, parse_topics, summary_text

XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<health-topics total="3" date-generated="09/29/2026 02:30:37">
<health-topic title="Malaria" url="https://medlineplus.gov/malaria.html" id="315"
    language="English" date-created="10/22/1998">
<also-called>Paludism</also-called>
<full-summary>&lt;p&gt;Malaria is a serious disease caused by a parasite. You get it when an
infected &lt;a href="https://medlineplus.gov/mosquitobites.html"&gt;mosquito&lt;/a&gt;
bites you.&lt;/p&gt;
&lt;p&gt;Malaria can be prevented. When traveling:&lt;/p&gt;&lt;ul&gt;
&lt;li&gt;Wear insect repellent with DEET &lt;/li&gt;
&lt;li&gt;Sleep under mosquito netting &lt;/li&gt;&lt;/ul&gt;
&lt;p class=""&gt;Centers for Disease Control and Prevention&lt;/p&gt;</full-summary>
<group url="https://medlineplus.gov/infections.html" id="12">Infections</group>
<site title="About Malaria" url="https://www.cdc.gov/malaria/about/">
<information-category>Start Here</information-category>
<organization>Centers for Disease Control and Prevention</organization>
<standard-description>THIRD PARTY TEXT</standard-description>
</site>
</health-topic>
<health-topic title="Malaria" url="https://medlineplus.gov/spanish/malaria.html" id="2019"
    language="Spanish" date-created="10/22/1998">
<full-summary>&lt;p&gt;La malaria es una enfermedad grave.&lt;/p&gt;</full-summary>
<group id="12">Infecciones</group>
</health-topic>
<health-topic title="Fitness" url="https://medlineplus.gov/fitness.html" id="9"
    language="English" date-created="01/02/2003">
<full-summary>&lt;p&gt;Exercise is good for you.&lt;/p&gt;</full-summary>
<group id="3">Fitness and Exercise</group>
</health-topic>
<health-topic title="Empty" url="https://medlineplus.gov/empty.html" id="10"
    language="English" date-created="01/02/2003">
<full-summary></full-summary>
</health-topic>
</health-topics>"""


def _malaria():
    return next(d for d in parse_topics(XML, groups=set()) if "Malaria" in d.title)


def test_english_only_and_empty_summaries_skipped():
    docs = parse_topics(XML, groups=set())
    assert [d.doc_id for d in docs] == ["medlineplus:315", "medlineplus:9"]


def test_group_filter():
    docs = parse_topics(XML, groups={"Infections"})
    assert [d.doc_id for d in docs] == ["medlineplus:315"]


def test_document_fields_for_citation():
    d = _malaria()
    assert d.source == "medlineplus"
    assert d.title == "MedlinePlus: Malaria"
    assert d.url == "https://medlineplus.gov/malaria.html"
    assert d.date == "1998-10-22"
    assert d.extra["groups"] == ["Infections"]
    assert d.extra["also_called"] == ["Paludism"]
    assert [s.heading for s in d.sections] == ["Summary"]


def test_summary_html_is_flattened_and_attribution_split_off():
    text = _malaria().sections[0].text
    assert text.startswith("Also called: Paludism.")       # lay synonym searchable
    assert "infected mosquito bites you." in text            # link text kept, tag gone
    assert "- Wear insect repellent with DEET" in text       # list items kept
    assert "<" not in text and "href" not in text
    assert "Centers for Disease Control" not in text         # attribution not body
    assert "THIRD PARTY TEXT" not in text                    # <site> records ignored
    assert _malaria().extra["summary_source"] == "Centers for Disease Control and Prevention"


def test_inline_links_do_not_leave_space_before_punctuation():
    text, _ = summary_text('<p>Risks include <a href="x">heart attack</a>, '
                           '<a href="y">stroke</a>, and more.</p>')
    assert text == "Risks include heart attack, stroke, and more."


def test_nested_lists_are_not_emitted_twice():
    text, _ = summary_text("<ul><li>Symptoms<ul><li>wheezing</li><li>cough</li>"
                           "</ul></li><li>Triggers</li></ul>")
    assert text.count("wheezing") == 1
    assert text == "- Symptoms wheezing cough\n- Triggers"


def test_summary_without_attribution_keeps_last_paragraph():
    text, attribution = summary_text("<p>First.</p><p>Last paragraph.</p>")
    assert text == "First.\n\nLast paragraph." and attribution == ""


def test_parse_file_reads_the_zip(tmp_path):
    z = tmp_path / "mplus_topics_compressed.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("mplus_topics_2026-09-29.xml", XML)
    assert len(parse_file(z)) == 2


def test_short_topic_chunks_into_one_citable_chunk():
    chunks = chunk_documents([_malaria()])
    assert len(chunks) == 1
    c = chunks[0]
    assert c.citation() == "MedlinePlus: Malaria [Summary] (1998-10-22)"
    assert c.url == "https://medlineplus.gov/malaria.html"


def test_latest_zip_url_picks_newest_date():
    html = ('<a href="https://medlineplus.gov/xml/mplus_topics_compressed_2026-09-26.zip">'
            '<a href="https://medlineplus.gov/xml/mplus_topics_compressed_2026-09-29.zip">'
            '<a href="https://medlineplus.gov/xml/mplus_topics_2026-09-30.xml">')
    url, date = latest_zip_url(html)
    assert date == "2026-09-29" and url.endswith("compressed_2026-09-29.zip")
