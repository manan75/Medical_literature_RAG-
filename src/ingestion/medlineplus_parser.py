"""MedlinePlus Health Topic XML -> Documents (one per English health topic).

Only public-domain fields are kept: title, URL, the NLM-written full summary,
"also called" names and group membership. The per-topic <site> records (links to
third-party pages, with their own descriptions) are ignored.

"Also called" names are written into the text itself ("Also called: high blood
pressure, HBP.") rather than only into metadata, so that BM25, the embedding and
the LLM all see the lay synonym -- that is the whole point of this source.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from lxml import etree

from src.data_sources import corpus_spec
from src.ingestion.normalize import normalize_text
from src.schema import Document, Section


def _iso(us_date: str) -> str:
    """'12/22/2015' -> '2015-12-22'."""
    parts = (us_date or "").split("/")
    return f"{parts[2]}-{parts[0]}-{parts[1]}" if len(parts) == 3 else us_date


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)])")


def _flat(el) -> str:
    """Element text, whitespace collapsed. Joining on " " keeps "Symptoms<ul>..."
    from gluing into one word; the regex then undoes the " ," that the same join
    leaves after inline links ("heart attack , stroke")."""
    return _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(el.get_text(" ").split()))


def summary_text(html: str) -> tuple[str, str]:
    """Flatten summary HTML to text. Returns (text, attribution).

    Summaries end with an attribution paragraph (<p class="">, e.g. "Centers for
    Disease Control and Prevention") naming the agency the text was adapted from.
    It is split off: as body text it would read as a stray sentence in a chunk.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    attribution = ""
    paras = soup.find_all("p")
    if paras and paras[-1].get("class") == [] and paras[-1].has_attr("class"):
        attribution = _flat(paras[-1])
        paras[-1].decompose()

    blocks = []
    for el in soup.find_all(["p", "ul", "ol"]):
        # Top-level blocks only: 140 summaries nest lists inside <li>/<p>, and
        # visiting the inner ones too would emit their text twice.
        if el.find_parent(["p", "ul", "ol", "li"]):
            continue
        if el.name == "p":
            blocks.append(_flat(el))
        else:
            blocks.append("\n".join(f"- {_flat(li)}"
                                    for li in el.find_all("li", recursive=False)))
    text = "\n\n".join(b for b in blocks if b) or _flat(soup)
    return normalize_text(text), attribution


def parse_topics(xml: bytes, language: str | None = None,
                 groups: set[str] | None = None) -> list[Document]:
    language = language or corpus_spec.MEDLINEPLUS_LANGUAGE
    groups = corpus_spec.MEDLINEPLUS_GROUPS if groups is None else groups
    root = etree.fromstring(xml, parser=etree.XMLParser(resolve_entities=False,
                                                        no_network=True))
    docs: list[Document] = []
    for t in root.iter("health-topic"):
        if t.get("language") != language:
            continue
        topic_groups = [g.text for g in t.findall("group") if g.text]
        if groups and not set(topic_groups) & groups:
            continue
        text, attribution = summary_text(t.findtext("full-summary") or "")
        if not text:
            continue
        also = [a.text.strip() for a in t.findall("also-called") if a.text]
        if also:
            text = f"Also called: {', '.join(also)}.\n\n{text}"
        docs.append(Document(
            doc_id=f"medlineplus:{t.get('id')}",
            source="medlineplus",
            title=f"MedlinePlus: {t.get('title')}",
            sections=[Section(heading="Summary", text=text)],
            url=t.get("url", ""),
            date=_iso(t.get("date-created", "")),
            extra={"also_called": also, "groups": topic_groups,
                   "summary_source": attribution},
        ))
    return docs


def parse_file(path: Path) -> list[Document]:
    """Read the harvested zip (it holds a single XML file)."""
    with zipfile.ZipFile(path) as zf:
        xml_name = next(n for n in zf.namelist() if n.endswith(".xml"))
        return parse_topics(zf.read(xml_name))
