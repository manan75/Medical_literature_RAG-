"""Parse NCBI E-utilities XML into Document records.

Handles two shapes:
  * PubmedArticleSet (db=pubmed) -- title + structured or plain abstract
  * pmc-articleset  (db=pmc)     -- JATS full text with real <sec> hierarchy

PubMed structured abstracts already carry section labels ("BACKGROUND", "METHODS",
"RESULTS", "CONCLUSIONS") in the NlmCategory/Label attribute. We preserve those as
Section headings rather than flattening the abstract, because those labels are
exactly the kind of context a retrieved chunk needs to be interpretable alone.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from src.ingestion.normalize import normalize_text
from src.schema import Document, Section


def _text(node) -> str:
    return normalize_text(node.get_text(" ", strip=True)) if node else ""


def parse_pubmed_xml(xml: str) -> list[Document]:
    """Parse a PubmedArticleSet into Documents (one per article)."""
    soup = BeautifulSoup(xml, "xml")
    docs: list[Document] = []

    for art in soup.find_all("PubmedArticle"):
        pmid_node = art.find("PMID")
        if not pmid_node:
            continue
        pmid = pmid_node.get_text(strip=True)

        title = _text(art.find("ArticleTitle"))
        journal = _text(art.find("Title"))

        # Date: prefer the article's own PubDate, fall back to the PubMed record date.
        pub = art.find("PubDate") or art.find("DateCompleted")
        date = ""
        if pub:
            y = pub.find("Year")
            m = pub.find("Month")
            date = "-".join(p.get_text(strip=True) for p in (y, m) if p)
            if not date:
                date = _text(pub)

        authors = []
        for a in art.find_all("Author")[:12]:
            last, initials = a.find("LastName"), a.find("Initials")
            if last:
                name = last.get_text(strip=True)
                if initials:
                    name += f" {initials.get_text(strip=True)}"
                authors.append(name)

        sections: list[Section] = []
        for ab in art.find_all("AbstractText"):
            label = ab.get("Label") or ab.get("NlmCategory") or "Abstract"
            body = _text(ab)
            if body:
                sections.append(Section(heading=label.title(), text=body))

        # Abstract-less records (editorials, letters) carry no retrievable content.
        if not sections:
            continue

        docs.append(Document(
            doc_id=f"pubmed:{pmid}",
            source="pubmed",
            title=title,
            sections=sections,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            date=date,
            authors=authors,
            journal=journal,
        ))

    return docs


def parse_pmc_xml(xml: str) -> list[Document]:
    """Parse a pmc-articleset (JATS full text) into Documents."""
    soup = BeautifulSoup(xml, "xml")
    docs: list[Document] = []

    for art in soup.find_all("article"):
        pmcid = ""
        for aid in art.find_all("article-id"):
            if aid.get("pub-id-type") == "pmc":
                pmcid = aid.get_text(strip=True)
                break
        if not pmcid:
            continue

        title = _text(art.find("article-title"))
        journal = _text(art.find("journal-title"))
        year = art.find("year")
        date = year.get_text(strip=True) if year else ""

        authors = []
        for contrib in art.find_all("contrib")[:12]:
            sn, gn = contrib.find("surname"), contrib.find("given-names")
            if sn:
                authors.append(
                    f"{sn.get_text(strip=True)} {gn.get_text(strip=True) if gn else ''}".strip()
                )

        sections: list[Section] = []

        abstract = art.find("abstract")
        if abstract:
            body = _text(abstract)
            if body:
                sections.append(Section(heading="Abstract", text=body))

        body_node = art.find("body")
        if body_node:
            for sec in body_node.find_all("sec", recursive=True):
                # Only leaf sections, so nested content is not emitted twice.
                if sec.find("sec"):
                    continue
                heading = _text(sec.find("title")) or "Body"
                paras = [_text(p) for p in sec.find_all("p")]
                text = "\n\n".join(p for p in paras if p)
                if text:
                    sections.append(Section(heading=heading, text=text))

        if not sections:
            continue

        docs.append(Document(
            doc_id=f"pmc:{pmcid}",
            source="pmc",
            title=title,
            sections=sections,
            url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid.lstrip('PMC')}/",
            date=date,
            authors=authors,
            journal=journal,
        ))

    return docs


def parse_file(path: Path) -> list[Document]:
    """Dispatch on the filename prefix written by src.data_sources.pubmed.harvest."""
    xml = path.read_text(encoding="utf-8", errors="replace")
    if path.name.startswith("pmc__"):
        return parse_pmc_xml(xml)
    return parse_pubmed_xml(xml)
