"""HTML extraction for web-sourced guidelines and drug pages.

Strategy: drop non-content elements outright, then walk the remaining tree in
document order, splitting on h1-h4. Walking in order (rather than collecting all
headings and then all paragraphs) is what keeps each paragraph attached to the
heading that actually preceded it.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from src.ingestion.normalize import normalize_text
from src.schema import Document, Section

_DROP = ["script", "style", "nav", "header", "footer", "aside", "form",
         "noscript", "iframe", "svg", "button"]

_HEADINGS = ["h1", "h2", "h3", "h4"]
_CONTENT = ["p", "li", "td", "th", "dd", "dt", "pre", "blockquote"]


def extract_html(html: str, doc_id: str, url: str = "") -> Document | None:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(_DROP):
        tag.decompose()

    page_title = normalize_text(soup.title.get_text()) if soup.title else ""
    root = soup.find("main") or soup.find("article") or soup.body or soup

    sections: list[Section] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        text = normalize_text("\n".join(buffer))
        if text:
            sections.append(Section(heading=heading or "Body", text=text))

    for el in root.find_all(_HEADINGS + _CONTENT):
        # Skip nested content already captured by an ancestor, e.g. a <p> inside a <li>.
        if el.name in _CONTENT and el.find_parent(_CONTENT):
            continue
        text = normalize_text(el.get_text(" ", strip=True))
        if not text:
            continue
        if el.name in _HEADINGS:
            flush()
            heading, buffer = text, []
        else:
            buffer.append(text)
    flush()

    if not sections:
        return None

    return Document(
        doc_id=doc_id,
        source="html",
        title=page_title or sections[0].heading,
        sections=sections,
        url=url,
    )


def extract_html_file(path: Path) -> Document | None:
    return extract_html(
        path.read_text(encoding="utf-8", errors="replace"),
        doc_id=f"html:{path.stem}",
        url=path.as_uri(),
    )
