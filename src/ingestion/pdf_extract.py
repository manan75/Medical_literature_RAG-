"""PDF extraction with heading detection, via PyMuPDF.

A flat text dump of a clinical PDF loses the section structure that makes a chunk
citable. So instead of a plain text dump, we walk the span dictionary and treat a
line as a heading when its font size is meaningfully larger than the document body
size, or when it is short, bold and title-like. That recovers real sections from
guideline PDFs without needing a document-specific parser.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf

from src.ingestion.normalize import normalize_text
from src.schema import Document, Section

_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold

# Once we hit this heading, the rest is bibliography, not retrievable content.
_REFERENCES_HEADING = re.compile(r"^\s*(references|bibliography|works cited)\s*$", re.I)


def _lines_with_style(page) -> list[tuple[str, float, bool]]:
    """Return (text, max_font_size, any_bold) for each visual line on the page."""
    out: list[tuple[str, float, bool]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            size = max((s.get("size", 0.0) for s in spans), default=0.0)
            bold = any(s.get("flags", 0) & _BOLD_FLAG for s in spans)
            out.append((text, size, bold))
    return out


def _is_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    if len(text) > 90 or text.endswith((".", ",", ";")):
        return False
    if size >= body_size + 1.0:
        return True
    # Same size but bold and short: almost always a run-in heading.
    return bold and len(text) < 60 and size >= body_size - 0.5


def _body_font_size(lines: list[tuple[str, float, bool]]) -> float:
    """Infer the body-text font size, weighted by character count rather than by
    line count. A document may have as many heading lines as body lines, but body
    text always dominates by volume -- counting lines lets a heading-heavy page
    elect the heading size as "body" and collapse every section into one.
    """
    weights: Counter[float] = Counter()
    for text, size, _ in lines:
        if size > 0:
            weights[round(size, 1)] += len(text)
    return weights.most_common(1)[0][0] if weights else 10.0


def extract_pdf(path: Path, drop_references: bool = True) -> Document | None:
    """Extract a PDF into an ordered list of Sections."""
    doc = pymupdf.open(path)
    try:
        all_lines: list[tuple[str, float, bool]] = []
        for page in doc:
            all_lines.extend(_lines_with_style(page))

        if not all_lines:
            return None

        body_size = _body_font_size(all_lines)

        sections: list[Section] = []
        heading = ""
        buffer: list[str] = []

        def flush() -> None:
            text = normalize_text("\n".join(buffer))
            if text:
                sections.append(Section(heading=heading or "Body", text=text))

        for text, size, bold in all_lines:
            if _is_heading(text, size, bold, body_size):
                if drop_references and _REFERENCES_HEADING.match(text):
                    flush()
                    buffer = []
                    break
                flush()
                heading, buffer = text, []
            else:
                buffer.append(text)
        else:
            flush()

        if not sections:
            return None

        meta = doc.metadata or {}
        title = normalize_text(meta.get("title") or "") or path.stem.replace("_", " ")

        return Document(
            doc_id=f"pdf:{path.stem}",
            source="pdf",
            title=title,
            sections=sections,
            url=path.as_uri(),
            date=(meta.get("creationDate") or "")[2:10],
            authors=[a for a in [meta.get("author")] if a],
            extra={"page_count": doc.page_count, "filename": path.name},
        )
    finally:
        doc.close()
