"""Data contracts shared across pipeline stages.

Two record types travel through the pipeline, both persisted as JSONL:

  Document -- one source item (a PubMed/PMC article, or one FDA drug label),
              already extracted and normalised into ordered sections.
  Chunk    -- one retrieval unit, carrying enough metadata to cite it.

Keeping these as dataclasses (rather than bare dicts) means a typo in a field
name fails loudly at the boundary instead of silently producing an uncitable chunk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Section:
    """A titled span of a document, e.g. 'Contraindications' or 'Methods'."""
    heading: str
    text: str


@dataclass
class Document:
    doc_id: str                 # stable, source-prefixed: "pubmed:12345", "fda:<set_id>"
    source: str                 # "pubmed" | "pmc" | "fda_label"
    title: str
    sections: list[Section] = field(default_factory=list)
    url: str = ""
    date: str = ""              # publication or label-effective date, ISO-ish
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    drug_names: list[str] = field(default_factory=list)   # populated for FDA labels
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"{s.heading}\n{s.text}" if s.heading else s.text
                           for s in self.sections)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        d = dict(d)
        d["sections"] = [Section(**s) for s in d.get("sections", [])]
        return cls(**d)


@dataclass
class Chunk:
    chunk_id: str               # "<doc_id>::<section_index>::<part_index>"
    doc_id: str
    text: str
    section: str                # heading this chunk came from
    source: str
    title: str
    url: str = ""
    date: str = ""
    drug_names: list[str] = field(default_factory=list)
    token_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)

    def citation(self) -> str:
        """Human-readable source label used in grounded answers."""
        bits = [self.title]
        if self.section:
            bits.append(f"[{self.section}]")
        if self.date:
            bits.append(f"({self.date})")
        return " ".join(bits)


# ---- JSONL helpers ----

def write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            payload = r.to_dict() if hasattr(r, "to_dict") else r
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_documents(path: Path) -> list[Document]:
    return [Document.from_dict(d) for d in read_jsonl(path)]


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.from_dict(d) for d in read_jsonl(path)]
