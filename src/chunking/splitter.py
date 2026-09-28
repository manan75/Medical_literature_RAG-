"""Section-aware chunking for medical text.

Strategy, and why
-----------------
Chunking happens **within a section, never across one**. Both of our primary
sources hand us real section boundaries for free — openFDA keys each SPL section
by name, PubMed labels structured-abstract segments — and those boundaries are
clinically load-bearing. A chunk that merges the tail of "Dosage and
Administration" with the head of "Contraindications" produces text that reads as
though a dose were being recommended in a context where it is contraindicated.
That is precisely the failure mode a medical RAG system cannot have, so the
section boundary is treated as a hard stop.

Within a section, long text is split on sentence boundaries with overlap, so no
chunk begins mid-sentence. Overlap exists because a fact and its qualifier
("...may be used in renal impairment" / "...except when eGFR < 30") frequently sit
in adjacent sentences, and a split between them would let the first be retrieved
without the second.

Tables (dosage schedules, interaction grids) are detected and kept whole: half a
dosage table is worse than no dosage table.
"""

from __future__ import annotations

import re

from src import config
from src.ingestion.normalize import estimate_tokens
from src.schema import Chunk, Document

# Abbreviations that end in a period but do not end a sentence. Without this guard
# a sentence splitter shatters "Administer 5 mg i.v. every 8 h" into fragments.
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "vs", "etc", "e.g", "i.e", "approx",
    "no", "fig", "ref", "al", "inc", "ltd", "co", "st", "mg", "ml", "kg", "mcg",
    "i.v", "i.m", "p.o", "b.i.d", "t.i.d", "q.d", "q.i.d", "p.r.n", "s.c",
    "u.s", "u.k", "ca", "cf", "min", "max", "hr", "sec", "wk", "mo", "yr",
}

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# A line looks tabular if it has an internal run of 2+ spaces or a pipe/tab -- the
# shapes a column layout collapses into once the PDF or SPL markup is flattened.
_TABLE_LINE = re.compile(r"\S(?:\s{2,}|\s*\|\s*|\t)\S")


def _ends_with_abbreviation(text: str) -> bool:
    """True if `text` ends in a known abbreviation or a single letter ('i.v.', '8 h.')."""
    m = re.search(r"([A-Za-z][A-Za-z.]*)\.$", text.strip())
    if not m:
        return False
    token = m.group(1).lower().rstrip(".")
    return token in _ABBREVIATIONS or len(token) == 1


def split_sentences(text: str) -> list[str]:
    """Split into sentences, rejoining splits that fell after an abbreviation.

    An abbreviation alone is not enough to rejoin: "Administer 5 mg i.v. every 8 h.
    Reduce the dose..." ends a real sentence on the abbreviation "h.". So we rejoin
    only when the following fragment also *starts* like a continuation -- lowercase
    or a digit. A capitalised follower means a new sentence began, abbreviation or
    not, which is also what keeps "Sharma R. The effect..." correctly split.
    """
    raw = _SENTENCE_END.split(text)
    out: list[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        continues = piece[0].islower() or piece[0].isdigit()
        if out and continues and _ends_with_abbreviation(out[-1]):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def _is_table_block(block: str) -> bool:
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    tabular = sum(1 for ln in lines if _TABLE_LINE.search(ln))
    return tabular >= max(2, len(lines) // 2)


def _split_units(text: str) -> list[str]:
    """Break section text into atomic units: whole tables, or single sentences.

    Returning tables as single indivisible units is what keeps a dosage schedule
    from being cut in half by a size limit.
    """
    units: list[str] = []
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            continue
        if _is_table_block(block):
            units.append(block)
        else:
            units.extend(split_sentences(block.replace("\n", " ")))
    return units


def _pack(units: list[str], target: int, overlap: int, min_tokens: int) -> list[str]:
    """Greedily pack units into chunks of ~`target` tokens with `overlap` carry-over."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)

        # A single oversized unit (a big table) becomes its own chunk rather than
        # being broken up -- oversize is the lesser harm.
        if unit_tokens >= target and not current:
            chunks.append(unit)
            continue

        if current and current_tokens + unit_tokens > target:
            chunks.append(" ".join(current))
            # Carry the tail of the finished chunk into the next one as overlap.
            carry: list[str] = []
            carry_tokens = 0
            for prev in reversed(current):
                prev_tokens = estimate_tokens(prev)
                if carry_tokens + prev_tokens > overlap:
                    break
                carry.insert(0, prev)
                carry_tokens += prev_tokens
            current, current_tokens = carry, carry_tokens

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current))

    # An orphan tail ("See full prescribing information.") is not independently
    # retrievable, so fold it back into the chunk it came from.
    if len(chunks) > 1 and estimate_tokens(chunks[-1]) < min_tokens:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {tail}"

    return chunks


def chunk_document(
    doc: Document,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
    min_tokens: int | None = None,
    noise_floor: int | None = None,
) -> list[Chunk]:
    """Split one Document into Chunks, never crossing a section boundary."""
    target = target_tokens or config.CHUNK_TARGET_TOKENS
    overlap = overlap_tokens or config.CHUNK_OVERLAP_TOKENS
    minimum = min_tokens or config.CHUNK_MIN_TOKENS
    floor = config.CHUNK_NOISE_FLOOR if noise_floor is None else noise_floor

    chunks: list[Chunk] = []
    for s_idx, section in enumerate(doc.sections):
        units = _split_units(section.text)
        if not units:
            continue
        for p_idx, text in enumerate(_pack(units, target, overlap, minimum)):
            # Drop stubs a whole section long -- a bare registration URL or a
            # one-line cross-reference is retrievable noise, not evidence.
            if estimate_tokens(text) < floor:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}::{s_idx}::{p_idx}",
                doc_id=doc.doc_id,
                text=text,
                section=section.heading,
                source=doc.source,
                title=doc.title,
                url=doc.url,
                date=doc.date,
                drug_names=doc.drug_names,
                token_estimate=estimate_tokens(text),
            ))
    return chunks


def chunk_documents(docs: list[Document], **kwargs) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, **kwargs))
    return out
