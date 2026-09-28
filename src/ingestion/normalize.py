"""Text normalisation shared by every extractor.

The goal is not pretty text -- it is text where a retrieved chunk still reads
correctly to a clinician. So we repair hyphenated line breaks and collapse
whitespace, but we deliberately do NOT strip numbers, units, or punctuation:
"2.5 mg/kg q12h" has to survive intact or the dosage chunks become useless.
"""

from __future__ import annotations

import re
import unicodedata

# Boilerplate lines that add noise to retrieval without adding clinical content.
_BOILERPLATE = re.compile(
    r"^\s*(downloaded from|this article is protected by copyright|"
    r"all rights reserved|see discussions, stats|page \d+ of \d+|"
    r"©\s*\d{4}|doi:\s*10\.)",
    re.IGNORECASE,
)

_WORD_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t\u00a0]{2,}")


def normalize_text(text: str) -> str:
    """Clean extracted text without destroying clinically meaningful detail."""
    if not text:
        return ""

    # NFKC folds ligatures and full-width forms; medical PDFs are full of "fi"/"ffi".
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # "hyper-\ntension" -> "hypertension" (a PDF line-wrap artifact, not a real hyphen)
    text = _WORD_HYPHEN_BREAK.sub(r"\1\2", text)

    lines = [ln.rstrip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not _BOILERPLATE.match(ln)]
    text = "\n".join(lines)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def prettify_heading(raw: str) -> str:
    """Turn an openFDA field name into a readable clinical heading.

    'drug_interactions' -> 'Drug Interactions'
    """
    words = raw.replace("_", " ").split()
    small = {"and", "in", "of", "to", "for", "with"}
    out = [w.capitalize() if i == 0 or w not in small else w
           for i, w in enumerate(words)]
    return " ".join(out)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens per whitespace word for English prose).

    Used only for chunk sizing, so an approximation avoids pulling in a tokenizer
    dependency at the ingestion stage.
    """
    return int(len(text.split()) * 1.3)
