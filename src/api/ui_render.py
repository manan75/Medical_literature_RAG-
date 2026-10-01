"""HTML fragments for the Streamlit UI (app.py). Pure functions, no Streamlit.

Every string that came from the corpus or the LLM is passed through html.escape
before it is placed in markup: chunk text is third-party content and model output
is untrusted, and both are rendered with unsafe_allow_html.
"""

from __future__ import annotations

import re
from collections import defaultdict
from html import escape

from src import config
from src.generation.answer import SOURCE_TYPES, Answer, Source, confidence_signals
from src.retrieval.types import RetrievalResult

# Rerank logits are drawn on this fixed scale so bars are comparable across answers.
SCORE_MIN, SCORE_MAX = -8.0, 10.0

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _e(text: str) -> str:
    return escape(text or "", quote=True)


def _type_class(source: str) -> str:
    return "src-" + re.sub(r"[^a-z]", "", source.lower())


def type_badge(source: str) -> str:
    """Badge for a Chunk.source ("medlineplus") or a display name ("MedlinePlus")."""
    label = SOURCE_TYPES.get(source, source)
    key = next((k for k, v in SOURCE_TYPES.items() if v == label), source)
    return f'<span class="badge {_type_class(key)}">{_e(label)}</span>'


def short_title(title: str) -> str:
    """Drop the "MedlinePlus: " / "FDA Label: " prefix; the badge already says it."""
    for prefix in ("MedlinePlus: ", "FDA Label: "):
        if title.startswith(prefix):
            return title[len(prefix):]
    return title


def link_citations(text: str, n_sources: int) -> str:
    """[1], [1][2] and [1, 2] -> superscript links to the reference list.
    Numbers outside 1..n_sources are left as plain text rather than linked."""
    def repl(m: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if not all(1 <= n <= n_sources for n in nums):
            return m.group(0)
        return "".join(f'<sup class="cite"><a href="#ref-{n}">{n}</a></sup>'
                       for n in nums)
    return _CITE.sub(repl, text)


def answer_markdown(ans: Answer) -> str:
    """Answer text as markdown with inline citation HTML. Escaped first, so model
    output cannot inject markup; markdown syntax (**, bullets) survives escaping."""
    text = _e(ans.text).replace("$", r"\$")          # "$5" is not LaTeX
    return link_citations(text, len(ans.sources))


def confidence_html(ans: Answer) -> str:
    level = ans.confidence
    filled = {"High": 3, "Medium": 2, "Low": 1}.get(level, 0)
    segs = "".join(f'<span class="seg{" on" if i < filled else ""}"></span>'
                   for i in range(3))
    if ans.chunks and level != "None":
        top, n_docs = confidence_signals(ans.chunks)
        strength = ("strong" if top >= config.HIGH_SCORE else
                    "moderate" if top > config.NOT_FOUND_THRESHOLD else "weak")
        signals = (
            f'<div class="sig"><span class="k">Strength</span>'
            f'<span class="v mono">{top:+.1f}</span><span class="d">{strength} '
            f'(&ge; {config.HIGH_SCORE:g} is strong)</span></div>'
            f'<div class="sig"><span class="k">Agreement</span>'
            f'<span class="v mono">{n_docs}</span><span class="d">distinct '
            f'source{"s" if n_docs != 1 else ""} within {config.CONFIDENCE_SPREAD:g} '
            f'of the best</span></div>')
    else:
        signals = f'<div class="sig"><span class="d">{_e(ans.confidence_reason)}</span></div>'
    return (f'<div class="confidence conf-{level.lower()}">'
            f'<div class="conf-head"><span class="eyebrow">Evidence confidence</span>'
            f'<span class="meter">{segs}</span><span class="level">{_e(level)}</span></div>'
            f'{signals}</div>')


def references_html(sources: list[Source]) -> str:
    items = []
    for s in sources:
        t = _e(short_title(s.title))
        title = (f'<a href="{_e(s.url)}" target="_blank" rel="noopener" title="{t}">{t}</a>'
                 if s.url else t)
        title = f'<span class="ref-title">{title}</span>'
        items.append(
            f'<li id="ref-{s.number}"><span class="num mono">{s.number}</span>'
            f'<div class="ref-body">{type_badge(s.source_type)} {title}'
            f'<span class="ref-section">{_e(s.section)}</span></div></li>')
    return (f'<div class="references"><div class="eyebrow">References</div>'
            f'<ol>{"".join(items)}</ol></div>')


def _bar_pct(score: float) -> float:
    clamped = min(max(score, SCORE_MIN), SCORE_MAX)
    return round(100 * (clamped - SCORE_MIN) / (SCORE_MAX - SCORE_MIN), 1)


def ledger_html(chunks: list[RetrievalResult]) -> str:
    """One row per passage the LLM saw: provenance, rerank score bar with the
    not-found threshold marked, and the full passage text behind a disclosure."""
    threshold = _bar_pct(config.NOT_FOUND_THRESHOLD)
    rows = []
    for i, r in enumerate(chunks, start=1):
        c = r.components
        chips = []
        if c.get("retrieval_rank"):
            chips.append(f'hybrid #{int(c["retrieval_rank"])}')
        chips += [f'{k.split("_")[0]} #{int(c[k])}'
                  for k in ("dense_rank", "bm25_rank") if k in c]
        if "label_scan" in c:
            chips.append("FDA label scan")
        chip_html = "".join(f'<span class="chip mono">{_e(x)}</span>' for x in chips)
        text = r.chunk.text.strip()
        rows.append(
            f'<div class="ledger-row">'
            f'<div class="ledger-meta"><span class="num mono">{i}</span>'
            f'{type_badge(r.chunk.source)}<span class="ledger-title">{_e(short_title(r.chunk.title))}</span>'
            f'<span class="ref-section">{_e(r.chunk.section)}</span></div>'
            f'<div class="scorebar" title="Rerank score {r.score:+.2f}">'
            f'<span class="fill{" below" if r.score < config.NOT_FOUND_THRESHOLD else ""}" '
            f'style="width:{_bar_pct(r.score)}%"></span>'
            f'<span class="tick" style="left:{threshold}%"></span></div>'
            f'<div class="ledger-foot"><span class="score mono">{r.score:+.2f}</span>'
            f'{chip_html}</div>'
            f'<details><summary>{_e(text[:180])}{"…" if len(text) > 180 else ""}</summary>'
            f'<p>{_e(text)}</p></details></div>')
    return (f'<div class="ledger"><div class="ledger-legend mono">'
            f'<span>rerank score {SCORE_MIN:+g}</span><span>threshold '
            f'{config.NOT_FOUND_THRESHOLD:g}</span><span>{SCORE_MAX:+g}</span></div>'
            f'{"".join(rows)}</div>')


PIPELINE = [
    ("Dense", "PubMedBERT embeddings, top 20 by cosine similarity"),
    ("Sparse", "BM25 keyword match, top 20"),
    ("Fuse", "reciprocal rank fusion of both lists"),
    ("Rerank", "cross-encoder re-scores each candidate with the question"),
    ("Filter", f"passages below {config.NOT_FOUND_THRESHOLD:g} are dropped; none left "
               f"means “not found” and no LLM call"),
    ("Generate", "Gemini answers only from the numbered passages and cites them"),
]


def pipeline_html() -> str:
    steps = "".join(f'<li><span class="step mono">{i:02d}</span><b>{_e(name)}</b>'
                    f'<span>{_e(desc)}</span></li>'
                    for i, (name, desc) in enumerate(PIPELINE, start=1))
    return f'<ol class="pipeline">{steps}</ol>'


def label_scan_html(findings: list[str]) -> str:
    cards = []
    for f in findings:
        if " names " in f:
            state, mark = "found", "●"
        elif "no FDA label" in f:
            state, mark = "missing", "○"
        else:
            state, mark = "absent", "○"
        cards.append(f'<div class="scan scan-{state}"><span class="mark">{mark}</span>'
                     f'<span>{_e(f)}</span></div>')
    return (f'<div class="eyebrow">FDA label scan</div>'
            f'<div class="scan-grid">{"".join(cards)}</div>')


def corpus_stats(chunks) -> list[tuple[str, int, int]]:
    """[(display name, documents, passages)] ordered by passages, largest first."""
    docs: dict[str, set] = defaultdict(set)
    passages: dict[str, int] = defaultdict(int)
    for c in chunks:
        docs[c.source].add(c.doc_id)
        passages[c.source] += 1
    return sorted(((SOURCE_TYPES.get(s, s), len(docs[s]), passages[s]) for s in docs),
                  key=lambda t: -t[2])


def stats_html(stats: list[tuple[str, int, int]]) -> str:
    rows = "".join(
        f'<tr><td>{type_badge(name)}</td><td class="mono">{d:,}</td>'
        f'<td class="mono">{p:,}</td></tr>' for name, d, p in stats)
    total_d, total_p = sum(s[1] for s in stats), sum(s[2] for s in stats)
    return (f'<table class="stats"><thead><tr><th>Source</th><th>Docs</th>'
            f'<th>Passages</th></tr></thead><tbody>{rows}</tbody><tfoot><tr>'
            f'<td>Total</td><td class="mono">{total_d:,}</td>'
            f'<td class="mono">{total_p:,}</td></tr></tfoot></table>')
