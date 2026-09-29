"""Streamlit demo UI over the grounded RAG pipeline.

    streamlit run app.py

Loads the models, Chroma store and BM25 index once per process (st.cache_resource)
and never rebuilds the index; run `python -m src.retrieval.build_index` for that.
"""

from __future__ import annotations

import streamlit as st

from src import config
from src.data_sources.corpus_spec import DRUGS
from src.generation.answer import SOURCE_TYPES, Answer, answer_question
from src.generation.providers import GeminiProvider, LLMError
from src.interactions.check import InteractionChecker

st.set_page_config(page_title="Medical Literature RAG", page_icon="⚕️", layout="wide")

st.warning(
    "**For medical information retrieval and education only.** This tool does not "
    "diagnose, prescribe, or give medical advice. Answers are generated from the "
    "cited sources below and can be incomplete or wrong. Always consult a "
    "qualified healthcare professional.", icon="⚠️")
st.title("Medical Literature RAG")
st.caption("MedlinePlus · PubMed · PMC · FDA drug labels → hybrid retrieval "
           "(PubMedBERT + BM25, RRF) → cross-encoder rerank → Gemini, answering only "
           "from cited passages. Plain-language topic summaries: Source: MedlinePlus, "
           "National Library of Medicine.")


@st.cache_resource(show_spinner="Loading embedding model, reranker and index...")
def load_pipeline():
    from src.embeddings import encoder
    from src.retrieval import rerank as rerank_mod
    from src.retrieval.rerank import RerankingRetriever

    retriever = RerankingRetriever()
    if retriever.hybrid.vector_store.count() == 0:
        raise RuntimeError("The vector index is empty. Build it first: "
                           "python -m src.retrieval.build_index")
    encoder.get_model()      # warm both models so the first question is not slow
    rerank_mod.get_model()
    return retriever


@st.cache_resource(show_spinner=False)
def load_provider():
    return GeminiProvider()


BADGE = {"High": "green", "Medium": "orange", "Low": "red", "None": "gray"}
TYPE_COLOR = {"MedlinePlus": "blue", "PubMed": "violet", "PMC": "violet",
              "FDA label": "green"}


def type_tag(source_type: str) -> str:
    return f":{TYPE_COLOR.get(source_type, 'gray')}-badge[{source_type}]"


def _md(text: str) -> str:
    return text.replace("$", r"\$")   # stop Streamlit reading "$5" as LaTeX


def render(ans: Answer) -> None:
    (st.markdown if ans.found else st.info)(_md(ans.text))
    color = BADGE.get(ans.confidence, "gray")
    st.markdown(f"**Confidence:** :{color}-background[{ans.confidence}]  "
                f"<small>{ans.confidence_reason}</small>", unsafe_allow_html=True)

    if ans.sources:
        st.markdown("**Sources**")
        st.markdown("\n".join(
            f"{s.number}. {type_tag(s.source_type)} [{_md(s.title)}]({s.url}) — "
            f"*{s.section}*" if s.url
            else f"{s.number}. {type_tag(s.source_type)} {_md(s.title)} — *{s.section}*"
            for s in ans.sources))

    if ans.chunks:
        with st.expander(f"Retrieved evidence: {len(ans.chunks)} passages "
                         f"(how retrieval works)"):
            st.caption(
                "Each passage was found by dense (PubMedBERT) and/or BM25 keyword "
                "search, fused with reciprocal rank fusion, then re-scored by a "
                "cross-encoder. Rerank score is the cross-encoder logit (higher = "
                f"more relevant; below {config.NOT_FOUND_THRESHOLD:g} is discarded).")
            for i, r in enumerate(ans.chunks, start=1):
                c = r.components
                ranks = ", ".join(f"{k.replace('_rank', '')} #{int(c[k])}"
                                  for k in ("dense_rank", "bm25_rank") if k in c)
                pre = ""
                if c.get("retrieval_rank"):
                    pre = f" · hybrid rank #{int(c['retrieval_rank'])}"
                if "label_scan" in c:
                    pre += " · found by FDA label scan"
                st.markdown(f"**[{i}] {_md(r.chunk.title)}** — *{r.chunk.section}*  \n"
                            f"rerank score **{r.score:+.2f}**{pre}"
                            + (f" ({ranks})" if ranks else "")
                            + " · " + type_tag(SOURCE_TYPES.get(r.chunk.source,
                                                                r.chunk.source)))
                st.text(r.chunk.text.strip())
                st.divider()


try:
    retriever = load_pipeline()
    provider = load_provider()
except (LLMError, RuntimeError, FileNotFoundError) as e:
    st.error(str(e))
    st.stop()

ask_tab, ddi_tab = st.tabs(["Ask a Question", "Drug Interaction Check"])

with ask_tab:
    with st.form("ask"):
        q = st.text_input("Question",
                          placeholder="What are the common side effects of metformin?")
        go = st.form_submit_button("Ask", type="primary")
    if go and q.strip():
        with st.spinner("Retrieving and generating..."):
            try:
                render(answer_question(q.strip(), retriever, provider))
            except LLMError as e:
                st.error(str(e))

with ddi_tab:
    st.caption(f"Covered drugs: {', '.join(DRUGS)}")
    with st.form("ddi"):
        col_a, col_b = st.columns(2)
        a = col_a.text_input("Drug A", placeholder="warfarin")
        b = col_b.text_input("Drug B", placeholder="aspirin")
        go = st.form_submit_button("Check interaction", type="primary")
    if go and a.strip() and b.strip():
        checker = InteractionChecker(retriever.hybrid.bm25.chunks, retriever, provider)
        with st.spinner("Scanning FDA labels and literature..."):
            try:
                report = checker.check(a.strip(), b.strip())
            except LLMError as e:
                st.error(str(e))
                st.stop()
        if report.label_findings:
            st.markdown("**FDA label scan**\n" + "\n".join(
                f"- {f}" for f in report.label_findings))
        render(report.answer)
