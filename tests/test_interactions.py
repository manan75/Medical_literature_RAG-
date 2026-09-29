"""Phase 7 tests: drug-interaction evidence lookup (no network, no models).

Synthetic label chunks stand in for the corpus, the cross-encoder is stubbed as in
test_rerank.py, and the LLM is a fake. A final test runs the label scan against
the real chunks.jsonl when it exists -- that part needs no model either.
"""

from __future__ import annotations

import pytest

from src import config
from src.interactions.check import NO_EVIDENCE, InteractionChecker
from src.retrieval import rerank as rerank_mod
from src.retrieval.types import RetrievalResult, renumber
from src.schema import Chunk, load_chunks
from tests.test_generation import FakeProvider


def _c(cid, title, section, text, source="fda_label", drugs=()):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], text=text,
                 section=section, source=source, title=title,
                 url=f"https://x.test/{cid}", drug_names=list(drugs))


CHUNKS = [
    _c("fda:w::0", "FDA Label: Warfarin", "Drug Interactions",
       "Aspirin and other antiplatelet agents increase bleeding risk.",
       drugs=["Warfarin", "Warfarin Sodium"]),
    _c("fda:w::1", "FDA Label: Warfarin", "Adverse Reactions", "Bleeding."),
    _c("fda:a::0", "FDA Label: Aspirin", "Warnings", "Stomach bleeding warning.",
       drugs=["Aspirin", "Low Dose Aspirin"]),
    _c("fda:s::0", "FDA Label: Simvastatin", "Contraindications",
       "Concomitant use with clarithromycin is contraindicated."),
    _c("fda:c::0", "FDA Label: Clarithromycin", "Drug Interactions",
       "Clarithromycin inhibits CYP3A; simvastatin exposure increases."),
    _c("fda:m::0", "FDA Label: Metformin Er 500 Mg", "Drug Interactions",
       "Carbonic anhydrase inhibitors may cause acidosis.",
       drugs=["Metformin Er 500 Mg", "Metformin"]),
    _c("fda:l::0", "FDA Label: Levothyroxine Sodium", "Drug Interactions",
       "Calcium carbonate reduces absorption."),
    _c("pubmed:9::0", "Warfarin and aspirin bleeding: a review", "Results",
       "Combining warfarin with aspirin raised major bleeding.", source="pubmed"),
]


class StubHybrid:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def search(self, query, top_k=None):
        self.queries.append(query)
        return renumber([RetrievalResult(c, 1.0) for c in self.chunks])


class StubRetriever:
    def __init__(self, chunks):
        self.hybrid = StubHybrid(chunks)


class StubCrossEncoder:
    def predict(self, pairs, **kw):
        return [5.0 if "Drug Interactions" in p else 1.0 for _, p in pairs]


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.setattr(rerank_mod, "get_model", lambda *a, **k: StubCrossEncoder())
    llm = FakeProvider("They interact: bleeding risk rises [1].")
    return InteractionChecker(CHUNKS, StubRetriever(CHUNKS), llm), llm


def test_warfarin_aspirin_finds_label_and_literature_evidence(checker):
    chk, llm = checker
    rep = chk.check("Warfarin", "aspirin")
    ids = [r.chunk_id for r in rep.answer.chunks]

    assert rep.answer.found
    assert ids[0] == "fda:w::0"                    # label DI section, reranked first
    assert "pubmed:9::0" in ids                    # literature naming both drugs
    assert "fda:w::1" not in ids                   # warfarin chunk not naming aspirin
    assert "Warfarin label names aspirin in: Drug Interactions" in rep.label_findings
    assert "Aspirin label does not mention warfarin" in rep.label_findings
    assert len(llm.calls) == 1
    assert rep.answer.sources[0].url == "https://x.test/fda:w::0"


def test_simvastatin_clarithromycin_found_in_both_labels(checker):
    chk, _ = checker
    rep = chk.check("simvastatin", "clarithromycin")
    assert rep.answer.found
    assert {r.chunk_id for r in rep.answer.chunks} == {"fda:s::0", "fda:c::0"}


def test_pair_without_evidence_says_so_and_never_calls_llm(checker):
    chk, llm = checker
    rep = chk.check("metformin", "levothyroxine")
    assert not rep.answer.found
    assert rep.answer.text.startswith(NO_EVIDENCE)
    assert "does NOT mean the combination is safe" in rep.answer.text
    assert llm.calls == []


def test_drug_not_in_corpus_is_reported(checker):
    chk, llm = checker
    rep = chk.check("warfarin", "unobtainium")
    assert not rep.answer.found
    assert "Not in our corpus: unobtainium" in rep.answer.text
    assert llm.calls == []


def test_normalize_uses_corpus_aliases_and_salt_forms(checker):
    chk, _ = checker
    assert chk.normalize("Low Dose Aspirin") == "aspirin"
    assert chk.normalize("warfarin sodium") == "warfarin"
    assert chk.normalize("Tramadol HCl 50 mg") == "tramadol"
    assert chk.normalize("  METFORMIN ") == "metformin"
    assert chk.normalize("coumadin") is None       # brand absent from corpus


@pytest.mark.skipif(not config.CHUNKS_FILE.exists(), reason="corpus not built")
@pytest.mark.parametrize("a,b", [("warfarin", "aspirin"),
                                 ("simvastatin", "clarithromycin"),
                                 ("warfarin", "ciprofloxacin"),
                                 ("sertraline", "tramadol")])
def test_real_labels_name_the_interacting_drug(a, b):
    chk = InteractionChecker(load_chunks(config.CHUNKS_FILE), None, None)
    hits, _ = chk._label_hits(a, b)
    assert hits, f"expected the {a}/{b} labels to name each other"


@pytest.mark.skipif(not config.CHUNKS_FILE.exists(), reason="corpus not built")
def test_real_labels_do_not_link_the_control_pair():
    chk = InteractionChecker(load_chunks(config.CHUNKS_FILE), None, None)
    hits, _ = chk._label_hits("amoxicillin", "gabapentin")
    assert hits == []
