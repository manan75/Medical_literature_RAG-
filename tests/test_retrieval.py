"""Phase 3-4 tests: embeddings, sparse retrieval, and hybrid fusion.

The vector store is exercised against an in-memory Chroma client with stub
embeddings, so the suite stays offline and never downloads the 420 MB encoder.
Fusion is tested on synthetic ranked lists, where the correct answer is arithmetic
rather than a matter of judgement.
"""

from __future__ import annotations

import pytest

from src.data_sources.openfda import _specificity
from src.embeddings.encoder import embedding_text
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.sparse import BM25Index, tokenize
from src.retrieval.types import RetrievalResult, renumber
from src.schema import Chunk


def _chunk(cid: str, text: str, section: str = "Adverse Reactions",
           title: str = "FDA Label: Metformin", drugs: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=f"doc:{cid.split('::')[0]}", text=text,
        section=section, source="fda_label", title=title,
        url="http://example.test", date="2024-01-01",
        drug_names=drugs or ["Metformin"], token_estimate=len(text.split()),
    )


def _result(cid: str, rank: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk=_chunk(cid, "text"), score=score, rank=rank)


# --------------------------------------------------------------------------
# Embedding input
# --------------------------------------------------------------------------

def test_embedding_text_carries_the_section_heading():
    """Two near-identical sentences under different headings must not embed alike."""
    c = _chunk("a::0::0", "Concomitant use increases bleeding risk.",
               section="Drug Interactions")
    text = embedding_text(c)
    assert text.startswith("FDA Label: Metformin | Drug Interactions")
    assert "Concomitant use increases bleeding risk." in text


def test_embedding_text_leaves_the_stored_chunk_untouched():
    c = _chunk("a::0::0", "Original body text.")
    embedding_text(c)
    assert c.text == "Original body text.", "decoration must not leak into citations"


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

def test_tokenizer_keeps_clinically_significant_tokens():
    tokens = tokenize("CYP3A4 inhibitors with 500 mg doses and eGFR 30")
    assert "cyp3a4" in tokens
    assert "500" in tokens and "30" in tokens
    assert "mg" in tokens
    assert "and" not in tokens and "with" not in tokens


def test_tokenizer_keeps_negation_words():
    """Stripping 'no'/'not' would invert the meaning of a contraindication."""
    tokens = tokenize("no known interaction and not contraindicated")
    assert "no" in tokens
    assert "not" in tokens


def test_tokenizer_preserves_hyphenated_terms():
    assert "first-line" in tokenize("first-line therapy")


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

@pytest.fixture
def bm25():
    return BM25Index([
        _chunk("a::0::0", "Metformin commonly causes diarrhea and nausea.",
               title="FDA Label: Metformin"),
        _chunk("b::0::0", "Warfarin requires regular INR monitoring.",
               title="FDA Label: Warfarin", drugs=["Warfarin"]),
        _chunk("c::0::0", "Simvastatin exposure rises with CYP3A4 inhibitors.",
               title="FDA Label: Simvastatin", drugs=["Simvastatin"]),
        _chunk("d::0::0", "Reduce the dose when eGFR falls below 30 mL/min.",
               section="Dosage and Administration", title="FDA Label: Metformin"),
    ])


def test_bm25_matches_exact_rare_tokens(bm25):
    results = bm25.search("CYP3A4", top_k=3)
    assert results and results[0].chunk_id == "c::0::0"


def test_bm25_matches_lab_thresholds(bm25):
    results = bm25.search("eGFR 30", top_k=3)
    assert results[0].chunk_id == "d::0::0"


def test_bm25_matches_on_section_heading(bm25):
    """A query naming a section must match even if the body never says the word."""
    results = bm25.search("dosage administration", top_k=4)
    assert results[0].chunk_id == "d::0::0"


def test_bm25_returns_nothing_for_absent_terms(bm25):
    assert bm25.search("chemotherapy oncology radiotherapy") == []


def test_bm25_ranks_are_sequential(bm25):
    results = bm25.search("metformin warfarin", top_k=4)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_bm25_empty_corpus_is_safe():
    assert BM25Index([]).search("anything") == []


# --------------------------------------------------------------------------
# Reciprocal rank fusion
# --------------------------------------------------------------------------

def test_rrf_rewards_agreement_over_a_single_strong_vote():
    """A chunk both retrievers rank 2nd beats one that only one retriever ranks 1st."""
    dense = renumber([_result("only-dense", 1), _result("both", 2)])
    sparse = renumber([_result("only-sparse", 1), _result("both", 2)])

    fused = reciprocal_rank_fusion({"dense": dense, "bm25": sparse}, k=60)
    assert fused[0].chunk_id == "both"

    # 2/(60+2) = 0.03226 beats 1/(60+1) = 0.01639
    assert fused[0].score == pytest.approx(2 / 62, rel=1e-6)
    assert fused[1].score == pytest.approx(1 / 61, rel=1e-6)


def test_rrf_ignores_raw_score_magnitude():
    """BM25 scores are unbounded; that must not let them dominate cosine scores."""
    dense = renumber([_result("d1", 1, score=0.51), _result("shared", 2, score=0.50)])
    sparse = renumber([_result("shared", 1, score=98.0), _result("s2", 2, score=97.0)])

    fused = reciprocal_rank_fusion({"dense": dense, "bm25": sparse}, k=60)
    assert fused[0].chunk_id == "shared"
    # d1 (rank 1, one list) and s2 (rank 2, one list) keep their rank-based order.
    assert [r.chunk_id for r in fused[1:]] == ["d1", "s2"]


def test_rrf_deduplicates_across_retrievers():
    dense = renumber([_result("x", 1), _result("y", 2)])
    sparse = renumber([_result("x", 1), _result("y", 2)])
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": sparse})
    assert len({r.chunk_id for r in fused}) == len(fused) == 2


def test_rrf_records_each_method_contribution():
    # Ranks set directly -- renumber() would reassign them by list position.
    dense = [_result("x", 1, score=0.9)]
    sparse = [_result("x", 3, score=14.2)]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": sparse})

    comp = fused[0].components
    assert comp["dense"] == pytest.approx(0.9)
    assert comp["bm25"] == pytest.approx(14.2)
    assert comp["dense_rank"] == 1 and comp["bm25_rank"] == 3


def test_rrf_respects_top_k_and_renumbers():
    dense = renumber([_result(f"c{i}", i) for i in range(1, 11)])
    fused = reciprocal_rank_fusion({"dense": dense}, top_k=3)
    assert len(fused) == 3
    assert [r.rank for r in fused] == [1, 2, 3]
    assert all(r.method == "hybrid" for r in fused)


def test_rrf_with_one_empty_list_still_works():
    dense = renumber([_result("x", 1)])
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": []})
    assert [r.chunk_id for r in fused] == ["x"]


def test_rrf_with_no_results_returns_empty():
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []


# --------------------------------------------------------------------------
# Vector store (in-memory Chroma, stub embeddings)
# --------------------------------------------------------------------------

@pytest.fixture
def store(monkeypatch):
    """A VectorStore backed by an ephemeral Chroma client and a toy encoder.

    The stub maps text to a 3-d vector by keyword, so nearest-neighbour ordering is
    predictable without loading the real model.
    """
    chromadb = pytest.importorskip("chromadb")
    from src.retrieval import vector_store as vs
    import uuid

    def fake_vector(text: str) -> list[float]:
        t = text.lower()
        return [
            1.0 if "metformin" in t else 0.0,
            1.0 if "warfarin" in t else 0.0,
            1.0 if "bleeding" in t else 0.0,
        ]

    monkeypatch.setattr(vs.encoder, "embed_query", lambda q: fake_vector(q))
    # A fresh collection name per test: chromadb reuses in-process clients, so a
    # fixed name leaks vectors from one test into the next.
    return vs.VectorStore(collection_name=f"test_{uuid.uuid4().hex}",
                          client=chromadb.EphemeralClient())


def test_vector_store_roundtrips_metadata(store):
    chunks = [
        _chunk("a::0::0", "Metformin causes diarrhea.", drugs=["Metformin", "Glucophage"]),
        _chunk("b::0::0", "Warfarin raises bleeding risk.", section="Warnings",
               title="FDA Label: Warfarin", drugs=["Warfarin"]),
    ]
    store.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    assert store.count() == 2

    results = store.search("metformin", top_k=1)
    assert len(results) == 1

    got = results[0].chunk
    assert got.chunk_id == "a::0::0"
    assert got.section == "Adverse Reactions"
    assert got.title == "FDA Label: Metformin"
    assert got.drug_names == ["Metformin", "Glucophage"], "list metadata must survive"
    assert got.text == "Metformin causes diarrhea."
    assert results[0].method == "dense"


def test_vector_store_scores_are_similarities_not_distances(store):
    store.add([_chunk("a::0::0", "Metformin causes diarrhea.")], [[1.0, 0.0, 0.0]])
    r = store.search("metformin", top_k=1)[0]
    assert r.score == pytest.approx(1.0, abs=1e-5), "identical vectors -> similarity 1"


def test_vector_store_rejects_mismatched_embeddings(store):
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        store.add([_chunk("a::0::0", "x"), _chunk("b::0::0", "y")], [[1.0, 0.0, 0.0]])


def test_vector_store_empty_search_is_safe(store):
    assert store.search("metformin") == []


def test_append_embeds_only_new_chunks(store, monkeypatch):
    from src.retrieval import build_index
    embedded = []

    def fake_embed(chunks, **kw):
        embedded.extend(c.chunk_id for c in chunks)
        return [[0.0, 0.0, 1.0]] * len(chunks)

    monkeypatch.setattr(build_index.encoder, "embed_chunks", fake_embed)
    old = [_chunk("a::0::0", "Metformin causes diarrhea.")]
    store.add(old, [[1.0, 0.0, 0.0]])
    new = [_chunk("medlineplus:1::0::0", "Malaria is caused by a parasite.")]

    build_index.append(chunks=old + new, store=store)
    assert embedded == ["medlineplus:1::0::0"]       # old chunk not re-embedded
    assert store.count() == 2
    assert store.ids() == {"a::0::0", "medlineplus:1::0::0"}

    build_index.append(chunks=old + new, store=store)  # idempotent
    assert store.count() == 2 and len(embedded) == 1


def test_vector_store_reset_clears_stale_vectors(store):
    store.add([_chunk("a::0::0", "Metformin causes diarrhea.")], [[1.0, 0.0, 0.0]])
    store.reset()
    assert store.count() == 0


# --------------------------------------------------------------------------
# Corpus quality: label selection
# --------------------------------------------------------------------------

def test_single_ingredient_label_beats_combination_product():
    plain = {"openfda": {"generic_name": ["METFORMIN HYDROCHLORIDE"]}}
    combo = {"openfda": {"generic_name": ["SITAGLIPTIN AND METFORMIN HYDROCHLORIDE"]}}
    assert _specificity(plain, "metformin") < _specificity(combo, "metformin")


def test_exact_generic_name_wins():
    exact = {"openfda": {"generic_name": ["METFORMIN"]}}
    salt = {"openfda": {"generic_name": ["METFORMIN HYDROCHLORIDE"]}}
    assert _specificity(exact, "metformin") < _specificity(salt, "metformin")
