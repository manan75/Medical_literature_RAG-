"""Phase 5 tests: cross-encoder reranking.

The cross-encoder itself is stubbed. What is under test is the plumbing around
it: that reordering is correct, that the pre-rerank score and rank survive for
inspection, and that the chunk (and therefore the citation) is never disturbed.
"""

from __future__ import annotations

import pytest

from src.retrieval import rerank as rerank_mod
from src.retrieval.rerank import rerank
from src.retrieval.types import RetrievalResult, renumber
from src.schema import Chunk


def _result(cid: str, score: float, section: str = "Adverse Reactions"
            ) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=cid, doc_id="doc:1", text=f"body of {cid}",
                    section=section, source="fda_label",
                    title="FDA Label: Metformin", url="http://example.test",
                    date="2024-01-01", token_estimate=3),
        score=score, method="hybrid",
    )


class _StubModel:
    """Scores each pair from a lookup keyed by the chunk id in the passage text."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs, **kwargs):
        self.seen = list(pairs)
        out = []
        for _query, passage in pairs:
            out.append(next((v for k, v in self.scores.items() if k in passage), 0.0))
        return out


@pytest.fixture
def stub(monkeypatch):
    def install(scores: dict[str, float]) -> _StubModel:
        model = _StubModel(scores)
        monkeypatch.setattr(rerank_mod, "get_model", lambda *a, **k: model)
        return model
    return install


def test_rerank_reorders_by_cross_encoder_score(stub):
    stub({"body of a": 0.1, "body of b": 9.5, "body of c": 4.0})
    results = renumber([_result("a", 0.9), _result("b", 0.5), _result("c", 0.2)])

    out = rerank("metformin side effects", results, top_k=3)
    assert [r.chunk_id for r in out] == ["b", "c", "a"]
    assert [r.rank for r in out] == [1, 2, 3]
    assert all(r.method == "reranked" for r in out)


def test_rerank_truncates_to_top_k(stub):
    stub({f"body of c{i}": float(i) for i in range(6)})
    results = renumber([_result(f"c{i}", 0.5) for i in range(6)])

    out = rerank("q", results, top_k=2)
    assert len(out) == 2
    assert [r.chunk_id for r in out] == ["c5", "c4"]


def test_rerank_preserves_the_original_retrieval_score_and_rank(stub):
    stub({"body of a": 0.1, "body of b": 9.5})
    results = renumber([_result("a", 0.87), _result("b", 0.42)])

    out = rerank("q", results, top_k=2)
    promoted = out[0]
    assert promoted.chunk_id == "b"
    assert promoted.score == pytest.approx(9.5)
    assert promoted.components["retrieval_score"] == pytest.approx(0.42)
    assert promoted.components["retrieval_rank"] == 2, "the move must stay inspectable"
    assert promoted.components["rerank"] == pytest.approx(9.5)


def test_rerank_sees_the_section_heading(stub):
    """The heading disambiguates near-identical text, so the reranker must see it."""
    model = stub({"body of a": 1.0})
    rerank("q", renumber([_result("a", 0.5, section="Drug Interactions")]), top_k=1)
    # A single result short-circuits, so use two to reach the model.
    model = stub({"body of a": 1.0, "body of b": 0.5})
    rerank("q", renumber([_result("a", 0.5, section="Drug Interactions"),
                          _result("b", 0.4)]), top_k=2)
    _query, passage = model.seen[0]
    assert "Drug Interactions" in passage
    assert "FDA Label: Metformin" in passage


def test_rerank_never_mutates_the_chunk(stub):
    stub({"body of a": 1.0, "body of b": 2.0})
    results = renumber([_result("a", 0.5), _result("b", 0.4)])

    for r in rerank("q", results, top_k=2):
        assert r.chunk.text.startswith("body of ")
        assert r.chunk.title == "FDA Label: Metformin"
        assert "FDA Label: Metformin" in r.chunk.citation()


def test_rerank_of_empty_list_is_empty(stub):
    assert rerank("q", [], top_k=5) == []


def test_rerank_of_single_result_skips_the_model(monkeypatch):
    """One candidate has nothing to reorder -- loading a 90 MB model would be waste."""
    def explode(*a, **k):
        raise AssertionError("the model must not be loaded for a single candidate")
    monkeypatch.setattr(rerank_mod, "get_model", explode)

    out = rerank("q", renumber([_result("a", 0.9)]), top_k=5)
    assert len(out) == 1 and out[0].method == "reranked"
