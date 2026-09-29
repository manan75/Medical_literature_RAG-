"""Phase 6 tests: grounded generation with a fake LLM provider (no network).

What is under test: the not-found path never calls the LLM, the prompt carries
numbered, citable context and the safety rules, sources line up with the numbers,
and the confidence label follows its documented rule.
"""

from __future__ import annotations

from src import config
from src.generation.answer import (
    NOT_FOUND_MESSAGE, NOT_FOUND_SENTINEL, SYSTEM_PROMPT, build_prompt,
    confidence, generate_answer)
from src.retrieval.types import RetrievalResult, renumber
from src.schema import Chunk

T = config.NOT_FOUND_THRESHOLD
HIGH = config.HIGH_SCORE


class FakeProvider:
    name = "fake"

    def __init__(self, reply: str = "Metformin commonly causes diarrhoea [1]."):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, prompt: str, system: str = "") -> str:
        self.calls.append((prompt, system))
        return self.reply


def _r(cid: str, score: float, doc: str = "fda:met",
       section: str = "Adverse Reactions") -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(chunk_id=cid, doc_id=doc, text=f"text of {cid}",
                    section=section, source="fda_label",
                    title="FDA Label: Metformin", url=f"https://x.test/{doc}",
                    date="2024-01-01"),
        score=score, method="reranked")


def test_below_threshold_returns_not_found_without_calling_llm():
    llm = FakeProvider()
    ans = generate_answer("q", renumber([_r("a", T - 1), _r("b", T - 5)]), llm)
    assert not ans.found
    assert ans.text == NOT_FOUND_MESSAGE
    assert ans.sources == []
    assert llm.calls == []


def test_only_chunks_above_threshold_reach_the_llm():
    llm = FakeProvider()
    ans = generate_answer("q", renumber([_r("good", T + 4), _r("junk", T - 3)]), llm)
    prompt, _ = llm.calls[0]
    assert "text of good" in prompt and "text of junk" not in prompt
    assert [s.number for s in ans.sources] == [1]


def test_exactly_one_llm_call_and_sources_match_prompt_numbers():
    llm = FakeProvider("Answer [1][2].")
    results = renumber([_r("a", HIGH + 1, "fda:met"),
                        _r("b", HIGH, "pubmed:1", "Results")])
    ans = generate_answer("side effects of metformin?", results, llm)

    assert len(llm.calls) == 1
    prompt, system = llm.calls[0]
    assert "[1] FDA Label: Metformin [Adverse Reactions]" in prompt
    assert "[2] FDA Label: Metformin [Results]" in prompt
    assert system == SYSTEM_PROMPT
    assert [(s.number, s.section, s.url) for s in ans.sources] == [
        (1, "Adverse Reactions", "https://x.test/fda:met"),
        (2, "Results", "https://x.test/pubmed:1")]
    assert [r.chunk_id for r in ans.chunks] == ["a", "b"]
    assert ans.found


def test_system_prompt_forbids_diagnosis_prescribing_and_safe_claims():
    s = SYSTEM_PROMPT.lower()
    assert "only from the numbered context" in s
    assert "do not diagnose" in s
    assert '"safe"' in s
    assert NOT_FOUND_SENTINEL in SYSTEM_PROMPT


def test_model_saying_not_found_is_mapped_to_not_found():
    ans = generate_answer("q", renumber([_r("a", HIGH)]),
                          FakeProvider(NOT_FOUND_SENTINEL))
    assert not ans.found and ans.text == NOT_FOUND_MESSAGE


def test_uncited_answer_is_downgraded_to_low_confidence():
    results = renumber([_r("a", HIGH + 2, "d1"), _r("b", HIGH + 1, "d2")])
    ans = generate_answer("q", results, FakeProvider("An answer with no refs."))
    assert ans.confidence == "Low"
    assert "no citations" in ans.confidence_reason


def test_confidence_high_needs_strong_score_and_two_agreeing_documents():
    assert confidence([_r("a", HIGH + 1, "d1"), _r("b", HIGH, "d2")])[0] == "High"


def test_confidence_medium_for_strong_single_source():
    level, why = confidence([_r("a", HIGH + 1, "d1"), _r("b", HIGH + 0.5, "d1")])
    assert level == "Medium"
    assert "1 distinct source" in why


def test_confidence_low_for_weak_single_source():
    far = HIGH - 1 - config.CONFIDENCE_SPREAD - 1
    assert confidence([_r("a", HIGH - 1, "d1"), _r("b", far, "d2")])[0] == "Low"


def test_prompt_contains_question_and_every_chunk():
    p = build_prompt("Why?", renumber([_r("a", 1), _r("b", 1)]))
    assert "Question: Why?" in p and "text of a" in p and "text of b" in p
