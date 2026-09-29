"""Phase 6: grounded answer generation -- the core RAG loop.

    python -m src.generation.answer "What are the common side effects of metformin?"

Flow for one question:
    rerank-retrieve top chunks
      -> drop chunks below the relevance threshold
      -> none left?  return "not found in corpus" WITHOUT calling the LLM
      -> number the chunks [1]..[n], ask the LLM to answer only from them
      -> return answer + numbered source list + the chunks + a confidence label

Exactly one LLM call per question (plus retries of that same call on 429/503).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from src import config
from src.generation.providers import LLMProvider
from src.retrieval.types import RetrievalResult

NOT_FOUND_MESSAGE = (
    "I could not find information about this in the indexed medical literature "
    "and drug labels, so I will not attempt an answer. Try rephrasing, or ask "
    "about a drug or condition covered by the corpus."
)
# The model replies with exactly this when the context does not answer the question.
NOT_FOUND_SENTINEL = "NOT_FOUND"

SYSTEM_PROMPT = f"""You are a medical literature retrieval assistant for education \
and information only.

Rules you must follow:
1. Answer ONLY from the numbered context passages supplied. Do not use outside \
knowledge, even if you are confident.
2. Cite every factual sentence with the passage number(s) it came from, like [1] or \
[2][3]. Never cite a number that is not in the context.
3. If the passages do not contain the answer, reply with exactly {NOT_FOUND_SENTINEL} \
and nothing else.
4. Do not diagnose, do not recommend or adjust doses for a specific person, and do \
not tell anyone to start, stop or combine medications. Only if the question asks \
for that kind of personal advice, say you cannot give it, then summarise what the \
sources say in general terms and suggest consulting a healthcare professional. \
General questions about a drug's effects, side effects or interactions are not \
personal advice: answer them directly without a disclaimer.
5. Never state that a drug or drug combination is "safe". If the passages do not \
describe an interaction, say that no interaction evidence was found in the \
provided sources.
6. If passages disagree, say so and cite both sides.
7. Be concise: a short paragraph or a few bullet points."""


@dataclass
class Source:
    number: int
    title: str
    section: str
    url: str
    citation: str


@dataclass
class Answer:
    question: str
    text: str
    found: bool
    confidence: str                 # "High" | "Medium" | "Low" | "None"
    confidence_reason: str
    sources: list[Source] = field(default_factory=list)
    chunks: list[RetrievalResult] = field(default_factory=list)


def build_prompt(question: str, results: list[RetrievalResult]) -> str:
    blocks = [f"[{i}] {r.chunk.citation()}\n{r.chunk.text.strip()}"
              for i, r in enumerate(results, start=1)]
    return ("Context passages:\n\n" + "\n\n---\n\n".join(blocks)
            + f"\n\nQuestion: {question}\n\nAnswer (with [n] citations):")


def confidence(results: list[RetrievalResult]) -> tuple[str, str]:
    """High / Medium / Low from two explainable signals:

    - strength: is the best reranker score clearly relevant (>= HIGH_SCORE)?
    - agreement: how many *distinct documents* have a chunk scoring close to the
      best one (within CONFIDENCE_SPREAD)? One label saying it is weaker evidence
      than a label and a paper saying it.
    """
    if not results:
        return "None", "No relevant passages were retrieved."
    top = max(r.score for r in results)
    agreeing_docs = {r.chunk.doc_id for r in results
                     if r.score >= top - config.CONFIDENCE_SPREAD}
    n = len(agreeing_docs)
    strong = top >= config.HIGH_SCORE
    band = ("strong" if strong else
            "moderate" if top > config.NOT_FOUND_THRESHOLD else "weak")
    why = (f"best reranker score {top:.1f} ({band}); "
           f"{n} distinct source document(s) score within "
           f"{config.CONFIDENCE_SPREAD:g} of it")
    if strong and n >= 2:
        return "High", why
    if strong or n >= 2:
        return "Medium", why
    return "Low", why


def generate_answer(question: str, results: list[RetrievalResult],
                    provider: LLMProvider, apply_threshold: bool = True,
                    not_found_message: str = NOT_FOUND_MESSAGE) -> Answer:
    """Turn already-retrieved chunks into a grounded, cited answer."""
    if apply_threshold:
        results = [r for r in results if r.score >= config.NOT_FOUND_THRESHOLD]
    if not results:
        return Answer(question, not_found_message, found=False,
                      confidence="None",
                      confidence_reason=(
                          f"No passage scored above the relevance threshold "
                          f"({config.NOT_FOUND_THRESHOLD:g}); the LLM was not called."))

    text = provider.generate(build_prompt(question, results), system=SYSTEM_PROMPT)

    if not text or text.strip().strip(".") == NOT_FOUND_SENTINEL:
        return Answer(question, not_found_message, found=False, confidence="None",
                      confidence_reason="Passages were retrieved, but the model "
                                        "judged that none of them answer the question.",
                      chunks=results)

    sources = [Source(i, r.chunk.title, r.chunk.section, r.chunk.url,
                      r.chunk.citation())
               for i, r in enumerate(results, start=1)]
    level, why = confidence(results)

    # Accept [1], [1][2] and [1, 2] -- models use all three.
    cited = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", text)
             for n in re.findall(r"\d+", group)}
    if not cited & set(range(1, len(results) + 1)):
        # Grounding check: an answer that cites nothing is not trusted.
        level, why = "Low", why + "; the answer contains no citations"

    return Answer(question, text, found=True, confidence=level,
                  confidence_reason=why, sources=sources, chunks=results)


def answer_question(question: str, retriever, provider: LLMProvider) -> Answer:
    """Retrieve (hybrid + rerank) then generate. `retriever` is a RerankingRetriever."""
    return generate_answer(question, retriever.search(question), provider)


def format_answer(ans: Answer) -> str:
    lines = [ans.text, "", f"Confidence: {ans.confidence} -- {ans.confidence_reason}"]
    if ans.sources:
        lines += ["", "Sources:"]
        lines += [f"  [{s.number}] {s.citation}" + (f"\n      {s.url}" if s.url else "")
                  for s in ans.sources]
    lines += ["", "For information and education only -- not medical advice."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask a grounded medical question.")
    ap.add_argument("question", nargs="+")
    args = ap.parse_args()

    from src.generation.providers import GeminiProvider, LLMError
    from src.retrieval.rerank import RerankingRetriever

    try:
        ans = answer_question(" ".join(args.question), RerankingRetriever(),
                              GeminiProvider())
    except LLMError as e:
        raise SystemExit(f"Error: {e}")
    print("\n" + format_answer(ans))


if __name__ == "__main__":
    main()
