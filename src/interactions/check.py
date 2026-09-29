"""Phase 7 (minimal): drug-drug interaction evidence lookup.

    python -m src.interactions.check warfarin aspirin

Evidence rule (deliberately strict and easy to explain): a chunk counts as
interaction evidence only if it mentions BOTH drugs -- or it is one drug's FDA
label and it names the other drug. Two sources of candidates:

  (a) the two FDA labels, scanned directly: Drug Interactions section first, then
      the rest of the label (sertraline's label names tramadol under Warnings,
      not under Drug Interactions);
  (b) hybrid retrieval over the whole corpus, so PubMed/PMC papers can contribute.

Candidates are reranked against the question and passed to the Phase 6 generation
path, which cites them. If there are none, the LLM is not called and the answer
says "No interaction evidence was found in our corpus" -- never that the
combination is safe. Absence of evidence in 169 documents is not evidence of safety.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from src.data_sources.corpus_spec import DRUGS
from src.generation.answer import Answer, generate_answer
from src.retrieval.rerank import rerank
from src.retrieval.types import RetrievalResult
from src.schema import Chunk

NO_EVIDENCE = "No interaction evidence was found in our corpus"
MAX_EVIDENCE = 6


def no_evidence_message(a: str, b: str) -> str:
    return (f"{NO_EVIDENCE} for {a} and {b}. This does NOT mean the combination "
            f"is safe: the corpus covers only {len(DRUGS)} FDA labels and a small "
            f"set of papers. Check a full interaction reference or ask a "
            f"pharmacist.")


@dataclass
class InteractionReport:
    drug_a: str
    drug_b: str
    answer: Answer
    # One line per label scan, e.g. "Warfarin label names aspirin in: Drug Interactions"
    label_findings: list[str] = field(default_factory=list)


def _mentions(text: str, drug: str) -> bool:
    return re.search(rf"\b{re.escape(drug)}\b", text, re.I) is not None


class InteractionChecker:
    def __init__(self, chunks: list[Chunk], retriever, provider):
        """`chunks`: the full chunk list; `retriever`: a RerankingRetriever."""
        self.chunks = chunks
        self.retriever = retriever
        self.provider = provider
        self.labels: dict[str, list[Chunk]] = {}   # generic name -> its label chunks
        self.aliases: dict[str, str] = {g: g for g in DRUGS}
        for c in chunks:
            if c.source != "fda_label":
                continue
            generic = next((g for g in DRUGS if _mentions(c.title, g)), None)
            if generic:
                self.labels.setdefault(generic, []).append(c)
                for name in c.drug_names:
                    self.aliases.setdefault(name.lower(), generic)

    def normalize(self, name: str) -> str | None:
        """Map user input to a corpus drug: exact alias ("warfarin sodium",
        "low dose aspirin"), else a generic name contained in the input
        ("Tramadol HCl 50 mg" -> tramadol). None if the drug is not covered."""
        key = " ".join(name.lower().split())
        if key in self.aliases:
            return self.aliases[key]
        return next((g for g in DRUGS if _mentions(key, g)), None)

    def _label_hits(self, a: str, b: str) -> tuple[list[Chunk], list[str]]:
        hits, findings = [], []
        for x, y in ((a, b), (b, a)):
            label = self.labels.get(x, [])
            if not label:
                findings.append(f"{x.title()}: no FDA label in corpus")
                continue
            found = [c for c in label if _mentions(c.text, y)]
            # Drug Interactions first: it is the section written for this question.
            found.sort(key=lambda c: c.section != "Drug Interactions")
            sections = list(dict.fromkeys(c.section for c in found))
            findings.append(
                f"{x.title()} label names {y} in: {', '.join(sections)}" if found
                else f"{x.title()} label does not mention {y}")
            hits += found
        return hits, findings

    def check(self, drug_a: str, drug_b: str) -> InteractionReport:
        a, b = self.normalize(drug_a), self.normalize(drug_b)
        missing = [raw for raw, n in ((drug_a, a), (drug_b, b)) if n is None]
        if missing:
            msg = (f"Not in our corpus: {', '.join(missing)}. This tool only covers "
                   f"these drugs: {', '.join(DRUGS)}.")
            return InteractionReport(drug_a, drug_b, Answer(
                f"{drug_a} + {drug_b}", msg, found=False, confidence="None",
                confidence_reason="Drug not covered by the corpus; nothing retrieved."))
        if a == b:
            return InteractionReport(a, b, Answer(
                f"{a} + {b}", "Please enter two different drugs.", found=False,
                confidence="None", confidence_reason="Same drug entered twice."))

        question = (f"Can {a} interact with {b}? Describe any interaction, its "
                    f"mechanism and consequences as stated in the sources.")
        label_chunks, findings = self._label_hits(a, b)

        # Literature: any retrieved chunk that names both drugs.
        retrieved = self.retriever.hybrid.search(f"{a} {b} interaction")
        both = [r.chunk for r in retrieved
                if _mentions(f"{r.chunk.title} {r.chunk.text}", a)
                and _mentions(f"{r.chunk.title} {r.chunk.text}", b)]

        unique = list({c.chunk_id: c for c in label_chunks + both}.values())
        if not unique:
            return InteractionReport(a, b, Answer(
                question, no_evidence_message(a, b), found=False, confidence="None",
                confidence_reason="No chunk in the corpus mentions both drugs; "
                                  "the LLM was not called."), findings)

        evidence = rerank(question, [RetrievalResult(c, 0.0) for c in unique],
                          top_k=MAX_EVIDENCE)
        # The mention rule already decided relevance, so no score threshold here.
        ans = generate_answer(question, evidence, self.provider,
                              apply_threshold=False,
                              not_found_message=no_evidence_message(a, b))
        return InteractionReport(a, b, ans, findings)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check two drugs for interaction evidence.")
    ap.add_argument("drug_a")
    ap.add_argument("drug_b")
    args = ap.parse_args()

    from src.generation.answer import format_answer
    from src.generation.providers import GeminiProvider, LLMError
    from src.retrieval.rerank import RerankingRetriever

    retriever = RerankingRetriever()
    try:
        checker = InteractionChecker(retriever.hybrid.bm25.chunks, retriever,
                                     GeminiProvider())
        report = checker.check(args.drug_a, args.drug_b)
    except LLMError as e:
        raise SystemExit(f"Error: {e}")
    print(f"\n{report.drug_a} + {report.drug_b}")
    for f in report.label_findings:
        print(f"  - {f}")
    print("\n" + format_answer(report.answer))


if __name__ == "__main__":
    main()
