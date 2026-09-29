"""Phase 1 entry point (network side): build the raw corpus on disk.

    python -m src.data_sources.harvest            # full corpus from corpus_spec
    python -m src.data_sources.harvest --small    # quick subset, for smoke tests
    python -m src.data_sources.harvest --medlineplus-only   # just MedlinePlus

Deliberately separate from ingestion so that parser changes never require
re-hitting NCBI or openFDA.
"""

from __future__ import annotations

import argparse

from src import config
from src.data_sources import corpus_spec, medlineplus, openfda, pubmed


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest the raw medical corpus.")
    ap.add_argument("--small", action="store_true",
                    help="Fetch a small subset (4 drugs, 3 queries) for a smoke test.")
    ap.add_argument("--per-query", type=int, default=12,
                    help="PubMed records to fetch per query (default: 12).")
    ap.add_argument("--skip-pmc", action="store_true",
                    help="Skip PMC full-text harvesting (it is the slowest step).")
    ap.add_argument("--medlineplus-only", action="store_true",
                    help="Fetch only MedlinePlus. PubMed/openFDA results are live, so "
                         "re-harvesting them changes the corpus under an existing index.")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.medlineplus_only:
        medlineplus.harvest()
        print("Next: python -m src.ingestion.pipeline")
        return

    drugs = corpus_spec.DRUGS[:4] if args.small else corpus_spec.DRUGS
    queries = corpus_spec.PUBMED_QUERIES[:3] if args.small else corpus_spec.PUBMED_QUERIES
    per_query = 3 if args.small else args.per_query

    if not config.NCBI_API_KEY:
        print("note: no NCBI_API_KEY set -- limited to 3 requests/sec.\n")

    print(f"[1/3] FDA drug labels ({len(drugs)} drugs)")
    fda_files = openfda.harvest(drugs)

    print(f"\n[2/3] PubMed abstracts ({len(queries)} queries x {per_query})")
    pm_files = pubmed.harvest(queries, per_query=per_query, db="pubmed")

    pmc_files = []
    if not args.skip_pmc and not args.small:
        print(f"\n[3/3] PMC full text ({len(corpus_spec.PMC_QUERIES)} queries)")
        pmc_files = pubmed.harvest(corpus_spec.PMC_QUERIES, per_query=3, db="pmc")
    else:
        print("\n[3/3] PMC full text -- skipped")

    print("\n[+] MedlinePlus health topics")
    medlineplus.harvest()

    print(f"\nHarvest complete: {len(fda_files)} FDA labels, "
          f"{len(pm_files)} PubMed batches, {len(pmc_files)} PMC batches, "
          f"MedlinePlus topics.")
    print("Next: python -m src.ingestion.pipeline")


if __name__ == "__main__":
    main()
