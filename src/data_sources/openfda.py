"""Fetch structured FDA drug label data from the openFDA API.

openFDA exposes SPL (Structured Product Labeling) sections as named JSON fields --
`warnings`, `drug_interactions`, `dosage_and_administration`, `contraindications`
and so on. Those field names ARE the clinical section headings, which is exactly
what the chunker needs in Phase 2 to keep "Contraindications" attached to its text.

No API key required for modest volumes. Docs: https://open.fda.gov/apis/drug/label/
"""

from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.data_sources.http import get

LABEL_ENDPOINT = "https://api.fda.gov/drug/label.json"

# Label sections worth retrieving. Ordered roughly as they appear on a real label.
LABEL_SECTIONS = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "boxed_warning",
    "drug_interactions",
    "adverse_reactions",
    "use_in_specific_populations",
    "clinical_pharmacology",
    "description",
]


def _specificity(record: dict, drug: str) -> tuple[int, int]:
    """Rank a candidate label by how specifically it is *about* `drug`.

    A search for "metformin" matches combination products such as "SITAGLIPTIN AND
    METFORMIN HYDROCHLORIDE" just as happily as the single-ingredient label, and
    openFDA returns them in no particular order. A combination label is the wrong
    evidence for both "side effects of metformin" and for drug-interaction lookups,
    where the whole point is to reason about one drug at a time.

    Sorts best-first: exact generic-name match, then fewest active ingredients,
    then shortest name.
    """
    names = [n.lower() for n in record.get("openfda", {}).get("generic_name", [])]
    target = drug.lower()

    if any(n == target for n in names):
        tier = 0                                   # exact match
    elif any(n.startswith(target) for n in names):
        tier = 1                                   # "metformin hydrochloride"
    elif any(" and " in n for n in names):
        tier = 3                                   # combination product
    else:
        tier = 2

    shortest = min((len(n) for n in names), default=999)
    return (tier, shortest)


def fetch_label(drug: str, limit: int = 1) -> list[dict]:
    """Fetch the most drug-specific label available, by generic then brand name."""
    for field in ("openfda.generic_name", "openfda.brand_name"):
        try:
            resp = get(
                LABEL_ENDPOINT,
                # Over-fetch so there is something to choose between.
                params={"search": f'{field}:"{drug}"', "limit": max(limit, 10)},
                min_interval=0.3,
            )
        except RuntimeError:
            continue
        results = resp.json().get("results", [])
        if results:
            results.sort(key=lambda r: _specificity(r, drug))
            return results[:limit]
    return []


def harvest(drugs: list[str], out_dir: Path | None = None) -> list[Path]:
    """Fetch one label per drug and save the raw JSON. Returns files written."""
    out_dir = out_dir or config.RAW_FDA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for drug in drugs:
        results = fetch_label(drug)
        if not results:
            print(f"  [skip] no FDA label found: {drug}")
            continue
        slug = "".join(c if c.isalnum() else "_" for c in drug.lower()).strip("_")
        path = out_dir / f"fda__{slug}.json"
        path.write_text(json.dumps(results[0], indent=2), encoding="utf-8")
        written.append(path)
        present = [s for s in LABEL_SECTIONS if s in results[0]]
        print(f"  [ok] {drug} -> {len(present)} sections -> {path.name}")

    return written
