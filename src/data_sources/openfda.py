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


def fetch_label(drug: str, limit: int = 1) -> list[dict]:
    """Fetch label record(s) for a drug by generic name, falling back to brand name."""
    for field in ("openfda.generic_name", "openfda.brand_name"):
        try:
            resp = get(
                LABEL_ENDPOINT,
                params={"search": f'{field}:"{drug}"', "limit": limit},
                min_interval=0.3,
            )
        except RuntimeError:
            continue
        results = resp.json().get("results", [])
        if results:
            return results
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
