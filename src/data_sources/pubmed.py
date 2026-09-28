"""Fetch biomedical literature from NCBI via the E-utilities API.

Two databases are used:
  * pubmed -- abstracts for any query (always available)
  * pmc    -- full text (JATS XML) for the open-access subset only

Raw responses are written verbatim to data/raw/pubmed/ so that ingestion (Phase 1)
is a pure, re-runnable transform over files on disk rather than over the network.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.data_sources.http import get

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _auth_params() -> dict:
    p = {"tool": config.NCBI_TOOL}
    if config.NCBI_EMAIL:
        p["email"] = config.NCBI_EMAIL
    if config.NCBI_API_KEY:
        p["api_key"] = config.NCBI_API_KEY
    return p


def _interval() -> float:
    # NCBI allows 10 req/s with a key, 3 req/s without.
    return 0.11 if config.NCBI_API_KEY else 0.34


def search(query: str, db: str = "pubmed", retmax: int = 20) -> list[str]:
    """Return a list of record IDs (PMIDs for db=pubmed, PMCIDs for db=pmc)."""
    resp = get(
        f"{EUTILS}/esearch.fcgi",
        params={**_auth_params(), "db": db, "term": query,
                "retmax": retmax, "retmode": "json", "sort": "relevance"},
        min_interval=_interval(),
    )
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_xml(ids: list[str], db: str = "pubmed") -> str:
    """Fetch full records as XML for a batch of IDs."""
    if not ids:
        return ""
    resp = get(
        f"{EUTILS}/efetch.fcgi",
        params={**_auth_params(), "db": db, "id": ",".join(ids), "retmode": "xml"},
        min_interval=_interval(),
    )
    return resp.text


def harvest(queries: list[str], per_query: int = 10, db: str = "pubmed",
            out_dir: Path | None = None) -> list[Path]:
    """Run each query, fetch the matching records, and save one XML file per query.

    Returns the list of files written.
    """
    out_dir = out_dir or config.RAW_PUBMED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for query in queries:
        ids = search(query, db=db, retmax=per_query)
        if not ids:
            print(f"  [skip] no results: {query!r}")
            continue
        xml = fetch_xml(ids, db=db)
        slug = "".join(c if c.isalnum() else "_" for c in query)[:60].strip("_")
        path = out_dir / f"{db}__{slug}.xml"
        path.write_text(xml, encoding="utf-8")
        # Sidecar manifest records provenance: which query produced which IDs.
        path.with_suffix(".meta.json").write_text(
            json.dumps({"query": query, "db": db, "ids": ids}, indent=2),
            encoding="utf-8",
        )
        written.append(path)
        print(f"  [ok] {query!r} -> {len(ids)} records -> {path.name}")

    return written
