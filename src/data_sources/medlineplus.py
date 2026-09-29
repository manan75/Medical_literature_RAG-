"""Download the MedlinePlus Health Topic XML (NLM), the plain-language disease source.

PubMed and FDA labels are written for clinicians; a lay question such as "what are
the symptoms of malaria?" had nothing to match. MedlinePlus health topic summaries
fill that gap and are public domain (medlineplus.gov/about/using/usingcontent/).
Only those summaries are ingested -- see src/ingestion/medlineplus_parser.py; the
copyrighted A.D.A.M. encyclopedia and ASHP drug monographs are not in this file's
scope and are never fetched.

NLM regenerates the file daily (Tue-Sat) under a dated name, so the current URL is
discovered from the index page rather than hard-coded. It is saved under a fixed
name so ingestion never sees two versions at once.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src import config
from src.data_sources.http import get

INDEX_URL = "https://medlineplus.gov/xml.html"
_ZIP_LINK = re.compile(
    r'https://medlineplus\.gov/xml/mplus_topics_compressed_(\d{4}-\d{2}-\d{2})\.zip')
ZIP_NAME = "mplus_topics_compressed.zip"


def latest_zip_url(index_html: str) -> tuple[str, str]:
    """Return (url, generation date) of the newest compressed topics file listed."""
    found = {m.group(1): m.group(0) for m in _ZIP_LINK.finditer(index_html)}
    if not found:
        raise RuntimeError(f"No compressed health topic XML link found on {INDEX_URL}")
    newest = max(found)                     # ISO dates sort correctly as strings
    return found[newest], newest


def harvest(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.RAW_MEDLINEPLUS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    url, generated = latest_zip_url(get(INDEX_URL).text)
    data = get(url, timeout=120).content
    path = out_dir / ZIP_NAME
    path.write_bytes(data)
    path.with_suffix(".meta.json").write_text(json.dumps({
        "source_url": url,
        "generated": generated,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "licence": "Health topic summaries: public domain (NLM). "
                   "Credit: Source: MedlinePlus, National Library of Medicine.",
    }, indent=2), encoding="utf-8")
    print(f"  [ok] MedlinePlus topics {generated} -> {path.name} "
          f"({len(data) / 1e6:.1f} MB)")
    return path
