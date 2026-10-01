"""Turn an openFDA drug-label JSON record into a Document.

openFDA returns each SPL section as a list of long strings keyed by the section
name. That mapping is the whole reason FDA labels are valuable here: the section
boundary is given to us, so "Drug Interactions" text never bleeds into "Dosage".
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data_sources.openfda import LABEL_SECTIONS
from src.ingestion.normalize import normalize_text, prettify_heading
from src.schema import Document, Section


def _iso_date(effective_time: str) -> str:
    """openFDA gives effective_time as YYYYMMDD; make it sortable and readable."""
    t = (effective_time or "").strip()
    if len(t) == 8 and t.isdigit():
        return f"{t[:4]}-{t[4:6]}-{t[6:]}"
    return t


def parse_label(record: dict) -> Document | None:
    openfda = record.get("openfda", {})
    set_id = record.get("set_id") or record.get("id") or ""
    if not set_id:
        return None

    generic = [n.title() for n in openfda.get("generic_name", [])]
    brand = [n.title() for n in openfda.get("brand_name", [])]
    drug_names = list(dict.fromkeys(generic + brand))
    primary = generic[0] if generic else (brand[0] if brand else "Unknown drug")
    title = f"FDA Label: {primary}"

    sections: list[Section] = []
    for field in LABEL_SECTIONS:
        value = record.get(field)
        if not value:
            continue
        raw = "\n\n".join(value) if isinstance(value, list) else str(value)
        text = normalize_text(raw)
        if text:
            sections.append(Section(heading=prettify_heading(field), text=text))

    if not sections:
        return None

    return Document(
        doc_id=f"fda:{set_id}",
        source="fda_label",
        title=title,
        sections=sections,
        url=f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}",
        date=_iso_date(record.get("effective_time", "")),
        drug_names=drug_names,
        extra={
            "manufacturer": openfda.get("manufacturer_name", []),
            "route": openfda.get("route", []),
            "substance_name": openfda.get("substance_name", []),
        },
    )


def parse_file(path: Path) -> list[Document]:
    record = json.loads(path.read_text(encoding="utf-8"))
    doc = parse_label(record)
    return [doc] if doc else []
