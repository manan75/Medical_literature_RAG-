"""Phase 1 entry point: every raw file in data/raw/ becomes a Document record.

This stage is a pure transform over the filesystem. Nothing here touches the
network, so it can be re-run freely after a parser change without re-harvesting.
Output: data/processed/documents.jsonl
"""

from __future__ import annotations

from pathlib import Path

from src import config
from src.ingestion import fda_parser, html_extract, pdf_extract, pubmed_parser
from src.schema import Document, write_jsonl


def ingest_all(raw_dir: Path | None = None) -> list[Document]:
    raw_dir = raw_dir or config.RAW_DIR
    docs: list[Document] = []
    failures: list[tuple[Path, str]] = []

    handlers: list[tuple[str, callable]] = [
        ("**/*.xml", pubmed_parser.parse_file),
        ("fda/*.json", fda_parser.parse_file),
        ("**/*.pdf", lambda p: [d] if (d := pdf_extract.extract_pdf(p)) else []),
        ("**/*.html", lambda p: [d] if (d := html_extract.extract_html_file(p)) else []),
        ("**/*.htm", lambda p: [d] if (d := html_extract.extract_html_file(p)) else []),
    ]

    for pattern, handler in handlers:
        for path in sorted(raw_dir.glob(pattern)):
            try:
                parsed = handler(path)
            except Exception as e:  # one malformed file must not sink the corpus
                failures.append((path, f"{type(e).__name__}: {e}"))
                continue
            docs.extend(parsed)
            if parsed:
                print(f"  [ok] {path.name} -> {len(parsed)} doc(s)")

    # Same article can arrive from two queries; keep the first occurrence.
    seen: set[str] = set()
    unique: list[Document] = []
    for d in docs:
        if d.doc_id not in seen:
            seen.add(d.doc_id)
            unique.append(d)

    if failures:
        print(f"\n  {len(failures)} file(s) failed to parse:")
        for path, err in failures:
            print(f"    - {path.name}: {err}")

    dropped = len(docs) - len(unique)
    if dropped:
        print(f"  deduplicated {dropped} repeat record(s)")

    return unique


def main() -> None:
    config.ensure_dirs()
    print("Ingesting raw documents...")
    docs = ingest_all()
    write_jsonl(config.DOCUMENTS_FILE, docs)

    by_source: dict[str, int] = {}
    total_sections = 0
    for d in docs:
        by_source[d.source] = by_source.get(d.source, 0) + 1
        total_sections += len(d.sections)

    print(f"\nWrote {len(docs)} documents ({total_sections} sections) "
          f"to {config.DOCUMENTS_FILE.relative_to(config.ROOT)}")
    for source, n in sorted(by_source.items()):
        print(f"  {source:12s} {n}")


if __name__ == "__main__":
    main()
