"""Extract per-page text from lecture slide PDFs into Markdown.

Usage:
    .venv/bin/python src/extract_slides.py data/slides/*     # folders or PDF files
    .venv/bin/python src/extract_slides.py data/slides/機器學習 -o processed/slides_text

Writes processed/slides_text/<folder>/<pdf name>.md with one "## p.N" section per page,
so a transcript's "look at page 8" can be matched to the slide. Pages with
little extractable text (mostly figures or formulas) are flagged so they can be
read as images instead. Math drawn as images or symbol fonts is not extracted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent  # project root (this file is in src/)
LOW_TEXT_CHARS = 80


def extract(pdf: Path, out: Path) -> tuple[int, int]:
    doc = pymupdf.open(pdf)
    lines = [f"# {pdf.stem}", "", f"Source: `{pdf.name}`, {len(doc)} pages", ""]
    low = 0
    for i, page in enumerate(doc, 1):
        text = "\n".join(l.rstrip() for l in page.get_text().splitlines() if l.strip())
        lines.append(f"## p.{i}")
        if len(text) < LOW_TEXT_CHARS:
            low += 1
            lines.append("_(mostly figures or formulas; little extractable text)_")
        lines += [text, ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return len(doc), low


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="PDF files or folders of PDFs")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "processed" / "slides_text", help="output folder")
    args = ap.parse_args()

    pdfs = []
    for p in args.inputs:
        pdfs += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 1
    for pdf in pdfs:
        out = args.out / pdf.parent.name / f"{pdf.stem}.md"
        pages, low = extract(pdf, out)
        print(f"{pdf.parent.name}/{pdf.name}: {pages} pages ({low} mostly figures) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
