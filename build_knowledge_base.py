"""Generate the reproducible PDF and persisted semantic index."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pymupdf as fitz
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, EMBEDDING_MODEL, VECTOR_STORE_DIR
from document_loader import load_documents, split_documents
from rag_pipeline import load_or_build_index


def _source_lines(markdown_path: Path, font: fitz.Font, max_width: float) -> list[str]:
    lines: list[str] = []
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip().upper()
        if not stripped:
            lines.append("")
        else:
            words = stripped.split()
            wrapped: list[str] = []
            current = ""
            for word in words:
                if font.text_length(word, fontsize=9.5) > max_width:
                    if current:
                        wrapped.append(current)
                        current = ""
                    segment = ""
                    for character in word:
                        if segment and font.text_length(segment + character, fontsize=9.5) > max_width:
                            wrapped.append(segment)
                            segment = character
                        else:
                            segment += character
                    if segment:
                        wrapped.append(segment)
                    continue
                candidate = f"{current} {word}".strip()
                if current and font.text_length(candidate, fontsize=9.5) > max_width:
                    wrapped.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                wrapped.append(current)
            lines.extend(wrapped)
    return lines


def _normalized_pdf_text(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def validate_knowledge_pdf(markdown_path: Path, pdf_path: Path) -> tuple[int, int]:
    """Verify every source content character is represented in the generated PDF."""
    expected_lines: list[str] = []
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip().upper()
        expected_lines.append(stripped)
    expected = _normalized_pdf_text("\n".join(expected_lines))

    with fitz.open(pdf_path) as document:
        page_texts: list[str] = []
        for page_number, page in enumerate(document, start=1):
            page_texts.append(page.get_text())
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        x0, y0, x1, y1 = span["bbox"]
                        if x0 < 47 or x1 > page.rect.width - 47 or y0 < 47 or y1 > page.rect.height - 47:
                            raise ValueError(f"PDF text exceeds page bounds on page {page_number}.")
        extracted = "\n".join(page_texts)
        page_count = document.page_count
    actual = _normalized_pdf_text(extracted)
    if actual != expected:
        shared_length = min(len(actual), len(expected))
        difference = next(
            (position for position in range(shared_length) if actual[position] != expected[position]),
            shared_length,
        )
        raise ValueError(
            "Generated PDF does not preserve the Markdown source "
            f"(expected {len(expected)} normalized characters, extracted {len(actual)}; "
            f"first difference at character {difference})."
        )
    return page_count, len(extracted)


def create_knowledge_pdf(markdown_path: Path, pdf_path: Path) -> None:
    """Render Markdown using measured line widths and add pages when content overflows."""
    page_width, page_height = 595, 842
    margin_x, margin_y = 48, 48
    font_size, line_height = 9.5, 12.5
    font = fitz.Font("helv")
    lines = _source_lines(markdown_path, font, page_width - 2 * margin_x)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page(width=page_width, height=page_height)
    baseline = margin_y + font_size + 2
    for line in lines:
        if baseline + font_size > page_height - margin_y:
            page = document.new_page(width=page_width, height=page_height)
            baseline = margin_y + font_size
        if line:
            page.insert_text(
                fitz.Point(margin_x, baseline),
                line,
                fontsize=font_size,
                fontname="helv",
            )
        baseline += line_height
    document.save(pdf_path, garbage=4, deflate=True)
    document.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild the index even if sources are unchanged.")
    args = parser.parse_args()

    pdf_path = DATA_DIR / "insurance_knowledge.pdf"
    create_knowledge_pdf(DATA_DIR / "insurance_knowledge.md", pdf_path)
    page_count, extracted_characters = validate_knowledge_pdf(DATA_DIR / "insurance_knowledge.md", pdf_path)
    print(f"PDF source validation passed: {page_count} pages, {extracted_characters} extracted characters.")

    source_documents = load_documents(DATA_DIR)
    expected_chunks = split_documents(source_documents)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    knowledge = load_or_build_index(DATA_DIR, embedder, force_rebuild=args.force)
    if len(knowledge.chunks) != len(expected_chunks):
        raise RuntimeError(
            f"Index passage count mismatch: expected {len(expected_chunks)}, got {len(knowledge.chunks)}."
        )
    loaded_characters = sum(len(document.text) for document in source_documents)
    print(
        f"Knowledge base ready: {len(source_documents)} extracted pages/documents, "
        f"{loaded_characters} source characters, {len(knowledge.chunks)} passages indexed."
    )


if __name__ == "__main__":
    main()
