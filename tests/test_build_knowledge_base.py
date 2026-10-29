from pathlib import Path

import pymupdf
import pytest

from build_knowledge_base import create_knowledge_pdf, validate_knowledge_pdf


def test_generated_pdf_preserves_full_source_and_wraps_across_pages(tmp_path: Path):
    source = tmp_path / "guide.md"
    pdf = tmp_path / "guide.pdf"
    source.write_text(
        "# Long guide\n\n"
        + ("Insurance terms depend on policy wording; check the complete document. " * 180)
        + "\n\n- Waiting periods vary by benefit.\n",
        encoding="utf-8",
    )

    create_knowledge_pdf(source, pdf)
    pages, extracted_characters = validate_knowledge_pdf(source, pdf)

    assert pages > 1
    assert extracted_characters > 10_000
    with pymupdf.open(pdf) as document:
        extracted = "\n".join(page.get_text() for page in document)
    assert "LONG GUIDE" in extracted
    assert "Waiting periods vary by benefit." in extracted


def test_pdf_validation_rejects_missing_source_content(tmp_path: Path):
    source = tmp_path / "guide.md"
    pdf = tmp_path / "guide.pdf"
    source.write_text("# Expected heading\n\nRequired insurance explanation.", encoding="utf-8")
    create_knowledge_pdf(source, pdf)

    with pymupdf.open(pdf) as document:
        document[0].insert_text((48, 120), "tampered", fontsize=9)
        document.save(tmp_path / "tampered.pdf")

    with pytest.raises(ValueError, match="does not preserve"):
        validate_knowledge_pdf(source, tmp_path / "tampered.pdf")
