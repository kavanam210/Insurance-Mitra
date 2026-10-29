from pathlib import Path

import pymupdf as fitz
import pytest

from document_loader import DocumentChunk, DocumentLoadError, extract_document, split_documents


def make_pdf(texts: list[str]) -> bytes:
    document = fitz.open()
    for text in texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def test_extracts_markdown_and_cleans_text():
    chunks = extract_document("guide.md", b"# Title\n\nA   useful   line.\n")
    assert len(chunks) == 1
    assert chunks[0].source == "guide.md"
    assert chunks[0].page is None
    assert chunks[0].text == "# Title\n\nA useful line."


def test_extracts_pdf_pages_with_page_numbers():
    chunks = extract_document("policy.pdf", make_pdf(["First page text", "Second page text"]))
    assert [(chunk.page, chunk.text) for chunk in chunks] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]


def test_chunks_preserve_source_page_and_overlap():
    document = DocumentChunk("abcdefghij klmnopqrst uvwxyz", "policy.pdf", 4)
    chunks = split_documents([document], chunk_size=12, overlap=4)
    assert len(chunks) > 1
    assert all(chunk.source == "policy.pdf" and chunk.page == 4 for chunk in chunks)
    assert chunks[0].text[-4:] in chunks[1].text


@pytest.mark.parametrize("name,content", [("notes.txt", b"no"), ("broken.md", b"\xff")])
def test_rejects_unsupported_or_invalid_documents(name, content):
    with pytest.raises(DocumentLoadError):
        extract_document(name, content)


def test_rejects_corrupt_pdf():
    with pytest.raises(DocumentLoadError):
        extract_document("broken.pdf", b"not a pdf")


def test_missing_directory_is_reported(tmp_path: Path):
    from document_loader import load_documents

    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
