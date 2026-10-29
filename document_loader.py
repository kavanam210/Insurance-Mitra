"""Load, normalize, and chunk PDF and Markdown knowledge documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    source: str
    page: int | None = None


class DocumentLoadError(ValueError):
    """Raised when a document cannot be safely loaded."""


def clean_text(text: str) -> str:
    """Normalize extracted text without removing meaningful paragraph boundaries."""
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_document(name: str, content: bytes) -> list[DocumentChunk]:
    """Extract a supported document from bytes, retaining PDF page numbers."""
    suffix = Path(name).suffix.lower()
    if suffix == ".md":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(f"Markdown document {Path(name).name!r} is not valid UTF-8.") from exc
        text = clean_text(text)
        return [DocumentChunk(text=text, source=Path(name).name)] if text else []

    if suffix == ".pdf":
        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                pages = [
                    DocumentChunk(text=clean_text(page.get_text()), source=Path(name).name, page=number)
                    for number, page in enumerate(pdf, start=1)
                ]
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise DocumentLoadError(f"Could not read PDF document {Path(name).name!r}.") from exc
        return [page for page in pages if page.text]

    raise DocumentLoadError(f"Unsupported document type for {Path(name).name!r}; use PDF or Markdown.")


def load_documents(directory: str | Path) -> list[DocumentChunk]:
    """Load all supported documents in a directory in stable filename order."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Knowledge directory does not exist: {root}")

    documents: list[DocumentChunk] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".md"}:
            documents.extend(extract_document(path.name, path.read_bytes()))
    return documents


def split_documents(
    documents: list[DocumentChunk],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[DocumentChunk]:
    """Split extracted text into bounded overlapping chunks with source metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks: list[DocumentChunk] = []
    for document in documents:
        text = clean_text(document.text)
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append(DocumentChunk(text=piece, source=document.source, page=document.page))
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
