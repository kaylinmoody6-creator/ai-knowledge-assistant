"""
document_processor.py
Handles uploading, parsing, and chunking of PDF and plain-text documents.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List

import streamlit as st


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    text: str
    file_name: str
    chunk_id: int
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    """
    Parses uploaded files and splits them into overlapping text chunks.

    Parameters
    ----------
    chunk_size : int
        Approximate number of words per chunk.
    overlap : int
        Number of words that adjacent chunks share.
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = max(0, min(overlap, chunk_size - 1))

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, uploaded_file) -> List[Chunk]:
        """Entry point: detect file type, extract text, then chunk."""
        name: str = uploaded_file.name.lower()
        raw_bytes: bytes = uploaded_file.read()

        if name.endswith(".pdf"):
            text = self._extract_pdf(raw_bytes, uploaded_file.name)
        else:
            text = self._extract_txt(raw_bytes)

        text = self._clean(text)
        return self._chunk(text, uploaded_file.name)

    # ── Extraction ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_pdf(raw: bytes, filename: str) -> str:
        """Extract text from a PDF using pypdf."""
        try:
            import pypdf  # lazy import so txt-only users don't need it

            reader = pypdf.PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "pypdf is required to parse PDFs. Run: pip install pypdf"
            )
        except Exception as exc:
            raise ValueError(f"Could not parse PDF '{filename}': {exc}") from exc

    @staticmethod
    def _extract_txt(raw: bytes) -> str:
        """Decode plain text, trying UTF-8 then latin-1 as fallback."""
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="replace")

    # ── Cleaning ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and remove junk characters."""
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces/tabs (but preserve newlines)
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk(self, text: str, file_name: str) -> List[Chunk]:
        """
        Sliding-window word chunker.

        Splits on whitespace, creates windows of `chunk_size` words,
        and advances by `chunk_size - overlap` words per step.
        """
        words = text.split()
        if not words:
            return []

        step = max(1, self.chunk_size - self.overlap)
        chunks: List[Chunk] = []
        chunk_id = 0

        for start in range(0, len(words), step):
            end = start + self.chunk_size
            window = words[start:end]
            if not window:
                break

            chunk_text = " ".join(window)
            # Approximate char positions (useful for highlighting later)
            char_start = len(" ".join(words[:start]))
            char_end = char_start + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    file_name=file_name,
                    chunk_id=chunk_id,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            chunk_id += 1

            # Stop if we've reached the end
            if end >= len(words):
                break

        return chunks
