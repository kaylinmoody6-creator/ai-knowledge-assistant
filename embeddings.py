"""
embeddings.py
Converts text chunks into OpenAI embeddings and stores them in a FAISS index
for fast semantic similarity search.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from src.document_processor import Chunk


class EmbeddingStore:
    """
    Manages embeddings for document chunks using OpenAI + FAISS.

    Usage
    -----
    store = EmbeddingStore()
    store.add_chunks(chunks)          # index all chunks
    results = store.search(query, k)  # retrieve top-k chunks
    """

    EMBED_MODEL = "text-embedding-3-small"  # fast & cheap; swap for -large if needed

    def __init__(self):
        self._chunks: List[Chunk] = []
        self._index = None          # FAISS index (populated after add_chunks)
        self._dim: int | None = None

        self._client = self._make_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and build a FAISS flat-L2 index."""
        if not chunks:
            raise ValueError("No chunks provided to index.")

        import faiss  # lazy import

        texts = [c.text for c in chunks]
        vectors = self._embed_batch(texts)

        matrix = np.array(vectors, dtype="float32")
        self._dim = matrix.shape[1]
        self._index = faiss.IndexFlatL2(self._dim)
        self._index.add(matrix)
        self._chunks = chunks

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Return the top-k chunks most relevant to *query*.

        Returns
        -------
        List of (Chunk, distance) tuples, sorted ascending by L2 distance.
        """
        if self._index is None:
            raise RuntimeError("Index is empty. Call add_chunks() first.")

        q_vec = np.array(self._embed_single(query), dtype="float32").reshape(1, -1)
        k = min(k, len(self._chunks))
        distances, indices = self._index.search(q_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(dist)))
        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _embed_single(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(input=[text], model=self.EMBED_MODEL)
        return resp.data[0].embedding

    def _embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Embed texts in batches to stay within API limits.
        OpenAI allows up to 2048 inputs per request; we use 100 to be safe.
        """
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(input=batch, model=self.EMBED_MODEL)
            # API returns embeddings in the same order as input
            batch_embeds = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embeds)
        return all_embeddings

    # ── Client factory ────────────────────────────────────────────────────────

    @staticmethod
    def _make_client():
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Enter your key in the sidebar."
            )
        return OpenAI(api_key=api_key)
