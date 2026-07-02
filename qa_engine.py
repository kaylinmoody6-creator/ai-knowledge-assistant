"""
qa_engine.py
Retrieves the most relevant chunks for a user question, then calls an LLM
to produce a grounded answer with source attribution.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from src.embeddings import EmbeddingStore


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise technical assistant.
Your ONLY job is to answer the user's question using the document excerpts provided below.

Rules you MUST follow:
1. Answer ONLY from the provided excerpts — do NOT use outside knowledge.
2. If the excerpts do not contain enough information, say clearly:
   "I couldn't find an answer in the uploaded documents."
3. Be concise and factual. Use bullet points or numbered lists when helpful.
4. Do not hallucinate details, figures, or conclusions not present in the excerpts.
5. You may quote short phrases from the excerpts when useful, but keep it concise.
"""

CONTEXT_TEMPLATE = """--- Document Excerpts ---
{context}
--- End of Excerpts ---

User Question: {question}

Provide a clear, grounded answer based solely on the excerpts above."""


class QAEngine:
    """
    Orchestrates retrieval → prompt construction → LLM generation.

    Parameters
    ----------
    store : EmbeddingStore
        Pre-populated vector store to search against.
    model : str
        OpenAI chat model to use for generation.
    top_k : int
        Number of chunks to retrieve per query.
    max_tokens : int
        Maximum tokens in the LLM response.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        model: str = "gpt-4o-mini",
        top_k: int = 5,
        max_tokens: int = 1024,
    ):
        self.store = store
        self.model = model
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._client = self._make_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Full RAG pipeline for a single question.

        Returns
        -------
        dict with keys:
            - "answer"  : str  — the LLM's response
            - "sources" : list — [{file, chunk_id, preview, distance}, ...]
        """
        question = question.strip()
        if not question:
            return {"answer": "Please enter a question.", "sources": []}

        # 1. Retrieve top-k relevant chunks
        results = self.store.search(question, k=self.top_k)
        if not results:
            return {
                "answer": "No documents are indexed yet. Please upload and index documents first.",
                "sources": [],
            }

        # 2. Build context string for the prompt
        context_parts = []
        sources = []
        for rank, (chunk, distance) in enumerate(results, start=1):
            context_parts.append(
                f"[Excerpt {rank} | File: {chunk.file_name} | Chunk #{chunk.chunk_id}]\n"
                f"{chunk.text}"
            )
            sources.append(
                {
                    "file": chunk.file_name,
                    "chunk_id": chunk.chunk_id,
                    "preview": chunk.text[:200].replace("\n", " ") + ("…" if len(chunk.text) > 200 else ""),
                    "distance": round(distance, 4),
                }
            )

        context = "\n\n".join(context_parts)

        # 3. Call the LLM
        user_message = CONTEXT_TEMPLATE.format(context=context, question=question)
        response_text = self._chat(user_message)

        return {"answer": response_text, "sources": sources}

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _chat(self, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=0.2,  # low temp for factual, reproducible answers
        )
        return response.choices[0].message.content.strip()

    # ── Client factory ────────────────────────────────────────────────────────

    @staticmethod
    def _make_client():
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=api_key)
