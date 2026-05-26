# 📚 AI Knowledge Assistant

> Upload technical documents (PDF or TXT), ask questions, and get **grounded answers** sourced directly from your files — no hallucinations, no outside knowledge.

---

## 🎯 Problem Statement

Engineers and researchers often need to query large technical documents — API references, research papers, internal wikis — but searching manually is slow. This app lets you drop in any PDF or text file and immediately start asking natural-language questions, with every answer tied back to the exact source excerpts.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | [Streamlit](https://streamlit.io) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector store** | [FAISS](https://github.com/facebookresearch/faiss) (CPU) |
| **LLM** | OpenAI `gpt-4o-mini` |
| **PDF parsing** | [pypdf](https://pypdf.readthedocs.io) |
| **Language** | Python 3.10+ |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)               │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────┐  │
│  │  File Upload│   │  Chat Input  │   │ Sidebar  │  │
│  └──────┬──────┘   └──────┬───────┘   └──────────┘  │
└─────────┼────────────────-┼───────────────────────--─┘
          │                 │
          ▼                 ▼
┌─────────────────┐   ┌─────────────────────────────┐
│DocumentProcessor│   │         QAEngine             │
│  - Parse PDF/TXT│   │  1. EmbeddingStore.search()  │
│  - Clean text   │   │  2. Build context prompt     │
│  - Chunk text   │   │  3. Call OpenAI Chat API     │
└────────┬────────┘   └────────────┬────────────────-┘
         │                         │
         ▼                         │
┌─────────────────┐                │
│  EmbeddingStore │◄───────────────┘
│  - Embed chunks │
│  - FAISS index  │
│  - Search()     │
└─────────────────┘
         │
         ▼
    OpenAI APIs
  (Embeddings + Chat)
```

**Flow:**
1. User uploads files → `DocumentProcessor` parses + chunks text
2. `EmbeddingStore` embeds chunks via OpenAI and indexes them in FAISS
3. User asks a question → question is embedded and searched against FAISS
4. Top-k matching chunks are assembled into a prompt
5. `QAEngine` sends prompt + system guardrail to GPT-4o-mini
6. Answer and source metadata are displayed in the chat UI

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/kaylinmoody6-creator/ai-knowledge-assistant.git
cd ai-knowledge-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Use it

1. Enter your **OpenAI API key** in the sidebar
2. Upload one or more **PDF or TXT** files
3. Click **Index Documents**
4. Ask questions in the chat box

---

## 📁 Project Structure

```
ai-knowledge-assistant/
├── app.py                    # Streamlit UI
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml           # Theme settings
├── src/
│   ├── __init__.py
│   ├── document_processor.py # PDF/TXT parsing + chunking
│   ├── embeddings.py         # OpenAI embeddings + FAISS index
│   └── qa_engine.py          # RAG pipeline + LLM generation
└── data/                     # Optional: drop test documents here
```

---

## ⚙️ Configuration

All tunable parameters are exposed in the sidebar:

| Parameter | Default | Effect |
|---|---|---|
| Chunk size | 300 words | Larger = more context per chunk, fewer chunks |
| Chunk overlap | 50 words | Higher = better cross-chunk continuity |
| Top-k retrieval | 5 | More chunks = more context, higher cost |
| LLM model | `gpt-4o-mini` | Swap for `gpt-4o` in `qa_engine.py` for higher quality |

---

## 🔒 Guardrails

The system prompt explicitly instructs the model to:
- Answer **only** from retrieved document excerpts
- Respond with `"I couldn't find an answer in the uploaded documents."` when information is absent
- Never fabricate figures, conclusions, or details

---

## 💡 What I Learned

- How **RAG (Retrieval-Augmented Generation)** pipelines work end-to-end
- Why chunk size and overlap matter for retrieval quality
- How FAISS flat-L2 indexing works for dense vector search
- Prompt engineering for grounded, hallucination-resistant answers
- Streamlit session state for managing multi-step workflows

---

## 🔮 Future Improvements

- [ ] **Chroma** as a persistent alternative to in-memory FAISS
- [ ] **Reranking** with a cross-encoder for better top-k precision
- [ ] **Streaming** LLM responses for faster perceived latency
- [ ] **Multi-modal**: support images and tables in PDFs
- [ ] **Export** chat history as Markdown or PDF
- [ ] **Authentication** for multi-user deployments
- [ ] **Evaluation harness**: RAGAS metrics for answer faithfulness + relevance

---

## 📄 License

MIT — do whatever you like, but a ⭐ is appreciated!
