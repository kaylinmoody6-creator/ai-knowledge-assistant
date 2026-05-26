import streamlit as st
import os
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingStore
from src.qa_engine import QAEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background: #f8fafc;
        border-left: 3px solid #6366f1;
        padding: 0.75rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #475569;
    }
    .stat-card {
        background: #f1f5f9;
        border-radius: 0.5rem;
        padding: 0.75rem;
        text-align: center;
    }
    .chunk-preview {
        font-size: 0.8rem;
        color: #64748b;
        background: #f8fafc;
        border-radius: 0.4rem;
        padding: 0.5rem 0.75rem;
        margin-top: 0.25rem;
        border: 1px solid #e2e8f0;
    }
    .answer-box {
        background: #fafafa;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "embedding_store": None,
        "qa_engine": None,
        "uploaded_files_info": [],
        "total_chunks": 0,
        "chat_history": [],
        "api_key_set": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored — it lives only in this session.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("API key loaded ✓", icon="🔑")

    st.markdown("---")
    st.markdown("## 📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more documents to index.",
    )

    chunk_size = st.slider("Chunk size (tokens ≈ words)", 100, 800, 300, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, 10)

    index_btn = st.button("🔍 Index Documents", use_container_width=True, type="primary")

    if index_btn:
        if not st.session_state.api_key_set:
            st.error("Please enter your OpenAI API key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing documents…"):
                try:
                    processor = DocumentProcessor(chunk_size=chunk_size, overlap=chunk_overlap)
                    all_chunks = []
                    file_info = []

                    for f in uploaded_files:
                        chunks = processor.process(f)
                        all_chunks.extend(chunks)
                        file_info.append({"name": f.name, "chunks": len(chunks)})

                    store = EmbeddingStore()
                    store.add_chunks(all_chunks)

                    st.session_state.embedding_store = store
                    st.session_state.qa_engine = QAEngine(store)
                    st.session_state.uploaded_files_info = file_info
                    st.session_state.total_chunks = len(all_chunks)
                    st.session_state.chat_history = []

                    st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s)!")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    # ── Stats panel ──
    if st.session_state.uploaded_files_info:
        st.markdown("---")
        st.markdown("## 📊 Index Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="stat-card">
                <div style="font-size:1.5rem;font-weight:700;color:#6366f1">{len(st.session_state.uploaded_files_info)}</div>
                <div style="font-size:0.75rem;color:#64748b">Files</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stat-card">
                <div style="font-size:1.5rem;font-weight:700;color:#8b5cf6">{st.session_state.total_chunks}</div>
                <div style="font-size:0.75rem;color:#64748b">Chunks</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("**Indexed files:**")
        for fi in st.session_state.uploaded_files_info:
            st.markdown(f"- `{fi['name']}` — {fi['chunks']} chunks")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📚 AI Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload technical documents, then ask questions grounded in their content.</div>', unsafe_allow_html=True)

if not st.session_state.api_key_set:
    st.info("👈 Enter your **OpenAI API key** in the sidebar to get started.", icon="🔑")
elif not st.session_state.embedding_store:
    st.info("👈 **Upload documents** and click **Index Documents** to begin.", icon="📄")
else:
    # ── Chat history ──
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources used", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"""<div class="source-box">
                            <strong>{src['file']}</strong> — chunk {src['chunk_id']}<br>
                            <span class="chunk-preview">{src['preview']}</span>
                        </div>""", unsafe_allow_html=True)

    # ── Chat input ──
    question = st.chat_input("Ask a question about your documents…")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.qa_engine.answer(question)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 Sources used", expanded=False):
                            for src in sources:
                                st.markdown(f"""<div class="source-box">
                                    <strong>{src['file']}</strong> — chunk {src['chunk_id']}<br>
                                    <span class="chunk-preview">{src['preview']}</span>
                                </div>""", unsafe_allow_html=True)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err = f"⚠️ Error generating answer: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err, "sources": []})

    # ── Clear chat ──
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()
