import streamlit as st
from rag_pipeline import process_document, query_document

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .answer-box {
        background: #f0fff4;
        border: 1px solid #38a169;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .source-box {
        background: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 8px 12px;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        margin: 4px 0;
    }
    .stat-pill {
        display: inline-block;
        background: #e9d8fd;
        color: #553c9a;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem;">📄 RAG Document Intelligence App</h1>
    <p style="margin:0.3rem 0 0;opacity:0.85;font-size:0.95rem;">
        Upload a document · Ask questions · Get grounded answers with sources
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com",
    )

    st.markdown("---")
    st.markdown("### 📂 Upload Document")

    uploaded = st.file_uploader(
        "PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        help="Max 200MB",
    )

    if uploaded and groq_key:
        if st.button("⚡ Process Document", use_container_width=True, type="primary"):
            with st.spinner("Loading → Chunking → Embedding…"):
                try:
                    pipeline = process_document(uploaded, groq_key)
                    st.session_state.pipeline = pipeline
                    st.session_state.messages = []
                    st.success("✅ Document ready!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.pipeline:
        p = st.session_state.pipeline
        st.markdown("---")
        st.markdown("### 📊 Document Stats")
        st.markdown(f"**File:** {p['file_name']}")
        st.markdown(
            f"<span class='stat-pill'>📄 {p['num_pages']} pages</span> "
            f"<span class='stat-pill'>🧩 {p['num_chunks']} chunks</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "**Built by** [Vasam Vamshi](https://github.com/vamshi1823)  \n"
        "🔗 [GitHub](https://github.com/vamshi1823/rag-document-intelligence)"
    )

# ── Main area ──────────────────────────────────────────────────────────────────
col_chat, col_examples = st.columns([3, 1])

with col_examples:
    st.markdown("### 💡 Example Questions")
    examples = [
        "What is the main topic of this document?",
        "Summarize the key points",
        "What are the conclusions?",
        "List the main findings",
        "Who are the authors or stakeholders?",
        "What data or statistics are mentioned?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:15]}", use_container_width=True):
            st.session_state._pending = ex
            st.rerun()

with col_chat:
    if not st.session_state.pipeline:
        st.info("👈 Upload a document and enter your Groq API key to get started.")
    else:
        # Render history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    st.markdown(
                        f'<div class="answer-box">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if msg.get("sources"):
                        with st.expander("📎 Source Passages"):
                            for i, src in enumerate(msg["sources"], 1):
                                st.markdown(
                                    f'<div class="source-box">'
                                    f'<b>Source {i} (Page {src["page"]}):</b> {src["snippet"]}…'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                else:
                    st.markdown(msg["content"])

        # Input
        pending = st.session_state.pop("_pending", None)
        user_input = st.chat_input("Ask a question about your document…")
        query = pending or user_input

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching document and generating answer…"):
                    try:
                        result = query_document(
                            st.session_state.pipeline["qa_chain"], query
                        )
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(
                            f'<div class="answer-box">{answer}</div>',
                            unsafe_allow_html=True,
                        )
                        if sources:
                            with st.expander("📎 Source Passages"):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(
                                        f'<div class="source-box">'
                                        f'<b>Source {i} (Page {src["page"]}):</b> {src["snippet"]}…'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    except Exception as e:
                        err = f"Error: {str(e)}"
                        st.error(err)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": err, "sources": []}
                        )
