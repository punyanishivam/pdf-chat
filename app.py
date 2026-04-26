import os
import re
import hashlib
import tempfile
import streamlit as st
from dotenv import load_dotenv

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Works locally (.env) and on Streamlit Cloud (Secrets)
GROQ_API_KEY = (
    st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
) or os.getenv("GROQ_API_KEY")

CHROMA_DIR = "./chroma_db"  # persisted to disk next to app.py

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
    .source-box {
        background: #f1efe8;
        border-left: 3px solid #378ADD;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-top: 0.4rem;
        font-size: 0.82rem;
        color: #555;
    }
    .pill {
        display: inline-block;
        background: #E6F1FB;
        color: #0C447C;
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Guard ─────────────────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your `.env` file and restart.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "rag_chain": None,
    "chat_history": [],
    "display_history": [],
    "pdf_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ───────────────────────────────────────────────────────────────────
def collection_name_for(pdf_name: str) -> str:
    """
    Chroma collection names: 3–63 chars, alphanumeric + hyphens,
    must start and end with alphanumeric. We derive a stable name
    from the filename + an 8-char hash to avoid collisions.
    """
    safe = re.sub(r"[^a-z0-9]+", "-", pdf_name.lower()).strip("-")[:50]
    h = hashlib.md5(pdf_name.encode()).hexdigest()[:8]
    name = f"{safe}-{h}"
    # Ensure it starts/ends with alphanumeric
    name = re.sub(r"^[^a-z0-9]+", "", name)
    name = re.sub(r"[^a-z0-9]+$", "", name)
    return name or h  # fallback to pure hash if name is empty


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load embedding model once and reuse across PDFs."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ── Build RAG pipeline ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_chain(pdf_bytes: bytes, pdf_name: str):
    """
    Returns a callable: invoke(input_text, chat_history) → {answer, context}

    On first call for a given PDF: chunks, embeds, and persists to Chroma.
    On subsequent calls (same PDF name): loads the existing collection instantly.
    """
    embeddings = get_embeddings()
    col_name = collection_name_for(pdf_name)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing_cols = [c.name for c in chroma_client.list_collections()]

    if col_name in existing_cols:
        # ── Fast path: collection already on disk ──────────────────────────
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=col_name,
            embedding_function=embeddings,
        )
    else:
        # ── Slow path: embed and persist ───────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        docs = PyPDFLoader(tmp_path).load()
        os.unlink(tmp_path)

        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            raise ValueError(
                "No text could be extracted from this PDF. "
                "It may be a scanned or image-only file. "
                "Try running it through OCR first (e.g. Adobe Acrobat, PDF24)."
            )

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
        ).split_documents(docs)

        if not chunks:
            raise ValueError("PDF was loaded but produced no usable text chunks.")

        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            client=chroma_client,
            collection_name=col_name,
        )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=GROQ_API_KEY
    )
    parser = StrOutputParser()

    condense_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Given the conversation above, rewrite the follow-up question as a "
                "self-contained standalone question. Return only the question.",
            ),
        ]
    )
    condense_chain = condense_prompt | llm | parser

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer using only the context below. "
                "If the answer is not in the context, say so clearly.\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    answer_chain = answer_prompt | llm | parser

    def invoke(input_text: str, chat_history: list) -> dict:
        standalone_q = (
            condense_chain.invoke({"input": input_text, "chat_history": chat_history})
            if chat_history
            else input_text
        )
        source_docs = retriever.invoke(standalone_q)
        context = "\n\n".join(doc.page_content for doc in source_docs)
        answer = answer_chain.invoke(
            {
                "input": input_text,
                "context": context,
                "chat_history": chat_history,
            }
        )
        return {"answer": answer, "context": source_docs}

    return invoke


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF Q&A Chatbot")
    st.markdown(
        '<div class="pill">Month 1 · AI Foundation</div>', unsafe_allow_html=True
    )
    st.markdown("---")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded:
        pdf_bytes = uploaded.read()
        if st.session_state.pdf_name != uploaded.name:
            col_name = collection_name_for(uploaded.name)
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            is_cached = col_name in [c.name for c in client.list_collections()]

            msg = (
                "Loading from cache…"
                if is_cached
                else f"Embedding **{uploaded.name}**…"
            )
            with st.spinner(msg):
                try:
                    st.session_state.rag_chain = build_rag_chain(
                        pdf_bytes, uploaded.name
                    )
                    st.session_state.pdf_name = uploaded.name
                    st.session_state.chat_history = []
                    st.session_state.display_history = []
                    label = (
                        "⚡ Loaded from cache" if is_cached else "✅ Embedded & ready"
                    )
                    st.success(f"{label} — ask anything about **{uploaded.name}**")
                except ValueError as e:
                    st.error(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"⚠️ Unexpected error: {e}")

    if st.session_state.pdf_name:
        st.markdown("---")
        st.markdown(f"**Active PDF:** {st.session_state.pdf_name}")
        if st.button("🗑 Clear chat"):
            st.session_state.chat_history = []
            st.session_state.display_history = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<small>LangChain · Chroma · Groq · Streamlit</small>", unsafe_allow_html=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("PDF Q&A Chatbot 📄")

if not st.session_state.rag_chain:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

# Render chat history
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    page = src.metadata.get("page", 0)
                    snippet = src.page_content[:300].replace("\n", " ")
                    st.markdown(
                        f'<div class="source-box"><strong>Chunk {i} · Page {page + 1}</strong>'
                        f"<br>{snippet}…</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if question := st.chat_input("Ask a question about your PDF…"):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = st.session_state.rag_chain(question, st.session_state.chat_history)
        answer = result["answer"]
        sources = result.get("context", [])

        st.markdown(answer)
        if sources:
            with st.expander("📎 Sources", expanded=False):
                for i, src in enumerate(sources, 1):
                    page = src.metadata.get("page", 0)
                    snippet = src.page_content[:300].replace("\n", " ")
                    st.markdown(
                        f'<div class="source-box"><strong>Chunk {i} · Page {page + 1}</strong>'
                        f"<br>{snippet}…</div>",
                        unsafe_allow_html=True,
                    )

    st.session_state.chat_history += [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]
    st.session_state.display_history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": sources},
    ]
