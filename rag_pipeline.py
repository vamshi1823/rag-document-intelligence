"""
rag_pipeline.py — RAG Document Intelligence core
LangChain + FAISS + Groq LLaMA-3 + HuggingFace embeddings
"""
from __future__ import annotations

import os
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ── Prompt ─────────────────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise document analyst. Use ONLY the provided context to answer.
If the answer is not in the context, say "I couldn't find that in the uploaded document."

Context:
{context}

Question: {question}

Answer:""",
)


# ── Document loading ───────────────────────────────────────────────────────────

def load_document(uploaded_file) -> list:
    suffix = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
    elif suffix in (".docx", ".doc"):
        loader = Docx2txtLoader(tmp_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    os.unlink(tmp_path)
    return docs


# ── Chunking ───────────────────────────────────────────────────────────────────

def split_documents(docs: list, chunk_size: int = 800, chunk_overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(docs)


# ── Vector store ───────────────────────────────────────────────────────────────

def build_vectorstore(chunks: list) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.from_documents(chunks, embeddings)


# ── QA Chain (LCEL — no deprecated chains) ────────────────────────────────────

def build_qa_chain(vectorstore: FAISS, groq_api_key: str, k: int = 4):
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return {"chain": chain, "retriever": retriever}


# ── Full pipeline ──────────────────────────────────────────────────────────────

def process_document(uploaded_file, groq_api_key: str) -> dict:
    docs = load_document(uploaded_file)
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    qa = build_qa_chain(vectorstore, groq_api_key)

    return {
        "chain": qa["chain"],
        "retriever": qa["retriever"],
        "num_pages": len(docs),
        "num_chunks": len(chunks),
        "file_name": uploaded_file.name,
    }


def query_document(pipeline: dict, question: str) -> dict:
    chain = pipeline["chain"]
    retriever = pipeline["retriever"]

    # Get answer
    answer = chain.invoke(question)

    # Get source docs separately
    source_docs = retriever.invoke(question)
    sources = [
        {
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:200],
        }
        for doc in source_docs
    ]
    return {"answer": answer, "sources": sources}
