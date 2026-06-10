# 📄 RAG Document Intelligence App

**Day 1 of 5 — AI Portfolio Sprint**  
Upload any PDF, TXT, or DOCX — ask questions — get grounded answers with source citations.

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://share.streamlit.io)
[![GitHub](https://img.shields.io/badge/GitHub-vamshi1823-blue?style=for-the-badge&logo=github)](https://github.com/vamshi1823/rag-document-intelligence)

---

## 🚀 Features

- 📄 Supports PDF, TXT, DOCX upload
- 🧩 Smart chunking with overlap (RecursiveCharacterTextSplitter)
- 🔍 Semantic search via FAISS + sentence-transformers (no OpenAI needed)
- 🤖 Answers grounded in document context via Groq LLaMA-3.3-70b
- 📎 Source passage citations with page numbers
- 💬 Multi-turn chat history within session

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| Vector Store | FAISS (in-memory) |
| Framework | LangChain `RetrievalQA` |
| UI | Streamlit |

---

## 📁 Project Structure

```
rag-document-intelligence/
├── app.py            # Streamlit UI + chat interface
├── rag_pipeline.py   # Load → chunk → embed → index → query
├── requirements.txt
└── README.md
```

---

## 🔧 Local Setup

```bash
git clone https://github.com/vamshi1823/rag-document-intelligence.git
cd rag-document-intelligence

pip install -r requirements.txt

streamlit run app.py
```

Enter your free Groq API key at [console.groq.com](https://console.groq.com).

---

## 👤 Author

**Vasam Vamshi** | AI/ML Engineer  
📧 vasamvamshi03@gmail.com  
🔗 [GitHub](https://github.com/vamshi1823) | [LinkedIn](https://linkedin.com/in/vasamvamshi)

---

*Part of a 5-day AI portfolio sprint showcasing LangChain, NLP, and Data Engineering skills.*
