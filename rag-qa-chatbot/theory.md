# Project 2 Theory — Retrieval-Augmented Generation (RAG)

---

## What is RAG?
Retrieval-Augmented Generation is a hybrid approach:
- Retrieve: Find relevant context from an external source (vector DB).
- Augment: Feed retrieved context to an LLM as part of the prompt.

This helps the LLM produce grounded, fact-based answers — reducing hallucination.

---

## Key Components
- **Embedding Model:** Converts text to dense vectors.
- **Vector DB (FAISS):** Stores and searches embeddings efficiently.
- **Retriever:** Finds top matching chunks.
- **LLM (GPT-3.5):** Generates the final response using the retrieved context.

---

## Why OpenAI API?
- GPT-3.5/4 are powerful at understanding + reasoning over provided context.
- Focus on learning retrieval without worrying about fine-tuning.

---

## Pipeline Flow
User Query → Embed → Vector Search → Retrieved Chunks
→ Prompt LLM with (Context + Question) → LLM Answer