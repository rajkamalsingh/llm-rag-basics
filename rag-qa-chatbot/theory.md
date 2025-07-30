# Project 2 Theory — Retrieval-Augmented Generation (RAG)

---

## What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that **combines search and generation** to improve the accuracy of responses from Large Language Models (LLMs).

Instead of relying only on what the LLM "knows" from training data, RAG:
1. **Retrieves** relevant information from an external source (documents, knowledge base).
2. **Augments** the user's query with this information.
3. **Generates** an answer based on both the query and the retrieved context.

This approach:
- Reduces **hallucinations** (wrong answers)
- Allows LLMs to access **fresh or private data**
- Makes chatbots **domain-specific**
---

## Components of RAG

### 1. **Embedding Model**
- Converts text (sentences, paragraphs) into high-dimensional vectors (numerical representations).
- Similar text → Similar vectors.
- We use `all-MiniLM-L6-v2` from [SentenceTransformers](https://www.sbert.net).

---

### 2. **Vector Database (FAISS)**
- Stores document embeddings.
- When a question comes in, it:
  - Embeds the question
  - Searches for similar embeddings
  - Returns top-k relevant chunks

---

### 3.  **Retriever**
- Finds the most relevant context for a query.
- Example:
```text
User: "What is RAG?"
Retriever finds:
- "RAG combines search with generation to produce context-aware answers."
```
---

## Why OpenAI API?
- GPT-3.5/4 are powerful at understanding + reasoning over provided context.
- Focus on learning retrieval without worrying about fine-tuning.

---

## Pipeline Flow
User Query → Embed → Vector Search → Retrieved Chunks
→ Prompt LLM with (Context + Question) → LLM Answer
---
## Key Parameters
- k (top_k): Number of relevant chunks retrieved.
- Embedding model choice: Affects semantic matching quality.
- Chunk size: Splitting large docs improves retrieval accuracy.

---
## Why Use RAG?
- LLMs have limited memory (context window).
- Pretrained LLMs lack private or updated knowledge.
- RAG provides dynamic knowledge injection without retraining.

