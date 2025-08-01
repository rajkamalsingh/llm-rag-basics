# Project 3: RAG Chatbot with Document Upload

---

##  Goal
To make RAG practical by allowing users to **upload their own documents (txt/pdf)** and dynamically build a knowledge base for Q&A.

---

## New Concepts Introduced
1. **Document Parsing** – Loading and reading different file types.
2. **Chunking** – Splitting text into manageable pieces for embeddings.
3. **Dynamic Indexing** – Creating FAISS vector search index on the fly.

---

## 1. Document Loading
- PDFs: Extract text from each page using **PyMuPDF**.
- Text files: Simple read operation.

---

## 2. Why Chunking?
- LLMs have context window limits (e.g., GPT-3.5 ≈ 16k tokens).
- Long documents cannot fit entirely in a prompt.
- Solution: Split into **chunks** of 300–500 tokens with **overlap**:
  - Overlap avoids losing context between chunks.
  - Example:
    ```
    Chunk 1: words[0:500]
    Chunk 2: words[450:950]
    ```

---

## 3. Building Embeddings
- Each chunk is converted into a **vector representation** using `SentenceTransformer`.
- Similar meaning = vectors closer in embedding space.

---

## 4. Vector Database (FAISS)
- Stores all document embeddings.
- Enables **fast nearest-neighbor search** for relevant chunks:
- Flow - User Query → Embedding → FAISS Search → Top-k chunks
---
## 5. Augmenting the Prompt
- Retrieved chunks are **added as context** to the user's question:
``` 
Context: <top relevant chunks>
Question: <user query>
Answer:
```
- This gives GPT-3.5 specific info to answer correctly.

---

## 6. Dynamic Knowledge Base
- Each time a new file is uploaded:
- Text → Chunks → Embeddings → New FAISS index
- No need to fine-tune the LLM — knowledge is **plugged in at runtime**.

---

## Workflow Diagram
``` 
Upload Doc → Extract Text → Chunking → Embeddings → FAISS Index
Query → Embed → Retrieve Top Chunks → GPT-3.5 → Answer
``` 

---

## Benefits of RAG with Upload
- Can **answer questions from private or custom data**.
- Avoids hallucinations by grounding answers in real text.
- Works for long docs beyond LLM context limits.

---

## Next Step
- Add **Memory**: Maintain conversation history between user and chatbot.
- Improve document processing (summaries, metadata).
- Try **multiple document support**.