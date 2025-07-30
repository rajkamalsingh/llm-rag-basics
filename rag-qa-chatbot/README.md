# Project 2: RAG Q&A Chatbot

## Goal
This project builds a **Question-Answering Chatbot** that:
- Uses a **custom knowledge base** (document chunks)
- Retrieves relevant information using **vector search (FAISS)**
- Uses **OpenAI GPT-3.5** to generate accurate, context-aware answers

---

## Tools and Technologies
- **OpenAI API (GPT-3.5 Turbo)** – For generating final answers
- **SentenceTransformers** (`all-MiniLM-L6-v2`) – To create text embeddings
- **FAISS** – Efficient similarity search to retrieve relevant document chunks
- **Gradio** – Simple web-based chatbot interface

---

## Installation
```bash
pip install openai transformers sentence-transformers faiss-cpu gradio
```

## How to Run
```bash
python rag_chatbot.py
```
## Example
### Question: "What is RAG in LLMs?"
### Answer: 
"RAG stands for Retrieval-Augmented Generation, a method where a model retrieves relevant context from an external database or document before generating its response. This ensures factual and context-aware answers."