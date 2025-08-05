# Project 3 – RAG Q&A Chatbot with Document Upload

---

## Overview

This project demonstrates a **Retrieval-Augmented Generation (RAG) chatbot** that allows you to:

- **Upload your own documents** (`.txt` or `.pdf`)  
- Dynamically **process, chunk, and embed text**  
- Perform **semantic search** using **FAISS**  
- Use **GPT-3.5 Turbo** to answer questions **grounded in the uploaded document**

This is a **step forward from Project 2**, making the chatbot practical for custom, private knowledge bases.

---

##  Project Structure
```
03-rag-qa-chatbot-upload/
│── rag_chatbot_upload.py # Main chatbot script
│── requirements.txt # Dependencies
│── README.md # This file
│── theory.md # Detailed explanations of RAG concepts
│── assets/
└── rag_upload_diagram.png # Flowchart of the RAG pipeline
```

---

## Technologies Used

- **Python 3.9+**
- [OpenAI GPT-3.5 Turbo](https://platform.openai.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS](https://faiss.ai/) – Vector similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) – PDF text extraction
- [Gradio](https://www.gradio.app/) – Simple web-based UI

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/llm-rag-basics.git
cd llm-rag-basics/03-rag-qa-chatbot-upload
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Set OpenAI API key
```
export OPENAI_API_KEY="your_api_key_here"
```
### 4. Run the chatbot
```
python rag_chatbot_upload.py
```
- This launches a Gradio interface.
- Upload your .txt or .pdf file and start asking questions!

### Example Run
1. Uploaded File: job_description.pdf

2. Question:
```
What skills are required according to this document?
```
3. Answer:
```
The document lists leadership, data analysis, teamwork, and communication skills as essential requirements for the role.
```
---

### RAG Pipeline Overview

- Document Upload – User provides .txt or .pdf file.
- Text Extraction – Convert document into raw text.
- Chunking – Split text into manageable pieces with overlap.
- Embedding – Convert chunks into vector representations.
- FAISS Search – Retrieve top-k relevant chunks for a query.
- Prompt Augmentation – Combine query + retrieved context.
- LLM Answer Generation – GPT-3.5 produces final grounded answer.

### Features
- Works with .txt and .pdf files.
- Automatically chunks large documents for better search.
- Uses semantic embeddings for accurate retrieval.
- Provides fact-based answers, reducing hallucinations.
- Simple web interface for interaction.

### Next Steps (Upcoming Projects)
- Project 4: Add memory support (multi-turn conversations).
- Support multiple document uploads.
- Replace GPT-3.5 with open-source LLMs (offline mode).

**See:** [theory.md](./theory.md) for detailed technical explanations of embeddings, FAISS, chunking, and the full RAG process.