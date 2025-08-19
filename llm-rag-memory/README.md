# Document RAG Chatbot with Memory

This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows users to:
- Upload a PDF document.
- Automatically process the document and extract knowledge.
- Ask questions about the content of the document.
- Maintain conversation history (memory) for a natural chat experience.

The chatbot uses LangChain, OpenAI GPT models, and Gradio for the user interface.

---

## Features
- Upload and process PDF documents.
- Conversational Retrieval Chain for intelligent context-aware Q&A.
- Memory support (chat history) so the chatbot remembers past questions and answers.
- Simple and clean Gradio UI.

---

## Tech Stack & Libraries
- Python 3.10+
- LangChain – for building RAG pipeline
- OpenAI – LLM provider (GPT models)
- Gradio – UI framework
- [PyPDF / FAISS / ChromaDB] (handled inside utils.py) – for document processing and vector search

---

## Project Structure
```
📦 Document-RAG-Chatbot-Memory
├── app.py          # Main Gradio app
├── utils.py        # Utility functions (e.g., process_pdf)
├── requirements.txt
├── README.md       # Project overview & usage
└── theory.md       # Detailed theoretical background
```

## Installation

### 1. Clone the repository
```
git clone https://github.com/yourusername/document-rag-chatbot-memory.git
cd document-rag-chatbot-memory
```

### 2. Create and activate virtual environment
```
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set your OpenAI API key
```
export OPENAI_API_KEY="your_api_key_here"   # Linux/Mac
setx OPENAI_API_KEY "your_api_key_here"     # Windows
```
--- 
## Usage
Run the app:
```
python app.py
```

The Gradio UI will open in your browser.

Steps in UI:
- Upload a PDF document.
- Click Process Document.
- Ask questions in the chatbox.
- Enjoy an interactive, memory-enabled RAG chatbot.

###  Example
- Upload a research paper (PDF).
- Ask: "What is the main contribution of this paper?"
- Follow up: "Explain it in simple terms."
- The chatbot will answer with context from the PDF and remember the conversation.

### Future Improvements
- Support for multiple file types (Word, TXT, etc.).
- Option to switch between memory-enabled and stateless chat.
- UI improvements (file history, chat export).
- Add embeddings database persistence (e.g., Chroma/FAISS).

### Contributing
Pull requests are welcome! If you’d like to add features or fix bugs, please fork the repo and submit a PR.

### License

MIT License.