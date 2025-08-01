import os
import fitz # PyMuPDF
import openai
import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss

# set api keys
openai.api_key = os.getenv("OPENAI_API_KEY", "your key here")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# document loader
def load_document(file_path):
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding = "utf-8") as f:
            text = f.read()

    elif file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()

    else:
        raise ValueError("Unsupported file format. please upload .txt or .pdf")
    return text

# split text into chunks
def chunk_text(text, chunk_size = 500, overlap= 50):
    words = text.split()
    chunks = []
    start = 0
    while start <len(words):
        end = start = chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size-overlap

    return chunks

# build faiss index
def build_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

# RAG function
def rag_answer(file, query):
    #load and process document
    text = load_document(file.name)
    chunks = chunk_text(text)

    # build faiss index
    index, chunk_list = build_index(chunks)

    # search for relevant chunks
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k=3)
    retrieved = "\n".join([chunk_list[i] for i in I[0]])

    # generate answer with gpt-3.5
    prompt = f"Context:\n{retrieved}\n\nQuestion: {query}\nAnswer:"

    from openai import OpenAI
    client = OpenAI(api_key = openai.api_key)

    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role":"system", "content":"You are a knowledgeable assistant that answers based on the given context only."},
            {"role":"user", "content":prompt}
        ],
        temperature = 0.2
    )
    return response.choices[0].message.content.strip()

# gradio interface
iface = gr.Interface(
    fn = rag_answer,
    inputs = [gr.File(file_types=[".txt",".pdf"]), gr.Textbox(label="Ask a question")],
    outputs = "text",
    title = "Rag chatbot with document upload",
    description= "Upload a .txt or .pdf file, then ask questions based on its content"

)

iface.launch(shre=True)