import openai
from sentence_traansformers import SentenceTransformer
import faiss
import gradio as gr

openai.api_key = ""

#load ebedding model
embedder = SentenceTransformer("all-MiniLM-L6-V2")


# some example document
docs = [
"Large Language Models are deep learning models trained on vast amounts of text data.",
    "Retrieval-Augmented Generation combines search with generation to produce context-aware responses.",
    "FAISS is a library for efficient similarity search of embeddings.",
    "OpenAI provides API access to powerful LLMs like GPT-3.5 and GPT-4."

]
# create embeddings
doc_embeddings = embedder.encode(docs)

# create FAISS index
dimensions =doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimensions)
index.add(doc_embeddings)

# search + generate function
def rag_answer(query):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k=2) # top 2 relevant docs
    retrieved = "\n".join([docs[i] for i in I[0]])

    prpompt = f"Context:\n{retrieved}\n\nQuestion: {query}\nAnswer:"

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role":"system", "content": "You are a helpful assistant."},
            {"role":"user", "content":prpompt}
        ],
        temperature = 0.2
    )
    return response['choices'][0]['message']['content'].strip()


# Gradio interface
gr.Interface(fn=rag_answer, inputs = "text", outputs="text").launch()