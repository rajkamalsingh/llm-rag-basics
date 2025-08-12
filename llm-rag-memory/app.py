import gradio as gr
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils import process_pdf

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="chat-history", return_messages=True)


def start_chat(upload_file):
    file_path = f"data/{upload_file.name}"
    upload_file.save(file_path)

    global qa_chain
    vectorstore = process_pdf(file_path)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return "File uploaded and processed. Ask me anything about it! "


def chat(message, history):
    if qa_chain:
        response = qa_chain.run(message)
        return response
    else:
        return "Please upoad a document first. "

with gr.Blocks() as demo:
    gr.Markdown("##RAG chatbot with memory")
    file_upload = gr.File(label="Upload PDF", type ="file")
    upload_btn = gr.Button("process Document")
    chat_box = gr.ChatInterface(fn=chat)
    upload_btn.click(start_chat, inputs=file_upload, outputs = chat_box.textbox)

demo.launch