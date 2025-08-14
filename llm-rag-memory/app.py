import gradio as gr
import os
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils import process_pdf

llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 openai_api_key="")
# Global variables
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
rag_chain = None  # will hold the RAG pipeline
chat_history = []

def upload_and_process(file_obj):
    """Handles file upload and RAG pipeline creation."""
    global rag_chain

    if not file_obj:
        return "No file uploaded."

    file_path = file_obj.name
    print(f"Processing file: {file_path}")

    # Process PDF (your utils.py function)
    retriever = process_pdf(file_path)

    # Create Conversational Retrieval Chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Replace with your LLM, e.g., ChatOpenAI(model_name="gpt-4o-mini")
        retriever=retriever.as_retriever(),
        memory=memory
    )
    chat_history.clear()
    return f"File '{os.path.basename(file_path)}' processed successfully! You can now chat."


def chat_with_rag(user_input):
    """Handles user chat with RAG pipeline."""
    global chat_history

    if rag_chain is None:
        return "Please upload and process a document first.", chat_history

    result = rag_chain({"question": user_input})
    answer = result["answer"]

    # Append as tuple for Gradio format
    chat_history.append((user_input, answer))
    return "", chat_history  # "" clears the input box



# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Document RAG Chatbot with Memory")

    with gr.Row():
        file_upload = gr.File(label="Upload PDF Document", file_types=[".pdf"])
        upload_btn = gr.Button("Process Document")

    status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Question")
    send_btn = gr.Button("Send")

    upload_btn.click(upload_and_process, inputs=file_upload, outputs=status)
    send_btn.click(chat_with_rag, inputs=msg, outputs=[msg, chatbot])

if __name__ == "__main__":
    demo.launch()