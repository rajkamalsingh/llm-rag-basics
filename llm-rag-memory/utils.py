from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY", "your_key ")
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
    chunks = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(openai_api_key="your key")
    vectorstore = FAISS.from_documents(chunks,embeddings)

    return vectorstore