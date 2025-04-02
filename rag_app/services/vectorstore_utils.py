# vectorstore_utils.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def split_documents(documents, chunk_size=2500, chunk_overlap=250):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def build_or_load_vectorstore(documents, persist_path, embedding_function):
    if not documents:
        raise ValueError("No documents found to embed. Ensure your input directory has valid .pdf or .txt files with readable content.")
    return Chroma.from_documents(documents, embedding=embedding_function, persist_directory=persist_path)
