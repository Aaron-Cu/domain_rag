import os
import hashlib
import shutil
from PyPDF2 import PdfReader
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

def generate_checksum(directory: str) -> str:
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            if filename.endswith(".pdf"):
                with open(os.path.join(root, filename), "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_checksum(path: str) -> str | None:
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return None

def save_checksum(path: str, checksum: str):
    with open(path, "w") as f:
        f.write(checksum)

def clear_vector_store(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def load_files_from_directory(directory: str) -> list[Document]:
    documents = []
    logger.info(f"Loading files from {directory}")
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filename.endswith(".pdf"):
                reader = PdfReader(filepath)
                text = "".join([page.extract_text() or "" for page in reader.pages])
            elif filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                continue
            documents.append(Document(page_content=text, metadata={"title": filename}))
    return documents
