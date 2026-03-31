from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag_app.file_registry import file_exists, register_file

import hashlib
import json
import os

REGISTRY_FILE = "file_registry.json"

CHROMA_PATH = "vectordb"

def ingest_file(file_path, file_name):
    print(f"Starting ingestion for '{file_name}'...")
    if file_exists(file_path):
        msg = f"File '{file_name}' already ingested. Skipping."
        print(msg)
        return {"status": "skipped", "message": msg}

    # Load document
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Add metadata
    for chunk in chunks:
        chunk.metadata["file_name"] = file_name

    embeddings = HuggingFaceEmbeddings()

    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    vectordb.add_documents(chunks)
    vectordb.persist()

    # Register file after ingestion
    register_file(file_path, file_name)

    return {"status": "completed", "message": f"File '{file_name}' ingested successfully."}