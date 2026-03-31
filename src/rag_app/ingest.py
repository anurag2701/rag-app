from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "vectordb"


def ingest_file(file_path, file_name):
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