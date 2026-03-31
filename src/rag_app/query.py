from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_app.llm_client import generate, stream_generate

CHROMA_PATH = "vectordb"


def _build_vectordb():
    embeddings = HuggingFaceEmbeddings()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


def _retrieve_documents(vectordb, question, selected_files=None):
    if selected_files:
        return vectordb.similarity_search(
            question,
            k=5,
            filter={"file_name": {"$in": selected_files}},
        )
    return vectordb.similarity_search(question, k=5)


def _build_prompt(question, context):
    return f"""
You are a helpful assistant. Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""


def ask_question(question, selected_files=None):
    vectordb = _build_vectordb()
    docs = _retrieve_documents(vectordb, question, selected_files)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = _build_prompt(question, context)

    answer = generate(prompt)

    return {
        "answer": answer,
        "sources": list({doc.metadata.get("file_name") for doc in docs if doc.metadata.get("file_name")}),
    }


def stream_question(question, selected_files=None):
    vectordb = _build_vectordb()
    docs = _retrieve_documents(vectordb, question, selected_files)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = _build_prompt(question, context)

    yield from stream_generate(prompt)