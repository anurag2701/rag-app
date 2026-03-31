from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag_app.llm_client import generate

CHROMA_PATH = "vectordb"


def ask_question(question, selected_files=None):
    embeddings = HuggingFaceEmbeddings()

    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Retrieve documents
    if selected_files:
        docs = vectordb.similarity_search(
            question,
            k=5,
            filter={"file_name": {"$in": selected_files}}
        )
    else:
        docs = vectordb.similarity_search(question, k=5)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful assistant. Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    answer = generate(prompt)

    return {
        "answer": answer,
        "sources": list(set([doc.metadata["file_name"] for doc in docs]))
    }