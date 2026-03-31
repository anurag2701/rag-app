from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Body
from fastapi.responses import StreamingResponse

import os
import shutil
import sqlite3
from datetime import datetime, timezone
from threading import Lock

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_app.ingest import ingest_file
from rag_app.query import ask_question, stream_question

from rag_app.file_registry import file_exists, register_file, remove_file, clear_registry

from typing import List
import time

app = FastAPI()

DOCS_PATH = "documents"
os.makedirs(DOCS_PATH, exist_ok=True)

VECTORDB_PATH = "./vectordb"
os.makedirs(VECTORDB_PATH, exist_ok=True)

STATUS_DB_PATH = os.path.join(VECTORDB_PATH, "file_statuses.sqlite3")
FILE_STATUS_LOCK = Lock()



def init_status_db() -> None:
    with sqlite3.connect(STATUS_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_statuses (
                file_name TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def set_file_status(file_name: str, status: str) -> None:
    updated_at = datetime.now(timezone.utc).isoformat()
    with FILE_STATUS_LOCK:
        with sqlite3.connect(STATUS_DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO file_statuses (file_name, status, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(file_name) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (file_name, status, updated_at),
            )
            conn.commit()


def get_file_status(file_name: str) -> str:
    with FILE_STATUS_LOCK:
        with sqlite3.connect(STATUS_DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT status FROM file_statuses WHERE file_name = ?",
                (file_name,),
            )
            row = cursor.fetchone()
    return row[0] if row else "unknown"





def ingest_file_with_status(file_path: str, file_name: str) -> None:
    set_file_status(file_name, "processing")
    try:
        ingest_file(file_path, file_name)
        set_file_status(file_name, "completed")
    except Exception:
        set_file_status(file_name, "failed")
        raise


init_status_db()


@app.get("/")
def home():

    return {"message": "RAG API running"}


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = os.path.join(DOCS_PATH, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check duplicate before scheduling ingestion
    if file_exists(file_path):
        # set_file_status(file.filename, "skipped")
        return {"message": f"File '{file.filename}' already ingested. Skipping."}

    set_file_status(file.filename, "queued")
    background_tasks.add_task(ingest_file_with_status, file_path, file.filename)

    return {"message": f"{file.filename} uploaded and queued for indexing"}


@app.get("/files")
def list_files():
    files = []
    for file_name in os.listdir(DOCS_PATH):
        files.append({
            "file_name": file_name,
            "status": get_file_status(file_name),
        })
    return {"files": files}



@app.post("/ask")
def ask(question: str, files: list[str] = []):
    answer = ask_question(question, files)
    return {"answer": answer}


def delete_uploaded_file(file_name: str) -> dict:
    file_path = os.path.join(DOCS_PATH, file_name)

    if os.path.exists(file_path):
        os.remove(file_path)

    with FILE_STATUS_LOCK:
        with sqlite3.connect(STATUS_DB_PATH) as conn:
            conn.execute(
                "DELETE FROM file_statuses WHERE file_name = ?",
                (file_name,),
            )
            conn.commit()

    removed_from_registry = remove_file(file_name)

    vectordb = Chroma(
        persist_directory=VECTORDB_PATH,
        embedding_function=HuggingFaceEmbeddings(),
    )
    try:
        vectordb.delete(where={"file_name": file_name})
    except Exception:
        pass

    return {
        "file_deleted": not os.path.exists(file_path),
        "registry_updated": removed_from_registry,
    }


def reset_files_storage() -> dict:
    documents_deleted = 0
    for entry_name in os.listdir(DOCS_PATH):
        entry_path = os.path.join(DOCS_PATH, entry_name)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)
        documents_deleted += 1

    vectordb_deleted = 0
    for entry_name in os.listdir(VECTORDB_PATH):
        entry_path = os.path.join(VECTORDB_PATH, entry_name)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)
        vectordb_deleted += 1

    clear_registry()
    init_status_db()

    return {
        "documents_deleted": documents_deleted,
        "vectordb_items_deleted": vectordb_deleted,
    }


@app.delete("/files/{file_name}")
def delete_file(file_name: str):
    result = delete_uploaded_file(file_name)
    return {
        "message": f"Deleted '{file_name}'",
        **result,
    }


@app.post("/reset-files")
def reset_files():
    result = reset_files_storage()
    return {
        "message": "All files and vector data have been reset.",
        **result,
    }


@app.post("/ask/stream")
def ask_stream(question: str, files: List[str] = Body(default=[])):
    def event_stream():
        for chunk in stream_question(question, files):
            yield chunk + "\n"
            time.sleep(0.01)   # helps flushing

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/test-stream")
def test_stream():
    def generate():
        import time
        for i in range(5):
            yield f"Hello {i}\n"
            time.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")