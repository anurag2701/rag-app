from fastapi import FastAPI, UploadFile, File, BackgroundTasks

import os
import shutil
import sqlite3
from datetime import datetime, timezone
from threading import Lock

from rag_app.ingest import ingest_file
from rag_app.query import ask_question



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