# rag-app

A small RAG-powered FastAPI application for uploading documents and querying them with an LLM.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup

Install dependencies:

```bash
uv sync
```

## Run the app

Start the development server with:

```bash
uv run python -m uvicorn main:app --reload
```

## Project structure

- `main.py` - application entry point
- `src/rag_app/app.py` - FastAPI app definition
- `src/rag_app/ingest.py` - document ingestion logic
- `src/rag_app/query.py` - query helpers
- `src/rag_app/llm_client.py` - LLM client utilities

## Notes

If you add new documents, place them in the document directories under `src/` and update the ingestion flow as needed.
