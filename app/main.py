"""
FastAPI backend for the RAG Document Assistant.

Run with:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .rag import answer_question

app = FastAPI(title="RAG Document Assistant")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: Question):
    """Answer a question using the indexed documents (RAG)."""
    return answer_question(payload.question)


@app.get("/")
def root():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
