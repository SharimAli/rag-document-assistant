# RAG Document Assistant

A small Retrieval-Augmented Generation (RAG) system: upload PDFs/text files,
it indexes them in a local vector database, and you can then ask questions
that get answered using only the content of those documents (with sources
cited) instead of the model's general knowledge.

Built to demonstrate the core skills behind LLM application development:
prompt engineering, embeddings, vector search, and wiring an LLM API into a
working backend + chat UI.

## How it works

```
PDF/TXT files
     |
     v
 chunking.py  -->  split into overlapping text chunks
     |
     v
embeddings.py -->  each chunk turned into a vector (OpenAI embeddings)
     |
     v
 ChromaDB     -->  vectors stored locally in a persistent vector database
     |
     |   (at question time)
     v
question --> embed question --> find closest chunks (cosine similarity)
     |
     v
  rag.py      -->  build a prompt: "answer using ONLY this context"
     |
     v
Claude / GPT  -->  generates the final answer
     |
     v
 FastAPI      -->  /ask endpoint returns {answer, sources}
     |
     v
 index.html   -->  simple chat UI in the browser
```

## Tech stack

- **Python** - backend logic
- **FastAPI** - REST API (`POST /ask`)
- **ChromaDB** - local vector database for storing embeddings
- **OpenAI embeddings API** - turns text into vectors
- **Claude (Anthropic) or GPT (OpenAI) API** - generates answers (swap with one config value)
- **pypdf** - extracts text from PDF files
- **Vanilla HTML/JS** - minimal chat interface, no framework needed

No LangChain/LlamaIndex was used on purpose - the chunking and retrieval
logic is written from scratch (see `app/chunking.py`, `app/rag.py`) so the
mechanics of RAG are fully visible and explainable rather than hidden
behind a framework. It's also straightforward to re-implement the same
pipeline with LangChain if a team prefers that.

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys
cp .env.example .env
# then edit .env and paste in your OPENAI_API_KEY and ANTHROPIC_API_KEY

# 4. Index the sample document (or drop your own PDFs into data/)
python -m app.ingest --path ./data

# 5. Start the server
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** and ask something like:
*"What is the probation period?"* or *"How many leave days do employees get?"*
— it will answer from `data/sample_docs/company_handbook.txt` and cite the source file.

## Running the tests

```bash
pytest
```

Tests cover the chunking logic and don't require any API keys.

## Project structure

```
rag-document-assistant/
├── app/
│   ├── config.py       # env-driven settings, no hardcoded secrets
│   ├── chunking.py      # splits text into overlapping chunks
│   ├── embeddings.py    # text -> vector via OpenAI
│   ├── ingest.py        # load docs -> chunk -> embed -> store in ChromaDB
│   ├── rag.py           # retrieve chunks -> build prompt -> call LLM
│   └── main.py           # FastAPI app (/ask endpoint + serves the UI)
├── static/index.html    # minimal chat UI
├── data/sample_docs/    # sample document to try it out immediately
├── tests/test_chunking.py
├── requirements.txt
└── .env.example
```

## Possible extensions

Things worth mentioning as "what I'd add next" in an interview:
- Swap ChromaDB for a hosted vector DB (Pinecone / Qdrant / pgvector) for production
- Stream the LLM's response token-by-token instead of waiting for the full answer
- Add conversation memory for multi-turn chat instead of single Q&A
- Re-rank retrieved chunks before sending them to the LLM
- Add a React/Next.js frontend instead of the plain HTML page
- Docker-ize the app for deployment

## Why this project

This mirrors the actual shape of client work in LLM application development:
document ingestion, embeddings, a vector database, prompt engineering, and an
LLM API wired into a real backend and UI - the same building blocks used for
AI chatbots, knowledge-base assistants, and RAG-based SaaS products.
