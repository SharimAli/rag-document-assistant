"""
The core RAG loop:
  1. Embed the user's question
  2. Retrieve the most similar chunks from ChromaDB
  3. Build a grounded prompt from those chunks
  4. Ask an LLM (Claude or OpenAI - configurable) to answer using only that context
"""

import chromadb

from . import config
from .embeddings import embed_query


def _get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection(config.COLLECTION_NAME)


def retrieve(question: str, top_k: int = None):
    """Return the chunks (and their source filenames) most relevant to `question`."""
    top_k = top_k or config.TOP_K
    collection = _get_collection()

    query_embedding = embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    chunks = results["documents"][0] if results["documents"] else []
    sources = [m["source"] for m in results["metadatas"][0]] if results["metadatas"] else []
    return chunks, sources


def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""You are a helpful assistant that answers questions using ONLY the context below.
If the answer isn't contained in the context, say you don't know instead of guessing.

Context:
{context}

Question: {question}

Answer:"""


def _call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=config.LLM_MODEL_ANTHROPIC,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_MODEL_OPENAI,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content


def generate_answer(prompt: str) -> str:
    if config.LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    return _call_anthropic(prompt)


def answer_question(question: str) -> dict:
    chunks, sources = retrieve(question)

    if not chunks:
        return {
            "answer": "I couldn't find anything relevant in the indexed documents. "
            "Run `python -m app.ingest` first to build the index.",
            "sources": [],
        }

    prompt = build_prompt(question, chunks)
    answer = generate_answer(prompt)

    # de-duplicate sources while preserving order
    unique_sources = list(dict.fromkeys(sources))

    return {"answer": answer, "sources": unique_sources}
