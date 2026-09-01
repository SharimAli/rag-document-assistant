"""
Ingestion pipeline: load documents -> chunk -> embed -> store in a
local ChromaDB vector database.

Run directly:
    python -m app.ingest --path ./data
"""

import argparse
import glob
import os

import chromadb
from pypdf import PdfReader

from . import config
from .chunking import chunk_text
from .embeddings import embed_texts


def load_documents(path: str):
    """Read every .pdf and .txt file under `path`, return (filename, text) pairs."""
    docs = []

    for filepath in glob.glob(os.path.join(path, "**/*.pdf"), recursive=True):
        reader = PdfReader(filepath)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        docs.append((os.path.basename(filepath), text))

    for filepath in glob.glob(os.path.join(path, "**/*.txt"), recursive=True):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            docs.append((os.path.basename(filepath), f.read()))

    return docs


def build_index(docs_dir: str = None) -> None:
    docs_dir = docs_dir or config.DOCS_DIR

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(config.COLLECTION_NAME)

    documents = load_documents(docs_dir)
    if not documents:
        print(f"No .pdf or .txt files found in '{docs_dir}'.")
        return

    all_chunks, all_ids, all_metadatas = [], [], []
    chunk_counter = 0

    for filename, text in documents:
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print(f"{filename}: {len(chunks)} chunk(s)")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk-{chunk_counter}")
            all_metadatas.append({"source": filename})
            chunk_counter += 1

    if not all_chunks:
        print("Documents were found but produced no text (e.g. scanned/image PDFs).")
        return

    # Embed and upsert in batches so we don't send an oversized request.
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        batch_meta = all_metadatas[i : i + batch_size]

        embeddings = embed_texts(batch_chunks)
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_chunks,
            metadatas=batch_meta,
        )
        print(f"  indexed chunks {i}-{i + len(batch_chunks) - 1}")

    print(f"\nDone. Indexed {len(all_chunks)} chunks from {len(documents)} document(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument("--path", default=config.DOCS_DIR, help="Folder containing PDF/TXT files")
    args = parser.parse_args()
    build_index(args.path)
