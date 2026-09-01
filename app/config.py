"""
Central configuration for the RAG Document Assistant.
Everything is driven by environment variables (loaded from a .env file)
so no secrets ever get hardcoded or committed to git.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- LLM provider settings ---
# Which model answers the question: "anthropic" or "openai".
# Swapping providers is a one-line change here - nothing else in the
# app needs to know which vendor is being used.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL_ANTHROPIC = os.getenv("LLM_MODEL_ANTHROPIC", "claude-sonnet-4-5")
LLM_MODEL_OPENAI = os.getenv("LLM_MODEL_OPENAI", "gpt-4o-mini")

# Embeddings use OpenAI, since Anthropic does not yet expose an
# embeddings endpoint. This can point at any embeddings provider.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- RAG / chunking settings ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))  # overlap between chunks
TOP_K = int(os.getenv("TOP_K", 4))                    # chunks retrieved per question

# --- Storage ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
DOCS_DIR = os.getenv("DOCS_DIR", "./data")
