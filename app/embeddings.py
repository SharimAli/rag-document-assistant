"""
Thin wrapper around the embeddings API. Kept in its own module so the
rest of the app never has to know which provider generates vectors.
"""

from typing import List
from openai import OpenAI

from . import config

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts in one API call."""
    client = _get_client()
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
