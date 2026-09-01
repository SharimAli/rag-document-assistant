"""
Splits raw text into overlapping chunks small enough to embed and
retrieve individually. Written from scratch (no LangChain) so the
mechanics of chunking are easy to see and explain.
"""

from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split `text` into overlapping chunks of roughly `chunk_size`
    characters, preferring to break on a sentence or line boundary
    rather than mid-word.

    Overlap keeps context from being lost at chunk edges - if an
    answer spans a boundary, both neighboring chunks still contain
    enough of it to be retrieved.
    """
    text = text.strip()
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        if end < text_len:
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            # Only trust the break point if it's not absurdly early -
            # otherwise we'd produce tiny, near-useless chunks.
            if break_point > chunk_size * 0.5:
                chunk = chunk[: break_point + 1]
                end = start + len(chunk)

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks
