"""
Simple unit tests for the chunking logic - no API keys required.
Run with: pytest
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=10) == []


def test_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_produces_multiple_chunks():
    text = "Sentence number {}. " * 200
    text = text.format(*range(200))
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    # every chunk should respect the size limit reasonably closely
    for c in chunks:
        assert len(c) <= 250


def test_overlap_must_be_smaller_than_chunk_size():
    try:
        chunk_text("some text", chunk_size=50, overlap=50)
        assert False, "expected a ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_empty_text_returns_no_chunks()
    test_short_text_returns_single_chunk()
    test_long_text_produces_multiple_chunks()
    test_overlap_must_be_smaller_than_chunk_size()
    print("All tests passed.")
