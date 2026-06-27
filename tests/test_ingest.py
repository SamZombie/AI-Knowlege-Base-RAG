import pytest
from langchain_core.documents import Document
from app.ingest import split_documents


def test_split_documents_returns_list():
    docs = [Document(page_content="This is a test document with some content.", metadata={"source": "test.pdf", "page": 0})]
    chunks = split_documents(docs)
    assert isinstance(chunks, list)


def test_split_documents_preserves_metadata():
    docs = [Document(page_content="This is a test document with some content.", metadata={"source": "test.pdf", "page": 0})]
    chunks = split_documents(docs)
    assert chunks[0].metadata["source"] == "test.pdf"


def test_split_documents_empty_input():
    chunks = split_documents([])
    assert chunks == []


def test_split_documents_chunk_count():
    long_text = "word " * 500
    docs = [Document(page_content=long_text, metadata={"source": "test.pdf", "page": 0})]
    chunks = split_documents(docs)
    assert len(chunks) > 1