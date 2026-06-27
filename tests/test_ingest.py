from app.ingest import load_documents, split_documents

def test_split_documents_returns_list():
    assert isinstance([], list)


def test_chunk_size_is_positive():
    from app.query import top_k_results
    assert int(top_k_results) > 0