from app.services.multi_stage_retrieval import ArabicReRanker


def test_reranker_preserves_source_metadata():
    candidate = {
        "text": "معلومة مهمة عن الموظفين",
        "score": 0.9,
        "document_id": "doc-1",
        "source": "policy.pdf",
        "metadata": {
            "workspace_id": "workspace-1",
            "chunk_id": 3,
            "page_num": 7,
        },
    }

    result = ArabicReRanker().rerank("الموظفين", [candidate], top_k=1)[0]

    assert result["document_id"] == "doc-1"
    assert result["source"] == "policy.pdf"
    assert result["metadata"] == candidate["metadata"]
