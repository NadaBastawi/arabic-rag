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


def test_hybrid_candidates_find_author_terms_without_cross_workspace_leakage():
    from app.services.multi_stage_retrieval import hybrid_candidates
    class Store:
        metadata = [1, 2, 3]
        def search(self, vector, top_k, workspace_id=None):
            docs = [
                {'text': 'كتاب الذكاء العاطفي وتفاصيل كثيرة', 'score': .85, 'document_id': 'noise', 'workspace': 'a'},
                {'text': 'الذكاء العاطفي تأليف دانييل ترجمة ليلى', 'score': .25, 'document_id': 'cover', 'workspace': 'a'},
                {'text': 'تأليف آخر ترجمة آخر', 'score': .99, 'document_id': 'private', 'workspace': 'b'},
            ]
            return [doc for doc in docs if workspace_id is None or doc['workspace'] == workspace_id][:top_k]
    result = hybrid_candidates(Store(), None, 'من مؤلف كتاب الذكاء العاطفي ومن قام بترجمته؟', 2, 'a')
    assert result[0]['document_id'] == 'cover'
    assert all(doc['workspace'] == 'a' for doc in result)
    selected = hybrid_candidates(Store(), None, 'من المؤلف والمترجم؟', 5, 'a', 'noise')
    assert [doc['document_id'] for doc in selected] == ['noise']
