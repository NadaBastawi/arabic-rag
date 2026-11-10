"""Safety checks for staging a separate, evidence-viewable PDF index."""
import sys
from pathlib import Path
import numpy as np
import pytest
from app import reindex_pdf as reindex
from app.utils.document_loader import DocumentChunk


def test_reindex_stages_source_and_refuses_original_paths(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'faiss', None)
    old_index = tmp_path / 'old-index'
    old_documents = tmp_path / 'old-documents'
    old_index.mkdir(); old_documents.mkdir()
    (old_index / 'sentinel').write_text('preserve')
    source = tmp_path / 'original.pdf'
    source.write_bytes(b'%PDF-original')
    monkeypatch.setattr(reindex.config, 'INDEX_PATH', str(old_index))
    monkeypatch.setattr(reindex.config, 'DOCS_PATH', str(old_documents))
    monkeypatch.setattr(reindex.config, 'EMBEDDING_MODEL', 'test-model')
    class Embedder:
        def __init__(self, **kwargs): pass
        def chunking_config(self): return {}
        def embed(self, texts): return np.ones((len(texts), 3), dtype='float32')
        def get_embedding_dimension(self): return 3
    class Loader:
        def __init__(self, **kwargs):
            self.last_report = {'ocr_required_pages': []}
        def load_file(self, path):
            return [DocumentChunk('supporting passage', 'old', 0, path, page_num=7)]
    monkeypatch.setattr(reindex, 'EmbeddingService', Embedder)
    monkeypatch.setattr(reindex, 'DocumentLoader', Loader)
    with pytest.raises(ValueError, match='separate'):
        reindex.reindex_pdf(source, old_index / 'nested', 'test')
    with pytest.raises(ValueError, match='workspace'):
        reindex.reindex_pdf(source, tmp_path / 'invalid', '../escape')
    result = reindex.reindex_pdf(source, tmp_path / 'staged', 'test')
    copied = Path(result['documents_path'], 'test', result['document_id'] + '.pdf')
    assert copied.read_bytes() == source.read_bytes() == b'%PDF-original'
    assert (old_index / 'sentinel').read_text() == 'preserve'
    store = reindex.FAISSVectorStore(result['index_path'], embedding_model='test-model')
    assert store.metadata[0]['page_num'] == 7
    assert store.metadata[0]['document_id'] == result['document_id']
    assert store.metadata[0]['workspace_id'] == 'test'
