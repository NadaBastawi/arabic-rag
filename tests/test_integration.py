"""Integration and API tests for the Arabic RAG system."""

import hashlib
import tempfile
from typing import List, Union

import numpy as np
import pytest
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from app import main as api_main
from app.services.cache import EmbeddingCache, QueryResultCache
from app.services.faiss_store import FAISSVectorStore
from app.services.rag_pipeline import RAGPipeline
from app.utils.document_loader import DocumentChunk, DocumentLoader


class MockEmbeddingService:
    """Deterministic local embedding service for integration tests."""

    def __init__(self, dim: int = 48):
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = text.split() or [text]
        for token in tokens:
            idx = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            return self._embed_one(texts)
        return np.vstack([self._embed_one(text) for text in texts])


class MockLLMService:
    def generate(self, prompt: str) -> str:
        return "mock-answer"


@pytest.fixture
def embedding_service():
    return MockEmbeddingService()


@pytest.fixture
def vector_store():
    with tempfile.TemporaryDirectory() as tmp:
        yield FAISSVectorStore(index_path=tmp)


def test_document_loader_chunks_text_with_overlap():
    loader = DocumentLoader(chunk_size=80, chunk_overlap=20, min_chunk_size=10)
    long_text = "\n\n".join(
        [
            "paragraph one has enough words for chunking",
            "paragraph two continues with additional content",
            "paragraph three ensures split and overlap",
            "paragraph four keeps the text long enough",
        ]
        * 4
    )

    chunks = loader._chunk_text(long_text, "doc1", "memory")

    assert len(chunks) > 1
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_document_loader_normalizes_arabic_pdf_glyphs_and_reports_quality():
    loader = DocumentLoader()

    normalized = loader._normalize_extracted_text("اﻟﺬﻛﺎء  \t اﻟﻌﺎﻃﻔﻲ\x00")
    assert normalized == "الذكاء العاطفي"

    usable = loader._assess_page_quality("الذكاء العاطفي " * 20)
    poor = loader._assess_page_quality("ABCDEFG262")
    assert usable["ocr_required"] is False
    assert poor["ocr_required"] is True


def test_vector_store_add_search_delete_and_persist(embedding_service):
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSVectorStore(index_path=tmp)

        texts = ["?????? ?????????", "?????? ?????", "?????? ????? ???????"]
        embeddings = embedding_service.embed(texts)
        store.add_documents(texts, embeddings, document_ids=["d1", "d2", "d3"])

        results = store.search(embedding_service.embed("??????"), top_k=2)
        assert len(results) >= 1
        assert "text" in results[0]

        assert store.delete_document("d2")
        assert all(item["document_id"] != "d2" for item in store.metadata)

        reloaded = FAISSVectorStore(index_path=tmp)
        assert len(reloaded.metadata) == len(store.metadata)


def test_snapshot_save_failure_keeps_committed_index_and_memory(tmp_path, monkeypatch):
    import os

    store = FAISSVectorStore(str(tmp_path), atomic=True)
    store.add_documents(["first"], np.array([[1.0, 0.0]], dtype=np.float32), ["first"])
    previous = (tmp_path / "CURRENT").read_text()

    def fail_replace(*args):
        raise OSError("injected commit failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="injected commit failure"):
            store.add_documents(["second"], np.array([[0.0, 1.0]], dtype=np.float32), ["second"])
    assert (tmp_path / "CURRENT").read_text() == previous
    assert [item["document_id"] for item in store.metadata] == ["first"]
    reopened = FAISSVectorStore(str(tmp_path), atomic=True)
    assert [item["document_id"] for item in reopened.metadata] == ["first"]
    store.add_documents(["third"], np.array([[0.0, 1.0]], dtype=np.float32), ["third"])
    assert len(list((tmp_path / "generations").iterdir())) >= 2
    assert [item["document_id"] for item in FAISSVectorStore(str(tmp_path), atomic=True).metadata] == ["first", "third"]


def test_rejected_remote_pointer_preserves_old_generation(tmp_path):
    class Pointer:
        value = ""
        rejected = False
        def read(self):
            return self.value
        def write(self, value):
            if self.rejected:
                raise RuntimeError("lease lost")
            self.value = value
    pointer = Pointer()
    store = FAISSVectorStore(str(tmp_path), atomic=True, pointer_store=pointer)
    store.add_documents(["first"], np.array([[1.0, 0.0]], dtype=np.float32), ["first"])
    previous = pointer.value
    pointer.rejected = True
    with pytest.raises(RuntimeError, match="lease lost"):
        store.add_documents(["second"], np.array([[0.0, 1.0]], dtype=np.float32), ["second"])
    assert pointer.value == previous
    assert [item["document_id"] for item in store.metadata] == ["first"]
    reloaded = FAISSVectorStore(str(tmp_path), atomic=True, pointer_store=pointer)
    assert [item["document_id"] for item in reloaded.metadata] == ["first"]


def test_vector_store_dimension_mismatch_can_reset(embedding_service):
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSVectorStore(index_path=tmp)
        texts = ["نص اول", "نص ثاني"]
        store.add_documents(texts, embedding_service.embed(texts))

        with pytest.raises(ValueError):
            store.ensure_embedding_dimension(64, reset_on_mismatch=False)

        store.ensure_embedding_dimension(64, reset_on_mismatch=True)
        stats = store.get_stats()

        assert stats["embedding_dim"] == 64
        assert stats["total_documents"] == 0
        assert stats["total_vectors"] == 0


def test_vector_store_dimension_mismatch_does_not_delete_data_by_default(embedding_service):
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSVectorStore(index_path=tmp)
        texts = ["نص محفوظ"]
        store.add_documents(texts, embedding_service.embed(texts), document_ids=["safe-doc"])

        with pytest.raises(ValueError, match="dimension mismatch"):
            store.ensure_embedding_dimension(64)

        reloaded = FAISSVectorStore(index_path=tmp)
        assert [item["document_id"] for item in reloaded.metadata] == ["safe-doc"]


def test_vector_store_rejects_embedding_model_change_without_deleting_data(embedding_service):
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSVectorStore(index_path=tmp, embedding_model="model-a")
        store.add_documents(["نص محفوظ"], embedding_service.embed(["نص محفوظ"]))

        changed_model = FAISSVectorStore(index_path=tmp, embedding_model="model-b")
        with pytest.raises(ValueError, match="model mismatch"):
            changed_model.ensure_embedding_dimension(embedding_service.dim)

        reloaded = FAISSVectorStore(index_path=tmp, embedding_model="model-a")
        assert len(reloaded.metadata) == 1


def test_embedding_and_query_cache():
    emb_cache = EmbeddingCache(max_size=2, ttl_hours=1)
    q_cache = QueryResultCache(max_size=2, ttl_hours=1)

    emb_cache.set("??", [0.1, 0.2])
    assert emb_cache.get("??") == [0.1, 0.2]
    assert emb_cache.get("?????") is None

    q_cache.set("????", 3, [{"text": "????"}])
    assert q_cache.get("????", 3) == [{"text": "????"}]
    assert q_cache.get("????", 2) is None


@pytest.fixture
def api_client(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        emb = MockEmbeddingService()
        llm = MockLLMService()
        vs = FAISSVectorStore(index_path=tmp)
        pipeline = RAGPipeline(
            embedding_service=emb,
            llm_service=llm,
            vector_store=vs,
            top_k=3,
            retrieval_mode="basic",
        )
        loader = DocumentLoader(chunk_size=120, chunk_overlap=20, min_chunk_size=10)

        api_main.embedding_service = emb
        api_main.llm_service = llm
        api_main.vector_store = vs
        api_main.rag_pipeline = pipeline
        api_main.embedding_cache = EmbeddingCache(max_size=100, ttl_hours=1)
        api_main.query_cache = QueryResultCache(max_size=100, ttl_hours=1)
        api_main.document_loader = loader
        api_main.document_registry = {}
        monkeypatch.setattr(api_main.config, "AUTH_ENABLED", False)

        def fake_get_services():
            return emb, llm, vs, pipeline

        monkeypatch.setattr(api_main, "get_services", fake_get_services)

        with TestClient(api_main.app) as client:
            yield client


def test_api_index_and_query(api_client):
    response = api_client.post(
        "/index",
        json={"texts": ["?????? ????????? ???", "?????? ????? ???"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["indexed_count"] == 2

    query_response = api_client.post(
        "/query",
        json={"question": "?? ?? ?????? ??????????", "retrieval_mode": "basic", "top_k": 2},
    )
    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert query_payload["answer"] == "mock-answer"
    assert isinstance(query_payload["retrieved_documents"], list)


def test_api_document_add_and_list(api_client):
    add_response = api_client.post(
        "/documents/add",
        json={
            "text": "????? ????? ????? ????????.",
            "document_id": "doc_test",
            "source": "unit-test",
        },
    )
    assert add_response.status_code == 200

    list_response = api_client.get("/documents")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["total_count"] == 1
    assert payload["documents"][0]["document_id"] == "doc_test"


def test_upload_duplicate_and_selected_document_evidence(api_client, monkeypatch, tmp_path):
    monkeypatch.setattr(api_main.config, "DOCS_PATH", str(tmp_path))
    text = ("الذكاء العاطفي يساعد الإنسان على فهم مشاعره. " * 8).encode("utf-8")
    first = api_client.post("/documents/upload", files={"file": ("first.txt", text, "text/plain")})
    assert first.status_code == 200, first.text
    duplicate = api_client.post("/documents/upload", files={"file": ("renamed.txt", text, "text/plain")})
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["document_id"] == first.json()["document_id"]
    assert api_client.get("/documents").json()["total_count"] == 1
    other = api_client.post("/documents/add", json={"text": "مستند آخر يحتوي على معلومات مختلفة تماما", "document_id": "other"})
    assert other.status_code == 200
    prompts = []
    monkeypatch.setattr(api_main.llm_service, "generate", lambda prompt: prompts.append(prompt) or "answer")
    answer = api_client.post("/query", json={"question": "ما الذكاء العاطفي؟", "document_id": first.json()["document_id"], "top_k": 2})
    assert answer.status_code == 200
    payload = answer.json()
    assert payload["citations"]
    assert all(c["document_id"] == first.json()["document_id"] for c in payload["citations"])
    assert all(d["text"] in prompts[0] for d in payload["retrieved_documents"])
    assert "مستند آخر يحتوي" not in prompts[0]
    assert api_client.post("/query", json={"question": "سؤال صالح", "document_id": "missing"}).status_code == 404
    assert api_client.post("/documents/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")}).status_code == 415


def test_upload_preserves_chunk_source_offsets(api_client, monkeypatch, tmp_path):
    monkeypatch.setattr(api_main.config, "DOCS_PATH", str(tmp_path))
    monkeypatch.setattr(api_main.document_loader, "load_file", lambda path: [
        DocumentChunk("مقطع عربي", "ignored", 0, path, page_num=3,
                      metadata={"char_start": 10, "char_end": 19,
                                "extraction_quality": "text", "workspace_id": "forged"})
    ])
    result = api_client.post("/documents/upload", files={"file": ("arbitrary.txt", b"example", "text/plain")})
    assert result.status_code == 200
    chunk = api_main.vector_store.metadata[0]
    assert (chunk["char_start"], chunk["char_end"], chunk["page_num"]) == (10, 19, 3)
    assert chunk["extraction_quality"] == "text"
    assert chunk["workspace_id"] != "forged"


def test_legacy_index_without_model_manifest_is_not_assumed_compatible(embedding_service):
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSVectorStore(tmp, embedding_model="model-a")
        store.add_documents(["نص محفوظ"], embedding_service.embed(["نص محفوظ"]))
        Path(tmp, "manifest.json").unlink()
        legacy = FAISSVectorStore(tmp, embedding_model="model-a")
        with pytest.raises(ValueError, match="model is unknown"):
            legacy.ensure_embedding_dimension(embedding_service.dim)
        assert len(legacy.metadata) == 1


def test_add_document_cannot_override_internal_metadata(api_client):
    response = api_client.post('/documents/add', json={
        'text': 'نص عربي صالح للاختبار', 'document_id': 'safe',
        'metadata': {'faiss_id': 123456, 'document_id': 'forged', 'source': 'forged', 'text': 'forged', 'workspace_id': 'other'}
    })
    assert response.status_code == 200
    assert api_client.get('/documents').json()['documents'][0]['document_id'] == 'safe'
    result = api_client.post('/query', json={'question': 'نص عربي', 'document_id': 'safe'})
    assert result.status_code == 200
    assert result.json()['retrieved_documents'][0]['text'] == 'نص عربي صالح للاختبار'


def test_long_single_paragraph_is_split_without_losing_words():
    loader = DocumentLoader(chunk_size=80, chunk_overlap=15, min_chunk_size=10)
    words = [f"word{i:03d}" for i in range(70)]
    chunks = loader._chunk_text(" ".join(words), "doc", "source", page_num=7)
    assert len(chunks) > 1
    assert all(len(c.text) <= 80 and c.page_num == 7 for c in chunks)
    assert all(any(word in chunk.text.split() for chunk in chunks) for word in words)
    assert set(chunks[0].text.split()) & set(chunks[1].text.split())


def test_arabic_token_chunks_respect_model_limit_prefix_overlap_and_page():
    class CharacterTokenizer:
        def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=False):
            result = {"input_ids": list(range(len(text) + (2 if add_special_tokens else 0)))}
            if return_offsets_mapping:
                result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
            return result
    tokenizer = CharacterTokenizer()
    text = "الجملة الأولى واضحة. الجملة الثانية تشرح الفكرة؟ الجملة الثالثة تؤكد التفاصيل. " * 3
    loader = DocumentLoader(min_chunk_size=1, tokenizer=tokenizer, max_tokens=40,
                            token_overlap=8, token_prefix="passage: ")
    chunks = loader._chunk_text(text, "doc", "source", page_num=9)
    assert len(chunks) > 2
    assert all(c.page_num == 9 for c in chunks)
    assert all(len(tokenizer("passage: " + c.text)["input_ids"]) <= 40 for c in chunks)
    assert any(c.text.endswith((".", "؟")) for c in chunks[:-1])
    assert any(set(chunks[i].text.split()) & set(chunks[i + 1].text.split())
               for i in range(len(chunks) - 1))
    assert text.strip()[-10:] in chunks[-1].text


def test_embedding_rejects_inputs_above_actual_model_limit():
    from app.services.embedding_service import EmbeddingService
    class Tokenizer:
        is_fast = True
        model_max_length = 12
        def __call__(self, text, add_special_tokens=True, truncation=False):
            return {"input_ids": list(range(len(text) + 2))}
    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 10
        default_prompt_name = "passage"
        prompts = {"passage": "P:"}
        def encode(self, texts, **kwargs):
            raise AssertionError("oversized text must not reach the model")
    service = EmbeddingService.__new__(EmbeddingService)
    service.model = Model()
    service._using_fallback = False
    service.allow_fallback = False
    service.model_name = "test"
    with pytest.raises(ValueError, match="token limit"):
        service.embed("1234567")


def test_e5_requires_explicit_migration():
    from app.services.embedding_service import EmbeddingService
    with pytest.raises(ValueError, match="explicit migration"):
        EmbeddingService(model_name="intfloat/multilingual-e5-large")


def test_embedding_rejects_nonfinite_vectors():
    from app.services.embedding_service import EmbeddingService
    class Tokenizer:
        is_fast = True
        model_max_length = 128
        def __call__(self, text, **kwargs): return {"input_ids": [1, 2, 3]}
    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 128
        default_prompt_name = None
        def get_sentence_embedding_dimension(self): return 2
        def encode(self, texts, **kwargs): return np.array([[float('nan'), 1.0]])
    service = EmbeddingService.__new__(EmbeddingService)
    service.model = Model(); service._using_fallback = False
    service.allow_fallback = False; service.model_name = 'test'
    with pytest.raises(RuntimeError, match='invalid vectors'):
        service.embed('نص عربي')


def test_token_chunks_cover_source_characters_the_tokenizer_skips():
    class SparseTokenizer:
        def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=False):
            offsets = [(i, i + 1) for i, ch in enumerate(text) if ch not in '# ']
            result = {'input_ids': list(range(len(offsets) + (2 if add_special_tokens else 0)))}
            if return_offsets_mapping:
                result['offset_mapping'] = offsets
            return result
    text = '# نص عربي # يحوي رموزا # يجب الاحتفاظ بها #'
    loader = DocumentLoader(min_chunk_size=1, tokenizer=SparseTokenizer(), max_tokens=12, token_overlap=2)
    chunks = loader._chunk_text(text, 'doc', 'source', page_num=12)
    covered = set()
    for chunk in chunks:
        a, b = chunk.metadata['char_start'], chunk.metadata['char_end']
        assert chunk.text == text[a:b]
        covered.update(range(a, b))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())


def test_stats_do_not_expose_other_workspaces_or_server_paths(api_client, monkeypatch):
    monkeypatch.setattr(api_main.config, 'AUTH_WORKSPACE_ID', 'a')
    store = api_main.vector_store
    store.add_documents(['نص أول', 'نص ثان'], np.ones((2, 48), dtype=np.float32),
                        ['a-doc', 'b-doc'], metadata_list=[{'workspace_id': 'a'}, {'workspace_id': 'b'}])
    response = api_client.get('/stats')
    assert response.status_code == 200
    assert response.json() == {'workspace_documents': 1, 'workspace_chunks': 1}


def test_pdf_page_images_are_scoped_and_match_page_number(api_client, monkeypatch, tmp_path):
    from PyPDF2 import PdfWriter
    monkeypatch.setattr(api_main.config, 'DOCS_PATH', str(tmp_path))
    monkeypatch.setattr(api_main.config, 'AUTH_WORKSPACE_ID', 'a')
    folder = tmp_path / 'a'
    folder.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=250)
    writer.add_blank_page(width=250, height=200)
    with (folder / 'doc.pdf').open('wb') as f:
        writer.write(f)
    api_main.vector_store.add_documents(['page one'], np.ones((1, 48), dtype=np.float32),
        ['doc'], ['sample.pdf'], [{'workspace_id': 'a', 'page_num': 1}])
    response = api_client.get('/documents/doc/pages')
    assert response.status_code == 200 and response.json() == {'page_count': 2}
    image = api_client.get('/documents/doc/pages/2.png')
    assert image.status_code == 200 and image.headers['content-type'] == 'image/png'
    assert image.content.startswith(b'\x89PNG\r\n\x1a\n')
    assert image.headers['cache-control'] == 'private, no-store'
    assert api_client.get('/documents/doc/pages/3.png').status_code == 404
    monkeypatch.setattr(api_main.config, 'AUTH_WORKSPACE_ID', 'b')
    assert api_client.get('/documents/doc/pages').status_code == 404
    assert api_client.get('/documents/doc/pages/1.png').status_code == 404


def test_index_rejects_changed_or_missing_model_revision_without_deleting_data(embedding_service):
    from pathlib import Path
    import json
    with tempfile.TemporaryDirectory() as tmp:
        store = FAISSVectorStore(tmp, embedding_model='model-a', embedding_revision='revision-a')
        store.add_documents(['نص محفوظ'], embedding_service.embed(['نص محفوظ']))
        wrong = FAISSVectorStore(tmp, embedding_model='model-a', embedding_revision='revision-b')
        with pytest.raises(ValueError, match='revision'):
            wrong.ensure_embedding_dimension(embedding_service.dim)
        manifest = Path(tmp, 'manifest.json')
        data = json.loads(manifest.read_text(encoding='utf-8'))
        del data['embedding_revision']
        manifest.write_text(json.dumps(data), encoding='utf-8')
        unknown = FAISSVectorStore(tmp, embedding_model='model-a', embedding_revision='revision-a')
        with pytest.raises(ValueError, match='revision'):
            unknown.ensure_embedding_dimension(embedding_service.dim)
        assert len(unknown.metadata) == 1
