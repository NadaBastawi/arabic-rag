"""Evaluate a staged index against user-supplied PDF/page expectations.

The case file is local JSON, not part of the application or production data.
No language model is called and no index is modified.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PyPDF2 import PdfReader

from app.config import config
from app.services.embedding_service import EmbeddingService
from app.services.faiss_store import FAISSVectorStore
from app.services.multi_stage_retrieval import AdaptiveRetrieval
from app.utils.document_loader import DocumentLoader
from app.utils.errors import InputValidator


def evaluate(index_path: Path, documents_path: Path, workspace: str,
             cases: list[dict], top_k: int = 5) -> list[dict]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", workspace):
        raise ValueError("Invalid workspace")
    if not (index_path / "manifest.json").is_file():
        raise ValueError("A staged index with a model manifest is required")
    if not cases or not 1 <= top_k <= 20:
        raise ValueError("At least one case and a top-k from 1 to 20 are required")
    embedder = EmbeddingService(model_name=config.EMBEDDING_MODEL,
                                device=config.EMBEDDING_DEVICE,
                                revision=config.effective_embedding_revision,
                                allow_fallback=False)
    store = FAISSVectorStore(str(index_path), embedding_model=config.EMBEDDING_MODEL,
                             schema_version=config.INDEX_SCHEMA_VERSION,
                             embedding_revision=config.effective_embedding_revision)
    store.ensure_embedding_dimension(embedder.get_embedding_dimension())
    retriever = AdaptiveRetrieval(embedder, store)
    reader_cache: dict[str, PdfReader] = {}
    pages_cache: dict[tuple[str, int], str] = {}
    report = []
    for number, case in enumerate(cases, 1):
        question = case.get("question")
        expected = case.get("expected_pages")
        doc_id = case.get("document_id")
        if not isinstance(question, str) or not question.strip() or not isinstance(expected, list) or not expected:
            raise ValueError("Each case needs a question and nonempty expected_pages")
        if not all(isinstance(page, int) and not isinstance(page, bool) and page > 0 for page in expected):
            raise ValueError("Expected PDF page numbers must be positive integers")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Each case must identify the expected document")
        InputValidator.validate_document_id(doc_id)
        if not any(m.get("workspace_id") == workspace and m.get("document_id") == doc_id for m in store.metadata):
            raise ValueError("Selected document is not in the workspace")
        docs = retriever.retrieve(question, top_k=top_k, workspace_id=workspace, document_id=doc_id)
        verified = True
        for item in docs:
            metadata = item.get("metadata", {})
            item_id = item.get("document_id", "")
            page = metadata.get("page_num")
            try:
                InputValidator.validate_document_id(item_id)
                start, end = metadata["char_start"], metadata["char_end"]
                if not isinstance(page, int) or page < 1 or not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError("Missing source offsets")
                source = Path(str(item.get("source", ""))).suffix.lower()
                if source != ".pdf":
                    raise ValueError("Source is not a PDF")
                file_path = documents_path / workspace / f"{item_id}{source}"
                if item_id not in reader_cache:
                    reader_cache[item_id] = PdfReader(str(file_path))
                key = (item_id, page)
                if key not in pages_cache:
                    extracted = reader_cache[item_id].pages[page - 1].extract_text() or ""
                    pages_cache[key] = DocumentLoader._normalize_extracted_text(extracted).strip()
                if start < 0 or end < start or pages_cache[key][start:end] != item.get("text"):
                    raise ValueError("Source offsets do not match")
            except (KeyError, IndexError, OSError, ValueError, TypeError):
                verified = False
        pages = [d.get("metadata", {}).get("page_num") for d in docs]
        report.append({"id": str(case.get("id", number)), "retrieved_pages": pages,
                       "expected_page_in_top_k": any(page in expected for page in pages),
                       "all_retrieved_passages_match_pdf": bool(docs) and verified})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--cases", type=Path, required=True,
                        help="Local JSON list with question, expected_pages and document_id")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        parser.error("Cases must be a JSON list")
    report = evaluate(args.index, args.documents, args.workspace, cases, args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(x["expected_page_in_top_k"] and x["all_retrieved_passages_match_pdf"] for x in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
