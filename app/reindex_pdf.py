"""Safely build a new metadata-preserving index from one PDF.

This command never writes to the configured production index. The caller must
provide a separate output directory and can verify it before switching paths.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import config
from app.services.embedding_service import EmbeddingService
from app.services.faiss_store import FAISSVectorStore
from app.utils.document_loader import DocumentLoader


def reindex_pdf(pdf_path: Path, output_path: Path, workspace_id: str) -> dict[str, object]:
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("A readable PDF path is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", workspace_id):
        raise ValueError("A valid workspace ID is required")
    output = output_path.resolve()
    for protected in (Path(config.INDEX_PATH).resolve(), Path(config.DOCS_PATH).resolve(), pdf_path.resolve()):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Output must be separate from existing data and the source PDF")
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError("Output directory must be new or empty")

    embedder = EmbeddingService(
        model_name=config.EMBEDDING_MODEL,
        device=config.EMBEDDING_DEVICE,
        allow_fallback=False,
    )
    loader = DocumentLoader(chunk_size=500, chunk_overlap=50, **embedder.chunking_config())
    chunks = loader.load_file(str(pdf_path))
    if not chunks:
        raise ValueError("No usable text was extracted from the PDF")
    # Generate first so a model failure cannot leave a misleading partial index.
    batches = []
    for offset in range(0, len(chunks), 64):
        batch = chunks[offset:offset + 64]
        batches.append(embedder.embed([chunk.text for chunk in batch]))
        if offset % 640 == 0 or offset + len(batch) == len(chunks):
            print(f"Embedded {offset + len(batch)}/{len(chunks)} chunks", flush=True)
    vectors = np.vstack(batches)
    index_path = output_path / "index"
    documents_path = output_path / "documents"
    document_id = uuid.uuid4().hex
    source_copy = documents_path / workspace_id / f"{document_id}.pdf"
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf_path, source_copy)
    source_hash = hashlib.sha256(source_copy.read_bytes()).hexdigest()
    store = FAISSVectorStore(
        index_path=str(index_path),
        embedding_model=config.EMBEDDING_MODEL,
        schema_version=config.INDEX_SCHEMA_VERSION,
    )
    now = datetime.now(timezone.utc).isoformat()
    store.add_documents(
        [chunk.text for chunk in chunks],
        vectors,
        [document_id] * len(chunks),
        [pdf_path.name] * len(chunks),
        [
            {
                "workspace_id": workspace_id,
                "chunk_id": chunk.chunk_id,
                "page_num": chunk.page_num,
                "added_at": now,
                "content_sha256": source_hash,
                **chunk.metadata,
            }
            for chunk in chunks
        ],
    )
    reopened = FAISSVectorStore(str(index_path), embedding_model=config.EMBEDDING_MODEL,
                                schema_version=config.INDEX_SCHEMA_VERSION)
    reopened.ensure_embedding_dimension(embedder.get_embedding_dimension())
    if reopened.get_stats()["total_vectors"] != len(chunks):
        raise RuntimeError("Reindexed vector count does not match extracted chunks")
    return {
        "output_path": str(output_path),
        "index_path": str(index_path),
        "documents_path": str(documents_path),
        "document_id": document_id,
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_count": len(chunks),
        "extraction": loader.last_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a separate PDF RAG index")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", default=config.AUTH_WORKSPACE_ID)
    args = parser.parse_args()
    result = reindex_pdf(args.pdf, args.output, args.workspace)
    print(f"Created separate index at {result['output_path']}")
    print(f"Model: {result['embedding_model']}; chunks: {result['chunk_count']}")
    print(f"Index path: {result['index_path']}; documents path: {result['documents_path']}")
    print(f"OCR-required pages: {result['extraction'].get('ocr_required_pages', [])}")
    print("Index remains inactive; evaluate retrieval before changing configuration.")


if __name__ == "__main__":
    main()
