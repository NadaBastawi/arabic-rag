"""Safely build a new metadata-preserving index from one PDF.

This command never writes to the configured production index. The caller must
provide a separate output directory and can verify it before switching paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from app.config import config
from app.services.embedding_service import EmbeddingService
from app.services.faiss_store import FAISSVectorStore
from app.utils.document_loader import DocumentLoader


def reindex_pdf(pdf_path: Path, output_path: Path, workspace_id: str) -> dict[str, object]:
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("A readable PDF path is required")
    if output_path.resolve() == Path(config.INDEX_PATH).resolve():
        raise ValueError("Output must be separate from the configured index path")
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError("Output directory must be new or empty")

    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    chunks = loader.load_file(str(pdf_path))
    if not chunks:
        raise ValueError("No usable text was extracted from the PDF")

    embedder = EmbeddingService(
        model_name=config.EMBEDDING_MODEL,
        device=config.EMBEDDING_DEVICE,
        allow_fallback=False,
    )
    store = FAISSVectorStore(
        index_path=str(output_path),
        embedding_model=config.EMBEDDING_MODEL,
        schema_version=config.INDEX_SCHEMA_VERSION,
    )
    now = datetime.now(timezone.utc).isoformat()
    store.add_documents(
        [chunk.text for chunk in chunks],
        embedder.embed([chunk.text for chunk in chunks]),
        [chunk.document_id] * len(chunks),
        [pdf_path.name] * len(chunks),
        [
            {
                "workspace_id": workspace_id,
                "chunk_id": chunk.chunk_id,
                "page_num": chunk.page_num,
                "added_at": now,
                **chunk.metadata,
            }
            for chunk in chunks
        ],
    )
    return {
        "output_path": str(output_path),
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
    print(f"Extraction: {result['extraction']}")


if __name__ == "__main__":
    main()
