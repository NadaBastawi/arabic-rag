"""
Index builder for Arabic RAG system.

Loads documents, generates embeddings, and saves FAISS index.
"""

import os
import json
import pickle
import logging
from typing import List, Optional
import numpy as np

from app.config import config
from app.services.embedding_service import EmbeddingService
from app.utils.arabic_preprocessing import preprocess_text

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ArabicIndexBuilder:
    """Builder for Arabic document index."""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        index_path: Optional[str] = None,
        docs_path: Optional[str] = None
    ):
        """
        Initialize index builder.
        
        Args:
            embedding_service: Embedding service instance.
            index_path: Path to save index.
            docs_path: Path to load documents from.
        """
        self.embedding_service = embedding_service or EmbeddingService(
            model_name=config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE
        )
        self.index_path = index_path or config.INDEX_PATH
        self.docs_path = docs_path or config.DOCS_PATH
        
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def load_documents(
        self,
        path: Optional[str] = None,
        limit: Optional[int] = None
    ) -> int:
        """
        Load documents from file or directory.
        
        Args:
            path: Path to load from (JSON, txt, or directory).
            limit: Maximum number of documents to load.
            
        Returns:
            Number of documents loaded.
        """
        load_path = path or self.docs_path
        
        if not os.path.exists(load_path):
            logger.warning(f"Path does not exist: {load_path}")
            return 0
        
        if os.path.isdir(load_path):
            self._load_from_directory(load_path, limit)
        elif load_path.endswith('.json'):
            self._load_from_json(load_path, limit)
        elif load_path.endswith('.txt'):
            self._load_from_txt(load_path, limit)
        else:
            logger.error(f"Unsupported file format: {load_path}")
            return 0
        
        logger.info(f"Loaded {len(self.texts)} documents")
        return len(self.texts)
    
    def _load_from_directory(self, dir_path: str, limit: Optional[int]):
        """Load documents from directory."""
        txt_files = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        
        for file_path in txt_files[:limit]:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.texts.extend(content.split('\n\n'))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
    
    def _load_from_json(self, json_path: str, limit: Optional[int]):
        """Load documents from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data[:limit]:
                if isinstance(item, str):
                    self.texts.append(item)
                elif isinstance(item, dict):
                    text = item.get('text', item.get('content', ''))
                    if text:
                        self.texts.append(text)
        elif isinstance(data, dict):
            for key in data:
                text = data[key]
                if isinstance(text, str) and text:
                    self.texts.append(text)
    
    def _load_from_txt(self, txt_path: str, limit: Optional[int]):
        """Load documents from text file."""
        with open(txt_path, 'r', encoding='utf-8') as f:
            self.texts = f.read().split('\n\n')
        
        if limit:
            self.texts = self.texts[:limit]
    
    def preprocess(self, remove_diacritics: bool = True) -> int:
        """
        Preprocess all loaded documents.
        
        Args:
            remove_diacritics: Whether to remove Arabic diacritics.
            
        Returns:
            Number of documents preprocessed.
        """
        if not self.texts:
            return 0
        
        processed = []
        for text in self.texts:
            if text and text.strip():
                if remove_diacritics:
                    text = preprocess_text(text)
                processed.append(text)
        
        self.texts = [t for t in processed if t]
        logger.info(f"Preprocessed {len(self.texts)} documents")
        return len(self.texts)
    
    def generate_embeddings(
        self,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for all documents.
        
        Args:
            batch_size: Batch size for embedding generation.
            show_progress: Whether to show progress bar.
            
        Returns:
            Numpy array of embeddings.
        """
        if not self.texts:
            raise ValueError("No documents to embed")
        
        logger.info(f"Generating embeddings for {len(self.texts)} documents...")
        
        self.embeddings = self.embedding_service.embed(self.texts)
        
        logger.info(f"Generated embeddings with shape {self.embeddings.shape}")
        return self.embeddings
    
    def save_index(self, path: Optional[str] = None) -> str:
        """
        Save index to file.
        
        Args:
            path: Path to save index.
            
        Returns:
            Path where index was saved.
        """
        save_path = path or self.index_path
        
        os.makedirs(save_path, exist_ok=True)
        
        embeddings_path = os.path.join(save_path, 'embeddings.npy')
        texts_path = os.path.join(save_path, 'texts.json')
        meta_path = os.path.join(save_path, 'metadata.json')
        
        np.save(embeddings_path, self.embeddings)
        
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(self.texts, f, ensure_ascii=False)
        
        metadata = {
            'num_documents': len(self.texts),
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'embedding_model': config.EMBEDDING_MODEL
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        
        logger.info(f"Index saved to {save_path}")
        return save_path
    
    def load_index(self, path: Optional[str] = None) -> int:
        """
        Load index from file.
        
        Args:
            path: Path to load index from.
            
        Returns:
            Number of documents loaded.
        """
        load_path = path or self.index_path
        
        embeddings_path = os.path.join(load_path, 'embeddings.npy')
        texts_path = os.path.join(load_path, 'texts.json')
        
        if not os.path.exists(embeddings_path):
            logger.error(f"Index not found: {load_path}")
            return 0
        
        self.embeddings = np.load(embeddings_path)
        
        with open(texts_path, 'r', encoding='utf-8') as f:
            self.texts = json.load(f)
        
        logger.info(f"Loaded {len(self.texts)} documents from index")
        return len(self.texts)
    
    def build_and_save(
        self,
        docs_path: Optional[str] = None,
        preprocess_docs: bool = True
    ) -> str:
        """
        Complete pipeline: load, preprocess, embed, save.
        
        Args:
            docs_path: Path to documents.
            preprocess_docs: Whether to preprocess.
            
        Returns:
            Path where index was saved.
        """
        self.load_documents(docs_path)
        
        if preprocess_docs:
            self.preprocess()
        
        self.generate_embeddings()
        
        return self.save_index()


def main():
    """Run index builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Arabic RAG index")
    parser.add_argument("--docs", type=str, help="Path to documents")
    parser.add_argument("--index", type=str, help="Path to save index")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing")
    args = parser.parse_args()
    
    builder = ArabicIndexBuilder()
    
    path = builder.build_and_save(
        docs_path=args.docs,
        preprocess_docs=not args.no_preprocess
    )
    
    print(f"Index built and saved to: {path}")


if __name__ == "__main__":
    main()
