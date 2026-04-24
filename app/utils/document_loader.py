"""
Document loader for PDF and text files with chunking.

Supports PDF extraction, text file loading, and intelligent chunking
with overlap for maintaining context.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    
    def __init__(
        self,
        text: str,
        document_id: str,
        chunk_id: int,
        source: str,
        page_num: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        self.text = text
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.source = source
        self.page_num = page_num
        self.metadata = metadata or {}


class DocumentLoader:
    """Loads documents from various formats and chunks them."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize document loader.
        
        Args:
            chunk_size: Target size for each chunk (in characters).
            chunk_overlap: Overlap between consecutive chunks.
            min_chunk_size: Minimum chunk size to keep.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def load_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Load a document file and return chunks.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of DocumentChunk objects.
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self._load_pdf(file_path)
        elif file_ext == '.txt':
            return self._load_text(file_path)
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return []
    
    def _load_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Load and chunk a PDF file."""
        try:
            import PyPDF2
            
            chunks = []
            doc_id = Path(file_path).stem
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text()
                        full_text += f"\n[Page {page_num + 1}]\n{text}"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {str(e)}")
            
            return self._chunk_text(full_text, doc_id, file_path)
            
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return []
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def _load_text(self, file_path: str) -> List[DocumentChunk]:
        """Load and chunk a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            doc_id = Path(file_path).stem
            return self._chunk_text(text, doc_id, file_path)
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return []
    
    def _chunk_text(
        self,
        text: str,
        document_id: str,
        source: str
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full text to chunk.
            document_id: Unique document identifier.
            source: Source filename/path.
            
        Returns:
            List of DocumentChunk objects.
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        chunks = []
        chunk_id = 0
        
        # Split by paragraphs first to preserve context
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph exceeds chunk size, save current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(
                        DocumentChunk(
                            text=current_chunk.strip(),
                            document_id=document_id,
                            chunk_id=chunk_id,
                            source=source
                        )
                    )
                    chunk_id += 1
                
                # Add overlap
                words = current_chunk.split()
                overlap_size = len(" ".join(words[-10:])) if len(words) > 10 else len(current_chunk) // 2
                current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else ""
            
            current_chunk += ("\n\n" if current_chunk else "") + paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(
                DocumentChunk(
                    text=current_chunk.strip(),
                    document_id=document_id,
                    chunk_id=chunk_id,
                    source=source
                )
            )
        
        logger.info(f"Created {len(chunks)} chunks from {document_id}")
        return chunks
    
    def load_directory(self, dir_path: str) -> List[DocumentChunk]:
        """
        Load all documents from a directory.
        
        Args:
            dir_path: Path to directory.
            
        Returns:
            List of DocumentChunk objects from all files.
        """
        all_chunks = []
        
        if not os.path.isdir(dir_path):
            logger.error(f"Directory not found: {dir_path}")
            return []
        
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            
            if os.path.isfile(file_path):
                file_ext = Path(file_path).suffix.lower()
                if file_ext in ['.pdf', '.txt']:
                    chunks = self.load_file(file_path)
                    all_chunks.extend(chunks)
        
        logger.info(f"Loaded {len(all_chunks)} chunks from {dir_path}")
        return all_chunks
