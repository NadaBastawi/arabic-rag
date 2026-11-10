"""
Document loader for PDF and text files with chunking.

Supports PDF extraction, text file loading, and intelligent chunking
with overlap for maintaining context.
"""

import os
import logging
import re
import unicodedata
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
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.last_report: Dict[str, object] = {}
    
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
        elif file_ext in ('.docx', '.docs'):
            return self._load_docx(file_path)
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return []
    
    def _load_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Load and chunk a PDF file."""
        try:
            import PyPDF2
            
            chunks = []
            doc_id = Path(file_path).stem
            poor_pages: List[int] = []
            page_reports: List[Dict[str, object]] = []

            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        text = self._normalize_extracted_text(page.extract_text() or "")
                        quality = self._assess_page_quality(text)
                        page_reports.append({"page_num": page_num + 1, **quality})
                        if quality["ocr_required"]:
                            poor_pages.append(page_num + 1)
                            logger.warning(
                                "PDF page %s requires OCR; skipping unusable extracted text",
                                page_num + 1,
                            )
                            continue
                        page_chunks = self._chunk_text(
                            text,
                            doc_id,
                            file_path,
                            page_num=page_num + 1,
                        )
                        chunks.extend(page_chunks)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {str(e)}")
            
            for chunk_id, chunk in enumerate(chunks):
                chunk.chunk_id = chunk_id
                chunk.metadata.update({"extraction_quality": "text"})
            self.last_report = {
                "format": "pdf",
                "page_count": len(reader.pages),
                "indexed_page_count": len(page_reports) - len(poor_pages),
                "ocr_required_pages": poor_pages,
                "pages": page_reports,
            }
            return chunks
            
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return []
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            self.last_report = {
                "format": "pdf",
                "page_count": 0,
                "indexed_page_count": 0,
                "ocr_required_pages": [],
                "error": "PDF extraction failed",
            }
            return []

    @staticmethod
    def _normalize_extracted_text(text: str) -> str:
        """Normalize PDF glyph variants while preserving Arabic word boundaries."""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r" *\n *", "\n", normalized)
        return normalized.strip()

    @staticmethod
    def _assess_page_quality(text: str) -> Dict[str, object]:
        """Identify pages whose extracted text is too sparse or mostly corrupted."""
        arabic_count = sum("\u0600" <= char <= "\u06ff" for char in text)
        meaningful_count = sum(char.isalnum() for char in text)
        suspicious_count = sum(char in "/\\[]" or ord(char) in {0xfffd} for char in text)
        arabic_ratio = arabic_count / max(meaningful_count, 1)
        ocr_required = (
            len(text) < 80
            or arabic_count < 10
            or (meaningful_count >= 100 and arabic_ratio < 0.03)
        )
        return {
            "character_count": len(text),
            "arabic_character_count": arabic_count,
            "arabic_ratio": round(arabic_ratio, 4),
            "suspicious_character_count": suspicious_count,
            "ocr_required": ocr_required,
        }
    
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
    
    def _load_docx(self, file_path: str) -> List[DocumentChunk]:
        """Load and chunk a DOCX file."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            full_text = ""
            
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text += para.text + "\n"
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            full_text += cell.text + "\n"
            
            doc_id = Path(file_path).stem
            return self._chunk_text(full_text, doc_id, file_path)
            
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            return []
        except Exception as e:
            logger.error(f"Error loading DOCX file {file_path}: {str(e)}")
            return []
    
    def _chunk_text(
        self,
        text: str,
        document_id: str,
        source: str,
        page_num: Optional[int] = None,
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
        start = 0
        text = text.strip()
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                # Prefer a word boundary without allowing an exceptionally long
                # word or a large overlap to stall progress.
                boundary = text.rfind(" ", start + self.chunk_size // 2, end)
                if boundary > start:
                    end = boundary
            part = text[start:end].strip()
            if part:
                chunks.append(DocumentChunk(
                    text=part, document_id=document_id, chunk_id=len(chunks),
                    source=source, page_num=page_num,
                ))
            if end == len(text):
                break
            next_start = max(start + 1, end - self.chunk_overlap)
            if self.chunk_overlap and next_start > start + 1:
                boundary = text.find(" ", next_start, end)
                if boundary >= 0:
                    next_start = boundary + 1
            start = next_start

        logger.info("Created %d chunks from %s", len(chunks), document_id)
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
