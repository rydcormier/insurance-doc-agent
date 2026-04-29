"""
ingestion/parser.py

Parse PDF insurance documents into structured chunks for embedding. Handles 
section-aware chunking to keep related policy content together.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class DocumentChunk:
    """A single chunk of text parsed from document."""
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls, 
        document_id: str, 
        document_name: str, 
        page_number: int, 
        text: str,
        metadata: Optional[dict] = None
    ) -> DocumentChunk:
        """Factory method to create a DocumentChunk with a unique chunk_id."""
        chunk_id = hashlib.md5(f"{document_id}:{page_number}:{text[:50]}".encode()).hexdigest()
        metadata = {} if not metadata else metadata
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            document_name=document_name,
            page_number=page_number,
            text=text,
            metadata=metadata
        )


class DocumentParser:
    """
    Parses PDF insurance documents into chunks for embedding.
    
    Usage:
        parser = DocumentParser(chunk_size=1000, chunk_overlap=200)
        chunks = parser.parse("path/to/insurance_policy.pdf")
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )


    def parse(self, file_path: str | Path) -> List[DocumentChunk]:
        """Parse a PDF document into chunks.
        
        Args:
            file_path: Path to the PDF document.

        Returns:
            A list of DocumentChunk ready for embedding.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")
        
        document_id = hashlib.md5(path.name.encode()).hexdigest()[:8]
        document_name = path.stem
        chunks = []

        with pdfplumber.open(path) as pdf:
            for page_numb, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text or not text.strip():
                    continue  # Skip empty pages

                # split page text into chunks
                page_chunks = self._splitter.split_text(text)

                for chunk_text in page_chunks:
                    if not chunk_text.strip():
                        continue  # Skip empty chunks
                    chunk = DocumentChunk.create(
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_numb,
                        text=chunk_text.strip(),
                        metadata={
                            "source": str(path), 
                            "page": page_numb, 
                            "total_pages": len(pdf.pages) 
                        }
                    )
                    chunks.append(chunk)
        return chunks


    def parse_directory(self, directory: str | Path) -> List[DocumentChunk]:
        """Parse all PDF documents in a directory.
        
        Args:
            directory: Path to the directory containing PDF documents.

        Returns:
            A list of DocumentChunk from all parsed documents.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")
        all_chunks = []
        for pdf_path in sorted(dir_path.glob("*.pdf")):
            print(f"Parsing: {pdf_path.name}...")
            chunks = self.parse(pdf_path)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks")
        return all_chunks
