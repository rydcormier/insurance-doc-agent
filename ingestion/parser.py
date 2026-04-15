"""
ingestion/parser.py

Parse PDF insurance documents into structured chunks for embedding. Handles 
section-aware chunking to keep related policy content together.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            document_name=document_name,
            page_number=page_number,
            text=text
        )


class DocumentParser:
    """
    Parses PDF insurance documents into chunks for embedding.
    
    Usage:
        parser = DocumentParser(chunk_size=1000, chunk_overlap=200)
        chunks = parser.parse("path/to/insurance_policy.pdf")
    """

    def __init__(self):
        pass 

    def parse(self):
        pass

    def parse_directory(self):
        pass
    