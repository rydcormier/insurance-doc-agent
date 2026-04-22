"""
tests/test_ingestion.py

Tests for the document parser and chunking logic.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parser import DocumentParser, DocumentChunk


class TestDocumentChunk:
    def test_create_generates_stable_id(self):
        pass 
    
    def test_create_different_text_different_id(self):
        pass
    
    def test_metadata_defaults_to_empty_dict(self):
        pass
    
    def test_metadata_stored_correctly(self):
        pass
    
    
class TestDocumentParser:
    def test_raises_on_missing_file(self):
        pass
    
    def test_raises_on_non_pdf(self, tmp_path):
        pass
    
    def test_chunk_size_respected(self):
        pass
    
    @patch("ingestion.parser.pdfplumber.open")
    def test_empty_pages_skipped(self, mock_open):
        pass
    
    def test_parse_directiry_returns_all_chunks(self, tmp_path):
        pass
    
    