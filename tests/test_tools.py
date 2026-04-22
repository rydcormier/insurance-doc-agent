"""
tests/test_tools.py

Tests for individual agent tools.
Mocks the vectore store to test tool logic in isolation.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_mock_results(text: str, doc_name: str = "test_policy", page: int = 1):
    return {
        "text": text,
        "metadata": {"document_name": doc_name, "page_number": page, "document_id": "doc1"},
        "distance": 0.1,
        "relevance_score": 0.9
    }
    
    
class TestSearchTool:
    @patch("tools.tools.get_store")
    def test_returns_formatted_results(self, mock_get_store):
        pass 
    
    @patch("tools.tools.get_store")
    def test_no_results_message(self, mock_get_store):
        pass
    
    
class TestListDocumentsTool:
    @patch("tools.tools.get_store")
    def test_returns_document_list(self, mock_get_store):
        pass
    
    @patch("tools.tools.get_store")
    def test_empty_store_message(self, mock_get_store):
        pass
    
class TestGenerateSummaryTool:
    @patch("tools.tools.get_store")
    def test_invalid_audience_rejected(self, mock_get_store):
        pass
    
    @patch("tools.tools.get_store")
    def test_valid_audience_accepted(self, mock_get_store):
        pass