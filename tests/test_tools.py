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


def make_mock_result(text: str, doc_name: str = "test_policy", page: int = 1):
    return {
        "text": text,
        "metadata": {"document_name": doc_name, "page_number": page, "document_id": "doc1"},
        "distance": 0.1,
        "relevance_score": 0.9
    }
    
    
class TestSearchTool:
    @patch("tools.tools.get_store")
    def test_returns_formatted_results(self, mock_get_store):
        mock_store = MagicMock()
        mock_store.search.return_value = [
            make_mock_result("The deductible is $500 per year."),
            make_mock_result("Coverage limit is $1,000,000.", page=2),
        ]
        mock_get_store.return_value = mock_store
        
        from tools.tools import search_policy_docuument
        result = search_policy_docuument.invoke("what is the deductible")
        
        assert "deductible" in result.lower()
        assert "Result 1" in result
        assert "Result 2" in result
    
    @patch("tools.tools.get_store")
    def test_no_results_message(self, mock_get_store):
        mock_store = MagicMock()
        mock_store.search.return_value = []
        mock_get_store.return_value = mock_store
        
        from tools.tools import search_policy_docuument
        result = search_policy_docuument.invoke("obscure query with no matches")
        
        assert "No relevant passeges" in result
    
    
class TestListDocumentsTool:
    @patch("tools.tools.get_store")
    def test_returns_document_list(self, mock_get_store):
        mock_store = MagicMock
        mock_store.list_documents.return_value = [
            {"document_id": "abc123", "document_name": "blue_shield_policy"},
            {"document_id": "def456", "document_name": "aetna_policy"},
        ]
        mock_get_store.return_value = mock_store
        
        from tools.tools import list_available_documents
        result = list_available_documents.invoke("")
        
        assert "blue_shield_policy" in result
        assert "aetna_policy" in result
    
    @patch("tools.tools.get_store")
    def test_empty_store_message(self, mock_get_store):
        mock_store = MagicMock()
        mock_store.list_documents.return_value = []
        mock_get_store.return_value = mock_store

        from tools.tools import list_available_documents
        result = list_available_documents.invoke("")

        assert "No documents" in result
    
class TestGenerateSummaryTool:
    @patch("tools.tools.get_store")
    def test_invalid_audience_rejected(self, mock_get_store):
        mock_store = MagicMock()
        mock_store.search.return_value = [make_mock_result("Some policy text.")]
        mock_get_store.return_value = mock_store

        from tools.tools import generate_summary
        result = generate_summary.invoke({"document_id": "doc1", "audience": "invalid_audience"})

        assert "Invalid audience" in result

    @patch("tools.tools.get_store")
    def test_valid_audiences_accepted(self, mock_get_store):
        mock_store = MagicMock()
        mock_store.search.return_value = [make_mock_result("Some policy text.")]
        mock_get_store.return_value = mock_store

        from tools.tools import generate_summary
        for audience in ["general", "technical", "executive"]:
            result = generate_summary.invoke({"document_id": "doc1", "audience": audience})
            assert "Invalid audience" not in result
