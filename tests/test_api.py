"""
tests/test_api.py

Tests for the FastAPI endpoints.
Mocks the agent and store to test routing and response shapes.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def client():
    with patch("api.app.get_agent") as mock_agent_factory, \
         patch("api.app.get_store") as mock_store_factory, \
         patch("api.app.get_parser"):
        
        mock_agent = MagicMock()
        mock_agent.run.return_value = "The deductible is $500 per year."
        mock_agent_factory.return_value = mock_agent
        
        mock_store = MagicMock()
        mock_store.list_documents.return_value = [{
            "document_id": "abc123", 
            "document_name": "test_policy", 
            "source": "test.pdf"
        }]
        mock_store.count = 42
        mock_store_factory.return_value = mock_store
        
        from api.app import app
        yield TestClient(app)
        
        
class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        
        
class TestDocumentEndpoint:
    def test_list_documents_returns_structure(self, client):
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total_chunks" in data
        assert data["total_chunks"] == 42
        
        
class TestQueryEndpoint:
    def test_query_returns_response(self, client):
        response = client.post("/query", json={"query": "What is the deductible?"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "deductible" in data["response"].lower()
        
    def test_query_echoes_question(self, client):
        response = client.post("/query", json={"query": "Test question."})
        assert response.json()["query"] == "Test question."
        
    def test_query_with_session_id(self, client):
        response = client.post(
            "/query", 
            json={"query": "Test question.", "session_id": "session-abc"}
        )
        assert response.json()["session_id"] == "session-abc"
        
        
class testSessionEndpoint:
    def test_clear_session_returns_ok(self, client):
        response = client.delete("/session")
        assert response.status_code == 200
        assert "cleared" in response.jsoin()["status"].lower()
        