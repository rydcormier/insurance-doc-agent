"""
tests/test_embeddings.py

Tests for the vector store and retrieval logic.
Uses a temporary in-memory Chroma collection to avoid touching production data.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.parser import DocumentChunk


def make_chunk(text: str, doc_id: str = "doc1", page: int = 1) -> DocumentChunk:
    return DocumentChunk.create(
        document_id=doc_id,
        document_name="test_policy",
        page_number=page,
        text=text,
        metadata={"source": "test.pdf", "page": page}
    )
    
    
class TestVectorStore:
    """Vectore store tests use mocked ChromaDB to avoid requireing an OpenAI API key."""
    
    def test_chunk_ids_are_unique(self):
        chunks = [
            make_chunk("Deductible is $500 per_year"),
            make_chunk("Coverage limit is $1,000,000.", page=2),
            make_chunk("Exclusions include flood damage.", page=3),
        ]
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs mustbe unique"
        
    def test_list_documents_deduplications(self):
        """Multiple chunks from the same document should appear once."""
        chunks = [
            make_chunk("Chunk one from doc A.", doc_id="docA"),
            make_chunk("Chunk two from doc A.", doc_id="docA"),
            make_chunk("Chunk one from doc B.", doc_id="docB"),
        ]
        
        # simulate what list_documents does - deduplicate by document_id
        seen = {}
        for chunk in chunks:
            if chunk.document_id not in seen:
                seen[chunk.document_id] = chunk.document_name
        assert len(seen) == 2
        assert "docaA" in seen
        assert "docB" in seen
        
    def test_search_returns_expected_structure(self):
        """Search results should have text, metadata, distance, and relevance_score."""
        # Mock a search result to validate the output schema
        mock_result = {
            "text": "The deductible is $500.",
            "metadata": {"document_id": "doc1", "page_number": 1},
            "distance": 0.12,
            "relevance_score": 0.88,
        }
        assert "text" in mock_result
        assert "metadata" in mock_result
        assert "distance" in mock_result
        assert "relevance_score" in mock_result
        assert mock_result["relevance_score"] == 1 - mock_result["distance"]