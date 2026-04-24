"""
embeddings/store.py

Vector store management — embed document chunks and enable semantic retrieval.
Wraps ChromaDB with a clean interface for the agent's retrieval tool.
"""

from __future__ import annotations

import os
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv, find_dotenv

from ingestion.parser import DocumentChunk

load_dotenv(find_dotenv)



class VectorStore:
    """
    Manage a ChromaDB vector store for insurance document chunks.
    
    Handles embedding, storage, and retrieval. Uses OpenAI's embeddings by 
    default; can be swapped for a local model by changing the embedding 
    function.
    
    Usage:
        store = VectorStore()
        store.add_chunks(chunks)
        results = store.search("What is my deductible?", n_results=5)
    """
    
    COLLECTION_NAME = "insurance_documents"

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", "./data/processed/chroma"
        )
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_chunks(self, chunks: list[DocumentChunk], batch_size: int = 100) -> None:
        """
        Embed and store document chunks.
        
        Args:
            chunks: List of DocumentChunk objects from the parser.
            batch_size: Number of chunks to embed per API call.
        """
        print(f"Adding {len(chunks)} chunks to the vector store...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in batch], 
                documents=[c.text for c in batch], 
                metadatas=[
                    {
                        **c.metadata,
                        "document_id": c.document_id,
                        "document_name": c.document_name,
                        "page_number": c.page_number,
                    }
                    for c in batch
                ],
            )
            print(f"  → Embedded chunks {i + 1}–{min(i + batch_size, len(chunks))}")
        print("Done.")
        
    def search(
        self,
        query: str,
        n_results: int = 5,
        document_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over stored chunks.
        
        Args:
            query: Natural language query.
            n_restuls: Number of top results to return.
            document_id: Optional filter to search within a specific document.
            
        Returns:
            List of dicts with 'text', 'metadata', and 'distance' keys.
        """
        where = {"document_id": document_id} if document_id else None
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for text, metadata, distance in zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        ):
            output.append({
                "text": text,
                "metadata": metadata,
                "distance": distance,
                "relevance_score": 1 - distance,  # cosine: 1 = identical
            })
        return output
    
    def list_documents(self) -> list[dict]:
        """Return a list of all ingested documents."""
        results = self._collection.get(include=["metadatas"])
        seen = {}
        for meta in results["metadatas"]:
            doc_id = meta["document_id"]
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "document_name": meta.get("document_name"),
                    "source": meta.get("source"),
                }
        return list(seen.values())
    
    @property
    def count(self) -> int:
        """Return the total number of chunks in the vector store."""
        return self._collection.count()