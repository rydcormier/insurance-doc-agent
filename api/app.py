"""
api/app.py

FastAPI serving layer for the Insurance Document Intelligence Agent.
Exposes endpoints for document ingestion, querying, and management.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from agent import InsuranceAgent
from ingestion.parser import DocumentParser
from embeddings.store import VectorStore

load_dotenv(find_dotenv())


app = FastAPI(
    title="Insurance Document Intelligence Agent",
    description="Agentic AI for insurance policy analysis using RAG and LLM tool use.",
    version="0.1.0"
)

# Module-level instances
_agent:  Optional[InsuranceAgent] = None
_store:  Optional[VectorStore]    = None
_parser: Optional[DocumentParser] = None

def get_agent() -> InsuranceAgent:
    global _agent
    if _agent is None:
        _agent = InsuranceAgent(verrbose=False)
    return _agent


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

def get_parser() -> DocumentParser:
    global _parser
    if _parser is None:
        _parser = DocumentParser(
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
        )
    return _parser


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    
    
class QueryResponse(BaseModel):
    query: str
    response: str
    session_id: Optional[str] = None
    
    
class DocumentListResponse(BaseModel):
    documents: list[dict]
    total_chunks: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """List all ingested documents."""
    store = get_store()
    return DocumentListResponse(
        documents=store.list_documents(),
        total_chunks=store.count
    )


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a PDF document into the knowledge base. Parses, chunks, embeds and
    stores the document for retrieval.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # save to temp location
    tmp_path = Path(f"tmp/{file.filename}")
    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        
        parser = get_parser()
        chunks = parser.parse(tmp_path)
        
        store = get_store()
        store.add_chunks(chunks)
        
        return {
            "filename":         file.filename,
            "chunks_created":   len(chunks),
            "status":           "ingested",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
            
    
@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """
    Query the insurance document agent.

    The agent will use its tools to search, extract, and reason over the
    ingested documents to answer the question.
    """
    try:
        agent = get_agent()
        response = agent.run(request.query)
        return QueryResponse(
            query=request.query,
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.delete("/session")
def clear_session():
    """Clear the agent's conversation memory."""
    get_agent().clear_memory()
    return {"status": "session cleared"}

