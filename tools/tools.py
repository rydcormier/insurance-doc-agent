"""
tools/tools.py

Agent tools - each function the LLM can call to interact with the insurance 
document knowledge base.

Each tool is a plain Python function decorated with @tool from LangChain. The
docstring IS the tool description the LLM reads to decide when to use it.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from embeddings.store import VectorStore

# shared vector store instance for all tools
_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    """Lazy-load the vector store."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

# ----------------------------------------------------------------------------
# Pydantic models for structured extraction output
# ----------------------------------------------------------------------------

class CoverageLimits(BaseModel):
    pass

# ----------------------------------------------------------------------------
# Tool definitions
# ----------------------------------------------------------------------------

@tool
def search_policy_docuument():
    pass


@tool
def list_available_documents():
    pass


@tool
def extract_coverage_limits():
    pass


@tool
def compare_policies():
    pass


@tool
def flag_anomolies():
    pass


@tool
def generate_summary():
    pass
