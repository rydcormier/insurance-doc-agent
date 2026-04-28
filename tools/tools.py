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

import instructor
import openai
from dotenv import find_dotenv, load_dotenv
from langchain.tools import tool
from pydantic import BaseModel, Field

from embeddings.store import VectorStore
from agent import SYSTEM_PROMPT

# shared vector store instance for all tools
_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    """Lazy-load the vector store."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store

# get key from env
load_dotenv(find_dotenv)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# ----------------------------------------------------------------------------
# Pydantic models for structured extraction output
# ----------------------------------------------------------------------------

class CoverageLimits(BaseModel):
    """Structured coverage information extracted from a policy document."""
    document_name:      str
    deductible:         Optional[str] = Field(None, description="The deductible amount")
    coverage_limit:     Optional[str] = Field(None, description="The maximum coverage amount")
    out_of_pocket_max:  Optional[str] = Field(None, description="Out of pocket maximum")
    copay:              Optional[str] = Field(None, description="Copay amount")
    coinsurance:        Optional[str] = Field(None, description="Coinsurance percentage")
    exclusions:         list[str]     = Field(default_factory=list, description="List of cove")
    notes:              Optional[str] = Field(None, description="Additional relevant notes")


# ----------------------------------------------------------------------------
# Tool definitions
# ----------------------------------------------------------------------------

@tool
def search_policy_docuument(query: str, document_id: Optional[str] = None) -> str:
    """
    Search the insurance policy knowledge base using semantic search. Use this
    tool to find relevant policy terms, coverage details, exclusions, or any 
    other information from ingested insurance documents.
    
    Args:
        query: The natural language question or search query.
        document_id: Optional. Restrict search to a specific document.
        
    Returns:
        Relevant text passages from the policy documents.
    """
    results = get_store().search(query, n_results=5, document_id=document_id)
    if not results:
        return "No relevant information found for this query."
    
    output_parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        output_parts.append(
            f"[Result {i} - {meta.get('document_name', 'Unknown')}, "
            f"Page {meta.get('page_number', '?')}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(output_parts)


@tool
def list_available_documents() -> str:
    """
    List all insurance documents currently available in the knowledge base. Use
    this tool first to understand what documents are available before searching
    or extracting information.
    
    Returns:
        A list of document names and their IDs.
    """
    docs = get_store().list_documents()
    if not docs:
        return "No documents have been ingested yet."
    lines = [f"- {d['document_name']} (ID: {d['document_id']})" for d in docs]
    return "Available documents:\n" + "\n".join(lines)


@tool
def extract_coverage_limits(document_id: str) -> str:
    """
    Extract stuructured coverage limit information from a specific policy 
    document. Returns deductibles, coverage limits, out-of-pockect maximums,
    copays, coinsurance rates, and key exclusions in a structured format. For
    exclusions, return a list of brief plain-English descriptions.
    
    Args:
        document_id: The ID of the document to extract from (use 
        list_available_documents first).
        
    Returns:
        Structured JSON with coverage limit details.
    """
    # Retrieve relevant chunks about coverage and limits
    queries = [
        "deductible amount coverage limit",
        "out of pocket maximum copay coinsurance",
        "exclusions not covered benefits",
    ]
    all_context = []
    for q in queries:
        results = get_store().search(q, n_results=3, document_id=document_id)
        all_context.extend([r["text"] for r in results])

    context = "\n\n".join(all_context)
    if not context.strip():
        return f"No coverage information found for document ID: {document_id}"
    
    # call llm
    client = instructor.from_openai(openai.OpenAI())
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=CoverageLimits,
        messages=[
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}"
            },
            {
                "role": "user",
                "content": (
                    "Extract all coverage limit information from the following insurance policy text.\n\n"
                    f"Policy text:\n{context}\n\n"
                    "Return the deductible, coverage limit, out-of-pocket maximum, copay, "
                    "coinsurance percentage, and any exclusions mentioned. For exclusions, "
                    "return a list of brief plain-English descriptions."
                )
            }
        ]
    )
    return result.model_dump_json(indent=2)


@tool
def compare_policies(document_id_1: str, document_id_2: str, aspect: str = "coverage") -> str:
    """
    Compare two insurance policy documents on a specific aspect such as 
    coverage limits, deductibles, exlusions, or premiums.
    
    Args:
        document_id_1: ID of the first document.
        document_id_2: ID of the second document.
        aspect: What to compare — e.g. 'coverage', 'deductibles', 'exclusions'.
        
    Returns:
        A structured comparison of the two policies on the requested aspect.
    """
    results_1 = get_store().search(aspect, n_results=3, document_id=document_id_1)
    results_2 = get_store().search(aspect, n_results=3, document_id=document_id_2)
    
    context_1 = "\n".join([r["text"] for r in results_1])
    context_2 = "\n".join([r["text"] for r in results_2])

    if not context_1 and not context_2:
        return f"No information found for aspect '{aspect}' in either document."
    
    # call llm
    client = instructor.from_openai(openai.OpenAI())
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=CoverageLimits,
        messages=[
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}"
            },
            {
                "role": "user",
                "content": (
                    f"Compare these two insurance policies on the following aspect: {aspect}\n\n"
                    f"--- POLICY 1: {document_id_1} ---\n"
                    f"{context_1}\n\n"
                    f"--- POLICY 2: {document_id_2} ---\n"
                    f"{context_2}\n\n"
                    "Structure your response as follows:\n"
                    "1. Policy 1 summary: what it says about this aspect (2-3 sentences)\n"
                    "2. Policy 2 summary: what it says about this aspect (2-3 sentences)\n"
                    "3. Key differences: the most important distinctions a decision-maker should know\n"
                    "4. Which is more favorable and why — or note if they are equivalent\n\n"
                    "If either policy does not address this aspect in the provided text, "
                    "state that clearly rather than inferring."
                )
            }
        ]
    )
    return result
    
    

@tool
def flag_anomalies(document_id: str) -> str:
    """
    Analyze an insurance policy document for potential anomalies, 
    inconsistencies, unusal terms, or missing standard provisions. Useful 
    for identifying non-standard policy language or coverage gaps.
    
    Args:
        document_id: The ID of the document to analyze.
        
    Returns:
        A list of flagged items with explanations.
    """
    results = get_store().search(
        "exclusion limitation restriction unusual clause",
        n_results=8,
        document_id=document_id
    )
    if not results:
        return f"No context found for document {document_id}."
    
    context = "\n\n".join([r["text"] for r in results])
    
    # call llm
    client = instructor.from_openai(openai.OpenAI())
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=CoverageLimits,
        messages=[
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}"
            },
            {
                "role": "user",
                "content": (
                    f"Review the following insurance policy text and identify anomalies, "
                    f"non-standard provisions, ambiguous language, or potential coverage gaps.\n\n"
                    f"Policy text:\n{context}\n\n"
                    "For each issue found, provide:\n"
                    "1. ISSUE: A short label for the anomaly (e.g. 'Non-standard exclusion')\n"
                    "2. QUOTED TEXT: The exact language from the policy that is concerning "
                    "(quote directly, do not paraphrase)\n"
                    "3. WHY IT'S UNUSUAL: How this deviates from standard policy language "
                    "or industry norms\n"
                    "4. POTENTIAL IMPACT: How this could affect the policyholder in practice\n"
                    "5. SEVERITY: Rate as LOW, MEDIUM, or HIGH\n\n"
                    "If no genuine anomalies are found, say so explicitly. "
                    "Do not manufacture concerns where none exist. "
                    "List findings from highest to lowest severity."
                )
            }
        ]
    )
    return result
    


@tool
def generate_summary(document_id: str, audience: str = "general") -> str:
    """
    Generate a plain-language summary of an insurance policy document tailored 
    to a specific audience.
    
    Args:
        document_id: The ID of the document to summarize.
        audience: One of 'general' (plain English), 'technical' (for actuaries/underwriters), 
                  or 'executive' (brief key points for leadership).
                  
    Returns:
        A structured summary appropriate for the requested audience.
    """
    valid_audiences = {"general", "technical", "executive"}
    if audience not in valid_audiences:
        return f"Invalid audience '{audience}'. Choose from : {', '.join(valid_audiences)}."
    
    results = get_store().search(
        "coverage benefits terms conditions premium",
        n_results=10,
        document_id=document_id
    )
    if not results:
        return f"No content found for document {document_id}."
    
    context = "\n\n".join([r["text"] for r in results])
    
    audience_instructions = {
        "general": "Write in plain English for a policyholder with no insurance background.",
        "technical": "Use precise actuarial and underwriting terminology.",
        "executive": "Provide a 5-bullet executive summary of key coverage and risks."
    }
    
    # call llm
    client = instructor.from_openai(openai.OpenAI())
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=CoverageLimits,
        messages=[
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}"
            },
            {
                "role": "user",
                "content": (
                    f"Summarize the following insurance policy for this audience: "
                    f"{audience.upper()}\n\n"
                    f"Audience guidance: {audience_instructions[audience]}\n\n"
                    f"Policy text:\n{context}\n\n"
                    "Base your summary only on what is explicitly stated in the provided text. "
                    "If important information such as premium or coverage limits is not present "
                    "in the retrieved text, note that it was not available rather than omitting "
                    "it silently."
                )
            }
        ]
    )
    return result