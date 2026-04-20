"""
agent/agent.py

LangChain agent that orchestrates tool use for insurance document Q&A. Uses 
OpenAPI function calling to decide which tolls to invoke and in what order.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are an expert insurance policy analyst with deep knowledge of
insurance products, policy terms, coverage structures, and regulatory requirements.

You have access to a knowledge base of insurance policy documents. Use your tools to:
- Search for specific policy information
- Extract structured coverage details
- Compare policies across different dimensions
- Identify unusual or potentially problematic policy language
- Generate summaries tailored to different audiences

Guidelines:
- Always use list_available_documents first if you're unsure what documents are available
- Use search_policy_document to find relevant context before making claims
- Be precise about coverage amounts, limits, and exclusions — these matter to policyholders
- When information is unclear or not found, say so rather than guessing
- For complex questions, break the problem into multiple tool calls

You are helpful, precise, and honest about the limits of what the documents contain."""

class InsuranceAgent:
    """
    Conversational agent for insurance document analysis.
    
    Maintains converation history across turns and uses LangChain's OpenAI 
    functions agent to orchestrate tool calls.
    
    Usage:
        agent = InsuranceAgent()
        response = agent.run("What is the deductible on the policy?")
        print(response)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        memory_window: int = 10,
        verrbose: bool = False
    ):
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.verbose = verrbose
        
        self._llm = ChatOpenAI(
            model=self.model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self._memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=memory_window
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(
            llm=self._llm,
            tools=ALL_TOOLS,
            prompt=self._prompt
        )
        
        self._executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            memory=self._memory,
            verbose=self.verbose,
            return_intermediate_steps=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
        
    def run(self, query: str) -> str:
        """
        Run the agent on a query and return the response.
        
        Args:
            query: Natural language question about insurance documents.
            
        Returns:
            Agent response string.
        """
        result = self._executor.invoke({"input": query})
        return result.get("output", "No response generated.")
    
    def run_with_steps(self, query: str) -> dict:
        """
        Run the agent and return both the response and intermediate tool calls.
        
        Args:
            query: Natural language question about insurance documents.
            
        Returns:
            Dictionary with 'output' and 'intermediate_steps' keys.
        """
        result = self._executor.invoke({"input": query})
        return {
            "output": result.get("output", ""),
            "intermediate_steps": result.get("intermediate_steps", [])
        }
        
    def clear_memory(self) -> None:
        """Clear the conversation history."""
        self._memory.clear()
