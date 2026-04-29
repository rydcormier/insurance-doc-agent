"""
agent/agent.py

LangChain agent that orchestrates tool use for insurance document Q&A. Uses 
OpenAPI function calling to decide which tolls to invoke and in what order.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from prompts import SYSTEM_PROMPT
from tools import ALL_TOOLS

load_dotenv()

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
        self._thread_id = str(uuid.uuid4())

        self._llm = ChatOpenAI(
            model=self.model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self._checkpointer = MemorySaver()
        self._executor = create_react_agent(
            model=self._llm,
            tools=ALL_TOOLS,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
            checkpointer=self._checkpointer,
        )
        
    def run(self, query: str) -> str:
        """
        Run the agent on a query and return the response.

        Args:
            query: Natural language question about insurance documents.

        Returns:
            Agent response string.
        """
        config = {"configurable": {"thread_id": self._thread_id}}
        result = self._executor.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else "No response generated."

    def run_with_steps(self, query: str) -> dict:
        """
        Run the agent and return both the response and intermediate tool calls.

        Args:
            query: Natural language question about insurance documents.

        Returns:
            Dictionary with 'output' and 'intermediate_steps' keys.
        """
        config = {"configurable": {"thread_id": self._thread_id}}
        result = self._executor.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        messages = result.get("messages", [])
        tool_messages = [m for m in messages if m.type == "tool"]
        return {
            "output": messages[-1].content if messages else "",
            "intermediate_steps": tool_messages,
        }

    def clear_memory(self) -> None:
        """Clear the conversation history by starting a new thread."""
        self._thread_id = str(uuid.uuid4())
