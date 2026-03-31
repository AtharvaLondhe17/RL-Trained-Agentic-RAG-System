"""
LangGraph Orchestrator.

Builds the agent graph with self-correction loop.
Entry point: run_agent(query, session_id) -> AgentState.
"""

import logging
import uuid

from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.nodes.decompose import decompose_query
from src.agents.nodes.retrieve import hybrid_retrieve
from src.agents.nodes.rerank import rerank_context
from src.agents.nodes.generate import generate_answer
from src.agents.nodes.verify import verify_and_score
from src.config import settings

logger = logging.getLogger(__name__)

# ── Build the StateGraph ──────────────────────────────────────────────

builder = StateGraph(AgentState)
builder.add_node("decompose", decompose_query)
builder.add_node("retrieve", hybrid_retrieve)
builder.add_node("rerank", rerank_context)
builder.add_node("generate", generate_answer)
builder.add_node("verify", verify_and_score)

builder.set_entry_point("decompose")
builder.add_edge("decompose", "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")
builder.add_edge("generate", "verify")


def route_after_verify(state: AgentState) -> str:
    """Route back to retrieve for self-correction or end."""
    if state.get("needs_retry", False):
        logger.info("Self-correction loop: routing back to retrieve (attempt %d).", state["retry_count"])
        return "retrieve"
    return END


builder.add_conditional_edges("verify", route_after_verify, {"retrieve": "retrieve", END: END})

graph = builder.compile()


# ── Public API ────────────────────────────────────────────────────────

async def run_agent(query: str, session_id: str) -> AgentState:
    """
    Run the full agentic RAG pipeline.
    Returns the final AgentState with answer, citations, confidence, etc.
    """
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "sub_questions": [],
        "route": "rag",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "retry_count": 0,
        "reward": 0.0,
        "task_id": str(uuid.uuid4()),
        "needs_retry": False,
    }

    try:
        result = await graph.ainvoke(initial_state)
        return result
    except Exception as e:
        logger.error("Agent graph execution failed: %s", e)
        initial_state["answer"] = f"An error occurred: {e}"
        initial_state["confidence"] = 0.0
        return initial_state
