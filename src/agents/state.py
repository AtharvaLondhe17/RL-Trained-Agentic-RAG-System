"""
Agent State definition using TypedDict.
All fields that flow through the LangGraph are defined here.
"""

from typing import TypedDict


class AgentState(TypedDict):
    query: str                    # original user query
    session_id: str               # for memory lookup
    sub_questions: list[str]      # decomposed sub-queries
    route: str                    # "rag" | "web" | "code"
    retrieved_chunks: list[dict]  # each: {text, source, score}
    reranked_chunks: list[dict]   # top-N after cross-encoder
    answer: str                   # generated response
    citations: list[str]          # source filenames used
    confidence: float             # 0.0 to 1.0
    retry_count: int              # increments on self-correction loop
    reward: float                 # set after verification; used by RL
    task_id: str                  # Agent Lightning span ID
    needs_retry: bool             # flag set by verify to indicate loop
