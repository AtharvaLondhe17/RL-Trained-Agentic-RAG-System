"""
Generate Answer Node.

Generates a grounded answer with citations using Gemini 1.5 Flash.
"""

import asyncio
import logging
import re

from src.utils import agl_compat as agl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import AgentState
from src.config import settings

logger = logging.getLogger(__name__)

GENERATE_SYSTEM_PROMPT = """You are a precise question-answering assistant. Answer the user's question using ONLY the provided context.
Rules:
- Cite your sources inline as [source_filename].
- If the context does not contain the answer, say "I cannot find this in the provided documents."
- Keep answers concise: under 300 words unless the question requires detail.
- Never invent facts not present in the context."""


async def generate_answer(state: AgentState) -> dict:
    """Generate a grounded answer with inline citations."""
    try:
        reranked = state.get("reranked_chunks", [])

        # Handle empty context
        if not reranked or (len(reranked) == 1 and reranked[0].get("source") == "system"):
            return {
                "answer": "No documents have been ingested yet. Please add documents via POST /ingest.",
                "citations": [],
            }

        # Build context
        context = "\n---\n".join(chunk["text"] for chunk in reranked)
        sources = list(set(chunk["source"] for chunk in reranked))

        # Rate limit respect for retries
        if state.get("retry_count", 0) > 0:
            await asyncio.sleep(1)

        # LLM call
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        )

        user_msg = f"Context:\n{context}\n\nQuestion: {state['query']}"
        messages = [
            SystemMessage(content=GENERATE_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        try:
            response = await llm.ainvoke(messages)
            answer = response.content
        except Exception as e:
            # Handle Gemini rate limit (429)
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str:
                logger.warning("Gemini rate limited. Waiting 60s before retry.")
                await asyncio.sleep(60)
                response = await llm.ainvoke(messages)
                answer = response.content
            else:
                raise

        # Extract citations: [filename] patterns
        citations = re.findall(r"\[([^\]]+)\]", answer)
        # Filter to only valid source citations
        citations = [c for c in citations if c in sources or c.endswith(('.txt', '.pdf', '.md', '.docx'))]
        citations = list(set(citations))

        agl.emit_tool_call(
            "generate",
            {"query": state["query"], "context_chunks": len(reranked)},
            {"answer_length": len(answer)},
        )

        logger.info("Generated answer: %d chars, %d citations", len(answer), len(citations))
        return {"answer": answer, "citations": citations}

    except Exception as e:
        logger.error("Generate node failed: %s", e)
        return {
            "answer": f"I encountered an error generating the answer: {e}",
            "citations": [],
        }
