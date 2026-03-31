"""
Decompose Query Node.

Breaks the user query into 1-3 focused sub-questions and classifies routing.
Uses Gemini 1.5 Flash via langchain-google-genai.
Includes session memory via sqlitedict.
"""

import json
import logging
import re

from src.utils import agl_compat as agl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from sqlitedict import SqliteDict

from src.agents.state import AgentState
from src.config import settings

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM_PROMPT = """You are a query analysis expert. Given a user question:
1. Break it into 1-3 focused sub-questions if needed. If the query is simple, return it as-is.
2. Classify the route: "rag" if answerable from documents, "web" if requires live data, "code" if requires computation.
Respond ONLY in JSON: {"sub_questions": [...], "route": "rag"|"web"|"code"}"""


async def decompose_query(state: AgentState) -> dict:
    """Decompose the user query into sub-questions and determine routing."""
    try:
        # Agent Lightning hook
        agl.emit_input(state["task_id"], state["query"])

        # ── Session Memory (Step 7) ──────────────────────────────
        history_context = ""
        try:
            with SqliteDict(settings.session_db_path, autocommit=False) as session_store:
                history = session_store.get(state["session_id"], [])
                if len(history) > 0:
                    last_turns = history[-3:]  # last 3 turns
                    history_lines = []
                    for turn in last_turns:
                        history_lines.append(f"Q: {turn.get('query', '')}")
                        history_lines.append(f"A: {turn.get('answer', '')[:200]}")
                    history_context = "\n\nPrevious conversation:\n" + "\n".join(history_lines)
        except Exception as e:
            logger.warning("Failed to load session history: %s", e)

        # ── LLM call ─────────────────────────────────────────────
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )

        user_msg = f"Question: {state['query']}{history_context}"
        messages = [
            SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        response = await llm.ainvoke(messages)
        raw = response.content

        # ── Parse JSON (handle markdown fences) ──────────────────
        # Strip ```json ... ``` fences if present
        cleaned = re.sub(r"```json\s*", "", raw)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            sub_questions = parsed.get("sub_questions", [state["query"]])
            route = parsed.get("route", "rag")
            if route not in ("rag", "web", "code"):
                route = "rag"
        except json.JSONDecodeError:
            logger.warning("Failed to parse decompose JSON, using fallback. Raw: %s", raw[:200])
            sub_questions = [state["query"]]
            route = "rag"

        logger.info("Decomposed into %d sub-questions, route=%s", len(sub_questions), route)
        return {"sub_questions": sub_questions, "route": route}

    except Exception as e:
        logger.error("Decompose node failed: %s", e)
        return {"sub_questions": [state["query"]], "route": "rag"}
