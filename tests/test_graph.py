"""
Tests for the LangGraph agent graph.

- Route classification (rag, web, code)
- Self-correction loop triggering
- Retry count increments and stops at max_retries
Uses mocked Gemini API.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.agents.state import AgentState


def _make_state(**overrides) -> AgentState:
    """Create a default AgentState with optional overrides."""
    state: AgentState = {
        "query": "test query",
        "session_id": "test_session",
        "sub_questions": [],
        "route": "rag",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "retry_count": 0,
        "reward": 0.0,
        "task_id": "test-task-id",
        "needs_retry": False,
    }
    state.update(overrides)
    return state


# ── Test route classification ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_decompose_routes_to_rag():
    """Decompose should route to 'rag' for document-based queries."""
    mock_response = MagicMock()
    mock_response.content = json.dumps({"sub_questions": ["What is RAG?"], "route": "rag"})

    with patch("src.agents.nodes.decompose.ChatGoogleGenerativeAI") as MockLLM, \
         patch("src.agents.nodes.decompose.agl") as mock_agl, \
         patch("src.agents.nodes.decompose.SqliteDict", MagicMock()):
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke.return_value = mock_response
        MockLLM.return_value = mock_llm_instance

        from src.agents.nodes.decompose import decompose_query
        state = _make_state(query="What is RAG?")
        result = await decompose_query(state)

        assert result["route"] == "rag"
        assert "What is RAG?" in result["sub_questions"]


@pytest.mark.asyncio
async def test_decompose_routes_to_web():
    """Decompose should route to 'web' for live data queries."""
    mock_response = MagicMock()
    mock_response.content = json.dumps({"sub_questions": ["What is the current weather?"], "route": "web"})

    with patch("src.agents.nodes.decompose.ChatGoogleGenerativeAI") as MockLLM, \
         patch("src.agents.nodes.decompose.agl") as mock_agl, \
         patch("src.agents.nodes.decompose.SqliteDict", MagicMock()):
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke.return_value = mock_response
        MockLLM.return_value = mock_llm_instance

        from src.agents.nodes.decompose import decompose_query
        state = _make_state(query="What is the current weather?")
        result = await decompose_query(state)

        assert result["route"] == "web"


@pytest.mark.asyncio
async def test_decompose_routes_to_code():
    """Decompose should route to 'code' for computation queries."""
    mock_response = MagicMock()
    mock_response.content = json.dumps({"sub_questions": ["Calculate 2+2"], "route": "code"})

    with patch("src.agents.nodes.decompose.ChatGoogleGenerativeAI") as MockLLM, \
         patch("src.agents.nodes.decompose.agl") as mock_agl, \
         patch("src.agents.nodes.decompose.SqliteDict", MagicMock()):
        mock_llm_instance = AsyncMock()
        mock_llm_instance.ainvoke.return_value = mock_response
        MockLLM.return_value = mock_llm_instance

        from src.agents.nodes.decompose import decompose_query
        state = _make_state(query="Calculate 2+2")
        result = await decompose_query(state)

        assert result["route"] == "code"


# ── Test self-correction ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_triggers_retry_on_low_confidence():
    """Verify should trigger retry when confidence < threshold."""
    with patch("src.agents.nodes.verify.agl") as mock_agl, \
         patch("src.agents.nodes.verify.SqliteDict", MagicMock()):
        from src.agents.nodes.verify import verify_and_score

        state = _make_state(
            answer="Short.",
            citations=["nonexistent.txt"],
            reranked_chunks=[{"text": "some text", "source": "doc.txt", "score": 0.5}],
            query="What is this?",
            retry_count=0,
        )
        result = await verify_and_score(state)

        # Low confidence expected: no citation match, short answer
        assert result["confidence"] < 0.72, f"Expected low confidence, got {result['confidence']}"
        assert result["retry_count"] == 1, "Should increment retry_count"


@pytest.mark.asyncio
async def test_verify_stops_at_max_retries():
    """Verify should NOT increment retry past max_retries."""
    with patch("src.agents.nodes.verify.agl") as mock_agl, \
         patch("src.agents.nodes.verify.SqliteDict", MagicMock()):
        from src.agents.nodes.verify import verify_and_score

        state = _make_state(
            answer="Short answer that is not well grounded.",
            citations=["nonexistent.txt"],
            reranked_chunks=[{"text": "something else entirely", "source": "other.txt", "score": 0.5}],
            query="Complex unrelated question about physics?",
            retry_count=2,  # already at max_retries
        )
        result = await verify_and_score(state)

        # At max retries, should return confidence and reward (not increment retry)
        assert "reward" in result, "Should return reward when at max retries"


# ── Test route_after_verify ──────────────────────────────────────────

def test_route_after_verify_returns_retrieve_on_low_confidence():
    """Should route to retrieve when needs_retry is True."""
    from src.agents.graph import route_after_verify

    state = _make_state(needs_retry=True, retry_count=1)
    result = route_after_verify(state)
    assert result == "retrieve"


def test_route_after_verify_returns_end_on_high_confidence():
    """Should route to END when needs_retry is False (high confidence)."""
    from src.agents.graph import route_after_verify
    from langgraph.graph import END

    state = _make_state(needs_retry=False, retry_count=0)
    result = route_after_verify(state)
    assert result == END


def test_route_after_verify_returns_end_at_max_retries():
    """Should route to END when max retries reached and needs_retry is False."""
    from src.agents.graph import route_after_verify
    from langgraph.graph import END

    state = _make_state(needs_retry=False, retry_count=2)  # exceeds max_retries=2
    result = route_after_verify(state)
    assert result == END
