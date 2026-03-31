"""
DSPy Modules for RL-Agentic RAG.

Defines DSPy Signatures and Modules that wrap the core RAG pipeline.
These are used by DSPy's optimizers (MIPROv2, BootstrapFewShot) to
automatically optimize prompts and few-shot examples.
"""

import dspy
from src.config import settings


# ── Configure DSPy with Gemini ────────────────────────────────────────

def configure_dspy():
    """Configure DSPy to use Gemini 1.5 Flash."""
    lm = dspy.LM(
        f"google/gemini-1.5-flash",
        api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    dspy.configure(lm=lm)
    return lm


# ══════════════════════════════════════════════════════════════════════
# SIGNATURES — Define input/output contracts
# ══════════════════════════════════════════════════════════════════════

class DecomposeQuery(dspy.Signature):
    """Break a user question into focused sub-questions and classify the routing."""
    query = dspy.InputField(desc="The user's original question")
    sub_questions = dspy.OutputField(desc="1-3 focused sub-questions as a Python list")
    route = dspy.OutputField(desc="One of: rag, web, code")


class GenerateAnswer(dspy.Signature):
    """Answer a question using ONLY the provided context with inline citations."""
    context = dspy.InputField(desc="Retrieved and reranked document passages")
    question = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="Concise answer with [source] citations, under 300 words")


class VerifyAnswer(dspy.Signature):
    """Assess whether an answer is well-grounded in the provided context."""
    question = dspy.InputField(desc="The original question")
    answer = dspy.InputField(desc="The generated answer")
    context = dspy.InputField(desc="The context used to generate the answer")
    assessment = dspy.OutputField(desc="Brief assessment of answer quality")
    is_grounded = dspy.OutputField(desc="yes or no")


# ══════════════════════════════════════════════════════════════════════
# MODULES — Composable DSPy programs
# ══════════════════════════════════════════════════════════════════════

class QueryDecomposer(dspy.Module):
    """Decomposes queries into sub-questions with routing."""

    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(DecomposeQuery)

    def forward(self, query: str):
        result = self.decompose(query=query)
        return dspy.Prediction(
            sub_questions=result.sub_questions,
            route=result.route,
        )


class AnswerGenerator(dspy.Module):
    """Generates grounded answers with citations from context."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str, context: str):
        result = self.generate(question=question, context=context)
        return dspy.Prediction(answer=result.answer)


class RAGPipeline(dspy.Module):
    """
    Full RAG pipeline as a DSPy Module.
    
    This wraps the decompose → generate flow so DSPy optimizers
    can optimize the prompts and few-shot examples.
    The retrieve and rerank steps use our custom hybrid retrieval
    (not DSPy's built-in retriever) since we want BM25+dense+RRF.
    """

    def __init__(self):
        super().__init__()
        self.decomposer = QueryDecomposer()
        self.generator = AnswerGenerator()

    def forward(self, question: str, context: str = ""):
        # Step 1: Decompose (optimized by DSPy)
        decomposition = self.decomposer(query=question)

        # Step 2: Generate (optimized by DSPy)
        # Context is provided externally from our hybrid retrieval pipeline
        prediction = self.generator(
            question=question,
            context=context,
        )

        return dspy.Prediction(
            answer=prediction.answer,
            sub_questions=decomposition.sub_questions,
            route=decomposition.route,
        )


# ══════════════════════════════════════════════════════════════════════
# METRICS — Used by DSPy optimizers to score outputs
# ══════════════════════════════════════════════════════════════════════

def answer_quality_metric(example, pred, trace=None) -> float:
    """
    DSPy-compatible metric function.
    
    Scores an answer on:
    - Length appropriateness (not too short, not too long)
    - Contains citations [source]
    - Doesn't refuse when context is available
    
    Returns float 0.0 to 1.0.
    """
    answer = pred.answer if hasattr(pred, 'answer') else str(pred)

    score = 0.0

    # Length check (20-350 words ideal)
    word_count = len(answer.split())
    if 20 < word_count < 350:
        score += 0.3
    elif 10 < word_count:
        score += 0.15

    # Contains citations
    import re
    citations = re.findall(r'\[([^\]]+)\]', answer)
    if citations:
        score += 0.3

    # Doesn't refuse when it shouldn't
    refusal_phrases = ["cannot find", "no information", "not available"]
    has_context = hasattr(example, 'context') and len(str(example.context)) > 50
    is_refusal = any(phrase in answer.lower() for phrase in refusal_phrases)

    if has_context and not is_refusal:
        score += 0.2
    elif not has_context and is_refusal:
        score += 0.2  # correct refusal

    # Answer isn't just repeating the question
    if hasattr(example, 'question'):
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, answer.lower(), example.question.lower()).ratio()
        if similarity < 0.5:
            score += 0.2

    return min(1.0, score)
