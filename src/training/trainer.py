"""
DSPy Training Loop.

Uses DSPy's BootstrapFewShot optimizer to automatically find the best
few-shot examples and prompt configurations for the RAG pipeline.
Replaces Agent Lightning's GRPO with Stanford NLP's proven approach.

Respects Gemini free tier with rate limiting (12 req/min).
"""

import asyncio
import json
import logging
import os
import time

import dspy

from src.config import settings
from src.training.dspy_modules import (
    RAGPipeline,
    configure_dspy,
    answer_quality_metric,
)
from src.training.reward import compute_reward

logger = logging.getLogger(__name__)

# Rate limiting: 12 requests per minute → 5 second gap
RATE_LIMIT_DELAY = 5.0


def _load_eval_queries() -> list[str]:
    """Load evaluation queries from file, or create defaults."""
    eval_path = os.path.join(settings.data_path, "eval_queries.txt")

    if not os.path.exists(eval_path):
        logger.info("Creating sample eval queries at %s", eval_path)
        os.makedirs(settings.data_path, exist_ok=True)
        sample_queries = [
            "What is retrieval augmented generation?",
            "How does BM25 scoring work?",
            "Explain the transformer architecture.",
            "What is cosine similarity?",
            "How do cross-encoders differ from bi-encoders?",
            "What is reinforcement learning from human feedback?",
            "Explain the attention mechanism in transformers.",
            "What is the purpose of document chunking?",
            "How does reciprocal rank fusion work?",
            "What are embedding models used for?",
            "Explain vector databases and their use cases.",
            "What is the difference between sparse and dense retrieval?",
            "How does prompt optimization work?",
            "What is prompt engineering?",
            "Explain chain-of-thought reasoning.",
            "What are the benefits of hybrid search?",
            "How does reranking improve retrieval quality?",
            "What is a knowledge graph?",
            "Explain few-shot learning.",
            "What is the role of confidence scoring in RAG?",
        ]
        with open(eval_path, "w") as f:
            f.write("\n".join(sample_queries))

    with open(eval_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _build_trainset(queries: list[str]) -> list[dspy.Example]:
    """Convert queries into DSPy Examples for the optimizer."""
    examples = []
    for q in queries:
        ex = dspy.Example(
            question=q,
            context="",  # Will be filled by retrieval during optimization
        ).with_inputs("question", "context")
        examples.append(ex)
    return examples


def run_dspy_optimization():
    """
    Run DSPy prompt optimization on the RAG pipeline.

    Uses BootstrapFewShot to find optimal few-shot examples
    that maximize answer quality across the evaluation set.
    """
    logger.info("=" * 60)
    logger.info("  DSPy Prompt Optimization")
    logger.info("=" * 60)

    # Configure DSPy with Gemini
    configure_dspy()

    # Load queries and build dataset
    queries = _load_eval_queries()
    trainset = _build_trainset(queries[:15])  # First 15 for training
    devset = _build_trainset(queries[15:])    # Last 5 for dev/eval

    logger.info("Training set: %d examples", len(trainset))
    logger.info("Dev set: %d examples", len(devset))

    # Create the pipeline module
    pipeline = RAGPipeline()

    # ── Optimizer: BootstrapFewShot ───────────────────────────────
    # This finds optimal few-shot examples by bootstrapping from
    # the training set, scoring with our metric, and selecting the
    # best demonstrations.
    logger.info("Starting BootstrapFewShot optimization...")

    optimizer = dspy.BootstrapFewShot(
        metric=answer_quality_metric,
        max_bootstrapped_demos=4,     # Max few-shot examples to add
        max_labeled_demos=4,          # Max labeled examples
        max_rounds=1,                 # Keep low for free tier
    )

    # Compile (optimize) the pipeline
    start_time = time.time()
    try:
        compiled_pipeline = optimizer.compile(
            pipeline,
            trainset=trainset,
        )
        elapsed = time.time() - start_time
        logger.info("Optimization completed in %.1fs", elapsed)
    except Exception as e:
        logger.error("Optimization failed: %s", e)
        logger.info("Falling back to unoptimized pipeline.")
        compiled_pipeline = pipeline
        elapsed = time.time() - start_time

    # ── Evaluate on dev set ──────────────────────────────────────
    logger.info("Evaluating optimized pipeline on dev set...")

    scores = []
    for i, example in enumerate(devset):
        try:
            pred = compiled_pipeline(
                question=example.question,
                context=example.context if hasattr(example, 'context') else "",
            )
            score = answer_quality_metric(example, pred)
            scores.append(score)
            logger.info(
                "  [%d/%d] Q: %s → Score: %.2f",
                i + 1, len(devset), example.question[:40], score,
            )
            time.sleep(RATE_LIMIT_DELAY)  # Rate limit
        except Exception as e:
            logger.warning("  [%d/%d] Failed: %s", i + 1, len(devset), e)
            scores.append(0.0)
            time.sleep(RATE_LIMIT_DELAY)

    avg_score = sum(scores) / max(len(scores), 1)

    # ── Save optimized pipeline ──────────────────────────────────
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "optimized_rag.json")

    try:
        compiled_pipeline.save(checkpoint_path)
        logger.info("Optimized pipeline saved to %s", checkpoint_path)
    except Exception as e:
        logger.warning("Failed to save checkpoint: %s", e)

    # ── Summary ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Optimization Results")
    logger.info("=" * 60)
    logger.info("  Time taken       : %.1fs", elapsed)
    logger.info("  Training examples: %d", len(trainset))
    logger.info("  Dev examples     : %d", len(devset))
    logger.info("  Avg dev score    : %.3f", avg_score)
    logger.info("  Checkpoint       : %s", checkpoint_path)
    logger.info("=" * 60)

    return {
        "avg_score": avg_score,
        "elapsed": elapsed,
        "checkpoint": checkpoint_path,
    }


async def run_agent_evaluation():
    """
    Run the full LangGraph agent on eval queries and collect rewards.
    Uses the actual agent pipeline (not DSPy) for end-to-end evaluation.
    """
    from src.agents.graph import run_agent

    queries = _load_eval_queries()
    logger.info("Running agent evaluation on %d queries...", len(queries))

    results = []
    for i, query in enumerate(queries):
        logger.info("  [%d/%d] %s", i + 1, len(queries), query[:50])
        try:
            result = await run_agent(query, session_id=f"eval_{i}")
            reward = result.get("reward", 0.0)
            confidence = result.get("confidence", 0.0)
            results.append({
                "query": query,
                "reward": reward,
                "confidence": confidence,
                "retries": result.get("retry_count", 0),
            })
            logger.info(
                "    → reward=%.3f, confidence=%.3f, retries=%d",
                reward, confidence, result.get("retry_count", 0),
            )
        except Exception as e:
            logger.error("    → Failed: %s", e)
            results.append({"query": query, "reward": 0.0, "confidence": 0.0, "retries": 0})

        await asyncio.sleep(RATE_LIMIT_DELAY)

    # Summary
    avg_reward = sum(r["reward"] for r in results) / max(len(results), 1)
    avg_confidence = sum(r["confidence"] for r in results) / max(len(results), 1)

    logger.info("=" * 60)
    logger.info("  Agent Evaluation Results")
    logger.info("=" * 60)
    logger.info("  Queries evaluated  : %d", len(results))
    logger.info("  Avg reward         : %.3f", avg_reward)
    logger.info("  Avg confidence     : %.3f", avg_confidence)
    logger.info("=" * 60)

    # Save results
    results_path = os.path.join(settings.data_path, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results


# ── CLI Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DSPy Training & Evaluation")
    parser.add_argument(
        "--mode",
        choices=["optimize", "evaluate", "both"],
        default="optimize",
        help="optimize: DSPy prompt optimization, evaluate: agent eval, both: run both",
    )
    args = parser.parse_args()

    if args.mode in ("optimize", "both"):
        run_dspy_optimization()

    if args.mode in ("evaluate", "both"):
        asyncio.run(run_agent_evaluation())
