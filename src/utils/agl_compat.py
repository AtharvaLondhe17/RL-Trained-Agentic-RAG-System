"""
Span Tracker — Lightweight observability layer.

Replaces Agent Lightning's emit_* functions with a SQLite-backed
span tracker. Tracks inputs, outputs, tool calls, and rewards
for every agent run. This data feeds into DSPy optimization.
"""

import datetime
import json
import logging
import os
import sqlite3
import threading

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get("SESSION_DB_PATH", "./session.db")
_lock = threading.Lock()
_initialized = False


def _init_db():
    """Create the spans table if it doesn't exist."""
    global _initialized
    if _initialized:
        return
    try:
        with _lock:
            conn = sqlite3.connect(_DB_PATH)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_spans_task
                ON spans(task_id)
            """)
            conn.commit()
            conn.close()
            _initialized = True
    except Exception as e:
        logger.warning("Span tracker DB init failed: %s", e)


def emit_input(task_id: str, data):
    """Record an agent input span."""
    _init_db()
    _store_span(task_id, "input", data)


def emit_output(data):
    """Record an agent output span."""
    _init_db()
    _store_span("_last", "output", data)


def emit_reward(task_id: str, reward: float):
    """Record a reward for a task span."""
    _init_db()
    _store_span(task_id, "reward", {"reward": reward})


def emit_tool_call(name: str, inputs: dict, outputs: dict):
    """Record a tool call span."""
    _init_db()
    _store_span("_tool", "tool_call", {
        "tool": name,
        "inputs": inputs,
        "outputs": outputs,
    })


def _store_span(task_id: str, event_type: str, data):
    """Store a span event in SQLite."""
    try:
        serialized = json.dumps(data, default=str)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with _lock:
            conn = sqlite3.connect(_DB_PATH)
            conn.execute(
                "INSERT INTO spans (task_id, event_type, data, timestamp) VALUES (?, ?, ?, ?)",
                (task_id, event_type, serialized, ts),
            )
            conn.commit()
            conn.close()
    except Exception as e:
        logger.debug("Span store failed (non-critical): %s", e)


def get_all_rewards() -> list[dict]:
    """Retrieve all reward spans for training analysis."""
    _init_db()
    try:
        with _lock:
            conn = sqlite3.connect(_DB_PATH)
            rows = conn.execute(
                "SELECT task_id, data, timestamp FROM spans WHERE event_type = 'reward' ORDER BY timestamp"
            ).fetchall()
            conn.close()
        return [
            {"task_id": r[0], "reward": json.loads(r[1]).get("reward", 0.0), "timestamp": r[2]}
            for r in rows
        ]
    except Exception:
        return []


def get_task_spans(task_id: str) -> list[dict]:
    """Retrieve all spans for a given task."""
    _init_db()
    try:
        with _lock:
            conn = sqlite3.connect(_DB_PATH)
            rows = conn.execute(
                "SELECT event_type, data, timestamp FROM spans WHERE task_id = ? ORDER BY timestamp",
                (task_id,),
            ).fetchall()
            conn.close()
        return [
            {"event_type": r[0], "data": json.loads(r[1]), "timestamp": r[2]}
            for r in rows
        ]
    except Exception:
        return []


AVAILABLE = True
