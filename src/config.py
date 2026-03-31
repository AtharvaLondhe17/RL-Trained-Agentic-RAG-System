"""
Configuration module using Pydantic BaseSettings.
Loads all values from .env file. Singleton pattern.
"""

import logging
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    gemini_api_key: str = "your_gemini_api_key_here"
    chroma_db_path: str = "./chroma_db"
    session_db_path: str = "./session.db"
    data_path: str = "./data"
    log_level: str = "INFO"
    confidence_threshold: float = 0.72
    max_retries: int = 2
    top_k_dense: int = 15
    top_k_sparse: int = 15
    rerank_top_n: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "all-mpnet-base-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gemini_model: str = "gemini-1.5-flash"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance — import this everywhere
settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rl-agentic-rag")
