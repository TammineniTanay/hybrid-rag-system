import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Central configuration pulled from environment variables.
    Every external service connection, model choice, and tunable
    parameter lives here so nothing is hardcoded in business logic.
    """

    # --- LLM Providers ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    default_llm: str = "claude"  # "claude" or "openai"

    # --- Embedding Model ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # --- Qdrant (Dense Vector Search) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "research_papers"

    # --- Elasticsearch (BM25 Sparse Search) ---
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "research_papers"

    # --- Neo4j (Knowledge Graph) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "your_password_here"

    # --- PostgreSQL (Feedback + Eval Storage) ---
    postgres_url: str = "postgresql://raguser:ragpass@localhost:5432/ragdb"

    # --- Tavily (Web Search Fallback for CRAG) ---
    tavily_api_key: str = ""

    # --- LangSmith Tracing ---
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "hybrid-rag-system"

    # --- Retrieval Tuning ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_per_retriever: int = 5
    rrf_k: int = 60  # constant for reciprocal rank fusion
    relevance_threshold: float = 0.5

    # --- CRAG Thresholds ---
    crag_relevant_ratio: float = 0.6   # if >60% chunks relevant, proceed
    crag_irrelevant_ratio: float = 0.6  # if >60% irrelevant, trigger correction
    max_crag_retries: int = 2

    # --- Application ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
