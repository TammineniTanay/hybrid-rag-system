"""
Hybrid RAG System — FastAPI Application

Entry point that wires together all services and exposes the REST API.
Run with: uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from app.config import get_settings
from app.api.routes import router, init_services

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("starting_hybrid_rag_system")
    init_services()
    yield
    logger.info("shutting_down")


app = FastAPI(
    title="Hybrid RAG System",
    description=(
        "Production-grade RAG with hybrid retrieval (dense + sparse + graph), "
        "Corrective RAG (CRAG), feedback-driven reward re-ranking, and "
        "RAGAS evaluation dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
settings = get_settings()
origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "Hybrid RAG System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
