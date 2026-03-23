"""
REST API routes for the hybrid RAG system.

Endpoints:
- POST /api/query       → ask a question, get a RAG-powered answer
- POST /api/feedback    → submit user rating on a response
- GET  /api/eval/metrics → current aggregate evaluation metrics
- GET  /api/eval/history → historical eval data for charting
- POST /api/ingest      → upload and index new documents
- GET  /api/health      → service health check
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
import uuid
import time
import structlog

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    FeedbackRequest,
    SourceReference,
    CRAGAction,
    EvalDashboardData,
    EvalSnapshot,
    RAGASMetrics,
)
from app.core.hybrid_retriever import HybridRetriever
from app.core.crag_pipeline import CRAGPipeline
from app.core.generator import AnswerGenerator
from app.evaluation.ragas_evaluator import RAGASEvaluator
from app.evaluation.reward_model import RewardModel
from app.services.database import DatabaseService

logger = structlog.get_logger()
router = APIRouter(prefix="/api")

# ── Service singletons (initialized in main.py lifespan) ─────
# These get set by the application startup handler.
hybrid_retriever: Optional[HybridRetriever] = None
crag_pipeline: Optional[CRAGPipeline] = None
generator: Optional[AnswerGenerator] = None
evaluator: Optional[RAGASEvaluator] = None
reward_model: Optional[RewardModel] = None
db: Optional[DatabaseService] = None


def init_services():
    """Called once during app startup to wire up all services."""
    global hybrid_retriever, crag_pipeline, generator, evaluator, reward_model, db

    hybrid_retriever = HybridRetriever()
    crag_pipeline = CRAGPipeline(hybrid_retriever)
    generator = AnswerGenerator()
    evaluator = RAGASEvaluator()
    reward_model = RewardModel()
    db = DatabaseService()

    logger.info("all_services_initialized")


# ── Query endpoint ────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """
    Main RAG endpoint. Receives a question, retrieves relevant
    context via hybrid search, optionally runs CRAG for quality
    correction, generates an answer, and evaluates the result.

    Flow:
    1. Hybrid retrieval (dense + sparse + graph)
    2. Reward model re-ranking (if trained)
    3. CRAG grading + correction (if enabled)
    4. LLM answer generation
    5. RAGAS evaluation
    6. Log everything to database
    """
    query_id = str(uuid.uuid4())
    start_total = time.perf_counter()

    # step 1: hybrid retrieval
    start_retrieval = time.perf_counter()
    fused_results = hybrid_retriever.retrieve(
        query=request.question,
        top_k=request.top_k,
    )
    retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

    if not fused_results:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # step 2: reward model re-ranking
    fused_results = reward_model.re_rank(request.question, fused_results)

    # step 3: CRAG (optional)
    crag_action = None
    if request.use_crag:
        crag_state = crag_pipeline.run(request.question, fused_results)
        answer = crag_state["answer"]
        crag_decision = crag_state.get("crag_decision", {})
        crag_action = crag_decision.get("action")
        final_chunks = crag_state.get("final_chunks", fused_results)
        generation_ms = 0  # timing is inside CRAG
    else:
        # step 4: direct generation without CRAG
        answer, generation_ms = generator.generate(request.question, fused_results)
        final_chunks = fused_results

    total_ms = (time.perf_counter() - start_total) * 1000

    # build source references for the response
    sources = []
    for fused in final_chunks[:request.top_k]:
        sources.append(
            SourceReference(
                chunk_id=fused.chunk.chunk_id,
                content_preview=fused.chunk.content[:200],
                source=fused.chunk.source,
                retriever=fused.contributing_retrievers[0] if fused.contributing_retrievers else "dense",
                relevance_score=round(fused.rrf_score, 4),
            )
        )

    # step 5: async evaluation (non-blocking in production)
    try:
        contexts = [f.chunk.content for f in final_chunks]
        eval_snap = evaluator.evaluate_single(
            question=request.question,
            answer=answer,
            contexts=contexts,
        )
        eval_snap.query_id = query_id

        db.save_eval_snapshot(
            query_id=query_id,
            faithfulness=eval_snap.metrics.faithfulness,
            answer_relevancy=eval_snap.metrics.answer_relevancy,
            context_precision=eval_snap.metrics.context_precision,
            context_recall=eval_snap.metrics.context_recall,
        )
    except Exception as exc:
        logger.warning("eval_failed", error=str(exc))

    # step 6: log the query
    chunk_data = [
        {
            "chunk_id": f.chunk.chunk_id,
            "content": f.chunk.content[:300],
            "rrf_score": f.rrf_score,
            "retrievers": [r.value for r in f.contributing_retrievers],
        }
        for f in final_chunks
    ]

    db.save_query_log(
        query_id=query_id,
        question=request.question,
        answer=answer,
        chunks=chunk_data,
        crag_action=crag_action,
        retrieval_time_ms=retrieval_ms,
        generation_time_ms=generation_ms,
    )

    return QueryResponse(
        query_id=query_id,
        question=request.question,
        answer=answer,
        sources=sources,
        crag_action_taken=crag_action,
        retrieval_time_ms=round(retrieval_ms, 1),
        generation_time_ms=round(generation_ms, 1),
        total_time_ms=round(total_ms, 1),
    )


# ── Feedback endpoint ─────────────────────────────────────────

@router.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Store user feedback on a query response."""
    # look up the original query to store alongside feedback
    chunks = db.get_chunks_by_query_id(request.query_id)
    chunk_ids = [c.get("chunk_id", "") for c in chunks] if chunks else []

    # we need the original question and answer from the query log
    # for now, store what we have
    feedback_id = db.save_feedback(
        query_id=request.query_id,
        question="",  # filled from query log in production
        answer="",
        chunk_ids=chunk_ids,
        rating=request.rating,
        comment=request.comment,
    )

    return {"feedback_id": feedback_id, "status": "recorded"}


# ── Evaluation endpoints ──────────────────────────────────────

@router.get("/eval/metrics")
def get_eval_metrics():
    """Return aggregate evaluation metrics for the dashboard."""
    metrics = db.get_aggregate_metrics()
    return metrics


@router.get("/eval/history")
def get_eval_history(days: int = 30, limit: int = 200):
    """Return eval snapshots over time for charting."""
    history = db.get_eval_history(days=days, limit=limit)
    return {"history": history, "count": len(history)}


# ── Health check ──────────────────────────────────────────────

@router.get("/health")
def health_check():
    """Basic liveness probe."""
    return {
        "status": "healthy",
        "services": {
            "retriever": hybrid_retriever is not None,
            "crag": crag_pipeline is not None,
            "generator": generator is not None,
            "evaluator": evaluator is not None,
            "database": db is not None,
        },
    }
