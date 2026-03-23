"""
Database service for feedback and evaluation storage.

Uses SQLAlchemy with PostgreSQL. Falls back to SQLite for local
development when PostgreSQL isn't available.

Tables:
- feedback: user ratings on query responses
- eval_snapshots: RAGAS metric snapshots per query
- query_log: full context for each query (for reward model retraining)
"""

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    JSON,
    func,
    desc,
)
from sqlalchemy.orm import Session, declarative_base
from datetime import datetime, timedelta
from typing import Optional
import uuid
import structlog

from app.config import get_settings

logger = structlog.get_logger()
Base = declarative_base()


class FeedbackRow(Base):
    __tablename__ = "feedback"

    feedback_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String, nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(JSON, default=list)
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvalSnapshotRow(Base):
    __tablename__ = "eval_snapshots"

    eval_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_id = Column(String, nullable=False, index=True)
    faithfulness = Column(Float, default=0.0)
    answer_relevancy = Column(Float, default=0.0)
    context_precision = Column(Float, default=0.0)
    context_recall = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryLogRow(Base):
    __tablename__ = "query_log"

    query_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    retrieved_chunks = Column(JSON, default=list)
    crag_action = Column(String, nullable=True)
    retrieval_time_ms = Column(Float, default=0.0)
    generation_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseService:
    """
    Handles all persistence for feedback, eval snapshots, and query logs.
    Connection is lazy — engine created on first use.
    """

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            settings = get_settings()
            try:
                self._engine = create_engine(settings.postgres_url)
                # test the connection
                with self._engine.connect() as conn:
                    conn.execute(func.now())
            except Exception:
                logger.warning("postgres_unavailable, falling back to sqlite")
                self._engine = create_engine("sqlite:///data/rag_feedback.db")

            Base.metadata.create_all(self._engine)
            logger.info("database_initialized")

        return self._engine

    def _session(self) -> Session:
        return Session(self._get_engine())

    # ── Feedback ──────────────────────────────────────────────

    def save_feedback(
        self,
        query_id: str,
        question: str,
        answer: str,
        chunk_ids: list[str],
        rating: int,
        comment: Optional[str] = None,
    ) -> str:
        feedback_id = str(uuid.uuid4())
        with self._session() as session:
            row = FeedbackRow(
                feedback_id=feedback_id,
                query_id=query_id,
                question=question,
                answer=answer,
                retrieved_chunk_ids=chunk_ids,
                rating=rating,
                comment=comment,
            )
            session.add(row)
            session.commit()
        logger.info("saved_feedback", query_id=query_id, rating=rating)
        return feedback_id

    def get_all_feedback(self, limit: int = 1000) -> list[dict]:
        with self._session() as session:
            rows = (
                session.query(FeedbackRow)
                .order_by(desc(FeedbackRow.created_at))
                .limit(limit)
                .all()
            )
            return [
                {
                    "feedback_id": r.feedback_id,
                    "query_id": r.query_id,
                    "question": r.question,
                    "answer": r.answer,
                    "chunk_ids": r.retrieved_chunk_ids,
                    "rating": r.rating,
                    "comment": r.comment,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                }
                for r in rows
            ]

    # ── Eval Snapshots ────────────────────────────────────────

    def save_eval_snapshot(
        self,
        query_id: str,
        faithfulness: float,
        answer_relevancy: float,
        context_precision: float,
        context_recall: float,
    ) -> str:
        eval_id = str(uuid.uuid4())
        with self._session() as session:
            row = EvalSnapshotRow(
                eval_id=eval_id,
                query_id=query_id,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
            )
            session.add(row)
            session.commit()
        return eval_id

    def get_eval_history(self, days: int = 30, limit: int = 500) -> list[dict]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._session() as session:
            rows = (
                session.query(EvalSnapshotRow)
                .filter(EvalSnapshotRow.created_at >= cutoff)
                .order_by(desc(EvalSnapshotRow.created_at))
                .limit(limit)
                .all()
            )
            return [
                {
                    "eval_id": r.eval_id,
                    "query_id": r.query_id,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                }
                for r in rows
            ]

    def get_aggregate_metrics(self) -> dict:
        with self._session() as session:
            eval_agg = session.query(
                func.count(EvalSnapshotRow.eval_id).label("total"),
                func.avg(EvalSnapshotRow.faithfulness).label("avg_faith"),
                func.avg(EvalSnapshotRow.answer_relevancy).label("avg_rel"),
                func.avg(EvalSnapshotRow.context_precision).label("avg_prec"),
                func.avg(EvalSnapshotRow.context_recall).label("avg_recall"),
            ).first()

            feedback_agg = session.query(
                func.count(FeedbackRow.feedback_id).label("total"),
                func.avg(FeedbackRow.rating).label("avg_rating"),
            ).first()

            query_count = session.query(func.count(QueryLogRow.query_id)).scalar() or 0
            crag_count = (
                session.query(func.count(QueryLogRow.query_id))
                .filter(QueryLogRow.crag_action.isnot(None))
                .filter(QueryLogRow.crag_action != "proceed")
                .scalar() or 0
            )

        return {
            "total_queries": query_count,
            "avg_faithfulness": round(float(eval_agg.avg_faith or 0), 4),
            "avg_relevancy": round(float(eval_agg.avg_rel or 0), 4),
            "avg_precision": round(float(eval_agg.avg_prec or 0), 4),
            "avg_recall": round(float(eval_agg.avg_recall or 0), 4),
            "avg_user_rating": round(float(feedback_agg.avg_rating or 0), 2),
            "feedback_count": feedback_agg.total or 0,
            "crag_trigger_rate": round(crag_count / max(query_count, 1), 4),
        }

    # ── Query Log ─────────────────────────────────────────────

    def save_query_log(
        self,
        query_id: str,
        question: str,
        answer: str,
        chunks: list[dict],
        crag_action: Optional[str],
        retrieval_time_ms: float,
        generation_time_ms: float,
    ):
        with self._session() as session:
            row = QueryLogRow(
                query_id=query_id,
                question=question,
                answer=answer,
                retrieved_chunks=chunks,
                crag_action=crag_action,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
            )
            session.add(row)
            session.commit()

    def get_chunks_by_query_id(self, query_id: str) -> list[dict]:
        with self._session() as session:
            row = session.query(QueryLogRow).filter_by(query_id=query_id).first()
            return row.retrieved_chunks if row else []
