"""
Feedback service: thin coordinator between the API and database layers.
"""

from typing import Optional
import structlog

from app.services.database import DatabaseManager
from app.evaluation.ragas_evaluator import RAGASEvaluator

logger = structlog.get_logger()


class FeedbackService:
    def __init__(self, db: DatabaseManager, evaluator: RAGASEvaluator):
        self.db = db
        self.evaluator = evaluator

    def submit_feedback(self, query_id: str, rating: int, comment: Optional[str] = None) -> dict:
        query_log = self.db.get_query_log(query_id)
        if not query_log:
            return {"error": f"Query {query_id} not found in log"}

        question = query_log["question"]
        answer = query_log["answer"]
        chunk_ids = [c.get("chunk_id", "") for c in query_log.get("chunks_data", [])]
        contexts = [c.get("content", "") for c in query_log.get("chunks_data", [])]

        feedback_id = self.db.save_feedback(
            query_id=query_id, question=question, answer=answer,
            chunk_ids=chunk_ids, rating=rating, comment=comment,
        )

        metrics = None
        if contexts and answer:
            try:
                snapshot = self.evaluator.evaluate_single(question=question, answer=answer, contexts=contexts)
                self.db.save_eval_snapshot(
                    query_id=query_id,
                    faithfulness=snapshot.metrics.faithfulness,
                    answer_relevancy=snapshot.metrics.answer_relevancy,
                    context_precision=snapshot.metrics.context_precision,
                    context_recall=snapshot.metrics.context_recall,
                )
                metrics = {
                    "faithfulness": snapshot.metrics.faithfulness,
                    "answer_relevancy": snapshot.metrics.answer_relevancy,
                    "context_precision": snapshot.metrics.context_precision,
                    "context_recall": snapshot.metrics.context_recall,
                }
            except Exception as exc:
                logger.warning("eval_after_feedback_failed", error=str(exc))

        return {"feedback_id": feedback_id, "query_id": query_id, "rating": rating, "metrics": metrics}
