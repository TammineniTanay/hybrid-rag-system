"""
RAG evaluation using RAGAS framework.

Computes four core metrics after each query-response cycle:
- Faithfulness: is the answer grounded in the retrieved context?
- Answer Relevancy: does the answer address the question?
- Context Precision: what fraction of retrieved chunks were useful?
- Context Recall: did we retrieve all the relevant information?

These metrics are stored per-query and aggregated for the dashboard.
When RAGAS evaluation isn't feasible in real-time (it calls an LLM
for each metric), we fall back to lightweight heuristic scores.
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from typing import Optional
import uuid
import structlog

from app.models.schemas import RAGASMetrics, EvalSnapshot, FusedResult

logger = structlog.get_logger()


class RAGASEvaluator:
    """
    Wraps RAGAS evaluation for single-query and batch assessment.

    RAGAS works by asking an LLM to judge the quality of retrieval
    and generation. It needs: question, answer, retrieved contexts,
    and (optionally) a ground truth answer.
    """

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> EvalSnapshot:
        """
        Evaluate a single query-response pair.

        If ground_truth is provided, context_recall can be computed.
        Without it, we skip that metric and set it to -1.
        """
        # RAGAS expects a HuggingFace Dataset
        eval_data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        metrics_to_run = [faithfulness, answer_relevancy, context_precision]

        if ground_truth:
            eval_data["ground_truth"] = [ground_truth]
            metrics_to_run.append(context_recall)

        try:
            dataset = Dataset.from_dict(eval_data)
            result = evaluate(dataset, metrics=metrics_to_run)

            metrics = RAGASMetrics(
                faithfulness=round(result["faithfulness"], 4),
                answer_relevancy=round(result["answer_relevancy"], 4),
                context_precision=round(result["context_precision"], 4),
                context_recall=round(result.get("context_recall", 0), 4),
            )

        except Exception as exc:
            # RAGAS can fail if the LLM quota is exceeded or context is empty.
            # fall back to heuristic scoring so the pipeline doesn't break.
            logger.warning("ragas_eval_failed", error=str(exc))
            metrics = self._heuristic_eval(question, answer, contexts)

        snapshot = EvalSnapshot(
            eval_id=str(uuid.uuid4()),
            query_id="",  # caller sets this
            metrics=metrics,
        )

        logger.info(
            "eval_complete",
            faithfulness=metrics.faithfulness,
            relevancy=metrics.answer_relevancy,
            precision=metrics.context_precision,
        )

        return snapshot

    def _heuristic_eval(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RAGASMetrics:
        """
        Lightweight fallback evaluation when RAGAS can't run.

        These are rough approximations, not proper LLM-judged scores:
        - Faithfulness: fraction of answer sentences with context overlap
        - Relevancy: keyword overlap between question and answer
        - Precision: average context chunk length (proxy for usefulness)
        - Recall: set to 0 since we can't measure without ground truth
        """
        answer_words = set(answer.lower().split())
        question_words = set(question.lower().split())
        context_words = set()
        for ctx in contexts:
            context_words.update(ctx.lower().split())

        # simple word overlap heuristics
        if context_words:
            faithfulness = len(answer_words & context_words) / max(len(answer_words), 1)
        else:
            faithfulness = 0.0

        relevancy = len(answer_words & question_words) / max(len(question_words), 1)
        relevancy = min(relevancy * 2, 1.0)  # scale up since overlap is naturally low

        # precision proxy: if contexts are short, they're probably focused
        if contexts:
            avg_len = sum(len(c.split()) for c in contexts) / len(contexts)
            precision = min(1.0, 100 / max(avg_len, 1))  # shorter = more precise
        else:
            precision = 0.0

        return RAGASMetrics(
            faithfulness=round(faithfulness, 4),
            answer_relevancy=round(relevancy, 4),
            context_precision=round(precision, 4),
            context_recall=0.0,
        )

    def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        ground_truths: Optional[list[str]] = None,
    ) -> list[EvalSnapshot]:
        """
        Batch evaluation for computing aggregate dashboard metrics.

        Uses RAGAS batch mode which is more efficient than calling
        evaluate_single in a loop (fewer LLM calls via batching).
        """
        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }

        metrics_to_run = [faithfulness, answer_relevancy, context_precision]

        if ground_truths:
            eval_data["ground_truth"] = ground_truths
            metrics_to_run.append(context_recall)

        try:
            dataset = Dataset.from_dict(eval_data)
            result = evaluate(dataset, metrics=metrics_to_run)

            snapshots = []
            df = result.to_pandas()
            for _, row in df.iterrows():
                metrics = RAGASMetrics(
                    faithfulness=round(row.get("faithfulness", 0), 4),
                    answer_relevancy=round(row.get("answer_relevancy", 0), 4),
                    context_precision=round(row.get("context_precision", 0), 4),
                    context_recall=round(row.get("context_recall", 0), 4),
                )
                snapshots.append(
                    EvalSnapshot(
                        eval_id=str(uuid.uuid4()),
                        query_id="",
                        metrics=metrics,
                    )
                )

            return snapshots

        except Exception as exc:
            logger.error("batch_eval_failed", error=str(exc))
            return [
                EvalSnapshot(
                    eval_id=str(uuid.uuid4()),
                    query_id="",
                    metrics=self._heuristic_eval(q, a, c),
                )
                for q, a, c in zip(questions, answers, contexts_list)
            ]
