"""Tests for evaluation and reward model."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.evaluation.ragas_evaluator import RAGASEvaluator
from app.evaluation.reward_model import RewardModel
from app.models.schemas import FusedResult, DocumentChunk, RetrieverSource


class TestHeuristicEval:
    def setup_method(self):
        self.evaluator = RAGASEvaluator()

    def test_good_overlap_high_faithfulness(self):
        metrics = self.evaluator._heuristic_eval(
            question="What is machine learning?",
            answer="Machine learning is a branch of AI that uses data and algorithms.",
            contexts=["Machine learning is a branch of artificial intelligence focusing on data and algorithms."],
        )
        assert metrics.faithfulness > 0.5

    def test_no_context_zero_scores(self):
        metrics = self.evaluator._heuristic_eval(
            question="What is deep learning?",
            answer="Deep learning uses neural networks.",
            contexts=[],
        )
        assert metrics.faithfulness == 0.0

    def test_scores_bounded(self):
        metrics = self.evaluator._heuristic_eval("q", "a with words", ["context with different words"])
        assert all(0 <= v <= 1 for v in [metrics.faithfulness, metrics.answer_relevancy, metrics.context_precision, metrics.context_recall])


class TestRewardModel:
    def setup_method(self):
        with patch("app.evaluation.reward_model.SentenceTransformer") as MockEncoder:
            mock_instance = MockEncoder.return_value
            mock_instance.encode.return_value = np.random.randn(384).astype(np.float32)
            self.reward = RewardModel()
            self.reward.encoder = mock_instance

    def test_feature_shape(self):
        features = self.reward.extract_features("query", "chunk content", 0.5, 2)
        assert features.shape == (6,)

    def test_rerank_without_model_is_identity(self):
        self.reward.model = None
        fused = [
            FusedResult(chunk=DocumentChunk(chunk_id=f"c{i}", content=f"content {i}"),
                        rrf_score=1.0 - i * 0.1, contributing_retrievers=[RetrieverSource.DENSE])
            for i in range(3)
        ]
        result = self.reward.re_rank("test query", fused)
        assert [r.chunk.chunk_id for r in result] == ["c0", "c1", "c2"]
