"""Tests for CRAG decision logic."""

import pytest
from unittest.mock import MagicMock, patch
from app.models.schemas import FusedResult, DocumentChunk, RetrieverSource


def make_fused(chunk_id, content="test content"):
    return FusedResult(
        chunk=DocumentChunk(chunk_id=chunk_id, content=content, source="test.pdf"),
        rrf_score=0.5, contributing_retrievers=[RetrieverSource.DENSE],
    )


class TestCRAGDecision:
    def setup_method(self):
        with patch("app.core.crag_pipeline.HybridRetriever"):
            with patch("app.core.crag_pipeline.ChatAnthropic"):
                with patch("app.core.crag_pipeline.WebSearchRetriever"):
                    from app.core.crag_pipeline import CRAGPipeline
                    self.pipeline = CRAGPipeline.__new__(CRAGPipeline)
                    self.pipeline.settings = MagicMock()
                    self.pipeline.settings.crag_relevant_ratio = 0.6
                    self.pipeline.settings.crag_irrelevant_ratio = 0.6
                    self.pipeline.settings.max_crag_retries = 2

    def test_mostly_relevant_proceeds(self):
        state = {
            "question": "What is attention?",
            "retrieved_chunks": [make_fused("c1"), make_fused("c2"), make_fused("c3")],
            "chunk_grades": [
                {"chunk_id": "c1", "grade": "relevant", "reasoning": "good"},
                {"chunk_id": "c2", "grade": "relevant", "reasoning": "good"},
                {"chunk_id": "c3", "grade": "irrelevant", "reasoning": "off topic"},
            ],
            "retry_count": 0,
        }
        result = self.pipeline._decide_action(state)
        assert result["crag_decision"]["action"] == "proceed"

    def test_mostly_irrelevant_rewrites(self):
        state = {
            "question": "What is attention?",
            "retrieved_chunks": [make_fused("c1"), make_fused("c2"), make_fused("c3")],
            "chunk_grades": [
                {"chunk_id": "c1", "grade": "irrelevant", "reasoning": "wrong"},
                {"chunk_id": "c2", "grade": "irrelevant", "reasoning": "wrong"},
                {"chunk_id": "c3", "grade": "relevant", "reasoning": "ok"},
            ],
            "retry_count": 0,
        }
        result = self.pipeline._decide_action(state)
        assert result["crag_decision"]["action"] == "rewrite"

    def test_max_retries_forces_proceed(self):
        state = {
            "question": "What is attention?",
            "retrieved_chunks": [make_fused("c1")],
            "chunk_grades": [{"chunk_id": "c1", "grade": "irrelevant", "reasoning": "wrong"}],
            "retry_count": 2,
        }
        result = self.pipeline._decide_action(state)
        assert result["crag_decision"]["action"] == "proceed"
