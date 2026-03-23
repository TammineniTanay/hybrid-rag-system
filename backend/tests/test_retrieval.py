"""Tests for the retrieval layer and RRF fusion."""

import pytest
from unittest.mock import MagicMock, patch
from app.models.schemas import DocumentChunk, RetrievedChunk, FusedResult, RetrieverSource
from app.core.hybrid_retriever import HybridRetriever


def make_chunk(chunk_id, content):
    return DocumentChunk(chunk_id=chunk_id, content=content, source="test.pdf")

def make_retrieved(chunk_id, content, score, rank, source):
    return RetrievedChunk(chunk=make_chunk(chunk_id, content), score=score, retriever=source, rank=rank)


class TestReciprocalRankFusion:
    def setup_method(self):
        with patch("app.core.hybrid_retriever.DenseRetriever"):
            with patch("app.core.hybrid_retriever.SparseRetriever"):
                with patch("app.core.hybrid_retriever.GraphRetriever"):
                    self.retriever = HybridRetriever()

    def test_single_retriever_preserves_order(self):
        dense = [
            make_retrieved("a", "chunk a", 0.9, 1, RetrieverSource.DENSE),
            make_retrieved("b", "chunk b", 0.7, 2, RetrieverSource.DENSE),
            make_retrieved("c", "chunk c", 0.5, 3, RetrieverSource.DENSE),
        ]
        fused = self.retriever._reciprocal_rank_fusion([dense, [], []], k=60)
        assert len(fused) == 3
        assert fused[0].chunk.chunk_id == "a"

    def test_overlapping_results_boost_rrf_score(self):
        dense = [make_retrieved("shared", "shared", 0.9, 1, RetrieverSource.DENSE)]
        sparse = [make_retrieved("shared", "shared", 0.85, 1, RetrieverSource.SPARSE)]
        fused = self.retriever._reciprocal_rank_fusion([dense, sparse, []], k=60)
        assert fused[0].chunk.chunk_id == "shared"
        assert len(fused[0].contributing_retrievers) == 2

    def test_empty_returns_empty(self):
        fused = self.retriever._reciprocal_rank_fusion([[], [], []], k=60)
        assert len(fused) == 0


class TestHybridRetrieverFlow:
    def setup_method(self):
        with patch("app.core.hybrid_retriever.DenseRetriever"):
            with patch("app.core.hybrid_retriever.SparseRetriever"):
                with patch("app.core.hybrid_retriever.GraphRetriever"):
                    self.retriever = HybridRetriever()
                    self.retriever.dense.search.return_value = [
                        make_retrieved("d1", "dense 1", 0.9, 1, RetrieverSource.DENSE),
                        make_retrieved("d2", "dense 2", 0.7, 2, RetrieverSource.DENSE),
                    ]
                    self.retriever.sparse.search.return_value = [
                        make_retrieved("d1", "dense 1", 0.6, 2, RetrieverSource.SPARSE),
                    ]
                    self.retriever.graph.search.return_value = []

    def test_retrieve_returns_fused_results(self):
        results = self.retriever.retrieve("test query", top_k=5)
        assert len(results) > 0
        assert all(isinstance(r, FusedResult) for r in results)

    def test_shared_chunk_ranked_higher(self):
        results = self.retriever.retrieve("test query", top_k=5)
        assert results[0].chunk.chunk_id == "d1"
