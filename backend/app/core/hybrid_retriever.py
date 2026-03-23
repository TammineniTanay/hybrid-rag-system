"""
Hybrid retriever: runs dense, sparse, and graph search in parallel,
then fuses their ranked results using Reciprocal Rank Fusion (RRF).

RRF is elegantly simple: for each document, sum 1/(k + rank_i) across
all retrievers where the document appears. The constant k (typically 60)
dampens the influence of absolute rank position so a doc ranked #1 in
one retriever and #10 in another still gets a decent combined score.

This approach is retriever-agnostic — you could swap in any search
backend and the fusion math stays the same.
"""

import asyncio
from collections import defaultdict
from typing import Optional
import time
import structlog

from app.config import get_settings
from app.retrieval.dense_search import DenseRetriever
from app.retrieval.sparse_search import SparseRetriever
from app.retrieval.graph_search import GraphRetriever
from app.models.schemas import (
    DocumentChunk,
    RetrievedChunk,
    FusedResult,
    RetrieverSource,
)

logger = structlog.get_logger()


class HybridRetriever:
    """
    Orchestrates three retrieval strategies and merges their output.

    The three retrievers cover complementary weaknesses:
    - Dense (Qdrant): catches semantic similarity, misses exact terms
    - Sparse (Elasticsearch): catches exact keywords, misses synonyms
    - Graph (Neo4j): catches multi-hop relationships, needs entities

    Each retriever is called with the same query. Their ranked results
    are merged via RRF into a single ranked list.
    """

    def __init__(self):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.graph = GraphRetriever()
        self.settings = get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        entity_names: Optional[list[str]] = None,
        source_filter: Optional[str] = None,
    ) -> list[FusedResult]:
        """
        Run all three retrievers and fuse results.

        entity_names is an optional list of entities extracted from
        the query (e.g. by an LLM). If provided, the graph retriever
        uses them for more precise traversal. If not, graph search
        falls back to fuzzy text matching on node properties.
        """
        start = time.perf_counter()

        # run all three searches
        # in production you'd use asyncio.gather for parallelism,
        # but synchronous is simpler to debug during development
        dense_results = self.dense.search(
            query, top_k=self.settings.top_k_per_retriever, source_filter=source_filter
        )
        sparse_results = self.sparse.search(
            query, top_k=self.settings.top_k_per_retriever, source_filter=source_filter
        )
        graph_results = self.graph.search(
            query, top_k=self.settings.top_k_per_retriever, entity_names=entity_names
        )

        # fuse with RRF
        fused = self._reciprocal_rank_fusion(
            [dense_results, sparse_results, graph_results],
            k=self.settings.rrf_k,
        )

        # take top_k from the fused list
        fused = fused[:top_k]

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "hybrid_retrieval",
            query=query[:80],
            dense_hits=len(dense_results),
            sparse_hits=len(sparse_results),
            graph_hits=len(graph_results),
            fused_hits=len(fused),
            time_ms=round(elapsed, 1),
        )

        return fused

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[RetrievedChunk]],
        k: int = 60,
    ) -> list[FusedResult]:
        """
        Merge multiple ranked lists using RRF.

        For each document d:
            RRF(d) = Σ 1 / (k + rank_i(d))

        where the sum is over all retrievers that returned d.

        Documents are identified by chunk_id. If the same logical
        content appears with different IDs (e.g. graph vs dense),
        we use content hashing as a fallback dedup key.
        """
        # accumulate RRF scores per chunk
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, DocumentChunk] = {}
        source_map: dict[str, set] = defaultdict(set)
        individual_scores: dict[str, dict[str, float]] = defaultdict(dict)

        for result_list in result_lists:
            for item in result_list:
                cid = item.chunk.chunk_id

                # RRF formula
                rrf_scores[cid] += 1.0 / (k + item.rank)

                # store the chunk object (first seen wins)
                if cid not in chunk_map:
                    chunk_map[cid] = item.chunk

                # track which retrievers contributed
                source_map[cid].add(item.retriever)
                individual_scores[cid][item.retriever.value] = item.score

        # sort by descending RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for cid in sorted_ids:
            fused_results.append(
                FusedResult(
                    chunk=chunk_map[cid],
                    rrf_score=rrf_scores[cid],
                    contributing_retrievers=list(source_map[cid]),
                    individual_scores=individual_scores[cid],
                )
            )

        return fused_results

    def index_chunks(self, chunks: list[DocumentChunk]):
        """Index chunks across both dense and sparse stores."""
        self.dense.index_chunks(chunks)
        self.sparse.index_chunks(chunks)
        logger.info("indexed_hybrid", count=len(chunks))
