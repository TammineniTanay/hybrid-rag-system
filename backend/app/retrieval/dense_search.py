"""
Dense (semantic) retrieval using Qdrant vector database.

This module handles embedding-based search where documents and queries
are converted to high-dimensional vectors. Similarity is measured by
cosine distance, catching semantic meaning even when surface-level
words differ entirely.

Example: query "terminate an employee" matches "letting go of staff"
because their embeddings are geometrically close.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
from typing import Optional
import uuid
import structlog

from app.config import get_settings
from app.models.schemas import DocumentChunk, RetrievedChunk, RetrieverSource

logger = structlog.get_logger()


class DenseRetriever:
    """
    Wraps Qdrant for vector similarity search.

    On initialization, loads the embedding model into memory and
    connects to Qdrant. The collection is created lazily on first
    index operation if it doesn't already exist.
    """

    def __init__(self):
        settings = get_settings()
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.qdrant_collection
        self.encoder = SentenceTransformer(settings.embedding_model)
        self.dimension = settings.embedding_dimension
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the vector collection if it's missing."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "created_qdrant_collection",
                collection=self.collection_name,
                dimension=self.dimension,
            )

    def index_chunks(self, chunks: list[DocumentChunk]):
        """
        Embed and store a batch of document chunks.

        Each chunk gets a vector computed by the sentence-transformer
        model. Metadata (source, page number, etc.) is stored as the
        Qdrant payload so we can return it during search without a
        second database lookup.
        """
        points = []
        texts = [c.content for c in chunks]

        # batch encode is much faster than encoding one by one
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        for chunk, vector in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.chunk_id,
                vector=vector.tolist(),
                payload={
                    "content": chunk.content,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "metadata": chunk.metadata,
                },
            )
            points.append(point)

        # upsert in batches of 100 to avoid overwhelming the server
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info("indexed_dense", count=len(points))

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Embed the query and find the closest document vectors.

        Returns ranked results with cosine similarity scores. An
        optional source_filter narrows results to a specific document.
        """
        query_vector = self.encoder.encode(query).tolist()

        # build an optional filter to scope by source document
        search_filter = None
        if source_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter),
                    )
                ]
            )

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
        )

        results = []
        for rank, hit in enumerate(hits, start=1):
            chunk = DocumentChunk(
                chunk_id=hit.id,
                content=hit.payload["content"],
                source=hit.payload.get("source", ""),
                page_number=hit.payload.get("page_number"),
                metadata=hit.payload.get("metadata", {}),
            )
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=hit.score,
                    retriever=RetrieverSource.DENSE,
                    rank=rank,
                )
            )

        logger.info("dense_search", query=query[:80], hits=len(results))
        return results

    def delete_collection(self):
        """Tear down the collection. Useful in tests and re-indexing."""
        self.client.delete_collection(self.collection_name)
        logger.warning("deleted_collection", collection=self.collection_name)
