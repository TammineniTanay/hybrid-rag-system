"""
Sparse (keyword) retrieval using Elasticsearch BM25.

BM25 is a bag-of-words ranking function that scores documents by
term frequency, inverse document frequency, and document length.
It shines where semantic search stumbles: exact matches for error
codes, product names, technical terms, and acronyms.

Example: "CUDA error 11" → BM25 nails the exact string match
while dense embeddings might return generic GPU troubleshooting.
"""

from elasticsearch import Elasticsearch
from typing import Optional
import structlog

from app.config import get_settings
from app.models.schemas import DocumentChunk, RetrievedChunk, RetrieverSource

logger = structlog.get_logger()


# Custom analyzer that handles technical content well:
# - lowercase filter for case-insensitive matching
# - asciifolding to normalize accented characters
# - word_delimiter_graph to split camelCase and hyphenated terms
INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "technical_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "word_delimiter_graph",
                    ],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "content": {
                "type": "text",
                "analyzer": "technical_analyzer",
            },
            "source": {"type": "keyword"},
            "page_number": {"type": "integer"},
            "metadata": {"type": "object", "enabled": False},
        }
    },
}


class SparseRetriever:
    """
    Wraps Elasticsearch for BM25 keyword search.

    The index uses a custom analyzer tuned for technical content.
    CamelCase splitting means "SentenceTransformer" also matches
    queries for "sentence" or "transformer" individually.
    """

    def __init__(self):
        settings = get_settings()
        self.es = Elasticsearch(settings.elasticsearch_url)
        self.index_name = settings.elasticsearch_index
        self._ensure_index()

    def _ensure_index(self):
        """Create the ES index with our custom mapping if missing."""
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
                index=self.index_name,
                body=INDEX_SETTINGS,
            )
            logger.info("created_es_index", index=self.index_name)

    def index_chunks(self, chunks: list[DocumentChunk]):
        """
        Bulk-index document chunks into Elasticsearch.

        Uses the bulk API for efficiency. Each chunk is stored with
        its full content and metadata so search results are self-
        contained without needing a join back to another store.
        """
        operations = []
        for chunk in chunks:
            # bulk format: action line, then document line
            operations.append(
                {"index": {"_index": self.index_name, "_id": chunk.chunk_id}}
            )
            operations.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "metadata": chunk.metadata,
                }
            )

        if operations:
            self.es.bulk(body=operations, refresh=True)
            logger.info("indexed_sparse", count=len(chunks))

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Run a BM25 query against the index.

        The query uses a multi_match across the content field. We
        also boost exact phrase matches (with slop 2) so documents
        containing the query as a near-exact phrase rank higher
        than those with scattered keyword hits.
        """
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["content"],
                    "type": "best_fields",
                }
            }
        ]

        # optional filter by source document
        filter_clauses = []
        if source_filter:
            filter_clauses.append({"term": {"source": source_filter}})

        # boost near-exact phrase matches
        should_clauses = [
            {
                "match_phrase": {
                    "content": {
                        "query": query,
                        "slop": 2,
                        "boost": 2.0,
                    }
                }
            }
        ]

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": should_clauses,
                    "filter": filter_clauses,
                }
            },
        }

        response = self.es.search(index=self.index_name, body=body)
        hits = response["hits"]["hits"]

        results = []
        for rank, hit in enumerate(hits, start=1):
            src = hit["_source"]
            chunk = DocumentChunk(
                chunk_id=src["chunk_id"],
                content=src["content"],
                source=src.get("source", ""),
                page_number=src.get("page_number"),
                metadata=src.get("metadata", {}),
            )

            # normalize ES scores to 0-1 range using sigmoid-style mapping
            raw_score = hit["_score"]
            normalized = raw_score / (1 + raw_score)

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=normalized,
                    retriever=RetrieverSource.SPARSE,
                    rank=rank,
                )
            )

        logger.info("sparse_search", query=query[:80], hits=len(results))
        return results

    def delete_index(self):
        """Remove the index entirely. Used in tests and re-indexing."""
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            logger.warning("deleted_es_index", index=self.index_name)
