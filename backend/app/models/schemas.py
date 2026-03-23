from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────

class RelevanceGrade(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    AMBIGUOUS = "ambiguous"


class CRAGAction(str, Enum):
    PROCEED = "proceed"
    REWRITE = "rewrite"
    WEB_SEARCH = "web_search"
    DECOMPOSE = "decompose"


class RetrieverSource(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    GRAPH = "graph"
    WEB = "web"


# ── Document Models ────────────────────────────────────────────

class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    source: str = ""
    page_number: Optional[int] = None


class RetrievedChunk(BaseModel):
    """A chunk returned by one of the retrievers, with scoring info."""
    chunk: DocumentChunk
    score: float
    retriever: RetrieverSource
    rank: int


class FusedResult(BaseModel):
    """A chunk after Reciprocal Rank Fusion across all retrievers."""
    chunk: DocumentChunk
    rrf_score: float
    contributing_retrievers: list[RetrieverSource]
    individual_scores: dict[str, float] = Field(default_factory=dict)


# ── CRAG Models ────────────────────────────────────────────────

class ChunkGrade(BaseModel):
    chunk_id: str
    grade: RelevanceGrade
    reasoning: str


class CRAGDecision(BaseModel):
    action: CRAGAction
    confidence: float
    relevant_chunks: list[str]  # chunk IDs that passed grading
    rewritten_query: Optional[str] = None
    sub_questions: Optional[list[str]] = None


# ── Query / Response Models ────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_crag: bool = True
    use_web_fallback: bool = True


class SourceReference(BaseModel):
    chunk_id: str
    content_preview: str  # first 200 chars
    source: str
    retriever: RetrieverSource
    relevance_score: float


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    sources: list[SourceReference]
    crag_action_taken: Optional[CRAGAction] = None
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Feedback Models ────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None


class FeedbackRecord(BaseModel):
    feedback_id: str
    query_id: str
    question: str
    answer: str
    retrieved_chunk_ids: list[str]
    rating: int
    comment: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Evaluation Models ─────────────────────────────────────────

class RAGASMetrics(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevancy: float = Field(ge=0, le=1)
    context_precision: float = Field(ge=0, le=1)
    context_recall: float = Field(ge=0, le=1)


class EvalSnapshot(BaseModel):
    eval_id: str
    query_id: str
    metrics: RAGASMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvalDashboardData(BaseModel):
    total_queries: int
    avg_faithfulness: float
    avg_relevancy: float
    avg_precision: float
    avg_recall: float
    avg_user_rating: float
    history: list[EvalSnapshot]
    crag_trigger_rate: float  # % of queries where CRAG corrected
    feedback_count: int


# ── Ingestion Models ──────────────────────────────────────────

class IngestionRequest(BaseModel):
    file_path: Optional[str] = None
    text_content: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class IngestionResult(BaseModel):
    document_id: str
    chunks_created: int
    entities_extracted: int
    status: str
