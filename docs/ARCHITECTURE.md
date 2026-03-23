# Architecture Documentation

## System Design

### Hybrid Retrieval Pipeline

The system runs three retrieval strategies in parallel for every query:

1. **Dense Search (Qdrant)** — Converts queries and documents into 384-dimensional vectors using `all-MiniLM-L6-v2`. Finds semantically similar content even when surface-level words differ. Best for natural language questions.

2. **Sparse Search (Elasticsearch BM25)** — Traditional keyword matching with TF-IDF scoring. Uses a custom analyzer that splits camelCase and handles technical terms. Best for exact matches: error codes, function names, acronyms.

3. **Graph Search (Neo4j)** — Traverses entity relationships extracted during ingestion. Nodes represent papers, authors, methods, datasets, and concepts. Edges capture relationships like AUTHORED, USES_METHOD, EVALUATED_ON. Best for multi-hop questions that require connecting facts across documents.

### Reciprocal Rank Fusion (RRF)

Results from all three retrievers are merged using RRF:

```
RRF_score(doc) = Σ 1/(k + rank_i(doc))
```

Where `k=60` is a dampening constant and `rank_i` is the document's position in retriever `i`. Documents found by multiple retrievers get higher combined scores.

### Corrective RAG (CRAG)

After retrieval, a grader LLM evaluates each chunk's relevance. Based on the grade distribution:

- **>60% relevant** → Proceed to answer generation
- **>60% irrelevant** → Rewrite query and re-retrieve, or fall back to web search
- **Mixed** → Decompose into sub-questions and retrieve for each

This is implemented as a LangGraph state machine with conditional edges.

### Feedback-Driven Reward Model

User ratings (1-5) are stored alongside the full retrieval context. A Gradient Boosted Classifier learns to predict chunk quality from features:
- Query-chunk embedding similarity
- RRF score
- Number of contributing retrievers
- Chunk and query lengths
- Lexical overlap ratio

The trained model re-ranks future retrieval results (70% original score + 30% predicted reward).

### RAGAS Evaluation

Every query-response pair is evaluated on four metrics:
- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — What fraction of retrieved chunks were useful?
- **Context Recall** — Did we retrieve all relevant information?

Results are displayed in a real-time dashboard with line charts, radar plots, and retriever utilization metrics.

## Data Flow

```
User Question
    │
    ├─→ Qdrant (dense embeddings)──────────┐
    ├─→ Elasticsearch (BM25 keywords)──────┤
    └─→ Neo4j (graph traversal)────────────┘
                                            │
                            Reciprocal Rank Fusion
                                            │
                              Reward Model Re-ranking
                                            │
                                CRAG Grading Pipeline
                               ╱       │        ╲
                          Rewrite   Web Search  Decompose
                               ╲       │        ╱
                                 Final Context
                                       │
                              LLM Answer Generation
                                       │
                            ┌──────────┴──────────┐
                            │                     │
                      User Feedback          RAGAS Eval
                            │                     │
                      PostgreSQL ←─────────────→ Dashboard
```
