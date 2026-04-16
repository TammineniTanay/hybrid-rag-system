# Hybrid RAG System with Self-Correcting Retrieval & Feedback Loop

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.26-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Zenodo_DOI-blue)](https://doi.org/10.5281/zenodo.19582347)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Publication-00CCBB)](https://www.researchgate.net/publication/403818727)

> 📄 **Published Research:** This repository is part of the **UniLLMOps** framework described in:
>
> *UniLLMOps: A Unified Framework for End-to-End Large Language Model Production Systems — From Distributed Fine-Tuning to Hybrid Retrieval-Augmented Inference*
>
> Tanay Tammineni, 2026 — [Paper (DOI)](https://doi.org/10.5281/zenodo.19582347) | [ResearchGate](https://www.researchgate.net/publication/403818727)

### 50-Query Controlled Validation Results

| Metric | Value |
|--------|-------|
| CRAG pass-through rate | 62% (31/50 queries) |
| CRAG correction rate | 38% (19/50 queries) |
| Mean retrieval latency | 163.5 ms |
| Mean total latency (proceed) | 55.9 s |
| Mean total latency (corrected) | 86.9 s |
| Dense retriever contribution | 42.7% |
| Sparse (BM25) contribution | 44.4% |
| Graph (Neo4j) contribution | 12.9% |
| Mean sources per query | 3.4 |
| Hybrid vs dense-only coverage | 100% vs 68% |

**LLM:** Ollama Llama 3.2 3B (local, no API costs) | **Embedding:** all-MiniLM-L6-v2 (384-dim) | **Corpus:** 5 papers, 783 chunks, 57 graph entities

Full per-query data: [`eval_results_50.json`](eval_results_50.json)

---

A production-grade Retrieval Augmented Generation system that goes beyond basic "chat with your PDFs." It combines three search strategies (dense embeddings, BM25 sparse search, knowledge graph traversal), implements Corrective RAG (CRAG) for self-healing retrieval, and continuously learns from user feedback through a reward-based re-ranking loop — with a real-time evaluation dashboard tracking faithfulness, relevance, and precision metrics.

> **Why this project?** Most RAG demos retrieve chunks and hope for the best. This system detects when retrieval fails, autonomously corrects itself, and improves over time from user feedback — the same challenges production teams at Anthropic, Cohere, and enterprise AI companies deal with daily.

---

## Demo

<!-- Replace these with actual screenshots after running the system -->
| Chat Interface | Evaluation Dashboard |
|:-:|:-:|
| ![Chat](docs/screenshots/chat_interface.png) | ![Dashboard](docs/screenshots/eval_dashboard.png) |

*Screenshots: Left — Query interface with retriever source badges and CRAG action indicators. Right — Real-time RAGAS metrics with line charts, radar plot, and retriever utilization.*

---

## Architecture Overview

```
                         ┌──────────────────┐
                         │   User Question   │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼               ▼
             ┌──────────┐  ┌──────────┐   ┌──────────┐
             │  Qdrant   │  │ Elastic- │   │  Neo4j   │
             │  Dense    │  │ search   │   │  Graph   │
             │ Embeddings│  │  BM25    │   │ Traversal│
             └─────┬─────┘  └────┬─────┘   └────┬─────┘
                   │             │               │
                   └─────────────┼───────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  Reciprocal Rank Fusion │
                    │    (RRF, k=60)         │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │  Reward Model Re-rank   │
                    │  (if trained)           │
                    └───────────┬────────────┘
                                ▼
               ┌────────────────────────────────┐
               │     CRAG: Corrective RAG        │
               │  ┌──────────────────────────┐   │
               │  │ Grade each chunk:         │   │
               │  │ RELEVANT / IRRELEVANT     │   │
               │  └─────────┬────────────────┘   │
               │            ▼                     │
               │  ┌──────────────────────────┐   │
               │  │ Decision Engine:          │   │
               │  │ • Proceed (>60% relevant) │   │
               │  │ • Rewrite query           │   │
               │  │ • Web search fallback     │   │
               │  │ • Decompose into sub-Qs   │   │
               │  └──────────────────────────┘   │
               └────────────┬───────────────────┘
                            ▼
               ┌────────────────────────────┐
               │   LLM Answer Generation     │
               │   (Claude / GPT-4)          │
               └────────────┬───────────────┘
                            ▼
                  ┌─────────┴──────────┐
                  ▼                    ▼
         ┌──────────────┐    ┌──────────────┐
         │ User Feedback │    │  RAGAS Eval   │
         │  (1-5 stars)  │    │  (4 metrics)  │
         └───────┬──────┘    └───────┬──────┘
                 │                   │
                 ▼                   ▼
         ┌──────────────────────────────┐
         │       PostgreSQL              │
         │  feedback + eval snapshots    │
         └──────────────┬───────────────┘
                        ▼
         ┌──────────────────────────────┐
         │    React Eval Dashboard       │
         │  Charts • Metrics • Trends    │
         └──────────────────────────────┘
```

---

## Key Features

### Hybrid Retrieval with Reciprocal Rank Fusion
Three retrievers cover each other's blind spots — dense search catches semantic meaning, BM25 nails exact keyword matches, and the knowledge graph answers multi-hop relationship questions. Results are fused using RRF: `score(d) = Σ 1/(k + rank_i(d))`.

### Corrective RAG (CRAG) via LangGraph
A state machine that detects bad retrieval before it reaches the LLM. Each chunk is graded for relevance. If >60% of chunks are irrelevant, the system autonomously rewrites the query, falls back to web search, or decomposes into simpler sub-questions — then retries. Max 2 correction loops to prevent runaway latency.

### Feedback-Driven Reward Model
User ratings train a Gradient Boosted Classifier that learns which (query, chunk) pairs lead to good answers. Features include embedding similarity, RRF score, retriever count, chunk length, and lexical overlap. The model blends its predictions with original scores (70/30 split) to re-rank future retrievals.

### Real-Time RAGAS Evaluation Dashboard
Every response is scored on four axes — Faithfulness, Answer Relevancy, Context Precision, Context Recall — displayed in a React dashboard with time-series charts, radar plots, and retriever utilization metrics.

---

## Tech Stack

| Layer           | Technology                                      | Why                                                    |
|-----------------|------------------------------------------------|--------------------------------------------------------|
| Orchestration   | LangGraph, LangChain                           | Stateful DAG with conditional edges for CRAG           |
| Vector Store    | Qdrant                                          | Fast cosine similarity, payload filtering              |
| Keyword Search  | Elasticsearch (BM25)                            | Custom analyzer for technical terms, camelCase split   |
| Knowledge Graph | Neo4j                                           | Cypher traversal for multi-hop entity relationships    |
| LLM             | Claude (Anthropic) / GPT-4                     | Configurable via settings, temperature-controlled      |
| Embeddings      | sentence-transformers (all-MiniLM-L6-v2)       | 384-dim, fast inference, good quality for retrieval    |
| Evaluation      | RAGAS                                           | Industry-standard RAG evaluation framework             |
| Tracing         | LangSmith                                       | Full observability on every chain step                 |
| Backend         | FastAPI, Python 3.11                            | Async support, auto-generated OpenAPI docs             |
| Frontend        | React 18, Recharts, TailwindCSS                | Interactive charts, responsive dashboard               |
| Database        | PostgreSQL (prod) / SQLite (dev fallback)       | Feedback storage, eval snapshots, query logs           |
| Deployment      | Docker Compose                                  | One command to start all 4 infrastructure services     |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- API keys: Anthropic (or OpenAI), Tavily (for web search fallback)

### Setup

```bash
# 1. Clone
git clone https://github.com/TammineniTanay/hybrid-rag-system.git
cd hybrid-rag-system

# 2. Environment variables
cp .env.example .env
# Edit .env → fill in ANTHROPIC_API_KEY, TAVILY_API_KEY, etc.

# 3. Start infrastructure (Qdrant, Elasticsearch, Neo4j, PostgreSQL)
docker-compose up -d

# 4. Install backend dependencies
cd backend
pip install -r requirements.txt

# 5. Ingest sample documents
python scripts/ingest_documents.py --source ../data/sample_papers/

# 6. Start the backend
uvicorn app.main:app --reload --port 8000

# 7. Start the frontend (new terminal)
cd ../frontend
npm install
npm run dev
```

Open `http://localhost:3000` → start querying your documents.

API docs available at `http://localhost:8000/docs` (Swagger UI).

---

## Project Structure

```
hybrid-rag-system/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── config.py                # Pydantic settings from .env
│   │   ├── api/
│   │   │   └── routes.py            # REST endpoints (query, feedback, eval, ingest)
│   │   ├── core/
│   │   │   ├── hybrid_retriever.py  # 3-way retrieval + RRF fusion
│   │   │   ├── crag_pipeline.py     # LangGraph state machine for CRAG
│   │   │   └── generator.py         # LLM answer generation with source citing
│   │   ├── retrieval/
│   │   │   ├── dense_search.py      # Qdrant vector similarity search
│   │   │   ├── sparse_search.py     # Elasticsearch BM25 keyword search
│   │   │   ├── graph_search.py      # Neo4j Cypher entity traversal
│   │   │   └── web_search.py        # Tavily web search (CRAG fallback)
│   │   ├── evaluation/
│   │   │   ├── ragas_evaluator.py   # RAGAS metrics + heuristic fallback
│   │   │   └── reward_model.py      # GBM re-ranker trained on user feedback
│   │   ├── models/
│   │   │   └── schemas.py           # Pydantic models (22 data classes)
│   │   └── services/
│   │       ├── database.py          # SQLAlchemy ORM (PostgreSQL/SQLite)
│   │       ├── feedback.py          # Feedback processing + eval trigger
│   │       └── ingestion.py         # Document chunking + entity extraction
│   ├── scripts/
│   │   ├── ingest_documents.py      # CLI: batch document ingestion
│   │   └── train_reward_model.py    # CLI: retrain reward model from feedback
│   ├── tests/
│   │   ├── test_retrieval.py        # RRF fusion + hybrid retriever tests
│   │   ├── test_crag.py             # CRAG decision logic tests
│   │   └── test_evaluation.py       # Heuristic eval + reward model tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Main app with tab navigation
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx    # Chat UI with retriever badges
│   │   │   ├── SourcePanel.jsx      # Retrieved sources with scores
│   │   │   ├── FeedbackWidget.jsx   # Star rating + comment input
│   │   │   └── EvalDashboard.jsx    # Charts: line, radar, bar
│   │   ├── hooks/useRAG.js          # Custom hook for query state
│   │   └── utils/api.js             # Backend API client
│   ├── package.json
│   └── Dockerfile
├── docs/
│   └── ARCHITECTURE.md              # Detailed system design documentation
├── configs/                          # Retrieval tuning configs
├── data/sample_papers/               # Place documents here for ingestion
├── eval_results_50.json              # 50-query validation results (paper Section X)
├── docker-compose.yml                # Qdrant + Elasticsearch + Neo4j + PostgreSQL
├── .env.example                      # Required environment variables
└── README.md
```

---

## API Endpoints

| Method | Endpoint              | Description                                    |
|--------|-----------------------|------------------------------------------------|
| POST   | `/api/query`          | Submit a question → hybrid retrieval → CRAG → answer |
| POST   | `/api/feedback`       | Submit user rating (1-5) for a previous response |
| GET    | `/api/eval/metrics`   | Aggregate RAGAS metrics for dashboard          |
| GET    | `/api/eval/history`   | Historical eval snapshots for time-series charts |
| POST   | `/api/ingest`         | Upload and index a document (PDF/TXT)          |
| POST   | `/api/ingest/text`    | Ingest raw text content directly               |
| POST   | `/api/reward/train`   | Trigger reward model training from feedback    |
| GET    | `/api/health`         | Service health check                           |

---

## How It Works

### 1. Hybrid Retrieval with Reciprocal Rank Fusion

Three retrievers run in parallel for every query:

- **Dense (Qdrant)**: Embeds query with `all-MiniLM-L6-v2`, finds nearest vectors by cosine similarity. Catches semantic meaning — "terminate an employee" matches "letting go of staff."
- **Sparse (Elasticsearch BM25)**: Keyword matching with a custom analyzer that splits camelCase and handles technical terms. Catches exact matches — "CUDA error 11" returns the exact doc.
- **Graph (Neo4j)**: Traverses entity relationships 2 hops deep. Catches connected facts — "What methods does the attention paper use?" follows `Paper → USES_METHOD → Method` edges.

Results are merged using Reciprocal Rank Fusion:
```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))    where k = 60
```

Documents found by multiple retrievers get boosted. A paper ranked #1 in dense and #3 in sparse scores higher than one ranked #1 in dense alone.

### 2. Corrective RAG (CRAG)

Implemented as a LangGraph state machine with 6 nodes and conditional edges:

```
grade_chunks → decide_action → [rewrite | web_search | decompose] → generate
```

The decision heuristic:
- **>60% chunks relevant** → proceed to answer generation
- **>60% chunks irrelevant** → rewrite query (attempt 1) or web search (attempt 2)
- **Mixed** → decompose into 2-3 sub-questions, retrieve for each, merge results
- **Max 2 retries** → proceed with best available context (prevent infinite loops)

### 3. Feedback-Driven Reward Model

User ratings are stored as `(query, chunks, answer, rating)` tuples. A Gradient Boosted Classifier trains on 6 features:

| Feature | Description |
|---------|-------------|
| Cosine similarity | Query-chunk embedding distance |
| RRF score | Original fusion score |
| Retriever count | How many retrievers found this chunk (1-3) |
| Chunk length | Word count of the chunk |
| Query length | Word count of the query |
| Lexical overlap | Fraction of query words in chunk |

Predictions blend with original scores: `final = 0.7 × RRF + 0.3 × reward`. Minimum 20 feedback records required to train.

### 4. RAGAS Evaluation

Every response is scored on four metrics:

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Is the answer supported by retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | What fraction of retrieved chunks were useful? |
| **Context Recall** | Did we retrieve all relevant information? |

Falls back to heuristic scoring (word overlap approximations) when RAGAS LLM quota is exhausted.

---

## Design Decisions

| Decision | Reasoning |
|----------|-----------|
| RRF over learned fusion | RRF is parameter-free and works out of the box. Learned fusion (cross-encoder) adds latency and needs training data we don't have at cold start. |
| LangGraph over hardcoded if/else | CRAG has conditional branching and retry loops. A state machine makes the flow explicit, testable, and extensible. |
| GBM over neural reward model | With <1000 feedback samples, gradient boosting generalizes better than a neural net and trains in seconds. |
| Heuristic eval fallback | RAGAS calls an LLM per metric. When quota is exhausted, word-overlap heuristics keep the dashboard functional. |
| SQLite dev fallback | Run the full pipeline locally without Docker by falling back to SQLite for feedback storage. |
| 70/30 reward blend | Aggressive weighting risks overfitting to early feedback. Conservative blend preserves retrieval quality while gradually learning. |

---

## Running Tests

```bash
cd backend
pip install pytest
pytest tests/ -v
```

Tests use mocked backends (no Qdrant/Elasticsearch/Neo4j required):
- `test_retrieval.py` — RRF fusion correctness, hybrid retriever flow
- `test_crag.py` — CRAG decision logic for all grade distributions
- `test_evaluation.py` — Heuristic eval bounds, reward model feature extraction

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{tammineni2026unillmops,
  title={UniLLMOps: A Unified Framework for End-to-End Large Language Model Production Systems -- From Distributed Fine-Tuning to Hybrid Retrieval-Augmented Inference},
  author={Tammineni, Tanay},
  year={2026},
  doi={10.5281/zenodo.19582347},
  url={https://doi.org/10.5281/zenodo.19582347}
}
```

## Related Repository

- [Distributed Fine-Tuning Pipeline](https://github.com/TammineniTanay/distributed-finetune-pipeline) — QLoRA + DeepSpeed ZeRO-3 + model merging + DPO alignment

---

## Future Roadmap

- [ ] Streaming responses via WebSocket
- [ ] Cross-encoder re-ranking before RRF
- [ ] Multi-modal ingestion (images, tables)
- [ ] A/B testing between retrieval strategies
- [ ] Kubernetes deployment with Helm charts
- [ ] Async parallel retrieval with `asyncio.gather`
- [ ] Conversation memory for multi-turn queries

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/cross-encoder-reranking`)
3. Commit changes (`git commit -m "Add cross-encoder re-ranking step"`)
4. Push to the branch (`git push origin feature/cross-encoder-reranking`)
5. Open a Pull Request

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

Built by [Tanay Tammineni](https://tanaytammineni.vercel.app) | [GitHub](https://github.com/TammineniTanay) | [LinkedIn](https://linkedin.com/in/tanay-tammineni)