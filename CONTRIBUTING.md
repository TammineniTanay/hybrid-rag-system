# Contributing to Hybrid RAG System

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `pytest tests/ -v`
6. Commit and push
7. Open a pull request

## Development Setup

```bash
git clone https://github.com/TammineniTanay/hybrid-rag-system.git
cd hybrid-rag-system
cp .env.example .env
# Fill in your API keys
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Add docstrings to all functions
- Keep functions focused and small