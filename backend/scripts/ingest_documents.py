"""
CLI script for batch document ingestion.

Usage:
    python scripts/ingest_documents.py --source data/sample_papers/
    python scripts/ingest_documents.py --source path/to/single_file.pdf
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.hybrid_retriever import HybridRetriever
from app.retrieval.graph_search import GraphRetriever
from app.services.ingestion import IngestionService


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("--source", required=True, help="Path to a file or directory to ingest")
    args = parser.parse_args()

    hybrid = HybridRetriever()
    graph = GraphRetriever()
    ingestion = IngestionService(hybrid, graph)

    source = args.source

    if os.path.isdir(source):
        print(f"Ingesting directory: {source}")
        results = ingestion.ingest_directory(source)
        total_chunks = sum(r["chunks_created"] for r in results)
        total_entities = sum(r["entities_extracted"] for r in results)
        print(f"\nDone. {len(results)} files, {total_chunks} chunks, {total_entities} entities.")
    elif os.path.isfile(source):
        print(f"Ingesting file: {source}")
        result = ingestion.ingest_file(source)
        print(f"Done. Chunks: {result['chunks_created']}, Entities: {result['entities_extracted']}")
    else:
        print(f"Error: {source} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
