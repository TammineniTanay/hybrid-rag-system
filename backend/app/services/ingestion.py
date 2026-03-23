"""
Document ingestion pipeline.

Takes raw documents (PDFs, text files), splits them into chunks,
extracts entities for the knowledge graph, and indexes everything
into the three retrieval stores (Qdrant, Elasticsearch, Neo4j).

Chunking: recursive character splitting with 512-token chunks and
64-token overlap — balances precision with enough context per chunk.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader
import json
import uuid
import os
import structlog

from app.config import get_settings
from app.models.schemas import DocumentChunk
from app.core.hybrid_retriever import HybridRetriever
from app.retrieval.graph_search import GraphRetriever

logger = structlog.get_logger()

ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Extract entities from this academic text. Return JSON with keys:
- authors: list of author names
- methods: algorithms, techniques, frameworks
- datasets: named datasets
- concepts: key technical topics

Return ONLY valid JSON."""),
    ("human", "{text}"),
])


class IngestionService:
    def __init__(self, hybrid_retriever: HybridRetriever, graph_retriever: GraphRetriever):
        settings = get_settings()
        self.retriever = hybrid_retriever
        self.graph = graph_retriever
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0,
        )
        self.entity_chain = ENTITY_EXTRACTION_PROMPT | self.llm | StrOutputParser()

    def ingest_file(self, file_path: str, metadata: dict = None) -> dict:
        metadata = metadata or {}
        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)

        if file_path.endswith(".pdf"):
            text = self._read_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        if not text.strip():
            return {"document_id": doc_id, "chunks_created": 0, "entities_extracted": 0, "status": "empty"}

        chunks = self._create_chunks(text, doc_id, file_name, metadata)
        self.retriever.index_chunks(chunks)
        entities_count = self._extract_and_index_entities(text, file_name, doc_id)

        return {"document_id": doc_id, "chunks_created": len(chunks), "entities_extracted": entities_count, "status": "success"}

    def ingest_directory(self, dir_path: str) -> list[dict]:
        results = []
        for fname in sorted(os.listdir(dir_path)):
            if fname.lower().endswith((".pdf", ".txt", ".md", ".tex")):
                results.append(self.ingest_file(os.path.join(dir_path, fname)))
        return results

    def _create_chunks(self, text, doc_id, source, metadata):
        raw = self.splitter.split_text(text)
        return [
            DocumentChunk(
                chunk_id=f"{doc_id}-chunk-{i:04d}",
                content=content,
                source=source,
                page_number=i // 3,
                metadata={**metadata, "document_id": doc_id, "chunk_index": i},
            )
            for i, content in enumerate(raw)
        ]

    def _read_pdf(self, path):
        reader = PdfReader(path)
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)

    def _extract_and_index_entities(self, text, source_name, doc_id):
        try:
            result = self.entity_chain.invoke({"text": text[:3000]})
            entities = json.loads(result)
            title = source_name.replace(".pdf", "").replace("_", " ")
            self.graph.add_paper(
                title=title,
                authors=entities.get("authors", []),
                methods=entities.get("methods", []),
                datasets=entities.get("datasets", []),
                concepts=entities.get("concepts", []),
                abstract=text[:500],
                source=source_name,
            )
            return sum(len(entities.get(k, [])) for k in ["authors", "methods", "datasets", "concepts"])
        except Exception as exc:
            logger.warning("entity_extraction_failed", error=str(exc))
            return 0
