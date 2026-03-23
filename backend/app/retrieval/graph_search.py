"""
Knowledge graph retrieval using Neo4j.

While dense and sparse search find individual chunks, the graph
retriever answers multi-hop questions by traversing relationships
between entities. If a user asks "What methods does the paper on
attention mechanisms use for evaluation?", the graph can follow:

    (Paper: Attention Is All You Need)
        --USES_METHOD--> (Method: BLEU Score)
        --USES_METHOD--> (Method: Perplexity)
        --EVALUATED_ON--> (Dataset: WMT 2014)

This structured traversal returns connected facts that would be
scattered across unrelated chunks in a flat vector store.
"""

from neo4j import GraphDatabase
from typing import Optional
import structlog

from app.config import get_settings
from app.models.schemas import DocumentChunk, RetrievedChunk, RetrieverSource

logger = structlog.get_logger()


# Schema for the knowledge graph.
# Nodes: Paper, Author, Method, Dataset, Concept
# Edges: AUTHORED, CITES, USES_METHOD, EVALUATED_ON, RELATES_TO
GRAPH_SCHEMA = """
// Constraints ensure no duplicate entities slip in during ingestion
CREATE CONSTRAINT paper_title IF NOT EXISTS
    FOR (p:Paper) REQUIRE p.title IS UNIQUE;
CREATE CONSTRAINT author_name IF NOT EXISTS
    FOR (a:Author) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT method_name IF NOT EXISTS
    FOR (m:Method) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT dataset_name IF NOT EXISTS
    FOR (d:Dataset) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT concept_name IF NOT EXISTS
    FOR (c:Concept) REQUIRE c.name IS UNIQUE;
"""


class GraphRetriever:
    """
    Wraps Neo4j for entity-relationship-based retrieval.

    Entities are extracted during document ingestion (by an LLM)
    and stored as graph nodes with typed edges. At query time,
    we identify entities in the question, find their graph
    neighborhood, and assemble the connected facts into synthetic
    document chunks that the rest of the pipeline can process
    identically to chunks from other retrievers.
    """

    def __init__(self):
        settings = get_settings()
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._ensure_schema()

    def _ensure_schema(self):
        """Apply uniqueness constraints on first run."""
        with self.driver.session() as session:
            for statement in GRAPH_SCHEMA.strip().split(";"):
                statement = statement.strip()
                if statement:
                    try:
                        session.run(statement)
                    except Exception:
                        pass  # constraint already exists

    def add_paper(
        self,
        title: str,
        authors: list[str],
        methods: list[str],
        datasets: list[str],
        concepts: list[str],
        abstract: str = "",
        source: str = "",
    ):
        """
        Insert a paper and all its relationships into the graph.

        Uses MERGE so re-ingesting the same paper is idempotent —
        nodes and edges are created only if they don't already exist.
        """
        with self.driver.session() as session:
            session.run(
                """
                MERGE (p:Paper {title: $title})
                SET p.abstract = $abstract,
                    p.source = $source
                """,
                title=title,
                abstract=abstract,
                source=source,
            )

            for author in authors:
                session.run(
                    """
                    MERGE (a:Author {name: $name})
                    MERGE (p:Paper {title: $title})
                    MERGE (a)-[:AUTHORED]->(p)
                    """,
                    name=author,
                    title=title,
                )

            for method in methods:
                session.run(
                    """
                    MERGE (m:Method {name: $name})
                    MERGE (p:Paper {title: $title})
                    MERGE (p)-[:USES_METHOD]->(m)
                    """,
                    name=method,
                    title=title,
                )

            for dataset in datasets:
                session.run(
                    """
                    MERGE (d:Dataset {name: $name})
                    MERGE (p:Paper {title: $title})
                    MERGE (p)-[:EVALUATED_ON]->(d)
                    """,
                    name=dataset,
                    title=title,
                )

            for concept in concepts:
                session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    MERGE (p:Paper {title: $title})
                    MERGE (p)-[:RELATES_TO]->(c)
                    """,
                    name=concept,
                    title=title,
                )

        logger.info("added_paper_to_graph", title=title)

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_names: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Find graph neighborhoods relevant to the query.

        Strategy:
        1. If entity_names provided (extracted from query by LLM),
           find those exact nodes and expand 2 hops outward
        2. Otherwise, do a fuzzy text search across all node names
           and expand from the best matches

        The connected subgraph is serialized into natural language
        "chunks" so downstream components don't need to know they
        came from a graph.
        """
        with self.driver.session() as session:
            if entity_names:
                results = self._search_by_entities(session, entity_names, top_k)
            else:
                results = self._search_by_text(session, query, top_k)

        logger.info("graph_search", query=query[:80], hits=len(results))
        return results

    def _search_by_entities(self, session, entities: list[str], top_k: int):
        """Find nodes matching entity names, expand their neighborhood."""
        records = session.run(
            """
            UNWIND $entities AS entity_name
            CALL {
                WITH entity_name
                // search across all node types
                OPTIONAL MATCH (n)
                WHERE (n:Paper OR n:Author OR n:Method OR n:Dataset OR n:Concept)
                  AND toLower(n.name) CONTAINS toLower(entity_name)
                   OR (n:Paper AND toLower(n.title) CONTAINS toLower(entity_name))
                RETURN n
                LIMIT 3
            }
            WITH n WHERE n IS NOT NULL
            // expand 2 hops to capture context
            MATCH path = (n)-[r*1..2]-(connected)
            RETURN n, collect(DISTINCT {
                node: connected,
                rel_type: type(r[0]),
                labels: labels(connected)
            }) AS neighborhood
            LIMIT $limit
            """,
            entities=entities,
            limit=top_k,
        )

        return self._records_to_chunks(records)

    def _search_by_text(self, session, query: str, top_k: int):
        """Fuzzy full-text search across all node names/titles."""
        # split query into individual terms for broader matching
        terms = query.lower().split()

        records = session.run(
            """
            UNWIND $terms AS term
            CALL {
                WITH term
                MATCH (n)
                WHERE (n:Paper AND toLower(n.title) CONTAINS term)
                   OR (n:Author AND toLower(n.name) CONTAINS term)
                   OR (n:Method AND toLower(n.name) CONTAINS term)
                   OR (n:Dataset AND toLower(n.name) CONTAINS term)
                   OR (n:Concept AND toLower(n.name) CONTAINS term)
                RETURN n, 1.0 AS match_score
                LIMIT 5
            }
            WITH n, sum(match_score) AS total_score
            ORDER BY total_score DESC
            LIMIT $limit
            MATCH path = (n)-[r*1..2]-(connected)
            RETURN n, total_score, collect(DISTINCT {
                node: properties(connected),
                rel_type: type(r[0]),
                labels: labels(connected)
            }) AS neighborhood
            """,
            terms=terms,
            limit=top_k,
        )

        return self._records_to_chunks(records)

    def _records_to_chunks(self, records) -> list[RetrievedChunk]:
        """
        Convert Neo4j records into RetrievedChunk objects.

        Serializes the graph neighborhood into readable sentences
        so the LLM can understand the relationships without needing
        to parse graph notation.
        """
        results = []
        for rank, record in enumerate(records, start=1):
            node = record["n"]
            neighborhood = record.get("neighborhood", [])

            # figure out what this node is
            node_props = dict(node.items()) if hasattr(node, "items") else node
            node_label = "Entity"
            node_name = node_props.get("title") or node_props.get("name", "Unknown")

            # build a natural language summary of the subgraph
            lines = [f"{node_label}: {node_name}"]

            if "abstract" in node_props and node_props["abstract"]:
                lines.append(f"Abstract: {node_props['abstract'][:300]}")

            for neighbor in neighborhood[:10]:  # cap to avoid huge chunks
                rel = neighbor.get("rel_type", "RELATED_TO")
                n_props = neighbor.get("node", {})
                n_name = n_props.get("title") or n_props.get("name", "unknown")
                n_labels = neighbor.get("labels", [])
                label_str = n_labels[0] if n_labels else "Entity"
                lines.append(f"  → {rel} → {label_str}: {n_name}")

            content = "\n".join(lines)
            chunk_id = f"graph-{hash(content) % 10**8:08d}"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=content,
                source=node_props.get("source", "knowledge_graph"),
                metadata={"type": "graph_traversal"},
            )

            # graph results get a flat relevance score based on rank
            score = 1.0 / rank

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    retriever=RetrieverSource.GRAPH,
                    rank=rank,
                )
            )

        return results

    def close(self):
        self.driver.close()
