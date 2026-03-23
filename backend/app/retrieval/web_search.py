"""
Web search fallback for Corrective RAG.

When the internal retrievers fail to find relevant context (detected
by the CRAG grader), this module queries the open web via Tavily to
fill the knowledge gap. Common scenario: user asks about something
recent that isn't in the indexed corpus yet.
"""

from tavily import TavilyClient
from typing import Optional
import structlog

from app.config import get_settings
from app.models.schemas import DocumentChunk, RetrievedChunk, RetrieverSource

logger = structlog.get_logger()


class WebSearchRetriever:
    """
    Thin wrapper around Tavily's search API.

    Tavily is purpose-built for LLM retrieval — it returns clean
    extracted content rather than raw HTML, saving us a parsing step.
    Results are converted into DocumentChunk objects so the rest of
    the pipeline handles them identically to internal chunks.
    """

    def __init__(self):
        settings = get_settings()
        self.client = TavilyClient(api_key=settings.tavily_api_key)

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_depth: str = "advanced",
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Search the web and return results as RetrievedChunks.

        search_depth="advanced" tells Tavily to do a deeper crawl
        with better content extraction at the cost of slightly
        higher latency (~2s vs ~1s for "basic").

        Domain filters let us scope results — e.g. only arxiv.org
        for academic questions, or exclude social media for factual
        queries.
        """
        try:
            response = self.client.search(
                query=query,
                search_depth=search_depth,
                max_results=top_k,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )
        except Exception as exc:
            logger.error("web_search_failed", error=str(exc))
            return []

        results = []
        for rank, item in enumerate(response.get("results", []), start=1):
            content = item.get("content", "")
            url = item.get("url", "")
            title = item.get("title", "")

            # prefix the content with title and source for context
            full_content = f"[Web: {title}]\nSource: {url}\n\n{content}"

            chunk = DocumentChunk(
                chunk_id=f"web-{hash(url) % 10**8:08d}",
                content=full_content,
                source=url,
                metadata={
                    "type": "web_search",
                    "title": title,
                    "url": url,
                    "score": item.get("score", 0),
                },
            )

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=item.get("score", 0.5),
                    retriever=RetrieverSource.WEB,
                    rank=rank,
                )
            )

        logger.info("web_search", query=query[:80], hits=len(results))
        return results
