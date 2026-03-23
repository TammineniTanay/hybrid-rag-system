"""
Corrective RAG (CRAG) pipeline built with LangGraph.

After the hybrid retriever returns chunks, CRAG checks whether
those chunks are actually relevant to the question. If they're not,
it takes corrective action: rewriting the query, falling back to
web search, or decomposing into sub-questions.

This is modeled as a state machine (LangGraph graph) with nodes:
    grade_chunks → decide_action → [rewrite | web_search | decompose] → generate

The conditional edges between nodes make the pipeline adaptive —
it does extra work only when needed, keeping latency low for
straightforward queries.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import structlog

from app.config import get_settings
from app.core.hybrid_retriever import HybridRetriever
from app.retrieval.web_search import WebSearchRetriever
from app.models.schemas import (
    FusedResult,
    RelevanceGrade,
    CRAGAction,
    CRAGDecision,
)

logger = structlog.get_logger()


# ── State definition ──────────────────────────────────────────
# LangGraph passes this dict through every node. Each node reads
# what it needs and writes its outputs back to the same dict.

class CRAGState(TypedDict):
    question: str
    retrieved_chunks: list[FusedResult]
    chunk_grades: list[dict]        # [{chunk_id, grade, reasoning}]
    crag_decision: dict             # CRAGDecision as dict
    final_chunks: list[FusedResult] # chunks that will reach the LLM
    answer: str
    retry_count: int


# ── Prompt templates ──────────────────────────────────────────

GRADING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a retrieval quality grader. Given a user question and
a retrieved document chunk, determine if the chunk contains information
relevant to answering the question.

Respond with a JSON object:
{{"grade": "relevant" | "irrelevant" | "ambiguous", "reasoning": "brief explanation"}}

Only mark as "relevant" if the chunk directly helps answer the question.
Mark "ambiguous" if the chunk is tangentially related but not sufficient.
Mark "irrelevant" if the chunk has nothing to do with the question."""),
    ("human", """Question: {question}

Document chunk:
{chunk_content}

Grade this chunk:"""),
])


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting specialist. The original query failed
to retrieve good results. Rewrite it to be more specific, use different
terminology, or approach the question from a different angle.

Return ONLY the rewritten query, nothing else."""),
    ("human", """Original question: {question}

The retrieved documents were mostly irrelevant. Here's what was retrieved:
{chunk_summaries}

Rewrite the query to find better results:"""),
])


DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a question decomposer. Break the complex question into
2-3 simpler sub-questions that, when answered together, would answer
the original question.

Return a JSON list of strings: ["sub-question 1", "sub-question 2", ...]"""),
    ("human", """Complex question: {question}

Break this into simpler sub-questions:"""),
])


GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise research assistant. Answer the question using
ONLY the provided context. If the context doesn't contain enough
information, say so honestly. Cite specific details from the context
to support your answer.

Do not make up information. Do not use prior knowledge."""),
    ("human", """Context:
{context}

Question: {question}

Answer:"""),
])


class CRAGPipeline:
    """
    Self-correcting retrieval pipeline.

    The LangGraph state machine runs through these stages:
    1. Grade each retrieved chunk for relevance
    2. Decide whether to proceed, rewrite, web-search, or decompose
    3. If correcting: execute the correction and re-grade
    4. Generate the final answer from approved chunks

    Maximum retry count prevents infinite correction loops.
    """

    def __init__(self, hybrid_retriever: HybridRetriever):
        settings = get_settings()
        self.retriever = hybrid_retriever
        self.web_search = WebSearchRetriever()

        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0,
        )

        self.settings = settings
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Construct the CRAG state machine.

        Graph topology:
            grade_chunks
                ↓
            decide_action
               ↙    ↓    ↘
          rewrite  web   decompose
               ↘    ↓    ↙
            grade_chunks (retry)  ← or → generate
                                           ↓
                                          END
        """
        workflow = StateGraph(CRAGState)

        # add nodes
        workflow.add_node("grade_chunks", self._grade_chunks)
        workflow.add_node("decide_action", self._decide_action)
        workflow.add_node("rewrite_and_retrieve", self._rewrite_and_retrieve)
        workflow.add_node("web_search_fallback", self._web_search_fallback)
        workflow.add_node("decompose_and_retrieve", self._decompose_and_retrieve)
        workflow.add_node("generate_answer", self._generate_answer)

        # set entry point
        workflow.set_entry_point("grade_chunks")

        # edges from grading → decision
        workflow.add_edge("grade_chunks", "decide_action")

        # conditional edges from decision
        workflow.add_conditional_edges(
            "decide_action",
            self._route_decision,
            {
                "proceed": "generate_answer",
                "rewrite": "rewrite_and_retrieve",
                "web_search": "web_search_fallback",
                "decompose": "decompose_and_retrieve",
            },
        )

        # correction paths loop back to grading (or straight to generate if max retries)
        workflow.add_edge("rewrite_and_retrieve", "grade_chunks")
        workflow.add_edge("web_search_fallback", "generate_answer")  # web results skip re-grading
        workflow.add_edge("decompose_and_retrieve", "grade_chunks")

        # generation → end
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    # ── Node implementations ──────────────────────────────────

    def _grade_chunks(self, state: CRAGState) -> dict:
        """Score each chunk for relevance to the question."""
        grades = []
        chain = GRADING_PROMPT | self.llm | StrOutputParser()

        for fused in state["retrieved_chunks"]:
            try:
                result = chain.invoke({
                    "question": state["question"],
                    "chunk_content": fused.chunk.content[:1000],
                })
                parsed = json.loads(result)
                grades.append({
                    "chunk_id": fused.chunk.chunk_id,
                    "grade": parsed.get("grade", "ambiguous"),
                    "reasoning": parsed.get("reasoning", ""),
                })
            except Exception as exc:
                logger.warning("grading_failed", chunk_id=fused.chunk.chunk_id, error=str(exc))
                grades.append({
                    "chunk_id": fused.chunk.chunk_id,
                    "grade": "ambiguous",
                    "reasoning": f"Grading failed: {exc}",
                })

        logger.info(
            "graded_chunks",
            total=len(grades),
            relevant=sum(1 for g in grades if g["grade"] == "relevant"),
            irrelevant=sum(1 for g in grades if g["grade"] == "irrelevant"),
        )

        return {"chunk_grades": grades}

    def _decide_action(self, state: CRAGState) -> dict:
        """
        Based on grade distribution, decide the corrective action.

        Heuristic:
        - >60% relevant → proceed to answer generation
        - >60% irrelevant → rewrite query or web search
        - mixed → decompose into sub-questions
        - if already retried max times → proceed anyway (best effort)
        """
        grades = state["chunk_grades"]
        retry_count = state.get("retry_count", 0)
        total = len(grades) if grades else 1

        relevant_count = sum(1 for g in grades if g["grade"] == "relevant")
        irrelevant_count = sum(1 for g in grades if g["grade"] == "irrelevant")

        relevant_ratio = relevant_count / total
        irrelevant_ratio = irrelevant_count / total

        # cap retries to prevent infinite loops
        if retry_count >= self.settings.max_crag_retries:
            action = CRAGAction.PROCEED
            confidence = relevant_ratio
        elif relevant_ratio >= self.settings.crag_relevant_ratio:
            action = CRAGAction.PROCEED
            confidence = relevant_ratio
        elif irrelevant_ratio >= self.settings.crag_irrelevant_ratio:
            # alternate between rewrite and web search on retries
            if retry_count == 0:
                action = CRAGAction.REWRITE
            else:
                action = CRAGAction.WEB_SEARCH
            confidence = 1.0 - relevant_ratio
        else:
            action = CRAGAction.DECOMPOSE
            confidence = 0.5

        relevant_ids = [g["chunk_id"] for g in grades if g["grade"] == "relevant"]

        decision = {
            "action": action.value,
            "confidence": confidence,
            "relevant_chunks": relevant_ids,
        }

        logger.info(
            "crag_decision",
            action=action.value,
            confidence=round(confidence, 2),
            relevant=relevant_count,
            total=total,
            retry=retry_count,
        )

        # filter chunks to only keep relevant ones for generation
        final_chunks = [
            f for f in state["retrieved_chunks"]
            if f.chunk.chunk_id in relevant_ids
        ]

        return {
            "crag_decision": decision,
            "final_chunks": final_chunks if final_chunks else state["retrieved_chunks"],
        }

    def _route_decision(self, state: CRAGState) -> str:
        """Conditional edge: route to the correct correction node."""
        return state["crag_decision"]["action"]

    def _rewrite_and_retrieve(self, state: CRAGState) -> dict:
        """Rewrite the query and re-run hybrid retrieval."""
        chain = REWRITE_PROMPT | self.llm | StrOutputParser()

        chunk_summaries = "\n".join(
            f"- {f.chunk.content[:150]}..."
            for f in state["retrieved_chunks"][:3]
        )

        rewritten = chain.invoke({
            "question": state["question"],
            "chunk_summaries": chunk_summaries,
        })

        logger.info("rewritten_query", original=state["question"][:80], rewritten=rewritten[:80])

        # re-retrieve with the new query
        new_results = self.retriever.retrieve(rewritten, top_k=5)

        return {
            "retrieved_chunks": new_results,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def _web_search_fallback(self, state: CRAGState) -> dict:
        """Fall back to web search when internal docs aren't enough."""
        web_results = self.web_search.search(state["question"], top_k=3)

        # convert web RetrievedChunks to FusedResults
        web_fused = []
        for wr in web_results:
            web_fused.append(
                FusedResult(
                    chunk=wr.chunk,
                    rrf_score=wr.score,
                    contributing_retrievers=[wr.retriever],
                    individual_scores={wr.retriever.value: wr.score},
                )
            )

        # merge web results with whatever relevant chunks we had
        combined = state.get("final_chunks", []) + web_fused

        logger.info("web_fallback", web_hits=len(web_results), total=len(combined))

        return {"final_chunks": combined}

    def _decompose_and_retrieve(self, state: CRAGState) -> dict:
        """Break question into sub-questions, retrieve for each."""
        chain = DECOMPOSE_PROMPT | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"question": state["question"]})
            sub_questions = json.loads(result)
        except Exception:
            # if decomposition fails, just try a simpler version
            sub_questions = [state["question"]]

        logger.info("decomposed", sub_questions=sub_questions)

        # retrieve for each sub-question and merge
        all_results = []
        seen_ids = set()

        for sub_q in sub_questions[:3]:  # cap at 3 sub-questions
            results = self.retriever.retrieve(sub_q, top_k=3)
            for r in results:
                if r.chunk.chunk_id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.chunk.chunk_id)

        return {
            "retrieved_chunks": all_results,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def _generate_answer(self, state: CRAGState) -> dict:
        """Generate the final answer from approved chunks."""
        chunks = state.get("final_chunks", state["retrieved_chunks"])

        # assemble context from chunks
        context_parts = []
        for i, fused in enumerate(chunks, 1):
            source = fused.chunk.source or "unknown"
            context_parts.append(
                f"[Source {i}: {source}]\n{fused.chunk.content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        chain = GENERATION_PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": state["question"],
        })

        return {"answer": answer}

    # ── Public interface ──────────────────────────────────────

    def run(self, question: str, retrieved_chunks: list[FusedResult]) -> CRAGState:
        """
        Execute the full CRAG pipeline.

        Takes the question and initial retrieved chunks, runs them
        through grading → decision → correction → generation,
        and returns the final state with the answer.
        """
        initial_state: CRAGState = {
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "chunk_grades": [],
            "crag_decision": {},
            "final_chunks": [],
            "answer": "",
            "retry_count": 0,
        }

        final_state = self.graph.invoke(initial_state)
        return final_state
