"""
Answer generation module.

Handles the final step: taking the approved context chunks and
the user's question, sending them to the LLM, and getting back
a grounded answer. Supports both Claude and OpenAI as backends.
"""

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import structlog

from app.config import get_settings
from app.models.schemas import FusedResult

logger = structlog.get_logger()


ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise research assistant. Your job is to answer
the user's question using the provided context documents.

Rules:
1. Use ONLY information from the provided context
2. If the context doesn't fully answer the question, say what you can
   and clearly state what information is missing
3. Cite which source(s) you drew each claim from using [Source N] notation
4. Be specific — prefer concrete details over vague summaries
5. If sources contradict each other, acknowledge the disagreement"""),
    ("human", """Context documents:

{context}

---

Question: {question}

Provide a thorough answer:"""),
])


class AnswerGenerator:
    """
    Generates grounded answers from retrieved context.

    Wraps LangChain's chat model abstraction so swapping between
    Claude and OpenAI is a config change, not a code change.
    """

    def __init__(self):
        settings = get_settings()

        if settings.default_llm == "claude":
            self.llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                anthropic_api_key=settings.anthropic_api_key,
                temperature=0.1,
                max_tokens=2048,
            )
        else:
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                openai_api_key=settings.openai_api_key,
                temperature=0.1,
                max_tokens=2048,
            )

        self.chain = ANSWER_PROMPT | self.llm | StrOutputParser()

    def generate(
        self,
        question: str,
        chunks: list[FusedResult],
    ) -> tuple[str, float]:
        """
        Generate an answer and return it with timing info.

        Returns:
            (answer_text, generation_time_ms)
        """
        # build context string from chunks
        context_parts = []
        for i, fused in enumerate(chunks, 1):
            source = fused.chunk.source or "unknown"
            retrievers = ", ".join(r.value for r in fused.contributing_retrievers)
            context_parts.append(
                f"[Source {i} | from: {source} | found by: {retrievers}]\n"
                f"{fused.chunk.content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        start = time.perf_counter()
        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "generated_answer",
            question=question[:80],
            answer_length=len(answer),
            time_ms=round(elapsed_ms, 1),
        )

        return answer, elapsed_ms
