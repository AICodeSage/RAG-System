"""Answer generation with citations and confidence scoring."""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.chunk import Chunk

LOGGER = logging.getLogger(__name__)

# Try to load OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


@dataclass
class Citation:
    """A source citation for an answer."""
    doc_id: str
    chunk_id: str
    text_snippet: str
    relevance_score: float
    source_file: str = ""


@dataclass
class AnswerResult:
    """Complete answer with metadata."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    sources_used: int = 0


class AnswerGenerator:
    """
    Generates answers with citations and confidence scores.
    
    Features:
    - Inline citations [1], [2], etc.
    - Confidence scoring based on retrieval quality
    - Source tracking for transparency
    - Streaming support
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None

        if OPENAI_AVAILABLE:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._client = OpenAI(api_key=api_key)

    def _calculate_confidence(
        self,
        chunks: List[Tuple[Chunk, float]],
        threshold: float = 0.3,
    ) -> float:
        """
        Calculate confidence based on retrieval scores.
        
        Returns a score between 0 and 1:
        - 1.0: High-quality matches found
        - 0.5: Moderate matches
        - 0.0: No good matches
        """
        if not chunks:
            return 0.0

        scores = [score for _, score in chunks]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Combine max and average
        confidence = 0.6 * max_score + 0.4 * avg_score

        # Penalize if best match is below threshold
        if max_score < threshold:
            confidence *= 0.5

        return min(1.0, max(0.0, confidence))

    def _build_context_with_citations(
        self,
        chunks: List[Tuple[Chunk, float]],
    ) -> Tuple[str, List[Citation]]:
        """Build context string with numbered sources."""
        citations = []
        context_parts = []

        for i, (chunk, score) in enumerate(chunks, 1):
            doc_id = chunk.metadata.get("doc_id", "unknown")
            source = chunk.metadata.get("source", "")
            snippet = chunk.text[:150].replace("\n", " ").strip()

            citation = Citation(
                doc_id=doc_id,
                chunk_id=chunk.id,
                text_snippet=snippet,
                relevance_score=score,
                source_file=source,
            )
            citations.append(citation)

            context_parts.append(f"[{i}] {chunk.text}")

        context = "\n\n".join(context_parts)
        return context, citations

    def generate(
        self,
        query: str,
        chunks: List[Tuple[Chunk, float]],
        conversation_context: str = "",
    ) -> AnswerResult:
        """
        Generate an answer with citations.
        
        Args:
            query: User's question
            chunks: List of (chunk, relevance_score) tuples
            conversation_context: Previous conversation for context
            
        Returns:
            AnswerResult with answer, citations, and confidence
        """
        if not chunks:
            return AnswerResult(
                answer="I don't have enough information to answer that question.",
                confidence=0.0,
                reasoning="No relevant documents found.",
            )

        confidence = self._calculate_confidence(chunks)
        context, citations = self._build_context_with_citations(chunks)

        if not self._client:
            # Fallback without LLM
            return AnswerResult(
                answer=self._fallback_answer(chunks),
                citations=citations,
                confidence=confidence,
                sources_used=len(chunks),
            )

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Only use information from the provided context
2. Cite sources using [1], [2], etc. when referencing information
3. If the context doesn't contain enough information, say so
4. Be concise but thorough
5. If you're uncertain, express that uncertainty"""

        user_prompt = f"""Context:
{context}

{f"Previous conversation:{chr(10)}{conversation_context}{chr(10)}{chr(10)}" if conversation_context else ""}Question: {query}

Answer the question using the context above. Include citations like [1], [2] where appropriate."""

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            answer = response.choices[0].message.content.strip()

            return AnswerResult(
                answer=answer,
                citations=citations,
                confidence=confidence,
                sources_used=len(chunks),
            )

        except Exception as e:
            LOGGER.error("Answer generation failed: %s", e)
            return AnswerResult(
                answer=self._fallback_answer(chunks),
                citations=citations,
                confidence=confidence * 0.5,  # Lower confidence for fallback
                reasoning=f"LLM error: {e}",
                sources_used=len(chunks),
            )

    def _fallback_answer(self, chunks: List[Tuple[Chunk, float]]) -> str:
        """Generate a simple answer without LLM."""
        if not chunks:
            return "No relevant information found."

        best_chunk = max(chunks, key=lambda x: x[1])[0]
        text = best_chunk.text.strip()
        
        # Truncate to reasonable length
        if len(text) > 500:
            text = text[:500] + "..."

        return f"Based on the documents: {text}"

    def generate_streaming(
        self,
        query: str,
        chunks: List[Tuple[Chunk, float]],
    ):
        """
        Generate answer with streaming output.
        Yields chunks of the response as they're generated.
        """
        if not self._client or not chunks:
            yield self.generate(query, chunks).answer
            return

        confidence = self._calculate_confidence(chunks)
        context, citations = self._build_context_with_citations(chunks)

        system_prompt = """You are a helpful assistant. Answer questions using the provided context.
Cite sources using [1], [2], etc. Be concise but thorough."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer with citations:"""

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            LOGGER.error("Streaming failed: %s", e)
            yield self._fallback_answer(chunks)

