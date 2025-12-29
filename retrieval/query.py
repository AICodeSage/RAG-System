"""Query enhancement and rewriting for better retrieval."""

import logging
import os
from typing import List, Optional

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


class QueryEnhancer:
    """
    Enhances queries for better retrieval through:
    - Query expansion (add related terms)
    - Query rewriting (clarify ambiguous queries)
    - Hypothetical document generation (HyDE)
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and OPENAI_AVAILABLE
        self._client = None

        if self.use_llm:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._client = OpenAI(api_key=api_key)
            else:
                self.use_llm = False
                LOGGER.debug("OpenAI API key not found, query enhancement disabled")

    def expand_query(self, query: str) -> str:
        """
        Expand query with related terms and synonyms.
        """
        if not self.use_llm or not self._client:
            return query

        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Expand the search query with related terms and synonyms. "
                            "Return ONLY the expanded query, no explanation. "
                            "Keep it concise (under 50 words)."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=100,
            )
            expanded = response.choices[0].message.content.strip()
            LOGGER.debug("Expanded query: '%s' -> '%s'", query, expanded)
            return expanded
        except Exception as e:
            LOGGER.warning("Query expansion failed: %s", e)
            return query

    def rewrite_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Rewrite ambiguous or conversational queries into clear search queries.
        """
        if not self.use_llm or not self._client:
            return query

        system_prompt = (
            "Rewrite the user's question as a clear, specific search query. "
            "Return ONLY the rewritten query, no explanation."
        )
        if context:
            system_prompt += f"\n\nConversation context:\n{context}"

        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=100,
            )
            rewritten = response.choices[0].message.content.strip()
            LOGGER.debug("Rewritten query: '%s' -> '%s'", query, rewritten)
            return rewritten
        except Exception as e:
            LOGGER.warning("Query rewriting failed: %s", e)
            return query

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer (HyDE technique).
        The hypothetical answer is then embedded and used for retrieval,
        often improving results for complex queries.
        """
        if not self.use_llm or not self._client:
            return query

        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate a brief, factual answer to this question as if you had "
                            "access to the relevant documents. This will be used to find "
                            "similar content. Keep it under 100 words."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.5,
                max_tokens=150,
            )
            hyde = response.choices[0].message.content.strip()
            LOGGER.debug("HyDE for query: '%s' -> '%s'", query, hyde[:100])
            return hyde
        except Exception as e:
            LOGGER.warning("HyDE generation failed: %s", e)
            return query

    def multi_query(self, query: str, n: int = 3) -> List[str]:
        """
        Generate multiple query variations for better recall.
        """
        if not self.use_llm or not self._client:
            return [query]

        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Generate {n} different versions of this search query. "
                            "Each should capture the same intent but use different words. "
                            "Return one query per line, no numbering or explanation."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
                max_tokens=200,
            )
            queries = response.choices[0].message.content.strip().split("\n")
            queries = [q.strip() for q in queries if q.strip()][:n]
            queries.insert(0, query)  # Include original
            LOGGER.debug("Multi-query: generated %d variations", len(queries))
            return queries
        except Exception as e:
            LOGGER.warning("Multi-query generation failed: %s", e)
            return [query]

