"""Dense embedding helpers using sentence-transformers."""

import os
import logging
from typing import List, Optional
import numpy as np

LOGGER = logging.getLogger(__name__)

# Try to load sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Try to load OpenAI for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


class EmbeddingModel:
    """Wrapper for embedding models - supports sentence-transformers and OpenAI."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_openai: bool = False,
    ):
        self.use_openai = use_openai
        self._model = None
        self._openai_client = None
        self._dimension = 384  # default for MiniLM

        if use_openai and OPENAI_AVAILABLE:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._openai_client = OpenAI(api_key=api_key)
                self._dimension = 1536  # text-embedding-3-small
                self._openai_model = model_name or "text-embedding-3-small"
                LOGGER.info("Using OpenAI embeddings: %s", self._openai_model)
            else:
                LOGGER.warning("OPENAI_API_KEY not set; falling back to sentence-transformers")
                self.use_openai = False

        if not self.use_openai and SENTENCE_TRANSFORMERS_AVAILABLE:
            model_name = model_name or "all-MiniLM-L6-v2"
            LOGGER.info("Loading sentence-transformer model: %s", model_name)
            self._model = SentenceTransformer(model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            LOGGER.info("Model loaded with dimension: %d", self._dimension)

        if self._model is None and self._openai_client is None:
            raise RuntimeError(
                "No embedding backend available. Install sentence-transformers "
                "(`pip install sentence-transformers`) or configure OpenAI."
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts efficiently."""
        if not texts:
            return np.array([])

        # Clean texts
        cleaned = [t.strip() if t else "" for t in texts]

        if self.use_openai and self._openai_client:
            return self._embed_openai(cleaned)
        else:
            return self._embed_local(cleaned)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Use sentence-transformers for local embedding."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Use OpenAI API for embeddings."""
        # OpenAI has a limit on batch size
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._openai_client.embeddings.create(
                model=self._openai_model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)
