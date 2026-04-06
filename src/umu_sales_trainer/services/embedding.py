"""Embedding service for text vectorization.

Uses sentence-transformers library with all-MiniLM-L6-v2 model to generate
semantic embeddings for text. Includes LRU caching for improved performance.
"""

import hashlib
from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Sentence-transformers based embedding service.

    Provides text vectorization using the all-MiniLM-L6-v2 model, optimized
    for semantic similarity tasks. Includes caching to reduce redundant
    encoding operations.

    Attributes:
        _model: The underlying sentence-transformer model instance.
        _device: Device to run the model on ("cpu" or "cuda").
        _cache: Dictionary-based cache for encoding results.

    Example:
        >>> service = EmbeddingService()
        >>> embeddings = service.encode(["Hello world", "Semantic search"])
        >>> query_embedding = service.encode_query("search query")
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformer model to use.
                Defaults to "all-MiniLM-L6-v2".
            device: Device to run the model on. Defaults to "cpu".
                Use "cuda" for GPU acceleration if available.
        """
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None
        self._cache: dict[str, List[float]] = {}

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformer model.

        Returns:
            The initialized SentenceTransformer model instance.
        """
        if self._model is None:
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a given text.

        Uses MD5 hash of the text to create a unique cache key.

        Args:
            text: Input text to generate cache key for.

        Returns:
            MD5 hash string as cache key.
        """
        return hashlib.md5(text.encode()).hexdigest()

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into embeddings.

        Results are cached to avoid redundant encoding of repeated texts.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        results: List[List[float]] = []
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                embedding = self.model.encode(text, normalize_embeddings=True)
                embedding_list = embedding.tolist()
                self._cache[cache_key] = embedding_list
                results.append(embedding_list)
        return results

    def encode_query(self, text: str) -> List[float]:
        """Encode a query string into an embedding vector.

        Optimized for query/text comparison tasks by normalizing the output.

        Args:
            text: Query text to encode.

        Returns:
            Query embedding vector as list of floats.

        Raises:
            ValueError: If text is empty.
        """
        if not text:
            raise ValueError("text cannot be empty")

        cache_key = f"query_{self._get_cache_key(text)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.model.encode(text, normalize_embeddings=True)
        embedding_list = embedding.tolist()
        self._cache[cache_key] = embedding_list
        return embedding_list

    def clear_cache(self) -> None:
        """Clear the encoding cache.

        Removes all cached embeddings to free memory.
        """
        self._cache.clear()
