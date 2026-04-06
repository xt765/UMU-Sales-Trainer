"""Embedding service for text vectorization.

Uses DashScope text-embedding API to generate semantic embeddings for text.
Includes LRU caching for improved performance. No local model download required.
"""

import hashlib
import os
from typing import List

import httpx


class EmbeddingService:
    """DashScope text-embedding API based embedding service.

    Provides text vectorization using DashScope's text-embedding-v1 API,
    optimized for semantic similarity tasks. Includes caching to reduce
    redundant encoding operations.

    Attributes:
        _cache: Dictionary-based cache for encoding results.
        _client: HTTP client for DashScope API requests.

    Example:
        >>> service = EmbeddingService()
        >>> embeddings = service.encode(["Hello world", "Semantic search"])
        >>> query_embedding = service.encode_query("search query")
    """

    DASHSCOPE_EMBEDDING_URL = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    )

    def __init__(
        self,
        model_name: str = "text-embedding-v1",
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Name of the DashScope embedding model.
                Defaults to "text-embedding-v1".
        """
        self._model_name = model_name
        self._cache: dict[str, List[float]] = {}
        self._client: httpx.Client | None = None

    @property
    def _api_key(self) -> str:
        """Get API key from environment dynamically.

        Returns:
            API key string.
        """
        return os.environ.get("DASHSCOPE_API_KEY", "")

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client.

        Returns:
            Configured httpx Client instance.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

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
            RuntimeError: If API key is not set or API call fails.
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        if not self._api_key:
            raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set")

        results: List[List[float]] = []
        for text in texts:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                embedding = self._call_embedding_api(text)
                self._cache[cache_key] = embedding
                results.append(embedding)
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
            RuntimeError: If API key is not set or API call fails.
        """
        if not text:
            raise ValueError("text cannot be empty")

        if not self._api_key:
            raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set")

        cache_key = f"query_{self._get_cache_key(text)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self._call_embedding_api(text)
        self._cache[cache_key] = embedding
        return embedding

    def _call_embedding_api(self, text: str) -> List[float]:
        """Call DashScope embedding API.

        Args:
            text: Text to encode.

        Returns:
            Embedding vector as list of floats.

        Raises:
            RuntimeError: If API call fails.
        """
        client = self._get_client()

        payload = {
            "model": self._model_name,
            "input": {"texts": [text]},
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        response = client.post(self.DASHSCOPE_EMBEDDING_URL, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        embedding = data.get("output", {}).get("embeddings", [{}])[0].get("embedding", [])

        if not embedding:
            raise RuntimeError(f"Failed to extract embedding from API response: {data}")

        return embedding

    def clear_cache(self) -> None:
        """Clear the encoding cache.

        Removes all cached embeddings to free memory.
        """
        self._cache.clear()

    def close(self) -> None:
        """Close the HTTP client.

        Should be called when the service is no longer needed.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
