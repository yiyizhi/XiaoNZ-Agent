"""Async client for any OpenAI-compatible /v1/embeddings endpoint.

Default config points at a local Ollama instance running bge-m3, but
any service speaking the OpenAI embeddings protocol works (HuggingFace
TEI, OpenAI's own text-embedding-3-*, etc). The bge-m3 family returns
unit vectors so dot product equals cosine similarity.

Failures never raise — callers get None and should log + skip.
Keeping embed calls soft-fail means embedding-server downtime can't
block message persistence.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        timeout: float = 30.0,
    ):
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._dim = dim
        self._timeout = timeout

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed a batch. Returns (N, dim) float32 array, or None on
        failure. Empty input returns an empty array, not None."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._url,
                    json={"model": self._model, "input": texts},
                )
        except httpx.HTTPError as e:
            logger.warning(
                "embedder.request_failed %s: %s n=%d",
                type(e).__name__, e, len(texts),
            )
            return None
        if resp.status_code != 200:
            logger.warning(
                "embedder.bad_status %s: %s",
                resp.status_code, resp.text[:200],
            )
            return None
        try:
            data = resp.json()
            items = data["data"]
            vectors = [item["embedding"] for item in items]
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("embedder.bad_payload %s", e)
            return None
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.shape != (len(texts), self._dim):
            logger.warning(
                "embedder.dim_mismatch shape=%s expected=(%d,%d)",
                arr.shape, len(texts), self._dim,
            )
            return None
        return arr

    async def embed_one(self, text: str) -> Optional[np.ndarray]:
        r = await self.embed([text])
        if r is None:
            return None
        return r[0]

    async def ping(self) -> bool:
        """Cheap check; used at startup to decide whether to log a
        warning about an unreachable embedding service."""
        v = await self.embed_one("ping")
        return v is not None
