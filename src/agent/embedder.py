"""Async client for any OpenAI-compatible /v1/embeddings endpoint.

Default config points at a local Ollama instance running bge-m3, but
any service speaking the OpenAI embeddings protocol works (HuggingFace
TEI, OpenAI's own text-embedding-3-*, etc). The bge-m3 family returns
unit vectors so dot product equals cosine similarity.

Failures never raise — callers get None and should log + skip.
Keeping embed calls soft-fail means embedding-server downtime can't
block message persistence.

Hardening (against bge-m3 503s seen in production):
- A shared httpx.AsyncClient with keep-alive, instead of building a
  fresh client on every call. Cuts socket churn, which itself was
  pushing the single-worker bge-m3 container into overload.
- A semaphore (default concurrency=1) serializes requests so we don't
  fan multiple in-flight POSTs at a single-worker model server.
- Transient failures (connect/read errors and 5xx) are retried with
  exponential backoff before falling back to soft-fail.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)


# Statuses we treat as "the server is busy/restarting, try again".
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class Embedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        dim: int,
        timeout: float = 30.0,
        max_concurrency: int = 1,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
    ):
        self._url = base_url.rstrip("/") + "/embeddings"
        self._model = model
        self._dim = dim
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._max_concurrency = max_concurrency
        # asyncio primitives (Semaphore/Lock) must be created on the
        # loop that will await them. This Embedder is built on the main
        # thread but used from a dedicated agent-thread loop, so we
        # defer creation until the first async call.
        self._sem: Optional[asyncio.Semaphore] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._init_lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_async_state(self) -> tuple[asyncio.Semaphore, httpx.AsyncClient]:
        # Called from inside an async method, so we're already on the
        # loop that will own these primitives. threading.Lock guards the
        # one-time init against the (rare) case of two coroutines racing
        # the very first call.
        if self._sem is not None and self._client is not None:
            return self._sem, self._client
        with self._init_lock:
            if self._sem is None:
                self._sem = asyncio.Semaphore(self._max_concurrency)
            if self._client is None:
                # trust_env=False bypasses HTTP_PROXY / macOS system proxy.
                # The embedder talks to localhost (or an internal LAN host),
                # so a system-wide HTTP proxy (ClashX et al) would just
                # 503 these requests — exactly what we saw in production.
                self._client = httpx.AsyncClient(
                    timeout=self._timeout,
                    trust_env=False,
                    limits=httpx.Limits(
                        max_keepalive_connections=4,
                        max_connections=8,
                    ),
                )
        return self._sem, self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed a batch. Returns (N, dim) float32 array, or None on
        failure. Empty input returns an empty array, not None."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        sem, client = self._ensure_async_state()
        async with sem:
            return await self._embed_with_retry(client, texts)

    async def _embed_with_retry(
        self, client: httpx.AsyncClient, texts: List[str],
    ) -> Optional[np.ndarray]:
        payload = {"model": self._model, "input": texts}

        for attempt in range(self._max_retries + 1):
            is_last = attempt == self._max_retries

            try:
                resp = await client.post(self._url, json=payload)
            except httpx.HTTPError as e:
                if is_last:
                    logger.warning(
                        "embedder.request_failed %s: %s n=%d",
                        type(e).__name__, e, len(texts),
                    )
                    return None
                delay = self._retry_base_delay * (2 ** attempt)
                logger.info(
                    "embedder.retry kind=%s attempt=%d/%d delay=%.1fs",
                    type(e).__name__, attempt + 1, self._max_retries, delay,
                )
                await asyncio.sleep(delay)
                continue

            if resp.status_code in _RETRY_STATUSES and not is_last:
                delay = self._retry_base_delay * (2 ** attempt)
                logger.info(
                    "embedder.retry status=%s attempt=%d/%d delay=%.1fs",
                    resp.status_code, attempt + 1, self._max_retries, delay,
                )
                await asyncio.sleep(delay)
                continue

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

        return None

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
