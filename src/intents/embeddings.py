"""Phase 3 — sentence embeddings for intent classification.

Wraps :class:`sentence_transformers.SentenceTransformer` behind a small,
seeded, cache-friendly facade so the rest of the project never touches the
encoder directly:

* one loader (``get_encoder``) so the ~90 MB model is read from disk once
  per process and the exact model id always comes from configuration
  (``EMBEDDING_MODEL``, default ``all-MiniLM-L6-v2``);
* ``embed_texts`` — deterministic CPU encoding (eval mode, fixed seed, no
  gradient tape);
* ``embed_texts_cached`` — an on-disk ``.npz`` cache keyed by
  (model id, texts hash) so repeated training/evaluation runs skip the
  encoder entirely. Golden-set embeddings are tiny (200 x 384 floats) and
  the cache keeps the evaluation protocol reproducible and fast.

Every downstream consumer (the Phase 3 classifier heads, Phase 4 FAISS
retrieval, the Phase 6 agent) goes through this module — the embedding
model is a single configuration point, never a magic constant.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)

_ENCODER = None  # module-level singleton; the model is ~90 MB on disk


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


def get_encoder():
    """Load (once per process) the configured SentenceTransformer model."""
    global _ENCODER
    if _ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "sentence-transformers is not installed.\n"
                "Phase 3 requires it: pip install -r requirements.txt"
            ) from exc
        settings = get_settings()
        model_id = settings.embedding_model
        logger.info("Loading embedding model: %s (CPU)", model_id)
        try:
            _ENCODER = SentenceTransformer(model_id, device="cpu")
        except Exception as exc:  # network/checkout failures etc.
            raise EmbeddingError(
                f"Failed to load embedding model {model_id!r}: {exc}"
            ) from exc
    return _ENCODER


def embed_texts(
    texts: Sequence[str], batch_size: int = 32, seed: int | None = None
) -> np.ndarray:
    """Embed texts deterministically on CPU; returns float32 (n, 384)."""
    import torch

    if seed is None:
        seed = get_settings().random_seed
    torch.manual_seed(seed)
    encoder = get_encoder()
    encoder.eval()
    with torch.no_grad():
        vectors = encoder.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    return np.asarray(vectors, dtype=np.float32)


def _cache_key(model_id: str, texts: Sequence[str]) -> str:
    """Stable cache key: model id + sha256 over the exact texts."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\x00")
    digest.update(model_id.encode("utf-8"))
    return digest.hexdigest()[:16]


def embed_texts_cached(
    texts: Sequence[str], cache_dir: Path, tag: str
) -> tuple[np.ndarray, bool]:
    """Embed texts with an on-disk .npz cache.

    Returns ``(vectors, cache_hit)``. The cache file is
    ``<cache_dir>/<tag>_<key>.npz`` and is validated by shape — a stale or
    corrupted cache is ignored and recomputed rather than trusted.
    """
    settings = get_settings()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(settings.embedding_model, texts)
    cache_path = cache_dir / f"{tag}_{key}.npz"
    n, dim = len(texts), 384
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            if vectors.shape == (n, dim):
                logger.info("Embedding cache hit: %s", cache_path.name)
                return vectors, True
            logger.warning(
                "Embedding cache shape mismatch (%s vs %s) — recomputing",
                vectors.shape, (n, dim),
            )
        except Exception:  # corrupted cache file
            logger.warning("Embedding cache unreadable — recomputing")
    vectors = embed_texts(texts)
    np.savez_compressed(cache_path, vectors=vectors)
    logger.info("Embedding cache written: %s", cache_path.name)
    return vectors, False
