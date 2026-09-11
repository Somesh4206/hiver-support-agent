"""Central configuration for the Apple Support AI project.

Every tunable value lives here and is read from the environment (optionally
via a local ``.env`` file — see ``.env.example``). Modules must never embed
magic constants; they import :func:`get_settings` instead.

The LLM API key is deliberately *not* exposed through this module. It is
read only by the code that needs it (Phase 5+, response generation) so that
settings objects can be logged safely at any time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at runtime, but listed in requirements.txt
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Fixed locations                                                             #
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
DEFAULT_RAW_CSV: Path = RAW_DIR / "twcs.csv"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


# --------------------------------------------------------------------------- #
# Environment readers                                                         #
# --------------------------------------------------------------------------- #
def _clean(value: str | None) -> str | None:
    """Trim an env value; empty strings count as unset."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_str(key: str, default: str | None = None) -> str | None:
    value = _clean(os.getenv(key))
    return value if value is not None else default


def _env_int(key: str, default: int) -> int:
    raw = _clean(os.getenv(key))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be an integer, got {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = _clean(os.getenv(key))
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be a number, got {raw!r}") from exc


def _resolve_path(raw: str | None, default: Path) -> Path:
    if raw is None:
        return default
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


# --------------------------------------------------------------------------- #
# Settings                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Snapshot of all configuration used by the pipeline and (later) agent."""

    # --- Phase 1: data pipeline -------------------------------------------
    apple_support_author_id: str | None
    raw_data_path: Path
    processed_dir: Path
    random_seed: int

    # --- Phase 2+: thresholds & models (documented now, consumed later) ---
    embedding_model: str
    intent_confidence_threshold: float
    retrieval_threshold: float
    top_k: int
    llm_model: str | None

    # --- Shared chunked-processing knobs (memory-safe at twcs scale) ------
    chunk_size: int = 200_000
    closure_max_passes: int = 6


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build (and cache) the project settings from the environment."""
    return Settings(
        apple_support_author_id=_env_str("APPLE_SUPPORT_AUTHOR_ID"),
        raw_data_path=_resolve_path(_env_str("RAW_DATA_PATH"), DEFAULT_RAW_CSV),
        processed_dir=_resolve_path(_env_str("PROCESSED_DIR"), PROCESSED_DIR),
        random_seed=_env_int("RANDOM_SEED", 42),
        chunk_size=_env_int("CHUNK_SIZE", 200_000),
        closure_max_passes=_env_int("CLOSURE_MAX_PASSES", 6),
        embedding_model=_env_str(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        intent_confidence_threshold=_env_float("INTENT_CONFIDENCE_THRESHOLD", 0.60),
        retrieval_threshold=_env_float("RETRIEVAL_THRESHOLD", 0.65),
        top_k=_env_int("TOP_K", 5),
        llm_model=_env_str("LLM_MODEL"),
    )


def require_apple_author_id(settings: Settings | None = None) -> str:
    """Return the configured AppleSupport author id or fail with instructions.

    The twcs dataset has no brand column, so the Apple support account must be
    identified explicitly. We refuse to guess: the id must come from
    configuration, and ``scripts/find_author_ids.py`` exists to discover it
    from the data itself.
    """
    settings = settings or get_settings()
    if settings.apple_support_author_id:
        return settings.apple_support_author_id
    raise ConfigurationError(
        "APPLE_SUPPORT_AUTHOR_ID is not configured.\n"
        "The Customer Support on Twitter dataset has no brand column, so the "
        "AppleSupport account must be configured explicitly (never guessed).\n"
        "1. Run:  python scripts/find_author_ids.py\n"
        "   -> lists the most frequent outbound author_ids in twcs.csv\n"
        "2. Identify the Apple support account in that list.\n"
        "3. Set it in .env (copy .env.example if needed):\n"
        "       APPLE_SUPPORT_AUTHOR_ID=<author_id>\n"
    )
