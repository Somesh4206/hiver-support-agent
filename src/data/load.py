"""Loading utilities for the Customer Support on Twitter dataset (twcs.csv).

The file is ~516 MB / ~2.8 M rows, so this module offers three strategies:

* :func:`load_twcs` — straightforward in-memory load (validation, missing-value
  handling, optional row limit / deterministic sampling). Suitable for
  development and for files up to a few hundred thousand rows.
* :func:`scan_dataset_stats` — chunked pass that computes dataset-level
  statistics (size, inbound/outbound, conversation starters, top outbound
  authors) without holding the file in memory.
* :func:`collect_author_threads` — memory-safe, multi-pass chunked collector
  that returns *complete conversation threads* involving a given support
  author (e.g. AppleSupport). This is the production path for full-scale runs.

Implementation notes
--------------------
* All identifiers are read as **strings**. twcs tweet ids would lose precision
  if parsed as numbers.
* ``keep_default_na=False`` so that "NA"-like author handles are not silently
  turned into NaN; empties are handled explicitly.
* ``inbound`` is normalised to a real boolean; malformed values are counted.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns required by every downstream stage (spec §1).
EXPECTED_COLUMNS: tuple[str, ...] = (
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
)

#: Reading the raw text column keeps tweet ids/author ids as exact strings.
_READ_KWARGS: dict = {
    "dtype": str,
    "keep_default_na": False,
    "na_filter": False,
}


class DatasetValidationError(RuntimeError):
    """Raised when the file does not match the expected twcs schema."""


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def validate_columns(columns) -> None:
    """Raise :class:`DatasetValidationError` if expected columns are missing."""
    missing = [c for c in EXPECTED_COLUMNS if c not in set(columns)]
    if missing:
        raise DatasetValidationError(
            f"twcs.csv is missing expected column(s): {missing}. "
            f"Expected schema: {list(EXPECTED_COLUMNS)}"
        )


def _normalise_inbound(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the inbound flag to bool; count malformed values."""
    raw = frame["inbound"].astype(str).str.strip().str.lower()
    malformed = int((~raw.isin(["true", "false"])).sum())
    if malformed:
        logger.warning("inbound: %d rows with malformed flag (treated as outbound)",
                       malformed)
    frame["inbound"] = raw.eq("true")
    return frame


def _strip_identifier_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for col in ("tweet_id", "author_id", "response_tweet_id",
                "in_response_to_tweet_id"):
        frame[col] = frame[col].astype(str).str.strip()
    return frame


def children_ids(response_tweet_id: str) -> list[str]:
    """Parse the comma-separated ``response_tweet_id`` field into child ids."""
    if not response_tweet_id:
        return []
    return [part.strip() for part in response_tweet_id.split(",") if part.strip()]


def _file_guidance(path: Path) -> str:
    return (
        f"Raw dataset not found at '{path}'.\n"
        "Download twcs.csv from Kaggle (thoughtvector/customer-support-on-twitter) "
        "into data/raw/ — see data/README.md for step-by-step instructions.\n"
        "For a quick pipeline smoke test without Kaggle access you can also run:\n"
        "    python scripts/make_synthetic_sample.py"
    )


# --------------------------------------------------------------------------- #
# Strategy 1 — simple in-memory load                                          #
# --------------------------------------------------------------------------- #
def load_twcs(
    path: str | Path,
    row_limit: int | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load twcs.csv into memory (optionally limited / sampled).

    Parameters
    ----------
    path:
        Path to ``twcs.csv``.
    row_limit:
        Read only the first N rows (development). Note that conversations
        crossing the boundary will appear truncated — full runs should not
        set this.
    sample_size:
        Deterministically sample N rows (development only; random row sampling
        breaks conversation boundaries and is never used for reported metrics).
    seed:
        Random seed for sampling (deterministic).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_file_guidance(path))

    size_mb = path.stat().st_size / 1e6
    if row_limit is None and size_mb > 300:
        logger.warning(
            "Loading a %.0f MB file fully into memory; prefer the chunked "
            "collector used by scripts/prepare_data.py for full-scale runs",
            size_mb,
        )

    logger.info("Loading %s (limit=%s, sample=%s)", path, row_limit, sample_size)
    frame = pd.read_csv(path, nrows=row_limit, **_READ_KWARGS)
    validate_columns(frame.columns)
    frame = _strip_identifier_columns(frame)
    frame = _normalise_inbound(frame)

    logger.info("Loaded %d rows x %d columns (%.0f MB file)",
                len(frame), frame.shape[1], size_mb)

    if sample_size is not None and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        logger.info("Sampled %d rows (seed=%d)", len(frame), seed)

    _log_missing(frame)
    return frame


def _log_missing(frame: pd.DataFrame) -> None:
    """Report empty values per column (twcs uses empties, not NaN)."""
    parts = []
    for col in EXPECTED_COLUMNS:
        if col not in frame.columns:
            continue
        empty = int((frame[col].astype(str).str.strip() == "").sum())
        if empty:
            parts.append(f"{col}={empty}")
    if parts:
        logger.info("Empty values: %s (response/in_response_to empties are "
                    "expected: they mark conversation roots and unanswered tweets)",
                    ", ".join(parts))


# --------------------------------------------------------------------------- #
# Strategy 2 — chunked dataset statistics                                     #
# --------------------------------------------------------------------------- #
@dataclass
class DatasetStats:
    """Dataset-level statistics computable in a single chunked pass."""

    total_rows: int = 0
    unique_tweet_ids: int = 0
    duplicate_tweet_ids: int = 0
    inbound_rows: int = 0
    outbound_rows: int = 0
    malformed_inbound: int = 0
    conversation_starters: int = 0  # rows with no in_response_to_tweet_id
    missing_by_column: dict[str, int] = field(default_factory=dict)
    top_outbound_authors: list[tuple[str, int]] = field(default_factory=list)
    file_size_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "unique_tweet_ids": self.unique_tweet_ids,
            "duplicate_tweet_ids": self.duplicate_tweet_ids,
            "inbound_rows": self.inbound_rows,
            "outbound_rows": self.outbound_rows,
            "malformed_inbound": self.malformed_inbound,
            "conversation_starters": self.conversation_starters,
            "missing_by_column": self.missing_by_column,
            "top_outbound_authors": [
                {"author_id": a, "tweets": n} for a, n in self.top_outbound_authors
            ],
            "file_size_bytes": self.file_size_bytes,
        }


def scan_dataset_stats(
    path: str | Path,
    chunk_size: int = 200_000,
    top_authors: int = 25,
) -> DatasetStats:
    """Compute dataset-level statistics without loading the file into memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_file_guidance(path))

    stats = DatasetStats(file_size_bytes=path.stat().st_size)
    seen_ids: set = set()
    author_counter: Counter[str] = Counter()
    logger.info("Scanning %s in chunks of %d rows ...", path, chunk_size)

    first = True
    for chunk in pd.read_csv(path, chunksize=chunk_size, **_READ_KWARGS):
        if first:
            validate_columns(chunk.columns)
            first = False
        chunk = _strip_identifier_columns(chunk)

        raw_inbound = chunk["inbound"].astype(str).str.strip().str.lower()
        stats.malformed_inbound += int((~raw_inbound.isin(["true", "false"])).sum())
        is_inbound = raw_inbound.eq("true")
        stats.inbound_rows += int(is_inbound.sum())
        stats.outbound_rows += int((~is_inbound).sum())
        stats.total_rows += len(chunk)

        # Conversation starters: tweets with no parent link.
        stats.conversation_starters += int(
            (chunk["in_response_to_tweet_id"] == "").sum()
        )

        # Missing values (empty strings) per column.
        for col in EXPECTED_COLUMNS:
            if col in ("response_tweet_id", "in_response_to_tweet_id"):
                continue  # empties are structurally meaningful here
            empty = int((chunk[col] == "").sum())
            if empty:
                stats.missing_by_column[col] = (
                    stats.missing_by_column.get(col, 0) + empty
                )

        # Outbound author frequency (support accounts).
        for author, count in (
            chunk.loc[~is_inbound, "author_id"].value_counts().items()
        ):
            author_counter[author] += int(count)

        # Duplicate tweet-id detection (int-keyed when possible to save memory).
        for tid in chunk["tweet_id"]:
            if tid == "":
                continue
            try:
                key = int(tid)
            except ValueError:
                key = tid
            seen_ids.add(key)

    stats.unique_tweet_ids = len(seen_ids)
    stats.duplicate_tweet_ids = stats.total_rows - stats.unique_tweet_ids
    stats.top_outbound_authors = author_counter.most_common(top_authors)

    logger.info(
        "Scan complete: %d rows (%d unique tweet ids, %d duplicates), "
        "%d inbound / %d outbound, %d conversation starters",
        stats.total_rows, stats.unique_tweet_ids, stats.duplicate_tweet_ids,
        stats.inbound_rows, stats.outbound_rows, stats.conversation_starters,
    )
    return stats


# --------------------------------------------------------------------------- #
# Strategy 3 — AppleSupport-first closure collection (memory-safe)            #
# --------------------------------------------------------------------------- #
def collect_author_threads(
    path: str | Path,
    author_id: str,
    max_passes: int = 6,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Collect *complete conversation threads* involving ``author_id``.

    Strategy (keeps peak memory ~hundreds of MB on a 2.8 M-row file):

    * **Pass 1** records every tweet authored by ``author_id`` and seeds a
      "wanted" set with their reply-partners (parents via
      ``in_response_to_tweet_id``, children via ``response_tweet_id``).
    * **Later passes** record any tweet whose id is wanted and expand the
      wanted set with *its* reply-partners.
    * Passes repeat until nothing new is found (threads are short, so this
      converges in a handful of passes).

    The result is the set of tweets reachable from an AppleSupport tweet
    through reply links in both directions — i.e. whole threads, ready for
    conversation reconstruction. Threads without any AppleSupport tweet are
    never touched, which is exactly the brand filter we want.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(_file_guidance(path))

    records: dict[str, dict[str, str]] = {}
    wanted: set[str] = set()

    for pass_no in range(1, max_passes + 1):
        added_rows = 0
        wanted_before = len(wanted)

        for chunk in pd.read_csv(path, chunksize=chunk_size, **_READ_KWARGS):
            chunk = _strip_identifier_columns(chunk)
            if pass_no == 1:
                mask = chunk["author_id"].eq(author_id)
            else:
                mask = chunk["tweet_id"].isin(wanted)
            if not mask.any():
                continue

            for row in chunk.loc[mask].to_dict("records"):
                tid = row["tweet_id"]
                if tid not in records:
                    records[tid] = {c: row[c] for c in EXPECTED_COLUMNS}
                    added_rows += 1
                # Grow the wanted set with this row's reply-partners.
                for child in children_ids(row["response_tweet_id"]):
                    wanted.add(child)
                parent = row["in_response_to_tweet_id"]
                if parent:
                    wanted.add(parent)

        wanted_after = len(wanted)
        logger.info(
            "Closure pass %d/%d: +%d rows recorded (%d total), wanted ids %d -> %d",
            pass_no, max_passes, added_rows, len(records),
            wanted_before, wanted_after,
        )
        if added_rows == 0 and wanted_after == wanted_before:
            break
    else:
        logger.warning(
            "Closure did not fully converge within %d passes; continuing with "
            "the threads collected so far", max_passes,
        )

    # Wanted ids that never appeared in the file: dangling references (noise).
    dangling = len(wanted - set(records))
    if dangling:
        logger.info("Dangling referenced ids not present in file: %d", dangling)

    logger.info("Collected %d tweets across AppleSupport threads", len(records))
    frame = pd.DataFrame(list(records.values()), columns=list(EXPECTED_COLUMNS))
    frame = _normalise_inbound(frame)
    return frame
