"""Conservative tweet cleaning for the Apple Support AI pipeline.

Policy (assignment §5): remove only *obvious Twitter artifacts* —

* HTML entities (``&amp;`` …),
* retweet prefixes (``RT @user:``),
* URLs,
* the leading ``@mention`` used to address the support account / customer
  (kept configurable — for retrieval evidence the addressing mention carries
  no signal, but the caller may keep it),
* duplicated / collapsed whitespace.

Everything else is **preserved on purpose**: spelling mistakes, natural
wording, casing, meaningful punctuation and product terms. Later phases
(embeddings, retrieval, intent classification) must see realistic customer
language, not over-normalised text.
"""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_RT_PREFIX_RE = re.compile(r"^RT\s+@\w+\s*:?\s*", re.IGNORECASE)
_LEADING_MENTION_RE = re.compile(r"^@\w+\s*")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class CleaningReport:
    """Counters for every cleaning decision (logged + persisted)."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_missing_tweet_id: int = 0
    dropped_missing_author_id: int = 0
    dropped_empty_text: int = 0
    dropped_duplicate_tweet_id: int = 0
    decoded_html_entities: int = 0
    removed_rt_prefix: int = 0
    removed_urls: int = 0
    removed_leading_mention: int = 0
    collapsed_whitespace: int = 0

    def as_dict(self) -> dict:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__  # type: ignore[attr-defined]
        }


def clean_text(
    text: str,
    *,
    strip_leading_mention: bool = True,
) -> str:
    """Clean one tweet while preserving natural customer language.

    The tweet is *not* lower-cased, punctuation is *not* stripped and typos
    are *not* corrected — those properties are features for later phases.
    """
    if not text:
        return ""

    original = text
    out = html.unescape(text)  # &amp; -> &, &gt; -> > …

    if _RT_PREFIX_RE.match(out):
        out = _RT_PREFIX_RE.sub("", out, count=1)

    if _URL_RE.search(out):
        out = _URL_RE.sub(" ", out)

    if strip_leading_mention and out.startswith("@"):
        out = _LEADING_MENTION_RE.sub("", out, count=1)

    out = _WHITESPACE_RE.sub(" ", out).strip()

    # A tweet that was nothing but "@AppleSupport" becomes empty here and is
    # dropped later (counted, never silently).
    _ = original
    return out


def clean_series(
    texts: pd.Series,
    *,
    strip_leading_mention: bool = True,
    report: CleaningReport | None = None,
) -> pd.Series:
    """Vectorised wrapper around :func:`clean_text` with report counters."""
    cleaned: list[str] = []
    for text in texts.astype(str):
        before = text
        after = clean_text(text, strip_leading_mention=strip_leading_mention)
        if report is not None:
            if html.unescape(before) != before:
                report.decoded_html_entities += 1
            if _RT_PREFIX_RE.match(before):
                report.removed_rt_prefix += 1
            if _URL_RE.search(before):
                report.removed_urls += 1
            if strip_leading_mention and before.startswith("@") and not after.startswith("@"):
                report.removed_leading_mention += 1
            if _WHITESPACE_RE.sub(" ", before).strip() != before.strip():
                report.collapsed_whitespace += 1
        cleaned.append(after)
    return pd.Series(cleaned, index=texts.index)


def preprocess_tweets(
    frame: pd.DataFrame,
    *,
    strip_leading_mention: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean text and drop malformed / duplicate rows.

    Steps (in order):
      1. drop rows with missing ``tweet_id`` or ``author_id`` (malformed),
      2. clean the text column (conservative, see :func:`clean_text`),
      3. drop rows whose text is empty after cleaning,
      4. drop duplicate ``tweet_id`` rows (keep first occurrence).

    Conversation structure (ids) is untouched; the caller reconstructs
    conversations afterwards.
    """
    report = CleaningReport(rows_in=len(frame))
    out = frame.copy()

    # 1. malformed identifiers.
    bad_ids = out["tweet_id"].eq("")
    bad_author = out["author_id"].eq("")
    report.dropped_missing_tweet_id = int(bad_ids.sum())
    report.dropped_missing_author_id = int((bad_author & ~bad_ids).sum())
    out = out.loc[~bad_ids & ~bad_author]

    # 2. text cleaning.
    out = out.assign(
        text=clean_series(
            out["text"], strip_leading_mention=strip_leading_mention, report=report
        )
    )

    # 3. empty text after cleaning (mention-only tweets, whitespace-only…).
    empty = out["text"].eq("")
    report.dropped_empty_text = int(empty.sum())
    out = out.loc[~empty]

    # 4. duplicate tweet ids (exact repeats of the same record).
    dup = out.duplicated(subset=["tweet_id"], keep="first")
    report.dropped_duplicate_tweet_id = int(dup.sum())
    out = out.loc[~dup].reset_index(drop=True)

    report.rows_out = len(out)
    logger.info(
        "Preprocessing: %d -> %d rows (dropped: %d missing ids, %d empty text, "
        "%d duplicate tweet ids)",
        report.rows_in, report.rows_out,
        report.dropped_missing_tweet_id + report.dropped_missing_author_id,
        report.dropped_empty_text, report.dropped_duplicate_tweet_id,
    )
    return out, report
