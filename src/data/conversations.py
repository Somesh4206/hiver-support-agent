"""Reconstruct support conversations from twcs reply links.

Data model
----------
``twcs.csv`` links tweets in two ways:

* ``in_response_to_tweet_id`` — the *single* tweet this tweet replies to
  (empty for conversation starters);
* ``response_tweet_id`` — comma-separated ids of tweets that reply to *this*
  tweet.

We build reply trees from the ``in_response_to`` links (each tweet has at
most one parent, so the structure is a tree/forest). ``response_tweet_id``
is used for closure expansion in :mod:`src.data.load` and for consistency
checks here. Known dataset noise is handled explicitly:

* tweets whose parent is missing from the frame become roots of their own
  (partially observed) conversation — this keeps boundaries stable when the
  frame is a subset (``--limit`` mode) instead of randomly merging threads;
* cycles (a ↔ b back-edges) are detected, broken and reported, never silently
  dropped;
* ordering inside a conversation is *structural* (depth-first from the root,
  siblings ordered by timestamp then file order) — tweet ids in twcs are not
  time-ordered, so they must not be used for sequencing.

Outputs
-------
* :func:`extract_support_pairs` — the customer-message ↔ historical-support-
  reply evidence pairs used for retrieval in Phase 4. Each pair keeps its
  ``conversation_id`` so that Phase 2 can split train/val/test at the
  *conversation* level and prevent leakage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.data.load import children_ids

logger = logging.getLogger(__name__)

_TWITTER_TS_FORMAT = "%a %b %d %H:%M:%S %z %Y"


@dataclass(frozen=True)
class Tweet:
    """A single tweet with the links needed for thread reconstruction."""

    tweet_id: str
    author_id: str
    inbound: bool
    created_at: str
    text: str
    parent_id: str  # "" when the tweet starts a conversation
    child_ids: tuple[str, ...]
    file_order: int  # original row position (deterministic tiebreaker)


@dataclass
class Conversation:
    """A reconstructed conversation thread, in reading order."""

    conversation_id: str  # root tweet id
    tweets: list[Tweet] = field(default_factory=list)


@dataclass
class ReconstructionStats:
    """Diagnostics for the reconstruction (persisted in pipeline_run.json)."""

    tweets_in: int = 0
    conversations: int = 0
    tweets_in_conversations: int = 0
    partial_roots: int = 0        # roots created because the parent is absent
    cyclic_tweets_skipped: int = 0
    parent_link_mismatches: int = 0  # in_response_to vs response_tweet_id noise
    dangling_child_links: int = 0    # response ids referencing absent tweets

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# Graph construction                                                          #
# --------------------------------------------------------------------------- #
def build_reply_graph(frame: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Return ``(parent_of, children_of, response_ids)`` maps keyed by tweet id.

    ``parent_of`` is authoritative for the tree structure; ``children_of`` is
    derived from it; ``response_ids`` (raw ``response_tweet_id`` field) is
    used for consistency diagnostics.
    """
    parent_of: dict[str, str] = {}
    children_of: dict[str, list[str]] = {}
    response_ids_by_tweet: dict[str, list[str]] = {}

    for row in frame.itertuples(index=False):
        tid = row.tweet_id
        parent = getattr(row, "in_response_to_tweet_id")
        parent_of[tid] = parent
        response_ids_by_tweet[tid] = children_ids(row.response_tweet_id)

    # Children map from the authoritative parent links.
    for tid, parent in parent_of.items():
        if parent and parent in parent_of and parent != tid:
            children_of.setdefault(parent, []).append(tid)

    return parent_of, children_of, response_ids_by_tweet


def _parse_timestamps(frame: pd.DataFrame) -> pd.Series:
    """Parse Twitter timestamps; unparseable values become NaT (counted)."""
    parsed = pd.to_datetime(frame["created_at"], format=_TWITTER_TS_FORMAT,
                            errors="coerce", utc=True)
    bad = int(parsed.isna().sum())
    if bad:
        logger.warning("%d rows have unparseable created_at values", bad)
    return parsed


# --------------------------------------------------------------------------- #
# Reconstruction                                                              #
# --------------------------------------------------------------------------- #
def reconstruct_conversations(frame: pd.DataFrame) -> tuple[list[Conversation], ReconstructionStats]:
    """Group a frame of tweets into conversation threads.

    Rules (see module docstring): roots are tweets with no parent *or* whose
    parent is absent from the frame; traversal is depth-first from the root
    with siblings ordered by timestamp (fallback: file order); back-edges
    forming cycles are skipped and counted.
    """
    stats = ReconstructionStats(tweets_in=len(frame))
    timestamps = _parse_timestamps(frame)

    tweets: dict[str, Tweet] = {}
    for order, row in enumerate(frame.itertuples(index=False)):
        tid = row.tweet_id
        tweets[tid] = Tweet(
            tweet_id=tid,
            author_id=row.author_id,
            inbound=bool(row.inbound),
            created_at=row.created_at,
            text=row.text,
            parent_id=getattr(row, "in_response_to_tweet_id"),
            child_ids=tuple(children_ids(row.response_tweet_id)),
            file_order=order,
        )

    parent_of, children_of, response_ids = build_reply_graph(frame)

    # Consistency diagnostics: in_response_to vs response_tweet_id disagreement
    # and response links pointing at tweets outside the frame.
    for tid, resp_ids in response_ids.items():
        for rid in resp_ids:
            if rid not in parent_of:
                stats.dangling_child_links += 1
            elif parent_of.get(rid) != tid:
                stats.parent_link_mismatches += 1

    # Roots: no parent, or parent absent from this frame.
    roots: list[str] = []
    for tid, tweet in tweets.items():
        if not tweet.parent_id:
            roots.append(tid)
        elif tweet.parent_id not in tweets:
            stats.partial_roots += 1
            roots.append(tid)

    # Sibling ordering: timestamp first (NaT last), then file order.
    ts_map = {
        tid: (0, timestamps.iloc[tweets[tid].file_order], tweets[tid].file_order)
        if not pd.isna(timestamps.iloc[tweets[tid].file_order])
        else (1, pd.Timestamp("1970-01-01", tz="UTC"), tweets[tid].file_order)
        for tid in tweets
    }

    def sort_key(tid: str):
        return ts_map[tid]

    visited: set[str] = set()
    conversations: list[Conversation] = []

    for root in sorted(roots, key=sort_key):
        if root in visited:
            continue
        conversation = Conversation(conversation_id=root)
        stack = [root]
        while stack:
            tid = stack.pop()
            if tid in visited:
                continue
            visited.add(tid)
            conversation.tweets.append(tweets[tid])
            for child in sorted(children_of.get(tid, []), key=sort_key, reverse=True):
                if child not in visited:
                    stack.append(child)
        conversations.append(conversation)

    stats.conversations = len(conversations)
    stats.tweets_in_conversations = len(visited)
    stats.cyclic_tweets_skipped = len(tweets) - len(visited)

    logger.info(
        "Reconstructed %d conversations from %d tweets (%d partial roots, "
        "%d cyclic/unreachable tweets)",
        stats.conversations, len(tweets), stats.partial_roots,
        stats.cyclic_tweets_skipped,
    )
    return conversations, stats


# --------------------------------------------------------------------------- #
# Filtering & frames                                                          #
# --------------------------------------------------------------------------- #
def filter_conversations_with_author(
    conversations: list[Conversation], author_id: str
) -> list[Conversation]:
    """Keep conversations in which ``author_id`` wrote at least one tweet."""
    kept = [
        conv for conv in conversations
        if any(t.author_id == author_id for t in conv.tweets)
    ]
    logger.info("Filtered to %d / %d conversations involving author %r",
                len(kept), len(conversations), author_id)
    return kept


def conversations_frame(conversations: list[Conversation]) -> pd.DataFrame:
    """One row per conversation: size, participants and tweet-id list."""
    rows = []
    for conv in conversations:
        tweets = conv.tweets
        rows.append({
            "conversation_id": conv.conversation_id,
            "tweets": len(tweets),
            "customer_tweets": sum(1 for t in tweets if t.inbound),
            "support_tweets": sum(1 for t in tweets if not t.inbound),
            "first_created_at": tweets[0].created_at if tweets else "",
            "last_created_at": tweets[-1].created_at if tweets else "",
            "tweet_ids": ";".join(t.tweet_id for t in tweets),
        })
    return pd.DataFrame(
        rows,
        columns=["conversation_id", "tweets", "customer_tweets", "support_tweets",
                 "first_created_at", "last_created_at", "tweet_ids"],
    )


def tweets_frame_with_conversation(
    conversations: list[Conversation],
) -> pd.DataFrame:
    """Explode conversations back into a tweet frame with thread metadata.

    Every tweet gets its ``conversation_id`` and its 1-based ``turn`` (its
    position in the conversation's reading order). This is the frame later
    phases (conversation-level splitting in Phase 2) consume.
    """
    rows = []
    for conv in conversations:
        for turn, tweet in enumerate(conv.tweets, start=1):
            rows.append({
                "tweet_id": tweet.tweet_id,
                "author_id": tweet.author_id,
                "inbound": tweet.inbound,
                "created_at": tweet.created_at,
                "text": tweet.text,
                "response_tweet_id": ",".join(tweet.child_ids),
                "in_response_to_tweet_id": tweet.parent_id,
                "conversation_id": conv.conversation_id,
                "turn": turn,
            })
    return pd.DataFrame(
        rows,
        columns=["tweet_id", "author_id", "inbound", "created_at", "text",
                 "response_tweet_id", "in_response_to_tweet_id",
                 "conversation_id", "turn"],
    )


# --------------------------------------------------------------------------- #
# Support pairs                                                               #
# --------------------------------------------------------------------------- #
PAIR_COLUMNS = [
    "conversation_id", "customer_tweet_id", "response_tweet_id", "tweet_ids",
    "customer_message", "support_response", "customer_created_at",
    "response_created_at", "created_at", "turn",
]


def extract_support_pairs(
    conversations: list[Conversation], support_author_id: str
) -> pd.DataFrame:
    """Extract customer-message ↔ support-reply evidence pairs.

    A pair is created for **every reply authored by ``support_author_id``**
    whose parent (``in_response_to_tweet_id``) is an inbound customer tweet
    in the same conversation. Two support replies to the same customer tweet
    therefore produce two pairs (both are legitimate historical evidence);
    each support reply is used in exactly one pair.

    The ``conversation_id`` column is what allows Phase 2 to split the data at
    conversation level — random tweet-level splits would leak replies of the
    same thread across train and test.
    """
    rows: list[dict] = []
    for conv in conversations:
        by_id = {t.tweet_id: t for t in conv.tweets}
        turn_of = {t.tweet_id: i for i, t in enumerate(conv.tweets, start=1)}
        for tweet in conv.tweets:
            if tweet.author_id != support_author_id:
                continue
            parent = by_id.get(tweet.parent_id)
            if parent is None or not parent.inbound:
                continue  # support tweet answering another support tweet / orphan
            rows.append({
                "conversation_id": conv.conversation_id,
                "customer_tweet_id": parent.tweet_id,
                "response_tweet_id": tweet.tweet_id,
                "tweet_ids": f"{parent.tweet_id};{tweet.tweet_id}",
                "customer_message": parent.text,
                "support_response": tweet.text,
                "customer_created_at": parent.created_at,
                "response_created_at": tweet.created_at,
                "created_at": parent.created_at,
                "turn": turn_of[tweet.tweet_id],
            })
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    logger.info("Extracted %d customer↔support pairs from %d conversations",
                len(pairs), len(conversations))
    return pairs
