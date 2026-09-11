"""Phase 1 exploratory data analysis.

Reads the processed outputs of ``scripts/prepare_data.py`` and writes:

* ``data/processed/eda_summary.json`` — the single source of truth consumed by
  the EDA notebook and the web dashboard (contract-locked);
* ``data/processed/charts/*.png`` — figures for the report and notebook.

Analyses (assignment §6): dataset size, AppleSupport volume, inbound/outbound
distribution, conversation count and length, message length, common terms and
bigrams, and *provisional keyword-based* support topics that inform — but do
not finalise — the Phase 2 intent taxonomy.

All numbers are computed from the processed data; nothing is hard-coded.
"""
from __future__ import annotations

import json
import re
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import get_settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("run_eda")

# --------------------------------------------------------------------------- #
# Provisional keyword topics (Phase 1 — informs the Phase 2 taxonomy)         #
# --------------------------------------------------------------------------- #
# Deliberately broad, transparent keyword matching over customer messages.
# Frequencies observed here justify which categories earn a place in the
# final labelled taxonomy (Phase 2); this is NOT the classifier.
PROVISIONAL_TOPICS: dict[str, tuple[str, ...]] = {
    "Account / Apple ID": (
        "apple id", "password", "passcode", "sign in", "sign-in", "login",
        "log in", "log-in", "account", "verification code", "verify",
        "two factor", "2fa",
    ),
    "iCloud / Backup": (
        "icloud", "backup", "back up", "backed up", "cloud", "sync",
        "storage full", "restore",
    ),
    "Billing / Payments": (
        "bill", "billing", "charge", "charged", "payment", "card", "invoice",
        "receipt", "credit", "double charg", "overcharg",
    ),
    "Refunds / Purchases": (
        "refund", "money back", "purchase", "bought", "paid", "in-app",
        "in app", "app store purchase",
    ),
    "Subscriptions": (
        "subscription", "subscribe", "apple music", "apple tv", "applecare",
        "trial", "renewal", "renew",
    ),
    "Device / Hardware": (
        "screen", "battery", "crack", "broke", "broken", "button", "water",
        "camera", "speaker", "microphone", "charger", "charging port",
        "headphone", "drop", "dropped", "dead", "wont turn on", "won't turn on",
        "black screen",
    ),
    "Software / Apps": (
        "update", "updated", "upgrade", "ios", "ipados", "macos", "bug",
        "glitch", "crash", "crashing", "frozen", "froze", "freeze", "hang",
        "stuck", "app", "reset", "reboot", "restart",
    ),
    "Connectivity": (
        "wifi", "wi-fi", "wi fi", "network", "internet", "bluetooth", "pair",
        "connect", "connection", "signal", "cellular", "hotspot", "airdrop",
    ),
    "Repair / Warranty": (
        "warranty", "repair", "replace", "replacement", "genius bar",
        "genius", "service", "out of warranty",
    ),
    "Order / Delivery": (
        "order", "delivery", "deliver", "ship", "shipping", "shipped",
        "tracking", "track", "arrive", "package", "parcel", "dispatch",
    ),
    "Security / Privacy": (
        "phishing", "scam", "fraud", "hack", "hacked", "hacker", "stolen",
        "steal", "stole", "compromised", "suspicious", "fake email",
        "fake text", "spam", "privacy",
    ),
}

# Display stopwords: standard English function words plus conversational
# filler that carries no topical signal. Applied ONLY to the term-frequency
# lists (customer wording itself is never modified).
STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "your", "youre", "youve",
    "with", "that", "this", "have", "from", "they", "them", "their", "there",
    "here", "what", "when", "where", "which", "while", "will", "just", "would",
    "could", "should", "than", "then", "been", "being", "were", "was", "has",
    "had", "does", "did", "doing", "into", "onto", "over", "under", "again",
    "more", "most", "some", "such", "only", "very", "much", "many", "also",
    "because", "about", "after", "before", "between", "through", "during",
    "above", "below", "both", "each", "few", "other", "same", "such", "only",
    "own", "too", "can", "cant", "cannot", "how", "why", "who", "whom", "its",
    "it's", "im", "ive", "id", "ill", "my", "me", "we", "our", "ours", "us",
    "he", "him", "his", "she", "her", "hers", "get", "got", "getting", "go",
    "going", "want", "wants", "need", "needs", "like", "know", "really",
    "still", "even", "ever", "never", "always", "please", "pls", "plz",
    "thanks", "thank", "thx", "hey", "hi", "hello", "sorry", "help", "back",
    "one", "two", "get", "got", "let", "lets", "try", "tried", "trying",
    "see", "look", "looks", "way", "thing", "things", "stuff", "anyone",
    "anybody", "someone", "somebody", "guys", "via", "dm", "pms",
    "now", "all", "any", "anything", "something", "nothing", "everything",
    "else", "again", "without", "within", "per",
})

TERMS_TOP_N = 15
BIGRAMS_TOP_N = 10
CONV_HIST_MAX_TURNS = 20  # turns > 20 aggregated into the last bucket

MESSAGE_BUCKETS: list[tuple[str, int, int | None]] = [
    ("1-25", 1, 25),
    ("26-50", 26, 50),
    ("51-75", 51, 75),
    ("76-100", 76, 100),
    ("101-150", 101, 150),
    ("151-200", 151, 200),
    ("201+", 201, None),
]


# --------------------------------------------------------------------------- #
# Analysis helpers                                                            #
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> list[str]:
    """Lower-cased alphabetic tokens (length >= 3) for term counting only."""
    return [tok for tok in
            (t.strip(".,!?;:'\"()[]{}…—-") for t in text.lower().split())
            if len(tok) >= 3 and tok.isalpha()]


def _term_frequencies(messages: pd.Series) -> tuple[list[dict], list[dict]]:
    """Top unigrams and bigrams over display-stopword-filtered tokens."""
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    for message in messages:
        tokens = [t for t in _tokenize(message) if t not in STOPWORDS]
        unigrams.update(tokens)
        bigrams.update(zip(tokens, tokens[1:]))
    top_terms = [{"term": t, "count": int(c)} for t, c in unigrams.most_common(TERMS_TOP_N)]
    top_bigrams = [{"term": " ".join(pair), "count": int(c)}
                   for pair, c in bigrams.most_common(BIGRAMS_TOP_N)]
    return top_terms, top_bigrams


def _provisional_topics(messages: pd.Series) -> tuple[list[dict], float]:
    """Keyword-topic tagging: matched counts, share, multi-topic rate.

    Keywords are matched on **word boundaries** (e.g. ``app`` must not match
    inside ``apple``). Phrases are matched as whole phrases.
    """
    total = len(messages)
    compiled = {
        topic: [re.compile(r"\b" + re.escape(key) + r"\b") for key in keys]
        for topic, keys in PROVISIONAL_TOPICS.items()
    }
    matched_counts: Counter = Counter()
    multi = 0
    for message in messages:
        hits = [topic for topic, patterns in compiled.items()
                if any(p.search(message) for p in patterns)]
        if hits:
            matched_counts.update(hits)
            if len(hits) >= 2:
                multi += 1
    topics = [
        {
            "topic": topic,
            "matched": int(matched_counts.get(topic, 0)),
            "share": round(matched_counts.get(topic, 0) / total, 4) if total else 0.0,
        }
        for topic in PROVISIONAL_TOPICS
    ]
    topics.sort(key=lambda item: item["matched"], reverse=True)
    multi_share = round(multi / total, 4) if total else 0.0
    return topics, multi_share


def _conversation_length_histogram(conversations: pd.DataFrame) -> list[dict]:
    """Tweet-count-per-conversation histogram (tail aggregated at 20+)."""
    counts = conversations["tweets"].astype(int)
    histogram: list[dict] = []
    for turns in range(1, CONV_HIST_MAX_TURNS + 1):
        if turns == CONV_HIST_MAX_TURNS:
            bucket = int((counts >= turns).sum())
        else:
            bucket = int((counts == turns).sum())
        if bucket or turns < 3:  # always show the head, even if zero
            histogram.append({"turns": turns, "count": bucket})
    return histogram


def _message_length_histogram(texts: pd.Series) -> list[dict]:
    lengths = texts.astype(str).str.len()
    histogram = []
    for label, low, high in MESSAGE_BUCKETS:
        if high is None:
            count = int((lengths >= low).sum())
        else:
            count = int(((lengths >= low) & (lengths <= high)).sum())
        histogram.append({"bucket": label, "count": count})
    return histogram


def _sample_conversations(
    pairs: pd.DataFrame, conversations: pd.DataFrame, seed: int, n: int = 6
) -> list[dict]:
    """Deterministically pick sample pairs stratified by conversation size."""
    if pairs.empty:
        return []
    merged = pairs.merge(
        conversations[["conversation_id", "tweets"]],
        on="conversation_id", how="left",
    )
    merged = merged.sort_values("conversation_id", kind="mergesort").reset_index(drop=True)
    turns = merged["tweets"].fillna(merged["turn"].astype(float))
    short = merged[turns <= 3]
    medium = merged[(turns > 3) & (turns <= 8)]
    long_ = merged[turns > 8]
    samples: list[pd.Series] = []
    rng = pd.Series(range(len(merged))).sample(frac=1.0, random_state=seed)  # deterministic shuffle key
    shuffled = merged.iloc[rng.index]
    for pool, take in ((short, 2), (medium, 2), (long_, 2)):
        pool_shuffled = shuffled[shuffled["conversation_id"].isin(pool["conversation_id"])]
        chosen = pool_shuffled.drop_duplicates("conversation_id").head(take)
        samples.extend(chosen.to_dict("records"))
    out = []
    seen: set[str] = set()
    for record in samples:
        if record["conversation_id"] in seen:
            continue
        seen.add(record["conversation_id"])
        out.append({
            "conversation_id": str(record["conversation_id"]),
            "turns": int(record["tweets"]) if pd.notna(record["tweets"]) else int(record["turn"]),
            "created_at": str(record["created_at"]),
            "customer_message": str(record["customer_message"]),
            "support_response": str(record["support_response"]),
        })
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #
def _render_charts(summary: dict, out_dir: Path) -> list[str]:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    dataset = summary["dataset"]
    inbound = summary["inbound_distribution"]

    # 1. Inbound vs outbound.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["inbound\n(customer)", "outbound\n(support)"],
           [inbound["inbound"], inbound["outbound"]],
           color=["#10b981", "#a1a1aa"])
    ax.set_title("Inbound vs outbound tweets (AppleSupport conversations)")
    for i, value in enumerate([inbound["inbound"], inbound["outbound"]]):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    path = charts_dir / "inbound_outbound.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # 2. Conversation length distribution.
    hist = summary["conversation_length_histogram"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([h["turns"] for h in hist], [h["count"] for h in hist], color="#10b981")
    ax.set_title(f"Conversation length (tweets per thread; {CONV_HIST_MAX_TURNS}+ grouped)")
    ax.set_xlabel("tweets in conversation")
    ax.set_ylabel("conversations")
    fig.tight_layout()
    path = charts_dir / "conversation_lengths.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # 3. Message length histogram.
    mhist = summary["message_length_histogram"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([m["bucket"] for m in mhist], [m["count"] for m in mhist], color="#10b981")
    ax.set_title("Message length (characters, cleaned text)")
    ax.set_xlabel("characters")
    ax.set_ylabel("messages")
    fig.tight_layout()
    path = charts_dir / "message_lengths.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # 4. Top terms.
    terms = summary["top_terms"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([t["term"] for t in reversed(terms)],
            [t["count"] for t in reversed(terms)], color="#10b981")
    ax.set_title("Most common terms in customer messages")
    ax.set_xlabel("occurrences")
    fig.tight_layout()
    path = charts_dir / "top_terms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # 5. Top bigrams.
    bigrams = summary["top_bigrams"]
    if bigrams:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh([b["term"] for b in reversed(bigrams)],
                [b["count"] for b in reversed(bigrams)], color="#10b981")
        ax.set_title("Most common bigrams in customer messages")
        ax.set_xlabel("occurrences")
        fig.tight_layout()
        path = charts_dir / "top_bigrams.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    # 6. Provisional topics.
    topics = summary["provisional_topics"]
    fig, ax = plt.subplots(figsize=(7, 5))
    names = [t["topic"] for t in reversed(topics)]
    shares = [t["share"] * 100 for t in reversed(topics)]
    ax.barh(names, shares, color="#10b981")
    ax.set_title("Provisional support topics (share of customer messages)")
    ax.set_xlabel("% of customer messages matching topic keywords")
    fig.tight_layout()
    path = charts_dir / "provisional_topics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    _ = dataset
    logger.info("Wrote %d charts to %s", len(written), charts_dir)
    return written


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    settings = get_settings()
    out_dir = settings.processed_dir
    run_path = out_dir / "pipeline_run.json"
    tweets_path = out_dir / "applesupport_tweets.csv"
    conv_path = out_dir / "conversations.csv"
    pairs_path = out_dir / "conversation_pairs.csv"

    required = [run_path, tweets_path, conv_path, pairs_path]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        logger.error(
            "Processed outputs missing (%s). Run scripts/prepare_data.py first.",
            ", ".join(missing),
        )
        return 1

    run_report = json.loads(run_path.read_text(encoding="utf-8"))
    tweets = pd.read_csv(tweets_path, dtype=str, keep_default_na=False)
    tweets["inbound"] = tweets["inbound"].astype(str).str.lower().eq("true")
    conversations = pd.read_csv(conv_path, dtype=str, keep_default_na=False)
    conversations["tweets"] = conversations["tweets"].astype(int)
    pairs = pd.read_csv(pairs_path, dtype=str, keep_default_na=False)

    scan = run_report.get("scan_stats") or {}
    texts = tweets["text"].astype(str)
    lengths_chars = texts.str.len()
    lengths_words = texts.str.split().str.len()
    timestamps = pd.to_datetime(tweets["created_at"], format="mixed",
                                errors="coerce", utc=True)

    summary: dict = {
        "generated_at": run_report.get("generated_at"),
        "source": {
            "file": run_report["input"]["file"],
            "path": run_report["input"]["path"],
            "synthetic": bool(run_report["input"].get("synthetic", False)),
            "apple_support_author_id": run_report["apple_support_author_id"],
        },
        "dataset": {
            "total_tweets": int(scan.get("total_rows", 0)) or int(len(tweets)),
            "total_conversations": int(scan.get("conversation_starters", 0))
            or int(len(conversations)),
            "apple_conversations": int(len(conversations)),
            "apple_tweets": int(len(tweets)),
            "apple_inbound": int(tweets["inbound"].sum()),
            "apple_outbound": int((~tweets["inbound"]).sum()),
            "support_pairs": int(len(pairs)),
            "avg_conversation_turns": round(float(conversations["tweets"].mean()), 2)
            if len(conversations) else 0.0,
            "median_conversation_turns": float(conversations["tweets"].median())
            if len(conversations) else 0.0,
            "max_conversation_turns": int(conversations["tweets"].max())
            if len(conversations) else 0,
            "avg_message_chars": round(float(lengths_chars.mean()), 1)
            if len(texts) else 0.0,
            "avg_message_words": round(float(lengths_words.mean()), 1)
            if len(texts) else 0.0,
            "date_start": str(tweets.loc[timestamps.idxmin(), "created_at"])
            if timestamps.notna().any() else "",
            "date_end": str(tweets.loc[timestamps.idxmax(), "created_at"])
            if timestamps.notna().any() else "",
        },
        "inbound_distribution": {
            "inbound": int(tweets["inbound"].sum()),
            "outbound": int((~tweets["inbound"]).sum()),
        },
        "conversation_length_histogram": _conversation_length_histogram(conversations),
        "message_length_histogram": _message_length_histogram(texts),
        "top_terms": [],
        "top_bigrams": [],
        "provisional_topics": [],
        "multi_topic_share": 0.0,
        "pipeline_stages": run_report.get("stages", []),
        "sample_conversations": _sample_conversations(
            pairs, conversations, seed=settings.random_seed
        ),
    }

    customer_messages = pairs["customer_message"].astype(str)
    summary["top_terms"], summary["top_bigrams"] = _term_frequencies(customer_messages)
    summary["provisional_topics"], summary["multi_topic_share"] = _provisional_topics(
        customer_messages
    )

    summary_path = out_dir / "eda_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote %s", summary_path)

    _render_charts(summary, out_dir)

    # Human-readable highlights.
    d = summary["dataset"]
    logger.info("=" * 72)
    logger.info("EDA SUMMARY (source=%s synthetic=%s)",
                summary["source"]["file"], summary["source"]["synthetic"])
    logger.info("  total tweets scanned:        %s", f"{d['total_tweets']:,}")
    logger.info("  AppleSupport conversations:  %s", f"{d['apple_conversations']:,}")
    logger.info("  AppleSupport tweets:         %s (in %s / out %s)",
                f"{d['apple_tweets']:,}", f"{d['apple_inbound']:,}",
                f"{d['apple_outbound']:,}")
    logger.info("  support pairs:               %s", f"{d['support_pairs']:,}")
    logger.info("  avg/median conv turns:       %s / %s",
                d["avg_conversation_turns"], d["median_conversation_turns"])
    logger.info("  avg message chars/words:     %s / %s",
                d["avg_message_chars"], d["avg_message_words"])
    logger.info("  top terms:                   %s",
                ", ".join(t["term"] for t in summary["top_terms"][:10]))
    logger.info("  top provisional topics:      %s",
                ", ".join(t["topic"] for t in summary["provisional_topics"][:5]))
    logger.info("  multi-topic share:           %.1f%%",
                summary["multi_topic_share"] * 100)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
