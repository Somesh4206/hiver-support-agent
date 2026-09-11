"""Phase 1 data preparation (end-to-end).

Pipeline
--------
1. load / scan the raw dataset (schema validation, dataset statistics);
2. collect complete conversation threads involving the configured
   AppleSupport author (memory-safe chunked closure on full-scale data);
3. clean text conservatively and drop malformed/duplicate rows;
4. reconstruct conversations from reply links;
5. filter to conversations the support author took part in;
6. extract customer-message ↔ historical-support-reply pairs;
7. run structural validations over every output;
8. persist outputs + a machine-readable run report (pipeline_run.json).

Usage (from the repository root):

    python scripts/prepare_data.py [--input PATH] [--author-id ID]
                                   [--limit N] [--sample N]

Full-scale runs use the chunked AppleSupport-first collector (~2.8 M rows in
a few passes, peak memory a few hundred MB). ``--limit`` / ``--sample`` are
development shortcuts that load a subset in memory first (conversations
crossing the subset boundary are reconstructed partially — never use these
modes for reported metrics).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    PROJECT_ROOT,
    ConfigurationError,
    get_settings,
    require_apple_author_id,
)
from src.data.conversations import (  # noqa: E402
    extract_support_pairs,
    filter_conversations_with_author,
    conversations_frame,
    reconstruct_conversations,
    tweets_frame_with_conversation,
)
from src.data.load import (  # noqa: E402
    collect_author_threads,
    load_twcs,
    scan_dataset_stats,
)
from src.data.preprocessing import preprocess_tweets  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_data")

OUTPUT_TWEETS = "applesupport_tweets.csv"
OUTPUT_CONVERSATIONS = "conversations.csv"
OUTPUT_PAIRS = "conversation_pairs.csv"
OUTPUT_RUN = "pipeline_run.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _collect_subset(path: Path, author_id: str, settings, limit, sample):
    """Return the tweet frame for the AppleSupport threads.

    Full-scale: chunked closure collector (scan stats computed alongside).
    Dev mode (``--limit``/``--sample``): in-memory load then in-frame closure.
    """
    if limit is None and sample is None:
        stats = scan_dataset_stats(path, chunk_size=settings.chunk_size)
        frame = collect_author_threads(
            path, author_id,
            max_passes=settings.closure_max_passes,
            chunk_size=settings.chunk_size,
        )
        return frame, stats, "full"

    frame = load_twcs(
        path, row_limit=limit, sample_size=sample, seed=settings.random_seed
    )
    # In-frame closure: keep tweets connected to an author tweet via links.
    wanted = set(frame.loc[frame["author_id"] == author_id, "tweet_id"])
    if not wanted:
        return frame.iloc[0:0], None, "dev-subset"
    all_ids = set(frame["tweet_id"])
    keep = set(wanted)
    while True:
        new = set()
        for row in frame.loc[frame["tweet_id"].isin(keep)].itertuples(index=False):
            parent = getattr(row, "in_response_to_tweet_id")
            if parent and parent in all_ids:
                new.add(parent)
            for child in str(row.response_tweet_id).split(","):
                child = child.strip()
                if child and child in all_ids:
                    new.add(child)
        new -= keep
        if not new:
            break
        keep |= new
    subset = frame.loc[frame["tweet_id"].isin(keep)].reset_index(drop=True)
    return subset, None, "dev-subset"


def _validate_outputs(tweets, conversations, pairs, author_id) -> list[dict]:
    """Structural validations — the phase's test gate (all must pass)."""
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
        logger.info("VALIDATION %-52s %s — %s", name, "PASS" if ok else "FAIL", detail)

    conv_ids = set(conversations["conversation_id"])
    tweet_ids = set(tweets["tweet_id"])

    record(
        "pair_ids_exist_in_tweets",
        bool(pairs["customer_tweet_id"].isin(tweet_ids).all()
             and pairs["response_tweet_id"].isin(tweet_ids).all()),
        f"{len(pairs)} pair endpoint ids resolved against the tweet frame",
    )
    record(
        "pair_conversations_exist",
        bool(pairs["conversation_id"].isin(conv_ids).all()) if len(pairs) else True,
        "every pair maps to a reconstructed conversation",
    )
    record(
        "pair_messages_non_empty",
        bool((pairs["customer_message"].str.len() > 0).all()
             and (pairs["support_response"].str.len() > 0).all()) if len(pairs) else True,
        "no empty customer messages or support responses in pairs",
    )
    record(
        "pair_response_ids_unique",
        bool(not pairs["response_tweet_id"].duplicated().any()),
        "each support reply is used in at most one pair",
    )
    record(
        "pair_responses_by_support_author",
        bool((tweets.set_index("tweet_id").loc[
            pairs["response_tweet_id"], "author_id"] == author_id).all())
        if len(pairs) else True,
        f"all pair responses authored by {author_id!r}",
    )
    record(
        "conversation_ids_unique",
        bool(not conversations["conversation_id"].duplicated().any()),
        f"{len(conversations)} unique conversation ids",
    )
    record(
        "tweets_have_conversation_id",
        bool((tweets["conversation_id"].str.len() > 0).all()),
        "every output tweet is attached to a conversation",
    )
    return checks


def main() -> int:
    settings = get_settings()
    started = time.time()

    parser = argparse.ArgumentParser(description="Phase 1 data preparation")
    parser.add_argument("--input", default=None, help="Path to twcs.csv")
    parser.add_argument("--author-id", default=None,
                        help="Override APPLE_SUPPORT_AUTHOR_ID")
    parser.add_argument("--limit", type=int, default=None,
                        help="DEV ONLY: read only the first N raw rows")
    parser.add_argument("--sample", type=int, default=None,
                        help="DEV ONLY: random-sample N raw rows (breaks "
                             "conversation boundaries; smoke tests only)")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else settings.raw_data_path
    if not input_path.is_file():
        logger.error("Input dataset not found: %s (see data/README.md)",
                     input_path)
        return 1

    try:
        author_id = args.author_id or require_apple_author_id(settings)
    except ConfigurationError as exc:
        logger.error("%s", exc)
        return 1

    synthetic = "synthetic" in input_path.name.lower()
    logger.info("AppleSupport author id: %r | input: %s (synthetic=%s)",
                author_id, input_path, synthetic)

    stages: list[dict] = []

    # --------------------------------------------------------------- #
    # 1–2. Load / scan + AppleSupport thread collection               #
    # --------------------------------------------------------------- #
    stage_started = time.time()
    subset, stats, mode = _collect_subset(input_path, author_id, settings,
                                          args.limit, args.sample)
    stages.append({
        "name": "Load raw dataset",
        "status": "OK",
        "detail": (
            f"{stats.total_rows:,} tweets scanned from {input_path.name} "
            f"({input_path.stat().st_size / 1e6:.0f} MB, {mode} mode)"
            if stats else
            f"loaded subset of {len(subset):,} tweets "
            f"({args.limit or args.sample} rows, {mode} mode)"
        ),
    })

    apple_rows = int((subset["author_id"] == author_id).sum())
    if apple_rows == 0:
        top = [f"{a} ({n:,})" for a, n in
               (stats.top_outbound_authors[:8] if stats else [])]
        logger.error(
            "No tweets authored by %r were found in %s. The configured author "
            "id is probably wrong. Frequent outbound authors: %s",
            author_id, input_path, ", ".join(top) or "run scripts/find_author_ids.py",
        )
        return 1
    logger.info("AppleSupport tweets found: %d (across %d collected thread tweets)",
                apple_rows, len(subset))

    schema_ok = stats is None or not stats.missing_by_column
    stages.append({
        "name": "Validate schema",
        "status": "OK" if schema_ok else "WARN",
        "detail": (
            "all 7 expected columns present"
            + (f"; empty values: {stats.missing_by_column}" if stats and stats.missing_by_column else "")
            + (f"; {stats.malformed_inbound:,} malformed inbound flags" if stats and stats.malformed_inbound else "")
        ),
    })

    # --------------------------------------------------------------- #
    # 3. Cleaning                                                     #
    # --------------------------------------------------------------- #
    cleaned, cleaning = preprocess_tweets(subset, strip_leading_mention=True)
    dropped = cleaning.rows_in - cleaning.rows_out
    stages.append({
        "name": "Clean & deduplicate",
        "status": "OK",
        "detail": (
            f"{dropped:,} rows dropped "
            f"(empty text: {cleaning.dropped_empty_text:,}, "
            f"missing ids: {cleaning.dropped_missing_tweet_id + cleaning.dropped_missing_author_id:,}, "
            f"duplicate tweet ids: {cleaning.dropped_duplicate_tweet_id:,}); "
            "URLs / RT prefixes / leading @mentions removed, natural language preserved"
        ),
    })

    # --------------------------------------------------------------- #
    # 4–5. Conversation reconstruction + AppleSupport filter          #
    # --------------------------------------------------------------- #
    conversations, recon_stats = reconstruct_conversations(cleaned)
    stages.append({
        "name": "Reconstruct conversations",
        "status": "OK" if recon_stats.cyclic_tweets_skipped == 0 else "WARN",
        "detail": (
            f"{recon_stats.conversations:,} conversations rebuilt from reply links "
            f"({recon_stats.partial_roots:,} partial roots, "
            f"{recon_stats.cyclic_tweets_skipped:,} cyclic tweets skipped, "
            f"{recon_stats.parent_link_mismatches:,} noisy links reported)")
        ,
    })

    apple_conversations = filter_conversations_with_author(conversations, author_id)
    apple_tweets_frame = tweets_frame_with_conversation(apple_conversations)
    stages.append({
        "name": "Filter AppleSupport",
        "status": "OK",
        "detail": (
            f"{len(apple_conversations):,} of {len(conversations):,} conversations "
            f"include {author_id!r} ({len(apple_tweets_frame):,} tweets kept)"
        ),
    })

    # --------------------------------------------------------------- #
    # 6. Evidence pairs                                               #
    # --------------------------------------------------------------- #
    pairs = extract_support_pairs(apple_conversations, author_id)
    stages.append({
        "name": "Extract support pairs",
        "status": "OK",
        "detail": f"{len(pairs):,} customer↔support pairs extracted (conversation-level ids preserved)",
    })

    # --------------------------------------------------------------- #
    # 7. Validation gate                                              #
    # --------------------------------------------------------------- #
    checks = _validate_outputs(apple_tweets_frame,
                               conversations_frame(apple_conversations),
                               pairs, author_id)
    all_passed = all(c["passed"] for c in checks)
    stages.append({
        "name": "Validate outputs",
        "status": "OK" if all_passed else "WARN",
        "detail": f"{sum(c['passed'] for c in checks)}/{len(checks)} structural checks passed",
    })

    # --------------------------------------------------------------- #
    # 8. Persist                                                      #
    # --------------------------------------------------------------- #
    out_dir = settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tweets_path = out_dir / OUTPUT_TWEETS
    conv_path = out_dir / OUTPUT_CONVERSATIONS
    pairs_path = out_dir / OUTPUT_PAIRS
    run_path = out_dir / OUTPUT_RUN

    apple_tweets_frame.to_csv(tweets_path, index=False)
    conversations_frame(apple_conversations).to_csv(conv_path, index=False)
    pairs.to_csv(pairs_path, index=False)

    run_report = {
        "generated_at": _utc_now(),
        "phase": 1,
        "mode": mode,
        "input": {
            "file": input_path.name,
            "path": str(input_path.relative_to(PROJECT_ROOT))
            if input_path.is_relative_to(PROJECT_ROOT)
            else str(input_path),
            "size_bytes": input_path.stat().st_size,
            "synthetic": synthetic,
        },
        "apple_support_author_id": author_id,
        "settings": {
            "random_seed": settings.random_seed,
            "chunk_size": settings.chunk_size,
            "closure_max_passes": settings.closure_max_passes,
            "note": "Phase 2+ thresholds (confidence/retrieval/top_k) are "
                    "documented in .env.example but not consumed yet.",
        },
        "scan_stats": stats.as_dict() if stats else None,
        "cleaning_report": cleaning.as_dict(),
        "reconstruction_stats": recon_stats.as_dict(),
        "outputs": {
            OUTPUT_TWEETS: {"rows": int(len(apple_tweets_frame)),
                            "path": str(tweets_path)},
            OUTPUT_CONVERSATIONS: {"rows": int(len(apple_conversations)),
                                   "path": str(conv_path)},
            OUTPUT_PAIRS: {"rows": int(len(pairs)), "path": str(pairs_path)},
        },
        "stages": stages,
        "validations": checks,
        "duration_seconds": round(time.time() - started, 1),
    }
    run_path.write_text(json.dumps(run_report, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # --------------------------------------------------------------- #
    # Summary                                                         #
    # --------------------------------------------------------------- #
    logger.info("=" * 72)
    logger.info("PHASE 1 PREPARATION COMPLETE in %.1fs", time.time() - started)
    logger.info("  conversations (all brands scanned): %s",
                f"{stats.total_rows:,} tweets" if stats else "dev subset")
    logger.info("  AppleSupport conversations:        %d", len(apple_conversations))
    logger.info("  AppleSupport thread tweets:        %d", len(apple_tweets_frame))
    logger.info("  customer↔support pairs:            %d", len(pairs))
    logger.info("  validations: %d/%d passed",
                sum(c["passed"] for c in checks), len(checks))
    logger.info("  outputs -> %s", out_dir)
    logger.info("=" * 72)
    if not all_passed:
        logger.error("Validation gate FAILED — inspect pipeline_run.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
