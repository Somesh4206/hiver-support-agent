"""Discover candidate support-account author ids in twcs.csv.

The Customer Support on Twitter dataset has no brand column. For outbound
(support) tweets the ``author_id`` field contains the account's screen name
(e.g. ``AppleSupport``, ``AmazonHelp``), while inbound (customer) tweets use
numeric ids. This script lists the most frequent outbound author ids so the
Apple support account can be identified from the data itself and configured
via ``APPLE_SUPPORT_AUTHOR_ID`` — it is never hard-coded or guessed.

Usage (from the repository root, with data/raw/twcs.csv present):

    python scripts/find_author_ids.py [--input PATH] [--top N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_RAW_CSV, get_settings  # noqa: E402
from src.data.load import scan_dataset_stats  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("find_author_ids")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List the most frequent outbound author_ids in twcs.csv."
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to twcs.csv (default: data/raw/twcs.csv or RAW_DATA_PATH)",
    )
    parser.add_argument("--top", type=int, default=25,
                        help="How many author ids to show (default: 25)")
    args = parser.parse_args()

    settings = get_settings()
    path = Path(args.input) if args.input else settings.raw_data_path
    if not path.is_file() and args.input is None:
        path = DEFAULT_RAW_CSV
    if not path.is_file():
        logger.error(
            "twcs.csv not found at %s — see data/README.md for download "
            "instructions", path,
        )
        return 1

    stats = scan_dataset_stats(path, chunk_size=settings.chunk_size)

    print("\n=== Outbound (support) accounts in twcs.csv, by tweet volume ===\n")
    print(f"{'rank':>4}  {'author_id':<28} {'tweets':>10}")
    print("-" * 46)
    for rank, (author, count) in enumerate(stats.top_outbound_authors[: args.top], 1):
        print(f"{rank:>4}  {author:<28} {count:>10,}")
    print("-" * 46)
    print(f"Dataset: {stats.total_rows:,} tweets "
          f"({stats.inbound_rows:,} inbound / {stats.outbound_rows:,} outbound)\n")
    print("Next steps:")
    print("  1. Identify the Apple support account in the list above.")
    print("  2. Copy .env.example to .env and set:")
    print("       APPLE_SUPPORT_AUTHOR_ID=<author_id from the list>")
    print("  3. Run: python scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
