"""Phase 2 — build the golden evaluation set for intent classification.

The golden set is a fixed, manually labelled sample of customer-initiated
conversation openers. It is the ONLY labelled data in the project and is
reserved for evaluation/training of the Phase 2 baselines (and later the
Phase 3 embedding classifier): Phase 4 retrieval must EXCLUDE these
conversations from its evidence index so evaluation never trains on its
own test material (golden-set isolation).

Subcommands
-----------
sample
    Deterministically sample ``--n`` (default 200) conversation openers
    from the Phase 1 artifact ``applesupport_tweets.csv`` (inbound roots
    only) and write the labelling worksheet
    ``evaluation/labeling_worksheet.jsonl``. The worksheet intentionally
    shows NO keyword hints — labels must come from reading the message.

build
    Validate the manual labels in ``evaluation/golden_labels.json``
    (``{"<conversation_id>": "<label>"}`` — written by the human labeler),
    apply the conversation-level stratified 70/15/15 train/val/test split
    (seed 42), and write:
      * ``evaluation/golden_set.csv``       (one row per labelled example)
      * ``evaluation/golden_conversation_ids.json`` (isolation manifest)
      * ``evaluation/split_assignments.csv`` (conversation_id -> split)

Design decisions (recorded in reports/decision_log.md):
  * Uniform random sampling (no rare-class boosting): the golden set must
    estimate performance on the NATURAL intent distribution; rare classes
    having low support is a documented limitation, not a sampling bug.
  * One example per conversation, sampled from openers only — matches the
    agent's real task (classify the first incoming message).
  * Splitting keys on conversation_id (spec: conversation-level split) and
    is stratified by label so every split reflects the class mix.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.intents.taxonomy import INTENT_TAXONOMY, TAXONOMY_LABELS  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("build_golden_set")

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
WORKSHEET = EVAL_DIR / "labeling_worksheet.jsonl"
LABELS_FILE = EVAL_DIR / "golden_labels.json"
GOLDEN_CSV = EVAL_DIR / "golden_set.csv"
SPLIT_CSV = EVAL_DIR / "split_assignments.csv"
ISOLATION_JSON = EVAL_DIR / "golden_conversation_ids.json"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


# --------------------------------------------------------------------------- #
# sample                                                                      #
# --------------------------------------------------------------------------- #
def cmd_sample(n: int) -> None:
    settings = get_settings()
    tweets_path = settings.processed_dir / "applesupport_tweets.csv"
    if not tweets_path.exists():
        sys.exit(
            f"Phase 1 artifact missing: {tweets_path}\n"
            "Run scripts/prepare_data.py first."
        )

    df = pd.read_csv(tweets_path, dtype=str, keep_default_na=False)
    df["turn_i"] = df["turn"].astype(int)
    openers = df[(df["turn_i"] == 1) & (df["inbound"] == "True")].copy()
    frame = len(openers)
    if frame < n:
        sys.exit(f"Sampling frame has only {frame} openers; cannot sample {n}.")

    sampled = openers.sample(n=n, random_state=settings.random_seed)
    # Sort by conversation id for a stable, browsable worksheet.
    sampled = sampled.sort_values("conversation_id")

    EVAL_DIR.mkdir(exist_ok=True)
    with WORKSHEET.open("w", encoding="utf-8") as fh:
        for row in sampled.itertuples():
            fh.write(
                json.dumps(
                    {
                        "conversation_id": row.conversation_id,
                        "tweet_id": row.tweet_id,
                        "created_at": row.created_at,
                        "author_id": row.author_id,
                        "text": row.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (EVAL_DIR / "sampling_frame.json").write_text(
        json.dumps(
            {
                "sampling_frame": "customer-initiated conversation openers (inbound turn-1 roots)",
                "sampling_frame_size": frame,
                "sample_size": n,
                "random_seed": settings.random_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Sampled %d openers from a frame of %d customer-initiated "
        "conversations (seed %d) -> %s",
        n,
        frame,
        settings.random_seed,
        WORKSHEET,
    )
    print(
        f"Worksheet written: {WORKSHEET} ({n} rows)\n"
        f"Next: label each row manually and write {LABELS_FILE} as a JSON "
        f"object mapping conversation_id -> one of {list(TAXONOMY_LABELS)}, "
        f"then run `build`."
    )


# --------------------------------------------------------------------------- #
# build                                                                       #
# --------------------------------------------------------------------------- #
def _stratified_split(labels: pd.Series, seed: int) -> pd.Series:
    """Assign train/val/test per label class, honouring the 70/15/15 ratios.

    Within each class the examples are shuffled deterministically; the
    per-class counts are rounded to the nearest integer that keeps the
    ratios (largest remainder goes to train). Classes with a single
    example go to train.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    split = pd.Series(index=labels.index, dtype="object")
    for label, group_idx in labels.groupby(labels).groups.items():
        idx = list(group_idx)
        rng.shuffle(idx)
        total = len(idx)
        n_val = round(total * SPLIT_RATIOS["val"])
        n_test = round(total * SPLIT_RATIOS["test"])
        n_train = total - n_val - n_test
        if n_train <= 0:  # single-example class -> train
            n_train, n_val, n_test = total, 0, 0
        split.loc[idx[:n_train]] = "train"
        split.loc[idx[n_train : n_train + n_val]] = "val"
        split.loc[idx[n_train + n_val :]] = "test"
    return split


def cmd_build() -> None:
    settings = get_settings()
    if not WORKSHEET.exists():
        sys.exit(f"Worksheet missing: {WORKSHEET}. Run `sample` first.")
    if not LABELS_FILE.exists():
        sys.exit(
            f"Manual labels missing: {LABELS_FILE}.\n"
            "Label every worksheet row by reading the message and write a "
            "JSON object {conversation_id: label}."
        )

    worksheet = [
        json.loads(line)
        for line in WORKSHEET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manual = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    # --- Validation gate (golden-set integrity) -------------------------- #
    errors: list[str] = []
    ws_ids = [w["conversation_id"] for w in worksheet]
    if len(ws_ids) != len(set(ws_ids)):
        errors.append("worksheet contains duplicate conversation ids")
    missing = [cid for cid in ws_ids if cid not in manual]
    if missing:
        errors.append(f"{len(missing)} worksheet rows have no label (e.g. {missing[:5]})")
    extra = [cid for cid in manual if cid not in set(ws_ids)]
    if extra:
        errors.append(f"{len(extra)} labels reference unknown ids (e.g. {extra[:5]})")
    invalid = {
        cid: label
        for cid, label in manual.items()
        if cid in set(ws_ids) and label not in TAXONOMY_LABELS
    }
    if invalid:
        errors.append(
            f"{len(invalid)} labels are not taxonomy classes "
            f"(e.g. {list(invalid.items())[:3]})"
        )
    if errors:
        sys.exit("GOLDEN SET VALIDATION FAILED:\n  - " + "\n  - ".join(errors))

    rows = []
    for w in worksheet:
        rows.append(
            {
                "conversation_id": w["conversation_id"],
                "tweet_id": w["tweet_id"],
                "created_at": w["created_at"],
                "author_id": w["author_id"],
                "text": w["text"],
                "label": manual[w["conversation_id"]],
            }
        )
    df = pd.DataFrame(rows)
    df["split"] = _stratified_split(df["label"], settings.random_seed)

    df.to_csv(GOLDEN_CSV, index=False)
    split_counts = df["split"].value_counts().to_dict()
    pd.DataFrame(
        {"conversation_id": df["conversation_id"], "split": df["split"]}
    ).to_csv(SPLIT_CSV, index=False)
    ISOLATION_JSON.write_text(
        json.dumps(
            {
                "purpose": (
                    "Golden-set isolation manifest: Phase 4 retrieval and any "
                    "future training corpus must exclude these conversation "
                    "ids from evidence indexes."
                ),
                "conversation_ids": sorted(df["conversation_id"].tolist()),
                "count": len(df),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Golden set: %d examples across %d classes", len(df), df["label"].nunique())
    for label in TAXONOMY_LABELS:
        sub = df[df["label"] == label]
        if len(sub):
            logger.info(
                "  %-18s total %3d  (train %d / val %d / test %d)",
                label,
                len(sub),
                (sub["split"] == "train").sum(),
                (sub["split"] == "val").sum(),
                (sub["split"] == "test").sum(),
            )
    logger.info(
        "Splits: train %d / val %d / test %d -> %s, %s, %s",
        split_counts.get("train", 0),
        split_counts.get("val", 0),
        split_counts.get("test", 0),
        GOLDEN_CSV,
        SPLIT_CSV,
        ISOLATION_JSON,
    )
    print(
        "GOLDEN SET BUILD PASSED — "
        f"{len(df)} examples, splits {split_counts.get('train', 0)}/"
        f"{split_counts.get('val', 0)}/{split_counts.get('test', 0)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_sample = sub.add_parser("sample", help="draw the deterministic sample")
    p_sample.add_argument("--n", type=int, default=200)
    sub.add_parser("build", help="validate labels, split, and write artifacts")
    args = parser.parse_args()

    if args.command == "sample":
        cmd_sample(args.n)
    elif args.command == "build":
        cmd_build()


if __name__ == "__main__":
    main()
