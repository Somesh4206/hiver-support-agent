"""Phase 2 — train and evaluate the intent-classification baselines.

Evaluates, on the golden set (``evaluation/golden_set.csv``):

1. ``keyword_rules`` — Phase 1 EDA keyword topics mapped to taxonomy labels
   (word-boundary matching, first match in EDA dict order wins, no match
   -> ``other``). Quantifies the gap between transparent heuristics and
   the taxonomy.
2. ``majority`` — always predict the TRAIN split's majority class.
3. ``tfidf_logreg`` — TF-IDF (word 1–2 grams, English stopwords) +
   LogisticRegression, fit on TRAIN, with stratified CV on TRAIN for
   stability, evaluated on VAL (sanity) and TEST (headline).

Writes:
  * ``evaluation/results/baseline_results.json``  (full metrics, all models)
  * ``evaluation/results/phase2_summary.json``    (dashboard-ready summary:
    taxonomy + golden-set stats + baseline metrics + provenance)

Every number in the outputs is measured; nothing is estimated or fabricated.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.intents.baselines import (  # noqa: E402
    build_tfidf_logreg,
    classification_report_dict,
    keyword_rule_predict,
    majority_predict,
    stratified_cv_macro_f1,
)
from src.intents.taxonomy import INTENT_TAXONOMY, TAXONOMY_LABELS  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("train_baselines")

EVAL_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
GOLDEN_CSV = EVAL_DIR / "golden_set.csv"

# Import the exact Phase 1 keyword definitions (single source of truth).
from run_eda import PROVISIONAL_TOPICS  # noqa: E402


def main() -> None:
    settings = get_settings()
    if not GOLDEN_CSV.exists():
        sys.exit(
            f"Golden set missing: {GOLDEN_CSV}\n"
            "Run `python scripts/build_golden_set.py sample` and `build` first."
        )
    df = pd.read_csv(GOLDEN_CSV, dtype=str, keep_default_na=False)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    logger.info(
        "Golden set loaded: %d examples (train %d / val %d / test %d)",
        len(df), len(train), len(val), len(test),
    )

    results: dict = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                     "random_seed": settings.random_seed}

    # ------------------------------------------------------------------ #
    # 1. Keyword rules (Phase 1 heuristics mapped to taxonomy labels)    #
    # ------------------------------------------------------------------ #
    kw_pred_test = keyword_rule_predict(test["text"], PROVISIONAL_TOPICS)
    kw_report_test = classification_report_dict(test["label"], kw_pred_test)
    kw_pred_train = keyword_rule_predict(train["text"], PROVISIONAL_TOPICS)
    kw_report_train = classification_report_dict(train["label"], kw_pred_train)
    results["keyword_rules"] = {
        "description": (
            "Phase 1 EDA provisional keyword topics mapped onto taxonomy "
            "labels; word-boundary matching, first match in EDA topic "
            "order wins, no match -> other."
        ),
        "train": kw_report_train,
        "test": kw_report_test,
    }
    logger.info(
        "keyword_rules  TEST accuracy=%.4f macro_f1=%.4f",
        kw_report_test["accuracy"], kw_report_test["macro_f1"],
    )

    # ------------------------------------------------------------------ #
    # 2. Majority class                                                  #
    # ------------------------------------------------------------------ #
    majority_class = train["label"].mode().iloc[0]
    maj_report_test = classification_report_dict(
        test["label"], majority_predict(train["label"], len(test))
    )
    maj_report_val = classification_report_dict(
        val["label"], majority_predict(train["label"], len(val))
    )
    results["majority"] = {
        "description": f"Always predict the train majority class ({majority_class}).",
        "majority_class": majority_class,
        "val": maj_report_val,
        "test": maj_report_test,
    }
    logger.info(
        "majority(%s) TEST accuracy=%.4f macro_f1=%.4f",
        majority_class, maj_report_test["accuracy"], maj_report_test["macro_f1"],
    )

    # ------------------------------------------------------------------ #
    # 3. TF-IDF + LogisticRegression                                     #
    #                                                                    #
    # Hyperparameters selected by 5-fold CV ON THE TRAIN SPLIT with the  #
    # composite criterion mean(accuracy, macro-F1) — both metrics matter #
    # for the assignment (intent accuracy headline + per-class fairness) #
    # and neither alone picks a useful model here: accuracy-only picks a #
    # majority-collapser, macro-only picks an accuracy-collapser. The    #
    # classic alternative (tune on val) is too noisy at n_val=29; CV on  #
    # train uses ~5x more data for selection. The VAL split is retained  #
    # as a held-out sanity check and TEST is untouched until the final   #
    # evaluation. An unregularized, unweighted fit memorises the         #
    # 142-example train split (train accuracy 1.0) and collapses to the  #
    # majority class — measured, and the reason this grid exists.        #
    # ------------------------------------------------------------------ #
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    grid: list[tuple[float, str | None, int]] = [
        (c, cw, md)
        for c in (0.01, 0.1, 1.0, 10.0)
        for cw in (None, "balanced")
        for md in (1, 2)
    ]
    cv_selector = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.random_seed)
    selection: list[dict] = []
    for c, cw, md in grid:
        cand = build_tfidf_logreg(settings.random_seed, c=c, class_weight=cw, min_df=md)
        oof_pred = cross_val_predict(cand, train["text"], train["label"], cv=cv_selector)
        cv_acc = round(float(accuracy_score(train["label"], oof_pred)), 4)
        cv_macro = round(
            float(f1_score(train["label"], oof_pred, labels=list(TAXONOMY_LABELS), average="macro", zero_division=0)), 4
        )
        selection.append(
            {
                "c": c,
                "class_weight": cw,
                "min_df": md,
                "cv_train_accuracy": cv_acc,
                "cv_train_macro_f1": cv_macro,
                "cv_train_mean_score": round((cv_acc + cv_macro) / 2, 4),
            }
        )
    # Selection criterion: CV-on-train mean(accuracy, macro-F1);
    # ties -> higher macro-F1.
    best = sorted(selection, key=lambda s: (s["cv_train_mean_score"], s["cv_train_macro_f1"]))[-1]
    logger.info(
        "Hyperparameter selection by 5-fold CV on train (16 configs): best C=%s "
        "class_weight=%s min_df=%s -> cv acc %.4f, cv macro_f1 %.4f",
        best["c"], best["class_weight"], best["min_df"],
        best["cv_train_accuracy"], best["cv_train_macro_f1"],
    )

    pipeline = build_tfidf_logreg(
        settings.random_seed,
        c=best["c"],
        class_weight=best["class_weight"],
        min_df=best["min_df"],
    )
    cv = stratified_cv_macro_f1(pipeline, train["text"], train["label"], settings.random_seed)
    pipeline.fit(train["text"], train["label"])

    lr_train = classification_report_dict(train["label"], pipeline.predict(train["text"]))
    lr_val = classification_report_dict(val["label"], pipeline.predict(val["text"]))
    lr_test = classification_report_dict(test["label"], pipeline.predict(test["text"]))
    results["tfidf_logreg"] = {
        "description": (
            "TfidfVectorizer(word 1-2 grams, English stopwords, sublinear tf) "
            "+ LogisticRegression(max_iter=2000), fit on the train split; "
            "hyperparameters selected by 5-fold CV on the train split "
            "(criterion: mean of accuracy and macro-F1)."
        ),
        "selected_hyperparameters": {
            "C": best["c"],
            "class_weight": best["class_weight"],
            "min_df": best["min_df"],
        },
        "selection_criterion": (
            "5-fold CV on train: mean(accuracy, macro-F1); ties -> macro-F1. "
            "Validation is too small (n=29) for stable tuning and is kept "
            "as a held-out sanity check; test untouched until final evaluation."
        ),
        "selection_grid": selection,
        "cv_on_train": cv,
        "train": lr_train,
        "val": lr_val,
        "test": lr_test,
    }
    logger.info(
        "tfidf_logreg  TEST accuracy=%.4f macro_f1=%.4f (train acc %.4f, CV macro_f1 %.3f±%.3f)",
        lr_test["accuracy"], lr_test["macro_f1"], lr_train["accuracy"],
        cv.get("macro_f1_mean", float("nan")), cv.get("macro_f1_std", float("nan")),
    )

    # ------------------------------------------------------------------ #
    # Deltas (test, headline)                                            #
    # ------------------------------------------------------------------ #
    results["deltas_test"] = {
        "logreg_vs_majority_accuracy": round(
            lr_test["accuracy"] - maj_report_test["accuracy"], 4
        ),
        "logreg_vs_majority_macro_f1": round(
            lr_test["macro_f1"] - maj_report_test["macro_f1"], 4
        ),
        "logreg_vs_keywords_accuracy": round(
            lr_test["accuracy"] - kw_report_test["accuracy"], 4
        ),
        "logreg_vs_keywords_macro_f1": round(
            lr_test["macro_f1"] - kw_report_test["macro_f1"], 4
        ),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "baseline_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", RESULTS_DIR / "baseline_results.json")

    # ------------------------------------------------------------------ #
    # Dashboard-ready summary                                            #
    # ------------------------------------------------------------------ #
    frame_meta_path = EVAL_DIR / "sampling_frame.json"
    frame_size = None
    if frame_meta_path.exists():
        frame_size = json.loads(frame_meta_path.read_text(encoding="utf-8")).get(
            "sampling_frame_size"
        )
    class_counts = df["label"].value_counts().to_dict()
    per_class_splits = {
        label: {
            "total": int((df["label"] == label).sum()),
            "train": int(((df["label"] == label) & (df["split"] == "train")).sum()),
            "val": int(((df["label"] == label) & (df["split"] == "val")).sum()),
            "test": int(((df["label"] == label) & (df["split"] == "test")).sum()),
        }
        for label in [c.label for c in INTENT_TAXONOMY]
    }
    dates = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )
    summary = {
        "phase": 2,
        "generated_at": results["generated_at"],
        "random_seed": settings.random_seed,
        "taxonomy": [
            {
                "label": c.label,
                "name": c.name,
                "description": c.description,
                "eda_anchor": c.eda_anchor,
                "signals": list(c.signals),
                "escalation_hint": c.escalation_hint,
                "typically_escalated": c.typically_escalated,
                "examples": list(c.examples),
            }
            for c in INTENT_TAXONOMY
        ],
        "golden_set": {
            "total": len(df),
            "sampling_frame": "customer-initiated conversation openers (inbound roots)",
            "sampling_frame_size": frame_size,
            "sampling": "uniform random sample, no rare-class boosting",
            "labeling": {
                "method": "manual",
                "labeler": "project author — every message read and labelled by meaning; "
                "the worksheet shows no keyword hints to avoid biasing labels",
                "worksheet": "evaluation/labeling_worksheet.jsonl",
                "labels_file": "evaluation/golden_labels.json",
                "validation": (
                    "automated gate: every worksheet row labelled exactly once, "
                    "labels restricted to taxonomy classes, no unknown ids "
                    "(build exits non-zero on failure)"
                ),
            },
            "class_counts": class_counts,
            "split_sizes": {
                "train": len(train), "val": len(val), "test": len(test)
            },
            "split_strategy": (
                "conversation-level, stratified by label, seed 42 "
                "(ratios 70/15/15, per-class rounding to train)"
            ),
            "per_class_splits": per_class_splits,
            "date_range": {
                "start": dates.min().strftime("%Y-%m-%d") if dates.notna().any() else None,
                "end": dates.max().strftime("%Y-%m-%d") if dates.notna().any() else None,
            },
            "period_note": (
                "99.5% of the 80,244 openers fall in Sep–Dec 2017 (the iOS 11 "
                "rollout window of the corpus), so the golden set — like the "
                "corpus itself — is dominated by iOS 11 era complaints. "
                "Metrics measure this historical window, not a balanced mix."
            ),
            "isolation_manifest": "evaluation/golden_conversation_ids.json",
        },
        "baselines": {
            "keyword_rules": {
                "description": results["keyword_rules"]["description"],
                "test": {
                    "accuracy": kw_report_test["accuracy"],
                    "macro_f1": kw_report_test["macro_f1"],
                    "weighted_f1": kw_report_test["weighted_f1"],
                },
            },
            "majority": {
                "description": results["majority"]["description"],
                "test": {
                    "accuracy": maj_report_test["accuracy"],
                    "macro_f1": maj_report_test["macro_f1"],
                    "weighted_f1": maj_report_test["weighted_f1"],
                },
            },
            "tfidf_logreg": {
                "description": results["tfidf_logreg"]["description"],
                "selected_hyperparameters": results["tfidf_logreg"]["selected_hyperparameters"],
                "selection_criterion": results["tfidf_logreg"]["selection_criterion"],
                "cv_on_train": cv,
                "val": {
                    "accuracy": lr_val["accuracy"],
                    "macro_f1": lr_val["macro_f1"],
                    "weighted_f1": lr_val["weighted_f1"],
                },
                "test": {
                    "accuracy": lr_test["accuracy"],
                    "macro_f1": lr_test["macro_f1"],
                    "weighted_f1": lr_test["weighted_f1"],
                },
                "per_class_test": lr_test["per_class"],
                "confusion_matrix_test": lr_test["confusion_matrix"],
            },
            "deltas_test": results["deltas_test"],
        },
        "artifacts": [
            "evaluation/golden_set.csv",
            "evaluation/golden_labels.json",
            "evaluation/labeling_worksheet.jsonl",
            "evaluation/split_assignments.csv",
            "evaluation/golden_conversation_ids.json",
            "evaluation/results/baseline_results.json",
            "evaluation/results/phase2_summary.json",
        ],
    }
    (RESULTS_DIR / "phase2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", RESULTS_DIR / "phase2_summary.json")
    print(
        "PHASE 2 BASELINES COMPLETE — "
        f"test accuracy: keyword {kw_report_test['accuracy']:.3f} / "
        f"majority {maj_report_test['accuracy']:.3f} / "
        f"tfidf-logreg {lr_test['accuracy']:.3f}; "
        f"test macro-F1: {lr_test['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
