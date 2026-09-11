"""Phase 3 — train and evaluate the embedding-based intent classifier.

The designed remedy for the Phase 2 finding (lexical TF-IDF features cannot
separate 10 classes from 142 train examples: test macro-F1 0.078): a frozen
``all-MiniLM-L6-v2`` sentence encoder + small learned heads.

Evaluated on the SAME golden set with the SAME protocol as Phase 2
(``scripts/train_baselines.py``):

1. ``logreg_head`` — LogisticRegression over the 384-d embeddings.
   Hyperparameters (C x class_weight) selected by 5-fold stratified CV ON
   THE TRAIN SPLIT with the composite criterion mean(accuracy, macro-F1)
   — identical to Phase 2, so the comparison isolates the *feature
   representation*, not the tuning protocol. VAL stays a held-out sanity
   check; TEST untouched until the final evaluation.
2. ``prototype`` — per-class centroid classifier with cosine confidence.
   Zero hyperparameters, the natural fallback for rare classes, and the
   confidence mechanism Phase 6 can threshold.

Also measured:
* rare-class handling — class_weight="balanced" (the linear-head equivalent
  of over-sampling rare classes) vs None, reported as a comparison, not an
  assumption;
* test-set error analysis with per-example confidence;
* a first confidence-threshold preview at the configured
  ``INTENT_CONFIDENCE_THRESHOLD`` (Phase 6 motivation, measured on test).

Writes:
  * ``evaluation/results/embedding_results.json``  (full metrics, both heads)
  * ``evaluation/results/phase3_summary.json``     (dashboard-ready summary)
  * ``models/embedding_logreg_head.joblib`` + ``.json`` metadata sidecar
  * ``models/embedding_prototype.joblib``          (for later phases)

Every number in the outputs is measured; nothing is estimated or fabricated.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.config import get_settings  # noqa: E402
from src.intents.baselines import classification_report_dict  # noqa: E402
from src.intents.embeddings import embed_texts_cached  # noqa: E402
from src.intents.embedding_classifier import (  # noqa: E402
    EmbeddingLogReg,
    PrototypeClassifier,
)
from src.intents.taxonomy import TAXONOMY_LABELS  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("train_embedding_classifier")

EVAL_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
GOLDEN_CSV = EVAL_DIR / "golden_set.csv"
BASELINE_JSON = RESULTS_DIR / "baseline_results.json"
MODELS_DIR = PROJECT_ROOT / "models"
EMBED_CACHE = MODELS_DIR / "embedding_cache"


def _report_from_predictions(
    y_true: pd.Series, y_pred: list[str]
) -> dict:
    return classification_report_dict(y_true, y_pred)


def _cv_select_logreg(
    train_vec: np.ndarray, train_labels: pd.Series, seed: int
) -> tuple[dict, list[dict]]:
    """Grid + 5-fold CV on train, composite criterion mean(acc, macro-F1)."""
    grid: list[tuple[float, str | None]] = [
        (c, cw) for c in (0.1, 1.0, 10.0, 100.0) for cw in (None, "balanced")
    ]
    cv_selector = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed
    )
    selection: list[dict] = []
    for c, cw in grid:
        head = EmbeddingLogReg(seed, c=c, class_weight=cw)
        oof_pred = cross_val_predict(
            head.model, train_vec, train_labels, cv=cv_selector
        )
        cv_acc = round(float(accuracy_score(train_labels, oof_pred)), 4)
        cv_macro = round(
            float(
                f1_score(
                    train_labels, oof_pred,
                    labels=list(TAXONOMY_LABELS),
                    average="macro", zero_division=0,
                )
            ),
            4,
        )
        selection.append(
            {
                "C": c,
                "class_weight": cw,
                "cv_train_accuracy": cv_acc,
                "cv_train_macro_f1": cv_macro,
                "cv_train_mean_score": round((cv_acc + cv_macro) / 2, 4),
            }
        )
    best = sorted(
        selection,
        key=lambda s: (s["cv_train_mean_score"], s["cv_train_macro_f1"]),
    )[-1]
    return best, selection


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

    # ------------------------------------------------------------------ #
    # 0. Embed (frozen public encoder; on-disk cache for reproducibility) #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    train_vec, _ = embed_texts_cached(train["text"], EMBED_CACHE, tag="golden_train")
    val_vec, _ = embed_texts_cached(val["text"], EMBED_CACHE, tag="golden_val")
    test_vec, _ = embed_texts_cached(test["text"], EMBED_CACHE, tag="golden_test")
    embed_seconds = round(time.perf_counter() - t0, 2)
    dim = int(train_vec.shape[1])
    logger.info(
        "Embeddings ready: train %s / val %s / test %s, dim=%d (%.1fs incl. cache read)",
        train_vec.shape, val_vec.shape, test_vec.shape, dim, embed_seconds,
    )

    results: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "random_seed": settings.random_seed,
        "embedding_model": settings.embedding_model,
        "embedding_dim": dim,
        "protocol": (
            "Frozen public encoder (no fine-tuning); both heads fit on the "
            "TRAIN split only; logreg_head hyperparameters selected by 5-fold "
            "stratified CV on train with criterion mean(accuracy, macro-F1) "
            "(identical to Phase 2 so the comparison isolates the feature "
            "representation); VAL is a held-out sanity check; TEST untouched "
            "until final evaluation."
        ),
    }

    # ------------------------------------------------------------------ #
    # 1. logreg_head — CV-on-train hyperparameter selection              #
    # ------------------------------------------------------------------ #
    best, selection = _cv_select_logreg(train_vec, train["label"], settings.random_seed)
    logger.info(
        "Hyperparameter selection by 5-fold CV on train (8 configs): best C=%s "
        "class_weight=%s -> cv acc %.4f, cv macro_f1 %.4f",
        best["C"], best["class_weight"],
        best["cv_train_accuracy"], best["cv_train_macro_f1"],
    )

    head = EmbeddingLogReg(
        settings.random_seed, c=best["C"], class_weight=best["class_weight"]
    )
    head.fit(train_vec, list(train["label"]))
    lr_train = _report_from_predictions(train["label"], head.predict(train_vec))
    lr_val = _report_from_predictions(val["label"], head.predict(val_vec))
    lr_test = _report_from_predictions(test["label"], head.predict(test_vec))

    # Rare-class over-sampling consideration: class_weight='balanced' is the
    # linear-head equivalent of over-sampling rare classes (same gradients,
    # reweighted). The grid above already covers both settings under the
    # SAME CV criterion — extract and compare the best of each, honestly.
    balanced_rows = [s for s in selection if s["class_weight"] == "balanced"]
    none_rows = [s for s in selection if s["class_weight"] is None]
    best_balanced = max(balanced_rows, key=lambda s: s["cv_train_mean_score"])
    best_none = max(none_rows, key=lambda s: s["cv_train_mean_score"])
    oversampling_note = {
        "question": (
            "Does rare-class over-sampling (class_weight='balanced' — the "
            "linear-head equivalent of duplicating rare examples) help at "
            "n_train=142 with a frozen encoder?"
        ),
        "best_unweighted_cv": {
            "C": best_none["C"],
            "cv_train_accuracy": best_none["cv_train_accuracy"],
            "cv_train_macro_f1": best_none["cv_train_macro_f1"],
            "cv_train_mean_score": best_none["cv_train_mean_score"],
        },
        "best_balanced_cv": {
            "C": best_balanced["C"],
            "cv_train_accuracy": best_balanced["cv_train_accuracy"],
            "cv_train_macro_f1": best_balanced["cv_train_macro_f1"],
            "cv_train_mean_score": best_balanced["cv_train_mean_score"],
        },
        "verdict": (
            "Selected by the same CV criterion — measurement, not assumption "
            "(see selection_grid for every candidate)."
        ),
    }

    results["logreg_head"] = {
        "description": (
            "LogisticRegression (multinomial, max_iter=5000) over frozen "
            f"{settings.embedding_model} {dim}-d embeddings; fit on the train "
            "split; hyperparameters selected by 5-fold CV on the train split "
            "(criterion: mean of accuracy and macro-F1 — identical protocol "
            "to the Phase 2 tfidf_logreg baseline)."
        ),
        "selected_hyperparameters": {
            "C": best["C"],
            "class_weight": best["class_weight"],
        },
        "selection_criterion": (
            "5-fold CV on train: mean(accuracy, macro-F1); ties -> macro-F1. "
            "Same criterion as Phase 2; VAL (n=29) kept as held-out sanity "
            "check; TEST untouched until final evaluation."
        ),
        "selection_grid": selection,
        "train": lr_train,
        "val": lr_val,
        "test": lr_test,
    }
    logger.info(
        "logreg_head  TEST accuracy=%.4f macro_f1=%.4f (train acc %.4f; val acc %.4f)",
        lr_test["accuracy"], lr_test["macro_f1"],
        lr_train["accuracy"], lr_val["accuracy"],
    )

    # ------------------------------------------------------------------ #
    # 2. prototype — centroid classifier, zero hyperparameters            #
    # ------------------------------------------------------------------ #
    proto = PrototypeClassifier(settings.random_seed)
    proto.fit(train_vec, list(train["label"]))
    proto_train = _report_from_predictions(train["label"], proto.predict(train_vec))
    proto_val = _report_from_predictions(val["label"], proto.predict(val_vec))
    proto_test = _report_from_predictions(test["label"], proto.predict(test_vec))
    results["prototype"] = {
        "description": (
            "Nearest-class-centroid classifier: L2-normalised mean embedding "
            "per class (fit on the train split), cosine similarity at "
            "inference, softmax over similarities as confidence. Zero "
            "hyperparameters; naturally handles classes with 2 examples; the "
            "confidence surface Phase 6 can threshold."
        ),
        "train": proto_train,
        "val": proto_val,
        "test": proto_test,
    }
    logger.info(
        "prototype    TEST accuracy=%.4f macro_f1=%.4f (val acc %.4f)",
        proto_test["accuracy"], proto_test["macro_f1"], proto_val["accuracy"],
    )

    # ------------------------------------------------------------------ #
    # 3. Head-to-head vs Phase 2 baselines on the SAME test split         #
    # ------------------------------------------------------------------ #
    if not BASELINE_JSON.exists():
        sys.exit(
            f"Phase 2 results missing: {BASELINE_JSON}\n"
            "Run `python scripts/train_baselines.py` first."
        )
    phase2 = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    comparison_test = {
        "keyword_rules": phase2["keyword_rules"]["test"],
        "majority": phase2["majority"]["test"],
        "tfidf_logreg": phase2["tfidf_logreg"]["test"],
        "logreg_head": {
            "accuracy": lr_test["accuracy"],
            "macro_f1": lr_test["macro_f1"],
            "weighted_f1": lr_test["weighted_f1"],
        },
        "prototype": {
            "accuracy": proto_test["accuracy"],
            "macro_f1": proto_test["macro_f1"],
            "weighted_f1": proto_test["weighted_f1"],
        },
    }
    results["comparison_test"] = comparison_test
    results["deltas_test"] = {
        "logreg_head_vs_tfidf_logreg": {
            "accuracy": round(
                lr_test["accuracy"] - phase2["tfidf_logreg"]["test"]["accuracy"], 4
            ),
            "macro_f1": round(
                lr_test["macro_f1"] - phase2["tfidf_logreg"]["test"]["macro_f1"], 4
            ),
        },
        "logreg_head_vs_majority": {
            "accuracy": round(
                lr_test["accuracy"] - phase2["majority"]["test"]["accuracy"], 4
            ),
            "macro_f1": round(
                lr_test["macro_f1"] - phase2["majority"]["test"]["macro_f1"], 4
            ),
        },
        "logreg_head_vs_keywords": {
            "accuracy": round(
                lr_test["accuracy"] - phase2["keyword_rules"]["test"]["accuracy"], 4
            ),
            "macro_f1": round(
                lr_test["macro_f1"] - phase2["keyword_rules"]["test"]["macro_f1"], 4
            ),
        },
        "prototype_vs_majority": {
            "accuracy": round(
                proto_test["accuracy"] - phase2["majority"]["test"]["accuracy"], 4
            ),
            "macro_f1": round(
                proto_test["macro_f1"] - phase2["majority"]["test"]["macro_f1"], 4
            ),
        },
    }

    # ------------------------------------------------------------------ #
    # 4. Confidence surface + error analysis on TEST (Phase 6 preview)    #
    # ------------------------------------------------------------------ #
    confidences = head.predict_confidence(test_vec)
    correct_mask = np.asarray(
        [c["label"] == t for c, t in zip(confidences, test["label"])]
    )
    mean_conf_correct = float(
        np.mean([c["confidence"] for c, m in zip(confidences, correct_mask) if m])
    ) if correct_mask.any() else None
    mean_conf_wrong = float(
        np.mean([c["confidence"] for c, m in zip(confidences, correct_mask) if not m])
    ) if (~correct_mask).any() else None
    threshold = settings.intent_confidence_threshold
    auto_mask = np.asarray([c["confidence"] >= threshold for c in confidences])
    auto_accuracy = (
        float(correct_mask[auto_mask].mean()) if auto_mask.any() else None
    )
    results["confidence_preview_test"] = {
        "threshold": threshold,
        "note": (
            "Measured on the 29-example test split with the selected "
            "logreg_head — a preview of the Phase 6 escalation input, not a "
            "tuned policy. Numbers at n=29 are indicative only."
        ),
        "mean_confidence_correct": round(mean_conf_correct, 4) if mean_conf_correct is not None else None,
        "mean_confidence_wrong": round(mean_conf_wrong, 4) if mean_conf_wrong is not None else None,
        "share_above_threshold": round(float(auto_mask.mean()), 4),
        "accuracy_above_threshold": round(auto_accuracy, 4) if auto_accuracy is not None else None,
    }

    errors: list[dict] = []
    for row, conf, ok in zip(test.itertuples(), confidences, correct_mask):
        if ok:
            continue
        text = str(row.text)
        errors.append(
            {
                "conversation_id": row.conversation_id,
                "text": text[:160] + ("…" if len(text) > 160 else ""),
                "true_label": row.label,
                "predicted_label": conf["label"],
                "confidence": conf["confidence"],
                "runner_up": conf["runner_up"],
                "runner_up_confidence": conf["runner_up_confidence"],
            }
        )
    results["error_analysis_test"] = {
        "n_test": len(test),
        "n_errors": int((~correct_mask).sum()),
        "errors": errors,
    }

    # ------------------------------------------------------------------ #
    # 5. Persist artifacts                                                #
    # ------------------------------------------------------------------ #
    MODELS_DIR.mkdir(exist_ok=True)
    head_path = MODELS_DIR / "embedding_logreg_head.joblib"
    head.save(head_path, settings.embedding_model)
    head_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "logreg_head",
                "embedding_model": settings.embedding_model,
                "embedding_dim": dim,
                "classes": head.classes_,
                "hyperparameters": {
                    "C": best["C"], "class_weight": best["class_weight"]
                },
                "test_metrics": {
                    "accuracy": lr_test["accuracy"],
                    "macro_f1": lr_test["macro_f1"],
                },
                "created_at": results["generated_at"],
                "regenerate": "python scripts/train_embedding_classifier.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    proto_path = MODELS_DIR / "embedding_prototype.joblib"
    proto.save(proto_path, settings.embedding_model)
    results["artifacts"] = [
        "models/embedding_logreg_head.joblib",
        "models/embedding_logreg_head.json",
        "models/embedding_prototype.joblib",
        "models/embedding_cache/",
        "evaluation/results/embedding_results.json",
        "evaluation/results/phase3_summary.json",
    ]

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "embedding_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", RESULTS_DIR / "embedding_results.json")

    # ------------------------------------------------------------------ #
    # 6. Dashboard-ready summary                                          #
    # ------------------------------------------------------------------ #
    summary = {
        "phase": 3,
        "generated_at": results["generated_at"],
        "random_seed": settings.random_seed,
        "embedding_model": {
            "id": settings.embedding_model,
            "dim": dim,
            "parameters": "22.3M (frozen; only the head is trained)",
            "trained_on": "external public corpus (not this dataset) — no golden leakage",
            "runtime": "CPU, float32",
            "embedding_seconds_for_golden_set": embed_seconds,
        },
        "protocol": results["protocol"],
        "heads": {
            "logreg_head": {
                "description": results["logreg_head"]["description"],
                "selected_hyperparameters": results["logreg_head"]["selected_hyperparameters"],
                "selection_criterion": results["logreg_head"]["selection_criterion"],
                "selection_grid": selection,
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
            "prototype": {
                "description": results["prototype"]["description"],
                "val": {
                    "accuracy": proto_val["accuracy"],
                    "macro_f1": proto_val["macro_f1"],
                    "weighted_f1": proto_val["weighted_f1"],
                },
                "test": {
                    "accuracy": proto_test["accuracy"],
                    "macro_f1": proto_test["macro_f1"],
                    "weighted_f1": proto_test["weighted_f1"],
                },
                "per_class_test": proto_test["per_class"],
                "confusion_matrix_test": proto_test["confusion_matrix"],
            },
        },
        "comparison_test": comparison_test,
        "deltas_test": results["deltas_test"],
        "oversampling_note": oversampling_note,
        "confidence_preview_test": results["confidence_preview_test"],
        "error_analysis_test": results["error_analysis_test"],
        "artifacts": results["artifacts"],
    }
    (RESULTS_DIR / "phase3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", RESULTS_DIR / "phase3_summary.json")

    print(
        "PHASE 3 EMBEDDING CLASSIFIER COMPLETE — "
        f"test accuracy: logreg_head {lr_test['accuracy']:.3f} / "
        f"prototype {proto_test['accuracy']:.3f} "
        f"(Phase 2: tfidf {comparison_test['tfidf_logreg']['accuracy']:.3f}, "
        f"majority {comparison_test['majority']['accuracy']:.3f}); "
        f"test macro-F1: logreg_head {lr_test['macro_f1']:.3f} / "
        f"prototype {proto_test['macro_f1']:.3f} "
        f"(tfidf {comparison_test['tfidf_logreg']['macro_f1']:.3f})"
    )


if __name__ == "__main__":
    main()
