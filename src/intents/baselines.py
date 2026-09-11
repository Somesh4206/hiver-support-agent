"""Phase 2 baseline intent classifiers and metrics helpers.

Three baselines are evaluated on the golden set, in increasing sophistication:

1. ``keyword_rules`` — the Phase 1 provisional keyword topics (EDA) mapped
   onto the taxonomy labels. This quantifies how far transparent keyword
   matching gets before any learning: it is the honest bridge from the
   Phase 1 EDA heuristics to the Phase 2 taxonomy.
2. ``majority`` — always predict the training majority class. The floor
   every learned model must beat.
3. ``tfidf_logreg`` — TF-IDF (word 1–2 grams) + multinomial Logistic
   Regression, the classical strong baseline for short-text classification.

All metrics are computed with sklearn; everything is seeded and every
number written to disk is measured, never estimated.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.intents.taxonomy import TAXONOMY_LABELS

# Priority order = Phase 1 EDA ``PROVISIONAL_TOPICS`` dict order; the first
# matching topic wins; no match -> "other". Kept in sync with run_eda.py.
EDA_TOPIC_TO_LABEL = {
    "Account / Apple ID": "account_icloud",
    "iCloud / Backup": "account_icloud",
    "Billing / Payments": "billing_purchases",
    "Refunds / Purchases": "billing_purchases",
    "Subscriptions": "billing_purchases",
    "Device / Hardware": "device_hardware",
    "Software / Apps": "software_bug",
    "Connectivity": "connectivity",
    "Repair / Warranty": "repair_warranty",
    "Order / Delivery": "billing_purchases",
    "Security / Privacy": "security_privacy",
}


def keyword_rule_predict(texts: pd.Series, keyword_topics: dict[str, tuple[str, ...]]) -> list[str]:
    """Predict labels with the Phase 1 keyword rules (first match wins).

    Matching is word-boundary regex, exactly like the Phase 1 EDA tagger
    (decision D11: plain substring matching inflated Software/Apps to 64.5%
    because ``app`` matched inside ``apple`` — that mistake must not be
    reintroduced here).
    """
    import re

    compiled = {
        topic: [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords]
        for topic, keywords in keyword_topics.items()
    }
    predictions: list[str] = []
    for text in texts:
        raw = str(text)
        label = "other"
        for topic, patterns in compiled.items():
            if any(p.search(raw) for p in patterns):
                label = EDA_TOPIC_TO_LABEL[topic]
                break
        predictions.append(label)
    return predictions


def majority_predict(train_labels: pd.Series, n: int) -> list[str]:
    """Always predict the training majority class."""
    majority = train_labels.mode().iloc[0]
    return [majority] * n


def build_tfidf_logreg(
    seed: int, c: float = 1.0, class_weight: str | None = None, min_df: int = 1
) -> Pipeline:
    """The classical TF-IDF + LogisticRegression short-text baseline.

    Hyperparameters are selected on the VALIDATION split by
    ``train_baselines.py`` (small grid: C x class_weight x min_df); the
    defaults here are the sklearn defaults so an un-tuned run stays honest.
    With only ~140 training examples, an unregularized model memorises the
    train split (train accuracy 1.0) and collapses to the majority class —
    the measured reason this baseline needs tuning.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    C=c,
                    class_weight=class_weight,
                    random_state=seed,
                ),
            ),
        ]
    )


def classification_report_dict(
    y_true: pd.Series, y_pred: list[str]
) -> dict[str, Any]:
    """Accuracy, macro/weighted F1, per-class rows, and confusion matrix."""
    labels = list(TAXONOMY_LABELS)
    present = [l for l in labels if l in set(y_true) | set(y_pred)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = [
        {
            "label": label,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    ]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4
        ),
        "classes_evaluated": present,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": [[int(v) for v in row] for row in cm],
        },
    }


def stratified_cv_macro_f1(
    pipeline: Pipeline, x: pd.Series, y: pd.Series, seed: int, folds: int = 5
) -> dict[str, Any]:
    """Stratified CV on the training split.

    Rare classes can make 5-fold stratification impossible (a class with 2
    training examples supports at most 2 folds); on ``ValueError`` we retry
    with 2 folds and record that the fold count was reduced.
    """
    attempted = folds
    while folds >= 2:
        try:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            scores = cross_val_score(
                pipeline, x, y, cv=cv, scoring="f1_macro"
            )
            return {
                "folds": folds,
                "folds_attempted": attempted,
                "fold_count_reduced": folds != attempted,
                "macro_f1_mean": round(float(np.mean(scores)), 4),
                "macro_f1_std": round(float(np.std(scores)), 4),
                "fold_scores": [round(float(s), 4) for s in scores],
            }
        except ValueError:
            folds -= 1
    return {"folds": 0, "error": "stratification impossible (class support < 2)"}
