"""Phase 3 — classifier heads over frozen sentence embeddings.

Two heads are evaluated on the golden set, mirroring the assignment's
"embedding classifier" requirement and its later confidence-thresholding
needs:

1. :class:`EmbeddingLogReg` — multinomial LogisticRegression on the 384-d
   MiniLM embeddings (frozen encoder, learned linear head). This is the
   classic transfer-learning recipe for n_train ~ 10^2 and the direct
   successor of the Phase 2 ``tfidf_logreg`` baseline (same head family,
   better features).
2. :class:`PrototypeClassifier` — per-class L2-normalised centroids with
   cosine-similarity confidence. Needs no training loop at all, naturally
   exposes a calibrated-ish confidence (max cosine), degrades gracefully
   on the 2-example ``security_privacy`` class, and is the exact
   mechanism the Phase 6 escalation policy can threshold
   (``INTENT_CONFIDENCE_THRESHOLD``).

Both heads implement the same interface: ``fit`` / ``predict`` /
``predict_confidence`` (label + probability-style confidence per input),
and ``save`` / ``load`` so later phases (and the Streamlit app) reuse the
trained artefact instead of refitting.

Rare-class handling is a measured comparison, not an assumption: with a
frozen encoder a linear head sees each rare-class example the same number
of times either way, so the "over-sampling" knob that can actually change
the fit is ``class_weight`` — ``train_embedding_classifier.py`` evaluates
both settings and reports the difference honestly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class EmbeddingLogReg:
    """LogisticRegression head over frozen sentence embeddings."""

    kind = "logreg_head"

    def __init__(self, seed: int, c: float = 1.0, class_weight: str | None = None):
        self.seed = seed
        self.c = c
        self.class_weight = class_weight
        self.model = LogisticRegression(
            max_iter=5000, C=c, class_weight=class_weight, random_state=seed
        )
        self.classes_: list[str] = []

    def fit(self, vectors: np.ndarray, labels: list[str]) -> "EmbeddingLogReg":
        self.model.fit(vectors, labels)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, vectors: np.ndarray) -> list[str]:
        return [str(p) for p in self.model.predict(vectors)]

    def predict_confidence(self, vectors: np.ndarray) -> list[dict]:
        """Per-row {label, confidence, top2} from softmax probabilities."""
        probs = self.model.predict_proba(vectors)
        order = np.argsort(-probs, axis=1)
        out: list[dict] = []
        for row, ranked in zip(probs, order):
            top = ranked[:2]
            out.append(
                {
                    "label": str(self.model.classes_[top[0]]),
                    "confidence": round(float(row[top[0]]), 4),
                    "runner_up": (
                        str(self.model.classes_[top[1]]) if len(top) > 1 else None
                    ),
                    "runner_up_confidence": (
                        round(float(row[top[1]]), 4) if len(top) > 1 else None
                    ),
                }
            )
        return out

    def save(self, path: Path, embedding_model: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": self.kind,
                "embedding_model": embedding_model,
                "classes": self.classes_,
                "model": self.model,
                "hyperparameters": {"C": self.c, "class_weight": self.class_weight},
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "EmbeddingLogReg":
        blob = joblib.load(path)
        head = cls(blob["seed"], c=blob["hyperparameters"]["C"],
                   class_weight=blob["hyperparameters"]["class_weight"])
        head.model = blob["model"]
        head.classes_ = list(blob["classes"])
        return head


class PrototypeClassifier:
    """Centroid (nearest-class-mean) classifier with cosine confidence."""

    kind = "prototype"

    def __init__(self, seed: int):
        self.seed = seed
        self.classes_: list[str] = []
        self.centroids_: dict[str, np.ndarray] = {}
        self.embedding_model: str | None = None

    def fit(self, vectors: np.ndarray, labels: list[str]) -> "PrototypeClassifier":
        vectors = np.asarray(vectors, dtype=np.float32)
        self.classes_ = sorted(set(labels))
        self.centroids_ = {}
        for label in self.classes_:
            rows = vectors[np.asarray(labels) == label]
            centroid = rows.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            self.centroids_[label] = (
                centroid / norm if norm > 0 else centroid
            ).astype(np.float32)
        return self

    def _similarity(self, vectors: np.ndarray) -> tuple[list[str], np.ndarray]:
        names = self.classes_
        matrix = np.stack([self.centroids_[name] for name in names])  # (k, d)
        unit = vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
        )
        sims = unit @ matrix.T  # cosine similarity, (n, k)
        best = np.argmax(sims, axis=1)
        return [names[i] for i in best], sims

    def predict(self, vectors: np.ndarray) -> list[str]:
        labels, _ = self._similarity(vectors)
        return labels

    def predict_confidence(self, vectors: np.ndarray) -> list[dict]:
        """Cosine similarity to each class centroid, softmax over the sims.

        Softmax over cosine scores keeps confidence in (0, 1) and comparable
        with the LogReg head's probabilities while preserving ranking.
        """
        labels, sims = self._similarity(vectors)
        probs = _softmax(sims)
        order = np.argsort(-probs, axis=1)
        out: list[dict] = []
        for row, ranked, label in zip(probs, order, labels):
            top = ranked[:2]
            out.append(
                {
                    "label": label,
                    "confidence": round(float(row[top[0]]), 4),
                    "runner_up": self.classes_[top[1]] if len(top) > 1 else None,
                    "runner_up_confidence": (
                        round(float(row[top[1]]), 4) if len(top) > 1 else None
                    ),
                }
            )
        return out

    def save(self, path: Path, embedding_model: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": self.kind,
                "embedding_model": embedding_model,
                "classes": self.classes_,
                "centroids": self.centroids_,
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "PrototypeClassifier":
        blob = joblib.load(path)
        head = cls(blob["seed"])
        head.classes_ = list(blob["classes"])
        head.centroids_ = {
            name: np.asarray(vec, dtype=np.float32)
            for name, vec in blob["centroids"].items()
        }
        head.embedding_model = blob.get("embedding_model")
        return head


def load_artifact_metadata(path: Path) -> dict:
    """Read the light metadata sidecar of a saved classifier head."""
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return {}
