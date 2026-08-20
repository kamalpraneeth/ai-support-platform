"""
ML Evaluation Module: cross-validation, classification metrics, and report saving.

This module provides reproducible evaluation of the TF-IDF + Logistic Regression
classifier using scikit-learn's standard evaluation tools.

Metrics computed:
  - Overall accuracy (train/test split)
  - Per-class Precision, Recall, F1-score
  - Macro-averaged and weighted-averaged F1
  - Confusion matrix
  - 5-fold cross-validation mean ± std accuracy

Important note on dataset size:
  With ~200 samples across 4 classes, evaluation metrics will naturally have
  higher variance than a large dataset. We use stratified splits to ensure
  all classes appear in both train and test sets. Cross-validation provides
  a more reliable estimate than a single train/test split.

These are real metrics computed from actual model execution.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_REPORT_PATH = ROOT / "models" / "evaluation_report.json"

LABEL_ORDER = [
    "Account",
    "Billing",
    "General",
    "Technical"]  # sorted alphabetically


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    """Per-class precision, recall, F1, and support."""
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass
class EvaluationMetrics:
    """Complete evaluation results for the ML classifier."""
    # Split evaluation
    test_accuracy: float = 0.0
    train_size: int = 0
    test_size: int = 0

    # Per-class metrics
    per_class: dict = field(default_factory=dict)  # {label: ClassMetrics dict}

    # Aggregate metrics
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0

    # Cross-validation (5-fold)
    cv_scores: list = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0

    # Confusion matrix
    confusion_matrix: list = field(default_factory=list)
    confusion_matrix_labels: list = field(default_factory=list)

    # Full sklearn classification report (human-readable string)
    classification_report_text: str = ""

    # Metadata
    model_type: str = "TF-IDF + Logistic Regression"
    dataset_size: int = 0
    num_classes: int = 4
    evaluation_version: str = "1.0"

    def summary(self) -> str:
        """Return a concise summary string for logging."""
        return (
            f"Accuracy: {self.test_accuracy:.4f} | "
            f"Macro F1: {self.macro_f1:.4f} | "
            f"CV: {self.cv_mean:.4f} ± {self.cv_std:.4f}"
        )


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline: Pipeline,
    texts: list[str],
    labels: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> EvaluationMetrics:
    """
    Evaluate a trained or untrained pipeline on the given dataset.

    The pipeline is re-trained on the training split to ensure honest evaluation
    (i.e., the test set is never seen during training).

    Args:
        pipeline: A fresh (untrained) sklearn Pipeline.
        texts: List of text samples.
        labels: List of category labels corresponding to texts.
        test_size: Fraction of data to use as test set (default: 0.2 = 20%).
        random_state: Seed for reproducible splits.
        cv_folds: Number of cross-validation folds.

    Returns:
        EvaluationMetrics with all computed scores.

    Note:
        With ~200 samples, a 20% test split gives ~40 test samples.
        This is a small test set — interpret individual class metrics with care.
        Cross-validation (5-fold) provides a more reliable accuracy estimate.
    """
    metrics = EvaluationMetrics(dataset_size=len(texts))

    # --- Train/test split (stratified to maintain class proportions) ---
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    metrics.train_size = len(X_train)
    metrics.test_size = len(X_test)

    # --- Train on training split ---
    pipeline.fit(X_train, y_train)

    # --- Evaluate on test split ---
    y_pred = pipeline.predict(X_test)
    metrics.test_accuracy = round(float(accuracy_score(y_test, y_pred)), 4)

    # --- Classification report ---
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    metrics.classification_report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    # Extract per-class metrics
    present_labels = sorted(set(labels))
    for label in present_labels:
        if label in report_dict:
            cls = report_dict[label]
            metrics.per_class[label] = {
                "precision": round(float(cls["precision"]), 4),
                "recall": round(float(cls["recall"]), 4),
                "f1_score": round(float(cls["f1-score"]), 4),
                "support": int(cls["support"]),
            }

    # Aggregate metrics
    macro = report_dict.get("macro avg", {})
    weighted = report_dict.get("weighted avg", {})
    metrics.macro_f1 = round(float(macro.get("f1-score", 0.0)), 4)
    metrics.weighted_f1 = round(float(weighted.get("f1-score", 0.0)), 4)
    metrics.macro_precision = round(float(macro.get("precision", 0.0)), 4)
    metrics.macro_recall = round(float(macro.get("recall", 0.0)), 4)

    # --- Confusion matrix ---
    present_labels_sorted = sorted(present_labels)
    cm = confusion_matrix(y_test, y_pred, labels=present_labels_sorted)
    metrics.confusion_matrix = cm.tolist()
    metrics.confusion_matrix_labels = present_labels_sorted

    # --- Cross-validation on full dataset ---
    # Use a fresh clone-like pipeline for CV to avoid data leakage
    from app.ml.classifier import build_pipeline as _build_pipeline
    cv_pipeline = _build_pipeline()
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state)
    cv_scores = cross_val_score(
        cv_pipeline,
        texts,
        labels,
        cv=skf,
        scoring="accuracy",
    )
    metrics.cv_scores = [round(float(s), 4) for s in cv_scores]
    metrics.cv_mean = round(float(np.mean(cv_scores)), 4)
    metrics.cv_std = round(float(np.std(cv_scores)), 4)

    logger.info("Evaluation complete: %s", metrics.summary())
    return metrics


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------

def save_evaluation_report(
    metrics: EvaluationMetrics,
    path: Path = EVALUATION_REPORT_PATH,
) -> None:
    """Serialize EvaluationMetrics to JSON on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)
    logger.info("Evaluation report saved to %s", path)


def load_evaluation_report(
    path: Path = EVALUATION_REPORT_PATH,
) -> Optional[dict]:
    """Load a previously saved evaluation report. Returns None if not found."""
    if not path.exists():
        logger.warning("Evaluation report not found at %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
