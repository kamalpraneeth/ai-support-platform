"""
Tests for the ML evaluation module (app/ml/evaluate.py).

Tests cover:
  - evaluate_model() returns correct schema
  - Metrics are in valid ranges [0, 1]
  - Confusion matrix shape matches number of classes
  - CV scores are valid
  - Save/load evaluation report round-trip
"""

import tempfile
from pathlib import Path

import pytest

from app.ml.classifier import build_pipeline
from app.ml.evaluate import (
    evaluate_model,
    save_evaluation_report,
    load_evaluation_report,
    EvaluationMetrics,
)


# Minimal dataset for testing evaluation (not for training quality).
# Needs at least cv_folds samples per class. We use 5 samples x 4 classes = 20 total.
TEXTS = [
    "I was charged twice for my subscription",
    "My payment failed but money was deducted",
    "I need a refund for this charge",
    "I was billed the wrong amount this month",
    "I want to cancel my subscription and get a refund",
    "The app keeps crashing on startup",
    "I am getting a 500 error on your website",
    "The mobile app crashes every time I open it",
    "Your website loads very slowly on my browser",
    "The login page is broken and I cannot sign in technically",
    "I cannot log into my account",
    "My account was suspended without warning",
    "I forgot my password reset email is not arriving",
    "How do I change my email address on my account?",
    "I need to update my account profile information",
    "What are the pricing plans?",
    "How do I get started with the platform?",
    "Do you offer a free trial?",
    "Where can I find your documentation online?",
    "Does your platform support third party integrations?",
]
LABELS = [
    "Billing", "Billing", "Billing", "Billing", "Billing",
    "Technical", "Technical", "Technical", "Technical", "Technical",
    "Account", "Account", "Account", "Account", "Account",
    "General", "General", "General", "General", "General",
]


class TestEvaluateModel:
    @pytest.fixture(scope="class")
    def metrics(self):
        """Run evaluation once for all tests in this class. Use 3-fold CV for small test dataset."""
        pipeline = build_pipeline()
        return evaluate_model(pipeline, TEXTS, LABELS, test_size=0.25, random_state=42, cv_folds=3)

    def test_returns_evaluation_metrics_instance(self, metrics):
        assert isinstance(metrics, EvaluationMetrics)

    def test_test_accuracy_in_valid_range(self, metrics):
        assert 0.0 <= metrics.test_accuracy <= 1.0

    def test_macro_f1_in_valid_range(self, metrics):
        assert 0.0 <= metrics.macro_f1 <= 1.0

    def test_weighted_f1_in_valid_range(self, metrics):
        assert 0.0 <= metrics.weighted_f1 <= 1.0

    def test_cv_mean_in_valid_range(self, metrics):
        assert 0.0 <= metrics.cv_mean <= 1.0

    def test_cv_std_non_negative(self, metrics):
        assert metrics.cv_std >= 0.0

    def test_cv_scores_has_correct_fold_count(self, metrics):
        assert len(metrics.cv_scores) == 3  # cv_folds=3 for small test dataset

    def test_per_class_metrics_populated(self, metrics):
        # Should have at least 1 class in per_class
        assert len(metrics.per_class) >= 1

    def test_per_class_scores_in_range(self, metrics):
        for label, cls in metrics.per_class.items():
            assert 0.0 <= cls["precision"] <= 1.0
            assert 0.0 <= cls["recall"] <= 1.0
            assert 0.0 <= cls["f1_score"] <= 1.0
            assert cls["support"] >= 0

    def test_confusion_matrix_is_square(self, metrics):
        n = len(metrics.confusion_matrix_labels)
        assert len(metrics.confusion_matrix) == n
        for row in metrics.confusion_matrix:
            assert len(row) == n

    def test_train_and_test_sizes_sum_to_dataset(self, metrics):
        # With 20 samples and 25% test: train=15, test=5
        assert metrics.train_size + metrics.test_size == len(TEXTS)

    def test_dataset_size_recorded(self, metrics):
        assert metrics.dataset_size == len(TEXTS)

    def test_summary_string_non_empty(self, metrics):
        summary = metrics.summary()
        assert len(summary) > 0
        assert "Accuracy" in summary


class TestSaveLoadReport:
    def test_save_and_load_round_trip(self, tmp_path):
        pipeline = build_pipeline()
        metrics = evaluate_model(pipeline, TEXTS, LABELS, test_size=0.25, random_state=42, cv_folds=3)

        report_path = tmp_path / "test_evaluation_report.json"
        save_evaluation_report(metrics, path=report_path)

        loaded = load_evaluation_report(path=report_path)
        assert loaded is not None
        assert "test_accuracy" in loaded
        assert abs(loaded["test_accuracy"] - metrics.test_accuracy) < 0.001

    def test_load_returns_none_for_missing_file(self, tmp_path):
        result = load_evaluation_report(path=tmp_path / "nonexistent.json")
        assert result is None
