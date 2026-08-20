"""
Tests for the data pipeline module (app/data_pipeline.py).

Tests cover:
  - CSV loading
  - Text validation (missing, too short, wrong category)
  - Text normalization
  - Duplicate detection
  - Quality report generation
  - Pipeline outputs correct structure
"""

import csv
from pathlib import Path

import pytest

from app.data_pipeline import (
    validate_record,
    normalize_text,
    run_pipeline,
    get_clean_texts_and_labels,
    DataQualityReport,
    VALID_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Helper: create a temp CSV file for testing
# ---------------------------------------------------------------------------

def _write_temp_csv(rows: list[dict], tmp_path: Path) -> Path:
    csv_path = tmp_path / "test_tickets.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ---------------------------------------------------------------------------
# validate_record tests
# ---------------------------------------------------------------------------

class TestValidateRecord:
    def test_valid_record_returns_true(self):
        errors = []
        result = validate_record(
            {"text": "I was charged twice", "category": "Billing"},
            row_index=0,
            errors=errors,
        )
        assert result is True
        assert len(errors) == 0

    def test_missing_text_returns_false(self):
        errors = []
        result = validate_record(
            {"text": "", "category": "Billing"},
            row_index=0,
            errors=errors,
        )
        assert result is False
        assert any(e.field == "text" for e in errors)

    def test_missing_category_returns_false(self):
        errors = []
        result = validate_record(
            {"text": "Valid ticket text here", "category": ""},
            row_index=1,
            errors=errors,
        )
        assert result is False
        assert any(e.field == "category" for e in errors)

    def test_invalid_category_returns_false(self):
        errors = []
        result = validate_record(
            {"text": "Valid ticket text here", "category": "Unknown"},
            row_index=2,
            errors=errors,
        )
        assert result is False
        assert any("Invalid category" in e.message for e in errors)

    def test_short_text_returns_false(self):
        errors = []
        result = validate_record(
            {"text": "hi", "category": "Billing"},
            row_index=3,
            errors=errors,
        )
        assert result is False

    def test_all_valid_categories_accepted(self):
        for category in VALID_CATEGORIES:
            errors = []
            result = validate_record(
                {"text": "This is a valid ticket text", "category": category},
                row_index=0,
                errors=errors,
            )
            assert result is True, f"Category '{category}' should be valid"


# ---------------------------------------------------------------------------
# normalize_text tests
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_strips_whitespace(self):
        result = normalize_text("  hello world  ")
        assert result == "hello world"

    def test_collapses_internal_whitespace(self):
        result = normalize_text("hello   world  test")
        assert result == "hello world test"

    def test_handles_normal_text(self):
        text = "I cannot login to my account"
        assert normalize_text(text) == text

    def test_handles_empty_string(self):
        result = normalize_text("")
        assert result == ""


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_pipeline_returns_records_and_report(self, tmp_path):
        rows = [
            {"text": "I was charged twice for my subscription", "category": "Billing"},
            {"text": "The app keeps crashing when I open it", "category": "Technical"},
            {"text": "I cannot login to my account", "category": "Account"},
            {"text": "What are your pricing plans?", "category": "General"},
        ]
        csv_path = _write_temp_csv(rows, tmp_path)
        records, report = run_pipeline(csv_path=csv_path, report_path=None)
        assert len(records) == 4
        assert isinstance(report, DataQualityReport)

    def test_pipeline_removes_duplicates(self, tmp_path):
        rows = [
            {"text": "I was charged twice for my subscription", "category": "Billing"},
            {"text": "I was charged twice for my subscription",
                "category": "Billing"},  # duplicate
            {"text": "The app crashes every time", "category": "Technical"},
        ]
        csv_path = _write_temp_csv(rows, tmp_path)
        records, report = run_pipeline(csv_path=csv_path, report_path=None)
        assert len(records) == 2
        assert report.duplicates_removed == 1

    def test_pipeline_removes_invalid_rows(self, tmp_path):
        rows = [
            {"text": "Valid billing ticket here", "category": "Billing"},
            # invalid: empty text
            {"text": "", "category": "Billing"},
            {"text": "Technical issue",
             "category": "InvalidCat"},
            # invalid: bad category
        ]
        csv_path = _write_temp_csv(rows, tmp_path)
        records, report = run_pipeline(csv_path=csv_path, report_path=None)
        assert len(records) == 1
        assert report.invalid_rows_removed == 2

    def test_report_has_correct_class_distribution(self, tmp_path):
        rows = [
            {"text": "Billing ticket text one", "category": "Billing"},
            {"text": "Billing ticket text two", "category": "Billing"},
            {"text": "Technical issue with the app", "category": "Technical"},
        ]
        csv_path = _write_temp_csv(rows, tmp_path)
        _, report = run_pipeline(csv_path=csv_path, report_path=None)
        assert report.class_distribution["Billing"] == 2
        assert report.class_distribution["Technical"] == 1

    def test_pipeline_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            run_pipeline(
                csv_path=Path("/nonexistent/path.csv"),
                report_path=None)

    def test_get_clean_texts_and_labels_returns_lists(self, tmp_path):
        rows = [
            {"text": "I was charged twice for my subscription", "category": "Billing"},
            {"text": "The app keeps crashing on startup", "category": "Technical"},
        ]
        csv_path = _write_temp_csv(rows, tmp_path)
        # Patch the DATA_PATH for this call
        import app.data_pipeline as dp
        original_path = dp.DATA_PATH
        dp.DATA_PATH = csv_path
        try:
            texts, labels = get_clean_texts_and_labels(csv_path)
            assert isinstance(texts, list)
            assert isinstance(labels, list)
            assert len(texts) == len(labels) == 2
        finally:
            dp.DATA_PATH = original_path
