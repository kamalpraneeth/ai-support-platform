"""
Data Pipeline: validation, cleaning, preprocessing, and quality reporting.

This module provides a reproducible data pipeline that transforms raw CSV
training data into a clean, validated dataset ready for ML model training.

Pipeline stages:
    Raw CSV
        ↓
    Load & Validate (schema, required columns, label set)
        ↓
    Clean (strip whitespace, normalize text, remove duplicates)
        ↓
    Preprocess (lowercase, collapse whitespace)
        ↓
    Feature Engineering (text_length, word_count features logged)
        ↓
    Quality Report (class balance, duplicates, missing values)
        ↓
    Clean Dataset ready for training

Design decisions:
- All validation errors are collected before raising, so the caller sees all
  issues at once rather than one-at-a-time.
- Text normalization is deliberately lightweight (lowercase, whitespace) to
  avoid discarding signal that TF-IDF relies on.
- The quality report is written to JSON so the /metrics endpoint can serve it.
"""

import csv
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Valid category labels (must match classifier.py CATEGORIES)
VALID_CATEGORIES = {"Billing", "Technical", "Account", "General"}

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tickets.csv"
REPORT_PATH = ROOT / "models" / "data_quality_report.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DataRecord:
    """A single validated and cleaned training record."""
    text: str
    category: str
    text_length: int = 0
    word_count: int = 0

    def __post_init__(self) -> None:
        self.text_length = len(self.text)
        self.word_count = len(self.text.split())


@dataclass
class ValidationError:
    """A single data validation error with row context."""
    row_index: int
    field: str
    message: str


@dataclass
class DataQualityReport:
    """Summary statistics and quality metrics for the dataset."""
    total_raw_rows: int = 0
    total_valid_rows: int = 0
    duplicates_removed: int = 0
    invalid_rows_removed: int = 0
    missing_text_count: int = 0
    missing_category_count: int = 0
    invalid_category_count: int = 0
    class_distribution: dict = field(default_factory=dict)
    class_balance_ratio: float = 0.0  # min_class / max_class; 1.0 = perfect balance
    avg_text_length: float = 0.0
    avg_word_count: float = 0.0
    min_text_length: int = 0
    max_text_length: int = 0
    validation_errors: list = field(default_factory=list)
    pipeline_version: str = "1.0"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_record(
    row: dict,
    row_index: int,
    errors: list[ValidationError],
) -> bool:
    """
    Validate a single CSV row.

    Returns True if the record is usable, False if it should be dropped.
    Appends ValidationError entries to `errors` for any issues found.
    """
    ok = True

    text = row.get("text", "").strip()
    category = row.get("category", "").strip()

    if not text:
        errors.append(
            ValidationError(
                row_index,
                "text",
                "Missing or empty text"))
        ok = False

    if not category:
        errors.append(
            ValidationError(
                row_index,
                "category",
                "Missing or empty category"))
        ok = False
    elif category not in VALID_CATEGORIES:
        errors.append(
            ValidationError(
                row_index,
                "category",
                f"Invalid category '{category}'. Expected one of: {
                    sorted(VALID_CATEGORIES)}",
            )
        )
        ok = False

    if text and len(text) < 5:
        errors.append(
            ValidationError(
                row_index,
                "text",
                f"Text too short ({
                    len(text)} chars)"))
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Cleaning & Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Apply lightweight text normalization.

    Operations:
    - Strip leading/trailing whitespace
    - Collapse multiple internal whitespace characters to single space
    - Normalize unicode quotation marks to ASCII
    - Remove non-printable characters

    We intentionally do NOT lowercase here — TF-IDF's lowercase=True handles
    that at vectorization time, and keeping original case helps during review.
    """
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\u00A0-\u024F]", " ", text)
    # Normalize unicode quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_raw_data(csv_path: Path = DATA_PATH) -> list[dict]:
    """Load raw rows from the CSV file without any processing."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def run_pipeline(
    csv_path: Path = DATA_PATH,
    report_path: Optional[Path] = REPORT_PATH,
) -> tuple[list[DataRecord], DataQualityReport]:
    """
    Execute the full data pipeline.

    Args:
        csv_path: Path to the raw CSV file.
        report_path: Where to save the JSON quality report. Pass None to skip saving.

    Returns:
        (clean_records, quality_report)

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If zero valid records remain after validation.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    logger.info("Data pipeline: loading from %s", csv_path)
    raw_rows = load_raw_data(csv_path)

    report = DataQualityReport(total_raw_rows=len(raw_rows))
    validation_errors: list[ValidationError] = []
    valid_rows: list[dict] = []

    # --- Stage 1: Validate ---
    for i, row in enumerate(raw_rows):
        if validate_record(row, i, validation_errors):
            valid_rows.append(row)
        else:
            report.invalid_rows_removed += 1

    # Count specific error types
    for err in validation_errors:
        if err.field == "text" and "Missing" in err.message:
            report.missing_text_count += 1
        elif err.field == "category" and "Missing" in err.message:
            report.missing_category_count += 1
        elif err.field == "category" and "Invalid" in err.message:
            report.invalid_category_count += 1

    # --- Stage 2: Clean & Deduplicate ---
    seen_texts: set[str] = set()
    clean_records: list[DataRecord] = []

    for row in valid_rows:
        raw_text = row["text"].strip()
        category = row["category"].strip()

        # Normalize text
        text = normalize_text(raw_text)

        # Deduplicate on normalized, lowercased text
        normalized_key = text.lower()
        if normalized_key in seen_texts:
            report.duplicates_removed += 1
            logger.debug("Duplicate removed: '%s...'", text[:40])
            continue
        seen_texts.add(normalized_key)

        record = DataRecord(text=text, category=category)
        clean_records.append(record)

    report.total_valid_rows = len(clean_records)

    if report.total_valid_rows == 0:
        raise ValueError(
            "Data pipeline produced zero valid records. "
            "Check the CSV file for correct format and valid category labels."
        )

    # --- Stage 3: Compute Statistics ---
    class_counts: dict[str, int] = {}
    total_length = 0
    total_words = 0
    lengths = []

    for record in clean_records:
        class_counts[record.category] = class_counts.get(
            record.category, 0) + 1
        total_length += record.text_length
        total_words += record.word_count
        lengths.append(record.text_length)

    report.class_distribution = dict(sorted(class_counts.items()))

    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        report.class_balance_ratio = round(
            min_count / max_count, 3) if max_count > 0 else 0.0

    report.avg_text_length = round(total_length / len(clean_records), 1)
    report.avg_word_count = round(total_words / len(clean_records), 1)
    report.min_text_length = min(lengths)
    report.max_text_length = max(lengths)
    report.validation_errors = [
        {"row": e.row_index, "field": e.field, "message": e.message}
        for e in validation_errors
    ]

    # --- Stage 4: Save Report ---
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("Data quality report saved to %s", report_path)

    _log_summary(report)
    return clean_records, report


def _log_summary(report: DataQualityReport) -> None:
    """Log a human-readable summary of the data quality report."""
    logger.info("=== Data Quality Report ===")
    logger.info("  Raw rows:          %d", report.total_raw_rows)
    logger.info("  Invalid removed:   %d", report.invalid_rows_removed)
    logger.info("  Duplicates removed:%d", report.duplicates_removed)
    logger.info("  Clean records:     %d", report.total_valid_rows)
    logger.info("  Class distribution: %s", report.class_distribution)
    logger.info(
        "  Balance ratio:     %.3f (1.0 = perfect)",
        report.class_balance_ratio)
    logger.info("  Avg text length:   %.1f chars", report.avg_text_length)
    logger.info("  Validation errors: %d", len(report.validation_errors))


def get_clean_texts_and_labels(
    csv_path: Path = DATA_PATH,
) -> tuple[list[str], list[str]]:
    """
    Convenience function: run the pipeline and return (texts, labels).

    This is the interface used by train.py — it abstracts the pipeline
    so the training script does not need to understand DataRecord internals.
    """
    records, _ = run_pipeline(csv_path=csv_path, report_path=REPORT_PATH)
    texts = [r.text for r in records]
    labels = [r.category for r in records]
    return texts, labels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    records, report = run_pipeline()
    print(f"\nPipeline complete: {report.total_valid_rows} clean records")
    print(f"Class distribution: {report.class_distribution}")
    print(f"Balance ratio: {report.class_balance_ratio}")
