import pytest
import csv
from app.analytics.data_quality import evaluate_data_quality


@pytest.fixture
def mock_csv(tmp_path):
    csv_file = tmp_path / "mock_dq.csv"
    data = [
        {"text": "A valid row", "category": "General"},
        {"text": "A valid row", "category": "General"},  # Duplicate text
        {"text": "", "category": "General"},  # Missing text
        {"text": "Another row", "category": ""},  # Missing category
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(data)
    return str(csv_file)


def test_evaluate_data_quality(mock_csv):
    dq = evaluate_data_quality(csv_path=mock_csv)

    assert "error" not in dq
    assert dq["total_records_evaluated"] == 4

    # 2 rows missing out of 4 -> 50.0%
    assert dq["missing_row_percent"] == 50.0

    # 1 duplicate out of 4 -> 25.0%
    assert dq["duplicate_percent"] == 25.0

    # Valid records: 4 - 2 missing - 1 duplicate = 1
    # 1 valid out of 4 -> 25.0%
    assert dq["valid_record_percent"] == 25.0

    assert dq["class_imbalance_ratio"] == 1.0  # Only General class is fully valid


def test_evaluate_data_quality_not_found():
    dq = evaluate_data_quality(csv_path="nonexistent.csv")
    assert "error" in dq
