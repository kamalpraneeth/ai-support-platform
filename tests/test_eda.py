import pytest
import csv
import os
from app.analytics.eda import generate_eda_report

@pytest.fixture
def mock_csv(tmp_path):
    csv_file = tmp_path / "mock_tickets.csv"
    data = [
        {"text": "Short issue", "category": "General"},
        {"text": "This is a longer issue about billing.", "category": "Billing"},
        {"text": "This is a longer issue about billing.", "category": "Billing"}, # Duplicate
        {"text": "", "category": "General"}, # Missing text
        {"text": "Urgent technical failure", "category": "Technical"},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(data)
    return str(csv_file)

def test_generate_eda_report_success(mock_csv):
    report = generate_eda_report(csv_path=mock_csv)
    
    assert "error" not in report
    assert report["total_records"] == 5
    assert report["missing_values"] == 1
    assert report["duplicate_records"] == 1
    
    # Text length stats
    assert report["text_length"]["min"] > 0
    assert report["text_length"]["max"] > 10
    
    # Category distribution
    assert report["category_distribution"]["General"] == 2
    assert report["category_distribution"]["Billing"] == 2
    assert report["category_distribution"]["Technical"] == 1
    
    # Class balance (min=1, max=2) -> 0.5
    assert report["class_balance_ratio"] == 0.5
    
    assert "sentiment_distribution" in report
    assert "urgency_distribution" in report

def test_generate_eda_report_not_found():
    report = generate_eda_report(csv_path="nonexistent.csv")
    assert "error" in report

def test_generate_eda_report_empty(tmp_path):
    csv_file = tmp_path / "empty.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category"])
        writer.writeheader()
    
    report = generate_eda_report(csv_path=str(csv_file))
    assert "error" in report
    assert report["error"] == "Empty dataset"
