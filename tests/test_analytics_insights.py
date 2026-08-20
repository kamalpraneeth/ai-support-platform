from unittest.mock import patch
from app.analytics.insights import generate_automated_insights


@patch("app.analytics.insights.generate_eda_report")
@patch("app.analytics.insights.evaluate_data_quality")
def test_generate_automated_insights_success(mock_quality, mock_eda):
    mock_eda.return_value = {
        "total_records": 100,
        "category_distribution": {"Technical": 40, "Billing": 60},
        "urgency_distribution": {"High": 15, "Low": 85},
        "text_length": {"mean": 120}
    }
    mock_quality.return_value = {
        "missing_row_percent": 0.0,
        "duplicate_percent": 2.0
    }

    result = generate_automated_insights()

    assert "error" not in result
    insights = result["insights"]
    assert len(insights) >= 4

    # Check specific strings generated
    assert any("Billing" in i and "60.0%" in i for i in insights)
    assert any("0%" in i for i in insights)  # No missing values
    assert any("2.0%" in i for i in insights)  # Duplicates
    assert any("15.0%" in i for i in insights)  # High urgency
    assert any("120" in i for i in insights)  # Mean text length


@patch("app.analytics.insights.generate_eda_report")
@patch("app.analytics.insights.evaluate_data_quality")
def test_generate_automated_insights_error(mock_quality, mock_eda):
    mock_eda.return_value = {"error": "Dataset not found"}
    mock_quality.return_value = {"error": "Dataset not found"}

    result = generate_automated_insights()
    assert "error" in result
    assert "insights" not in result
