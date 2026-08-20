from .eda import generate_eda_report
from .data_quality import evaluate_data_quality

def generate_automated_insights() -> dict:
    """
    Generate automated plain-text insights from the EDA and data quality metrics.
    """
    eda = generate_eda_report()
    quality = evaluate_data_quality()

    if "error" in eda or "error" in quality:
        return {"error": "Could not generate insights due to missing or empty dataset."}

    insights = []

    # Category insights
    total_records = eda.get("total_records", 0)
    category_dist = eda.get("category_distribution", {})
    if total_records > 0 and category_dist:
        top_category = max(category_dist, key=category_dist.get)
        top_pct = round((category_dist[top_category] / total_records) * 100, 1)
        insights.append(f"The '{top_category}' category is the most common, accounting for {top_pct}% of all tickets.")

    # Data quality insights
    missing_pct = quality.get("missing_row_percent", 0)
    duplicate_pct = quality.get("duplicate_percent", 0)
    
    if missing_pct > 0:
        insights.append(f"Data Quality Warning: {missing_pct}% of records have missing values.")
    else:
        insights.append("Excellent Data Quality: 0% missing values detected in the dataset.")
        
    if duplicate_pct > 0:
        insights.append(f"Data Quality Warning: {duplicate_pct}% of records are exact text duplicates.")

    # Urgency insights
    urgency_dist = eda.get("urgency_distribution", {})
    high_urgency = urgency_dist.get("High", 0)
    if total_records > 0:
        high_pct = round((high_urgency / total_records) * 100, 1)
        insights.append(f"{high_pct}% of tickets are flagged as High urgency based on heuristic analysis.")

    # Text length insights
    mean_len = eda.get("text_length", {}).get("mean", 0)
    insights.append(f"The average ticket text length is {mean_len} characters.")

    return {
        "insights": insights
    }
