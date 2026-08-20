import csv
import statistics
import os
import logging
from collections import Counter
from app.ml.sentiment import analyze_sentiment
from app.ml.urgency import score_urgency

logger = logging.getLogger(__name__)


def generate_eda_report(csv_path: str = "data/tickets.csv") -> dict:
    """
    Generate Exploratory Data Analysis (EDA) metrics from the raw dataset.
    Reads tickets.csv and computes text lengths, distributions, and heuristic labels.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"EDA dataset not found: {csv_path}")
        return {"error": "Dataset not found"}

    total_records = 0
    categories = []
    text_lengths = []
    sentiments = []
    urgencies = []
    seen_texts = set()
    duplicates = 0
    missing_values = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_records += 1
            text = row.get("text", "").strip()
            category = row.get("category", "").strip()

            if not text or not category:
                missing_values += 1

            if text in seen_texts:
                duplicates += 1
            else:
                seen_texts.add(text)

            if category:
                categories.append(category)

            if text:
                text_lengths.append(len(text))
                # Heuristic feature distribution
                sentiments.append(analyze_sentiment(text))
                urgencies.append(score_urgency(text))

    if total_records == 0:
        return {"error": "Empty dataset"}

    # Calculations
    category_counts = dict(Counter(categories))

    # Class balance ratio (min / max category count)
    class_counts = list(category_counts.values())
    if class_counts:
        min_class = min(class_counts)
        max_class = max(class_counts)
        balance_ratio = round(min_class / max_class, 3) if max_class > 0 else 0
    else:
        balance_ratio = 0

    # Text length stats
    if text_lengths:
        text_stats = {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "mean": round(statistics.mean(text_lengths), 1),
            "median": round(statistics.median(text_lengths), 1),
        }
    else:
        text_stats = {"min": 0, "max": 0, "mean": 0, "median": 0}

    # Text length distribution (bins)
    length_distribution = {"0-20": 0, "21-50": 0, "51-100": 0, "100+": 0}
    for length in text_lengths:
        if length <= 20:
            length_distribution["0-20"] += 1
        elif length <= 50:
            length_distribution["21-50"] += 1
        elif length <= 100:
            length_distribution["51-100"] += 1
        else:
            length_distribution["100+"] += 1

    return {
        "total_records": total_records,
        "category_distribution": category_counts,
        "class_balance_ratio": balance_ratio,
        "missing_values": missing_values,
        "duplicate_records": duplicates,
        "text_length": text_stats,
        "text_length_distribution": length_distribution,
        "sentiment_distribution": dict(Counter(sentiments)),
        "urgency_distribution": dict(Counter(urgencies))
    }
