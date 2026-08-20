import csv
import os
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def evaluate_data_quality(csv_path: str = "data/tickets.csv") -> dict:
    """
    Evaluate dataset quality and return percentage-based health metrics.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"Data quality dataset not found: {csv_path}")
        return {"error": "Dataset not found"}

    total_records = 0
    missing_values = 0
    duplicates = 0
    seen_texts = set()
    categories = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_records += 1
            text = row.get("text", "").strip()
            category = row.get("category", "").strip()

            is_missing = not text or not category
            if is_missing:
                missing_values += 1
            
            is_duplicate = False
            if text:
                if text in seen_texts:
                    duplicates += 1
                    is_duplicate = True
                else:
                    seen_texts.add(text)
            
            if category and not is_missing:
                categories.append(category)

    if total_records == 0:
        return {"error": "Empty dataset"}

    missing_percent = round((missing_values / total_records) * 100, 2)
    duplicate_percent = round((duplicates / total_records) * 100, 2)
    valid_records = total_records - missing_values - duplicates
    valid_percent = round((valid_records / total_records) * 100, 2)

    # Class imbalance ratio
    category_counts = list(Counter(categories).values())
    if category_counts:
        min_class = min(category_counts)
        max_class = max(category_counts)
        imbalance_ratio = round(min_class / max_class, 3) if max_class > 0 else 0
    else:
        imbalance_ratio = 0

    return {
        "missing_row_percent": missing_percent,
        "duplicate_percent": duplicate_percent,
        "valid_record_percent": valid_percent,
        "class_imbalance_ratio": imbalance_ratio,
        "total_records_evaluated": total_records
    }
