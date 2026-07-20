"""
Training script: trains the TF-IDF + Logistic Regression classifier
on data/tickets.csv and saves the fitted pipeline to models/classifier.pkl.

Run with:
    python -m app.ml.train

This script is also called by the Dockerfile at build time so the model
is baked into the Docker image (no training needed at container startup).
"""

import csv
import pickle
import sys
from pathlib import Path

# Ensure the project root is on the Python path when run as a script
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.ml.classifier import build_pipeline, MODEL_PATH

DATA_PATH = ROOT / "data" / "tickets.csv"


def load_data(csv_path: Path) -> tuple[list[str], list[str]]:
    """Load texts and labels from the CSV training file."""
    texts, labels = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(row["category"].strip())
    return texts, labels


def train_and_save() -> None:
    """Train the pipeline and serialize it to disk."""
    print(f"Loading data from {DATA_PATH} ...")
    texts, labels = load_data(DATA_PATH)
    print(f"  {len(texts)} samples, {len(set(labels))} classes: {sorted(set(labels))}")

    print("Building and training pipeline ...")
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model saved to {MODEL_PATH}")

    # Quick sanity check
    test_cases = [
        ("I was charged twice for my subscription", "Billing"),
        ("The app keeps crashing on startup", "Technical"),
        ("I cannot log into my account", "Account"),
        ("Do you offer a free trial?", "General"),
    ]
    print("\nSanity checks:")
    for text, expected in test_cases:
        pred = pipeline.predict([text])[0]
        status = "[OK]" if pred == expected else "[FAIL]"
        print(f"  {status} '{text[:45]}' -> {pred} (expected {expected})")


if __name__ == "__main__":
    train_and_save()
