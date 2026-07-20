"""
ML Classifier: TF-IDF + Logistic Regression for ticket category prediction.

Trained on data/tickets.csv (80 labeled samples across 4 categories).
Model is serialized to models/classifier.pkl after training.
"""

import os
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Path to the serialized model
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "classifier.pkl"

# Valid category labels
CATEGORIES = ["Billing", "Technical", "Account", "General"]


def load_model() -> Pipeline:
    """Load the trained pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python -m app.ml.train` to train and save the model."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_category(text: str, model: Pipeline | None = None) -> str:
    """
    Predict the support ticket category.

    Args:
        text: Raw ticket text from the user.
        model: Optional pre-loaded pipeline (avoids disk I/O on every call).

    Returns:
        One of: 'Billing', 'Technical', 'Account', 'General'
    """
    if model is None:
        model = load_model()
    prediction = model.predict([text])[0]
    return prediction


def build_pipeline() -> Pipeline:
    """
    Build a fresh TF-IDF + Logistic Regression pipeline.

    Design decisions:
    - TfidfVectorizer: ngram_range=(1,2) captures bigrams like 'credit card'
      and 'cannot login' which are discriminative for support tickets.
    - max_features=5000: keeps the vocabulary manageable on a small dataset.
    - LogisticRegression: fast, interpretable, works well on text with TF-IDF.
      C=1.0 is the default regularization; no tuning needed for 80 samples.
    - max_iter=1000: prevents convergence warnings on small datasets.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,   # log(1 + tf) — helps with varying doc lengths
            strip_accents="unicode",
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        )),
    ])
