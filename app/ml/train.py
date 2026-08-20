"""
Training script: trains the TF-IDF + Logistic Regression classifier
on data/tickets.csv and saves the fitted pipeline to models/classifier.pkl.

Run with:
    python -m app.ml.train

This script is also called by the Dockerfile at build time so the model
is baked into the Docker image (no training needed at container startup).

Pipeline:
    Raw CSV
        ↓
    Data Pipeline (validation + cleaning via app.data_pipeline)
        ↓
    Evaluation (train/test split + cross-validation)
        ↓
    Full-dataset retraining (final model uses all clean data)
        ↓
    Save classifier.pkl + evaluation_report.json
"""

from app.ml.evaluate import evaluate_model, save_evaluation_report
from app.ml.classifier import build_pipeline, MODEL_PATH
from app.data_pipeline import get_clean_texts_and_labels, DATA_PATH
import logging
import pickle
import sys
import warnings
from pathlib import Path

# Ensure the project root is on the Python path when run as a script
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suppress sklearn convergence/optimize warnings in train output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def train_and_save() -> None:
    """
    Full training pipeline:
    1. Load + validate + clean data via data_pipeline
    2. Evaluate on a held-out test split (honest evaluation)
    3. Retrain on the full clean dataset for the production model
    4. Save classifier.pkl and evaluation_report.json
    """
    print(f"[1/4] Loading and validating data from {DATA_PATH} ...")
    texts, labels = get_clean_texts_and_labels(DATA_PATH)
    n_classes = len(set(labels))
    print(
        f"      {
            len(texts)} clean samples | {n_classes} classes: {
            sorted(
                set(labels))}")

    # --- Evaluation on train/test split ---
    print("[2/4] Evaluating model (80/20 stratified split + 5-fold CV) ...")
    eval_pipeline = build_pipeline()
    metrics = evaluate_model(
        eval_pipeline,
        texts,
        labels,
        test_size=0.2,
        random_state=42)

    print("\n      -- Evaluation Results -----------------------------------------")
    print(f"      Test accuracy (20% hold-out): {metrics.test_accuracy:.4f}")
    print(f"      Macro F1:                     {metrics.macro_f1:.4f}")
    print(f"      Weighted F1:                  {metrics.weighted_f1:.4f}")
    print(
        f"      CV accuracy (5-fold):         {metrics.cv_mean:.4f} ± {metrics.cv_std:.4f}")
    print()
    print("      Per-class metrics:")
    for label, cls in metrics.per_class.items():
        print(f"        {label:<12} P={cls['precision']:.3f}  R={cls['recall']:.3f}  "
              f"F1={cls['f1_score']:.3f}  support={cls['support']}")
    print()
    print("      Confusion matrix (rows=actual, cols=predicted):")
    print(f"      Labels: {metrics.confusion_matrix_labels}")
    for row_label, cm_row in zip(
            metrics.confusion_matrix_labels, metrics.confusion_matrix):
        print(f"        {row_label:<12} {cm_row}")

    save_evaluation_report(metrics)
    print("\n      Evaluation report saved to models/evaluation_report.json")

    # --- Full-dataset retrain (final production model) ---
    print("\n[3/4] Retraining on full dataset for production model ...")
    final_pipeline = build_pipeline()
    final_pipeline.fit(texts, labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_pipeline, f)
    print(f"      Model saved to {MODEL_PATH}")

    # --- Sanity checks ---
    print("\n[4/4] Sanity checks:")
    test_cases = [
        ("I was charged twice for my subscription", "Billing"),
        ("The app keeps crashing on startup", "Technical"),
        ("I cannot log into my account", "Account"),
        ("Do you offer a free trial?", "General"),
    ]
    all_pass = True
    for text, expected in test_cases:
        pred = final_pipeline.predict([text])[0]
        proba = final_pipeline.predict_proba([text])[0].max()
        status = "[OK]" if pred == expected else "[FAIL]"
        if pred != expected:
            all_pass = False
        print(
            f"      {status} '{text[:45]}' -> {pred} (conf={proba:.3f}, expected {expected})")

    print()
    if all_pass:
        print("      All sanity checks passed. [PASS]")
    else:
        print("      WARNING: Some sanity checks failed — inspect training data.")


if __name__ == "__main__":
    train_and_save()
