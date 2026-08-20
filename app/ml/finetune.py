"""
Fine-tuning workflow for an AI Virtual Assistant.

This script demonstrates fine-tuning a small open-source LLM (distilbert-base-uncased)
using PEFT (Parameter-Efficient Fine-Tuning) with LoRA.

It also compares the new LoRA model against the existing TF-IDF + Logistic Regression
baseline on a held-out test set, saving the results to `results/model_comparison.json`.
"""

import os
import json
import logging
from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from app.ml.classifier import load_model, predict_category
from app.data_pipeline import get_clean_texts_and_labels

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "distilbert-base-uncased"
DATA_PATH = Path("data/tickets.csv")
OUTPUT_DIR = "models/distilbert-lora"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "model_comparison.json")

# Category mapping
CATEGORIES = ["Billing", "Technical", "Account", "General"]
ID2LABEL = {i: label for i, label in enumerate(CATEGORIES)}
LABEL2ID = {label: i for i, label in enumerate(CATEGORIES)}


def load_and_split_data():
    """Load the dataset and split it identically for both models."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    texts, labels = get_clean_texts_and_labels(DATA_PATH)
    df = pd.DataFrame({"text": texts, "category": labels})

    # 80/20 train/test split, stratified by category
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["category"]
    )
    return train_df, test_df


def evaluate_baseline(test_df: pd.DataFrame) -> dict:
    """Evaluate the existing TF-IDF + LogReg model."""
    logger.info("Evaluating baseline model (TF-IDF + Logistic Regression)...")
    try:
        baseline_model = load_model()
    except FileNotFoundError:
        logger.warning("Baseline model not found. Run app/ml/train.py first.")
        return {}

    y_true = test_df["category"].tolist()
    y_pred = []

    for text in test_df["text"]:
        pred = predict_category(text, model=baseline_model)
        y_pred.append(pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    results = {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }
    logger.info("Baseline Results: %s", results)
    return results


def train_and_evaluate_lora(train_df: pd.DataFrame,
                            test_df: pd.DataFrame) -> dict:
    """Fine-tune DistilBERT using LoRA and evaluate."""
    logger.info("Preparing data for LoRA fine-tuning...")

    # Prepare datasets
    def prepare_dataset(df: pd.DataFrame):
        return Dataset.from_dict({
            "text": df["text"].tolist(),
            "label": [LABEL2ID[c] for c in df["category"]]
        })

    train_dataset = prepare_dataset(train_df)
    test_dataset = prepare_dataset(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize_func(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(
        tokenize_func,
        batched=True).remove_columns(
        ["text"])
    test_dataset = test_dataset.map(
        tokenize_func,
        batched=True).remove_columns(
        ["text"])

    # Load Base Model
    logger.info(f"Loading base model: {MODEL_ID}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Configure LoRA
    logger.info("Injecting LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]  # Standard targets for DistilBERT
    )

    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()

    # Define Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0)
        acc = accuracy_score(labels, predictions)
        return {
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }

    # Setup Trainer
    training_args = TrainingArguments(
        output_dir="models/tmp_trainer",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=5,
        remove_unused_columns=False,
        use_cpu=True  # Force CPU to prevent CUDA OOM on standard setups
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    # Evaluate
    logger.info("Evaluating fine-tuned LoRA model on test set...")
    eval_results = trainer.evaluate()

    metrics = {
        "accuracy": round(eval_results.get("eval_accuracy", 0.0), 4),
        "f1_score": round(eval_results.get("eval_f1_score", 0.0), 4),
        "precision": round(eval_results.get("eval_precision", 0.0), 4),
        "recall": round(eval_results.get("eval_recall", 0.0), 4)
    }

    logger.info("LoRA Results: %s", metrics)

    # Save Adapter
    logger.info("Saving LoRA adapter to %s", OUTPUT_DIR)
    lora_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, test_df = load_and_split_data()
    logger.info(
        "Dataset split: %d train, %d test",
        len(train_df),
        len(test_df))

    # Evaluate baseline
    baseline_metrics = evaluate_baseline(test_df)

    # Train and evaluate LoRA
    lora_metrics = train_and_evaluate_lora(train_df, test_df)

    # Compare
    comparison = {
        "dataset_size_train": len(train_df),
        "dataset_size_test": len(test_df),
        "baseline_tfidf_logreg": baseline_metrics,
        "finetuned_distilbert_lora": lora_metrics,
        "winner": "baseline" if baseline_metrics.get("accuracy", 0) > lora_metrics.get("accuracy", 0) else "lora"
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(comparison, f, indent=4)

    logger.info("Model comparison saved to %s", RESULTS_FILE)
    logger.info("Winner: %s", comparison["winner"])


if __name__ == "__main__":
    main()
