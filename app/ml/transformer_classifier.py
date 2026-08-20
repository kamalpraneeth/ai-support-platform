"""
Transformer Classifier Inference.

This module provides the same interface as the baseline `classifier.py`
(`predict_with_confidence`), but uses the fine-tuned DistilBERT + LoRA model.
"""

import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

logger = logging.getLogger(__name__)

# Constants (Must match finetune.py)
BASE_MODEL_ID = "distilbert-base-uncased"
LORA_ADAPTER_DIR = "models/distilbert-lora"
CATEGORIES = ["Billing", "Technical", "Account", "General"]
ID2LABEL = {i: label for i, label in enumerate(CATEGORIES)}
LABEL2ID = {label: i for i, label in enumerate(CATEGORIES)}


class TransformerClassifier:
    """Wrapper for the fine-tuned DistilBERT model."""

    def __init__(self, base_model_id: str = BASE_MODEL_ID,
                 adapter_dir: str = LORA_ADAPTER_DIR):
        if not os.path.exists(adapter_dir):
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_dir}. Please run finetune.py first.")

        logger.info(f"Loading base model: {base_model_id}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_id,
            num_labels=len(CATEGORIES),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )

        logger.info(f"Loading LoRA adapter from {adapter_dir}")
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()  # Set to evaluation mode

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict the category of the text.
        Returns: (Category String, Confidence Score)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class_id = torch.max(probs, dim=-1)

        category = ID2LABEL[predicted_class_id.item()]
        return category, confidence.item()


# Global singleton cache similar to _classifier in main.py
_transformer_instance = None


def load_transformer_model() -> TransformerClassifier:
    """Load the transformer model into memory."""
    global _transformer_instance
    if _transformer_instance is None:
        _transformer_instance = TransformerClassifier()
    return _transformer_instance


def predict_with_confidence(
        text: str, model: TransformerClassifier | None = None) -> tuple[str, float]:
    """
    Drop-in replacement for the TF-IDF predict_with_confidence.
    """
    if model is None:
        model = load_transformer_model()
    return model.predict(text)
