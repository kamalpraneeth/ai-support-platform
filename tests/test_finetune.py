"""
Tests for the Fine-Tuning Module.

We use unittest.mock to avoid downloading or loading actual PyTorch models
during the CI/CD test run, which would consume massive bandwidth and memory.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.ml.transformer_classifier import TransformerClassifier, predict_with_confidence


@pytest.fixture
def mock_peft_model():
    with patch("app.ml.transformer_classifier.AutoModelForSequenceClassification.from_pretrained") as mock_base:
        with patch("app.ml.transformer_classifier.PeftModel.from_pretrained") as mock_peft:
            with patch("app.ml.transformer_classifier.AutoTokenizer.from_pretrained") as mock_tokenizer:

                # Mock tokenizer return
                mock_tokenizer.return_value = MagicMock(
                    return_value={"input_ids": [1, 2, 3]})

                # Mock model return
                import torch
                # Create fake logits for 4 categories. Make index 1
                # ("Technical") the highest.
                fake_logits = torch.tensor([[-2.0, 5.0, 0.0, 1.0]])

                mock_model_instance = MagicMock()
                mock_model_instance.return_value = MagicMock(
                    logits=fake_logits)
                mock_peft.return_value = mock_model_instance

                yield mock_base, mock_peft, mock_tokenizer


def test_transformer_classifier_initialization_fails_if_no_adapter():
    """Ensure it fails cleanly if the user hasn't run finetune.py yet."""
    with pytest.raises(FileNotFoundError):
        TransformerClassifier(adapter_dir="invalid/path/that/does/not/exist")


def test_transformer_classifier_predict(mock_peft_model, tmp_path):
    """Test that the predict method correctly maps logits to categories and confidence."""
    # Create a fake adapter dir to pass the os.path.exists check
    adapter_dir = tmp_path / "fake-lora"
    adapter_dir.mkdir()

    classifier = TransformerClassifier(adapter_dir=str(adapter_dir))

    category, confidence = classifier.predict("My app crashed!")

    assert category == "Technical"
    # Softmax of [-2.0, 5.0, 0.0, 1.0] -> index 1 is extremely close to 1.0
    assert confidence > 0.95


def test_predict_with_confidence_function(mock_peft_model, tmp_path):
    """Test the drop-in replacement function for main.py integration."""
    adapter_dir = tmp_path / "fake-lora"
    adapter_dir.mkdir()

    classifier = TransformerClassifier(adapter_dir=str(adapter_dir))

    category, confidence = predict_with_confidence(
        "Test ticket", model=classifier)

    assert category == "Technical"
    assert confidence > 0.95
