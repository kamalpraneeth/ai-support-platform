import pytest
import io
import os
from unittest.mock import patch, MagicMock

from app.cv.validator import validate_image_file, ImageValidationError
from app.cv.processor import load_and_preprocess_image
from app.cv.detector import YOLOObjectDetector


def test_validate_image_file_valid():
    """Test validator passes valid images"""
    from PIL import Image

    # Create a simple valid 10x10 PNG in memory
    img = Image.new('RGB', (10, 10), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')

    # Should not raise
    validate_image_file(img_bytes.getvalue(), "test.png", "image/png")


def test_validate_image_file_oversized():
    """Test validator rejects oversized files"""
    large_bytes = b"0" * (5 * 1024 * 1024 + 1)
    with pytest.raises(ImageValidationError, match="File size exceeds"):
        validate_image_file(large_bytes, "test.png", "image/png")


def test_validate_image_file_invalid_mime():
    """Test validator rejects invalid MIME types"""
    with pytest.raises(ImageValidationError, match="Unsupported media type"):
        validate_image_file(b"fake data", "test.pdf", "application/pdf")


def test_validate_image_file_resolution_too_high():
    """Test validator rejects images with massive resolution"""
    from PIL import Image
    # Create a 5000x5000 image
    img = Image.new('RGB', (5000, 5000), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')

    with pytest.raises(ImageValidationError, match="exceeds maximum allowed"):
        validate_image_file(img_bytes.getvalue(), "test.png", "image/png")


def test_validate_image_file_corrupted():
    """Test validator catches corrupted image bytes"""
    with pytest.raises(ImageValidationError, match="valid image or is corrupted"):
        validate_image_file(b"this is not an image", "test.png", "image/png")


def test_processor_load_and_preprocess():
    """Test OpenCV processing resizes to 640x640"""
    from PIL import Image
    # Create 100x100 image
    img = Image.new('RGB', (100, 100), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')

    arr, latency = load_and_preprocess_image(img_bytes.getvalue())

    # YOLO requires 640x640
    assert arr.shape == (640, 640, 3)
    assert latency > 0


@patch("ultralytics.YOLO")
@patch.dict(os.environ, {"ENABLE_COMPUTER_VISION": "false"})
def test_detector_lazy_loading_disabled(mock_yolo):
    """Test YOLO doesn't load or infer when ENABLE_COMPUTER_VISION is false"""
    detector = YOLOObjectDetector()

    # Inference should return empty lists and 0 latency immediately
    objects, latency = detector.detect(None)
    assert objects == []
    assert latency == 0.0
    mock_yolo.assert_not_called()


@patch("ultralytics.YOLO")
@patch.dict(os.environ, {"ENABLE_COMPUTER_VISION": "true"})
def test_detector_inference(mock_yolo):
    """Test YOLO parses bounding boxes correctly"""

    # Mock the YOLO model output
    mock_model_instance = MagicMock()
    mock_model_instance.names = {0: "person", 1: "laptop", 2: "cell phone"}

    # Mock the result object
    mock_box1 = MagicMock()
    mock_box1.conf = [0.95]
    mock_box1.cls = [1]  # laptop

    mock_box2 = MagicMock()
    mock_box2.conf = [0.2]  # below threshold (0.5)
    mock_box2.cls = [2]  # cell phone

    mock_result = MagicMock()
    mock_result.boxes = [mock_box1, mock_box2]

    mock_model_instance.return_value = [mock_result]
    mock_yolo.return_value = mock_model_instance

    detector = YOLOObjectDetector(confidence_threshold=0.5)

    # Dummy array
    dummy_img = MagicMock()
    objects, latency = detector.detect(dummy_img)

    assert len(objects) == 1
    assert objects[0]["label"] == "laptop"
    assert objects[0]["confidence"] == 0.95
    assert latency > 0
