import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.cv.ocr import extract_text


@pytest.fixture
def mock_easyocr():
    with patch("app.cv.ocr.get_reader") as mock_get_reader:
        mock_reader = MagicMock()
        mock_get_reader.return_value = mock_reader

        # By default, pretend it found some text
        mock_reader.readtext.return_value = [
            "ERR_CONNECTION_REFUSED", "Please", "contact", "support"]

        yield mock_reader


def test_extract_text_success(mock_easyocr):
    # Dummy image array
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    text = extract_text(img)

    assert "ERR_CONNECTION_REFUSED" in text
    assert "Please contact support" in text
    mock_easyocr.readtext.assert_called_once_with(img, detail=0)


def test_extract_text_empty(mock_easyocr):
    mock_easyocr.readtext.return_value = []
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    text = extract_text(img)
    assert text == ""


def test_ocr_influences_classification_and_rag(mock_easyocr, client):
    """
    Practical test:
    If a user just says "It is broken", the text alone is vague (probably 'General' category).
    But if they attach a screenshot containing "ERR_CONNECTION_REFUSED",
    OCR extracts it, appends it to the text, and the classifier should confidently route it to 'Technical'.
    """
    # Create a valid dummy image (e.g. 1x1 white pixel in PNG format)
    import cv2
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    success, encoded_image = cv2.imencode('.png', img)
    assert success
    file_bytes = encoded_image.tobytes()

    # We also need to mock YOLO so it doesn't crash or load heavily
    with patch("app.main.cv_detector") as mock_yolo:
        mock_yolo.detect.return_value = ([], 10.0)

        # Provide a specific OCR return value
        mock_easyocr.readtext.return_value = [
            "ERR_CONNECTION_REFUSED", "Traceback", "Exception"]

        response = client.post(
            "/ticket/with-image",
            data={"text": "It is broken"},
            files={"image": ("screenshot.png", file_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()

        # The OCR text should be appended to the ticket text
        assert "ERR_CONNECTION_REFUSED" in data["text"]

        # The ML classifier should now see the 'ERR_CONNECTION_REFUSED' and output Technical
        # (Assuming the ML classifier is loaded in app.main)
        assert data["category"] == "Technical"
