"""
OCR Module.

Uses easyocr to lazily load and detect text (such as error codes) from images.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy loaded reader
_reader = None


def get_reader():
    global _reader
    if _reader is None:
        logger.info("Initializing EasyOCR reader (lazy load)...")
        import easyocr
        # Use English. Disable GPU if you want it strictly CPU bound,
        # but easyocr defaults to GPU if available.
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def extract_text(image: np.ndarray) -> str:
    """
    Extract text from an OpenCV image array.
    Returns a single string with all detected text separated by spaces.
    """
    try:
        reader = get_reader()
        # detail=0 returns just a list of text strings, not bounding boxes
        results = reader.readtext(image, detail=0)

        if not results:
            return ""

        extracted = " ".join(results)
        logger.info(f"OCR extracted {len(results)} text fragments.")
        return extracted
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""
