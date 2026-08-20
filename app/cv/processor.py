import time
from typing import Tuple
import numpy as np

# We lazy-import cv2 to avoid overhead if CV is disabled,
# but for type hints and module scope we can import it locally within
# functions.


def load_and_preprocess_image(file_content: bytes) -> Tuple[np.ndarray, float]:
    """
    Decodes raw image bytes into an OpenCV BGR array, resizes it for YOLO (640x640),
    and measures preprocessing latency.

    Returns:
        (image_array, latency_ms)
    """
    start_time = time.perf_counter()
    import cv2

    # 1. Decode bytes to numpy array
    nparr = np.frombuffer(file_content, np.uint8)

    # 2. Decode into cv2 image (BGR format)
    # cv2.IMREAD_COLOR ensures 3 channels, stripping alpha if present
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("OpenCV failed to decode the image array.")

    # 3. Resize to standard YOLOv8 size (640x640) to constrain memory during inference
    # Note: Ultralytics YOLOv8 does its own resizing under the hood, but doing it here
    # ensures consistent memory usage regardless of original image size.
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return img_resized, latency_ms
