import time
import os
import logging
from typing import Tuple, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class YOLOObjectDetector:
    """
    Wrapper for Ultralytics YOLOv8n object detection.
    Supports lazy loading to avoid memory usage when CV is disabled or unused.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._model = None
        
        # Check if CV is globally enabled via config
        # We read directly from environ to avoid circular imports, 
        # though ideally it comes from app config.
        self.is_enabled = os.getenv("ENABLE_COMPUTER_VISION", "false").lower() == "true"

    def _load_model(self):
        """Lazily loads the YOLOv8 nano model into memory."""
        if not self.is_enabled:
            raise RuntimeError("Computer Vision is disabled in configuration (ENABLE_COMPUTER_VISION=false)")
            
        if self._model is None:
            logger.info("Loading YOLOv8n model for the first time...")
            start = time.perf_counter()
            from ultralytics import YOLO
            # 'yolov8n.pt' will be downloaded automatically by ultralytics to the current directory
            # on the very first run if it doesn't exist.
            self._model = YOLO('yolov8n.pt')
            latency = (time.perf_counter() - start) * 1000.0
            logger.info(f"YOLOv8n model loaded in {latency:.2f}ms")

    def detect(self, image_array: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs object detection on the provided OpenCV BGR image array.
        
        Returns:
            (detected_objects, inference_latency_ms)
            detected_objects is a list of dicts: [{"label": "laptop", "confidence": 0.91}, ...]
        """
        if not self.is_enabled:
            return [], 0.0
            
        self._load_model()
        
        start_time = time.perf_counter()
        
        # verbose=False to prevent ultralytics from spamming stdout on every request
        results = self._model(image_array, verbose=False)
        
        detected_objects = []
        
        # results is a list of Results objects (one per image, we only pass 1)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    cls_id = int(box.cls[0])
                    label = self._model.names[cls_id]
                    detected_objects.append({
                        "label": label,
                        "confidence": round(conf, 3)
                    })
                    
        # Deduplicate labels (e.g., if there are 2 laptops, just report laptop once with max confidence)
        # Though for some cases count matters, we just need to know *what* is in the image.
        unique_objects = {}
        for obj in detected_objects:
            lbl = obj["label"]
            if lbl not in unique_objects or obj["confidence"] > unique_objects[lbl]["confidence"]:
                unique_objects[lbl] = obj
                
        final_objects = list(unique_objects.values())
        
        inference_latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return final_objects, inference_latency_ms

# Global singleton instance for the application to share
cv_detector = YOLOObjectDetector(confidence_threshold=0.5)
