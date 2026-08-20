# AI Support Platform — Computer Vision (CV)

The platform supports an optional Computer Vision pipeline that utilizes OpenCV and Ultralytics YOLOv8 for automated object detection on images uploaded by customers in support tickets.

## 1. Capabilities & Limitations

**What it does:**
- Detects the presence of common objects (laptops, cell phones, monitors, etc.) in user-uploaded images.
- Extracts the detected labels and their confidence scores.
- Feeds this context into the LLM prompt so the AI can explicitly acknowledge the customer's hardware context (e.g., *"I see you uploaded a picture of your laptop..."*).

**What it DOES NOT do:**
- It does **not** perform damage assessment (e.g., cracked screens, water damage). YOLOv8n is an object detector, not an anomaly detector.
- It does **not** replace the text-based ML classifier. It strictly augments the context available to the Generative AI orchestration.

## 2. Architecture & Pipeline

The pipeline is triggered only on the `POST /ticket/with-image` endpoint.

1. **Security Validation (`validator.py`)**: The uploaded `multipart/form-data` image is checked for safe MIME types (PNG, JPEG, WebP), file size limits (5MB max), and extreme resolution boundaries to prevent OOM DOS attacks. It is safely decoded using Pillow before OpenCV touches it.
2. **Preprocessing (`processor.py`)**: OpenCV reads the bytes, strips alpha channels, and resizes the image to 640x640 (standard YOLO size) while tracking latency.
3. **Inference (`detector.py`)**: The YOLOv8 nano model (`yolov8n.pt`) performs object detection, filtering results below a 0.5 confidence threshold.
4. **Data Persistence (`models.py`)**: The detected objects and CV latency metrics are saved as JSON strings to the `Ticket` database record.
5. **Orchestration (`ai_orchestrator.py`)**: If CV data is present, the Prompt Builder automatically injects a note about the detected items into the system context for the LLM.

## 3. Configuration & Lazy Loading

Computer Vision requires heavy dependencies (`torch`, `ultralytics`, `opencv-python-headless`). To prevent unnecessary memory bloat in environments where CV is unused, the YOLO model is **lazily loaded**.

You must explicitly enable it in your `.env` file:
```env
ENABLE_COMPUTER_VISION=true
```
If enabled, the ~6MB YOLOv8n weights will be loaded into RAM on the *first* request to `/ticket/with-image`. If disabled, the API will still function, but the CV steps will return empty object lists with 0ms latency.

## 4. Measured Metrics

We do not claim theoretical metrics (like COCO mAP) for our specific use case. Instead, we measure real-time API latency metrics internally on every request, which are persisted to the database:
- `preprocessing_ms`: Time taken by OpenCV to decode and resize the image.
- `inference_ms`: Time taken by YOLOv8n to compute bounding boxes.
- `total_cv_latency_ms`: Combined CV pipeline delay injected into the ticket submission route.

These metrics ensure we maintain transparency regarding the performance impact of the CV modality on our REST APIs.
