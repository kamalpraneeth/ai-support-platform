import io


class ImageValidationError(Exception):
    """Raised when an image fails validation checks."""


# Maximum allowed file size (5 MB)
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

# Allowed MIME types and their extensions
ALLOWED_MIME_TYPES = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "image/webp": [".webp"],
}

# Maximum pixel dimensions to prevent memory bombs before resizing
MAX_DIMENSION = 4096


def validate_image_file(file_content: bytes, filename: str,
                        content_type: str) -> None:
    """
    Validates an uploaded image for security and size limits.
    Raises ImageValidationError if the image fails any checks.
    """
    # 1. Check file size
    if len(file_content) > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError(
            f"File size exceeds maximum limit of {
                MAX_FILE_SIZE_BYTES / 1024 / 1024:.1f}MB")

    # 2. Check content type
    if content_type not in ALLOWED_MIME_TYPES:
        raise ImageValidationError(
            f"Unsupported media type: {content_type}. Allowed types: {
                ', '.join(
                    ALLOWED_MIME_TYPES.keys())}")

    # 3. Safe decode using Pillow to check resolution and corruption before
    # handing to OpenCV
    try:
        from PIL import Image, UnidentifiedImageError

        # PIL reads lazily, but we wrap in memory view
        with Image.open(io.BytesIO(file_content)) as img:
            # Check dimensions
            width, height = img.size
            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise ImageValidationError(
                    f"Image resolution ({width}x{height}) exceeds maximum allowed ({MAX_DIMENSION}x{MAX_DIMENSION})"
                )

            # Verify the image is not corrupted (actually loads data)
            img.verify()

    except UnidentifiedImageError:
        raise ImageValidationError(
            "File does not appear to be a valid image or is corrupted.")
    except ImageValidationError:
        raise
    except Exception as e:
        raise ImageValidationError(
            f"Failed to decode image securely: {
                str(e)}")
