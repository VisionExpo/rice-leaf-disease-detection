import io
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.inference.predict import predict_image

app = FastAPI(
    title="Rice Leaf Disease Detection API",
    description="API for classifying rice leaf diseases using MobileNetV2.",
    version="0.1.0"
)

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


class PredictionResponse(BaseModel):
    predicted_class : str
    confidence: float
    probabilities: dict[str, float]

@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Check whether the API service is running."""
    return {
        "status": "healthy",
        "service": "rice-leaf-disease-detection"
    }

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)

async def predict(file: UploadFile = File(...)) -> PredictionResponse: # noqa: B008
    """Predict the disease shown in an uploaded rice leaf image."""

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Upload JPG, JPEG or PNG."
        )

    try:
        contents = await file.read(MAX_UPLOAD_SIZE + 1)
    finally:
        await file.close()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the maximum size of 10 MB."
        )

    # Verify that the uploaded content is actually an image.
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        ) from exc

    # The existing inference function accepts a filesystem path.
    # Use a temporary file, then remove it after prediction.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:
            temp_file.write(contents)
            temp_path = Path(temp_file.name)

        result = predict_image(temp_path)
        return PredictionResponse(**result)

    except FileNotFoundError as exc:
        logger.exception(
            "Prediction failed because a required file was not found."
        )
        raise HTTPException(
            status_code=500,
            detail="The model or a required file could not be found."
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error during rice leaf disease prediction."
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again later."
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
