from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from src.preprocessing.image import preprocess_image

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenetv2_final.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"

@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    """Load the trained model once and reuse it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

@lru_cache(maxsize=1)
def load_class_names() -> list[str]:
    """Load class names once and reuse them."""
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

def predict_image(image_path: str | Path) -> dict:
    """
    Predict the rice leaf disease for a single image.

    Returns:
        {
            "predicted_class": str,
            "confidence": float,
            "probabilities": dict[str, float]
        }
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Use the exact same preprocessing used during training
    image_array = preprocess_image(image_path)

    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)

    # Load model and class names
    model = load_model()
    class_names = load_class_names()

    #Predict
    probabilities = model.predict(
        image_batch,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    probability_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_dict
    }