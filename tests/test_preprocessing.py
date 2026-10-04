from pathlib import Path
from unittest import result

import numpy as np
from PIL import Image

from src.preprocessing.image import preprocess_image

def test_preprocess_image_output_shape(tmp_path: Path) -> None:
    """Processed image should have shape (128, 448, 3)."""
    image_path = tmp_path / "test_image.jpg"

    image = Image.new("RGB", (565, 233), (50, 80, 100))
    image.save(image_path)

    result = preprocess_image(image_path)

    assert result.shape == (128, 448, 3)


def test_preprocess_image_dtype(tmp_path) -> None:
    """Processed image should use float32."""
    image_path = tmp_path / "test_image.jpg"

    image = Image.new("RGB", (565, 233), (50, 80, 100))
    image.save(image_path)

    result = preprocess_image(image_path)

    assert result.dtype == np.float32


def test_preprocess_image_range(tmp_path) -> None:
    """Processed pixel values should be between 0 to 1."""
    image_path = tmp_path / "test_image.jpg"

    image = Image.new("RGB", (565, 233), (50, 80, 100))
    image.save(image_path)

    result = preprocess_image(image_path)

    assert result.min() >= 0.0
    assert result.max() <= 1.0

def test_preprocess_image_handles_dark_background(tmp_path: Path) -> None:
    """Preprocessing should handle images with a dark background."""
    image_path = tmp_path / "dark_image.jpg"

    image = Image.new("RGB", (565, 233), (20, 25, 20))
    image.save(image_path)

    result = preprocess_image(image_path)

    assert result.shape == (128, 448, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0
