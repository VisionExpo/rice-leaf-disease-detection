from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TARGET_WIDTH = 448
TARGET_HEIGHT = 128


def estimate_padding_color(
        image: Image.Image
) -> tuple[int, int, int]:
    """Estimate background color from the four image corners."""
    image_array = np.asarray(image.convert("RGB"))

    height, width, _ = image_array.shape

    patch_size = max(1, min(height, width) // 20)

    corner_patches = np.concatenate(
        [
            image_array[:patch_size, :patch_size].reshape(-1, 3),
            image_array[:patch_size, -patch_size:].reshape(-1, 3),
            image_array[-patch_size:, :patch_size].reshape(-1, 3),
            image_array[-patch_size:, -patch_size:].reshape(-1, 3)
        ]
    )

    color = np.median(corner_patches, axis=0)

    return tuple(color.astype(np.uint8))

def preprocess_image(
        image_path: Path,
        target_size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)
) -> np.ndarray:
    """
    Load and preprocess one rice leaf image.

    Steps:
    1. Convert to RGB.
    2. Estimate padding color.
    3. Resize while preserving aspect ratio.
    4. Pad to target size.
    5. Convert to float32.
    6. Normalize pixels to [0, 1].
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        padding_color = estimate_padding_color(image)

        image = ImageOps.pad(
            image,
            target_size,
            method=Image.Resampling.LANCZOS,
            color=padding_color,
            centering=(0.5, 0.5)
        )

        image_array =np.asarray(
            image,
            dtype=np.float32
        )

    image_array /= 255.0

    return image_array

def preprocess_dataset(
        dataframe,
        project_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess all images referenced by a dataframe.

    Returns:
        X: image arrays with shape(N, 128, 448, 3)
        y: class labels
    """
    images = []
    labels = []

    for _, row in dataframe.iterrows():
        image_path = project_root / Path(row["image_path"])

        image = preprocess_image(image_path)

        images.append(image)
        labels.append(row["class"])

    X = np.stack(images)
    y = np.array(labels)

    return X, y