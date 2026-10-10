import io
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main


client = TestClient(api_main.app)


def create_test_jpeg() -> bytes:
    """Create a small, valid JPEG image for testing."""
    buffer = io.BytesIO()

    image = Image.new("RGB", (32, 32), color=(80, 160, 60))
    image.save(buffer, format="JPEG")

    return buffer.getvalue()


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_success(monkeypatch):
    expected_result = {
        "predicted_class": "Leaf smut",
        "confidence": 0.8,
        "probabilities": {
            "Bacterial leaf blight": 0.1,
            "Brown spot": 0.1,
            "Leaf smut": 0.8,
        },
    }

    # Mock inference so this test focuses on API behavior.
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda image_path: expected_result,
    )

    response = client.post(
        "/predict",
        files={
            "file": (
                "rice_leaf.jpg",
                create_test_jpeg(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == expected_result
    assert sum(response.json()["probabilities"].values()) == pytest.approx(1.0)


def test_predict_rejects_unsupported_extension():
    response = client.post(
        "/predict",
        files={
            "file": (
                "document.txt",
                b"This is a text file.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsupported file extension. Upload JPG, JPEG or PNG."
    )


def test_predict_rejects_invalid_image():
    response = client.post(
        "/predict",
        files={
            "file": (
                "corrupted.jpg",
                b"This is not a valid image.",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The uploaded file is not a valid image."
    )


def test_predict_rejects_empty_upload():
    response = client.post(
        "/predict",
        files={
            "file": (
                "empty.jpg",
                b"",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The uploaded file is empty."
    )

def test_predict_rejects_oversized_upload():
    large_content = b"x" * (api_main.MAX_UPLOAD_SIZE + 1)

    response = client.post(
        "/predict",
        files={
            "file": (
                "large.jpg",
                large_content,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 413

def test_predict_rejects_missing_file():
    response = client.post("/predict")

    assert response.status_code == 422

def test_predict_returns_generic_error_when_inference_fails(monkeypatch):
    def fail_prediction(image_path):
        raise RuntimeError("internal model details")

    monkeypatch.setattr(api_main, "predict_image", fail_prediction)

    image_buffer = BytesIO()
    Image.new("RGB", (10, 10), color="green").save(
        image_buffer, format="JPEG"
    )

    response = client.post(
        "/predict",
        files={
            "file": (
                "leaf.jpg",
                image_buffer.getvalue(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Prediction failed. Please try again later."
    )
    assert "internal model details" not in response.text