import io

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