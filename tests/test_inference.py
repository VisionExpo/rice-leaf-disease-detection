import json

import numpy as np
import pytest

import src.inference.predict as inference


def test_class_names_exist():
    class_names = inference.load_class_names()

    assert isinstance(class_names, list)
    assert len(class_names) == 3


def test_class_names_are_expected():
    class_names = inference.load_class_names()

    assert class_names == [
        "Bacterial leaf blight",
        "Brown spot",
        "Leaf smut",
    ]


@pytest.mark.parametrize(
    "invalid_data",
    [
        "not-a-list",
        [],
        ["Bacterial leaf blight", 42],
        ["Bacterial leaf blight", "Bacterial leaf blight"],
        [""],
    ],
)
def test_load_class_names_rejects_invalid_data(
    tmp_path, monkeypatch, invalid_data
):
    class_names_path = tmp_path / "class_names.json"
    class_names_path.write_text(
        json.dumps(invalid_data),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        inference,
        "CLASS_NAMES_PATH",
        class_names_path,
    )

    inference.load_class_names.cache_clear()

    try:
        with pytest.raises(ValueError):
            inference.load_class_names()
    finally:
        inference.load_class_names.cache_clear()


def test_predict_image_rejects_mismatched_class_count(
    tmp_path, monkeypatch
):
    class FakeModel:
        def predict(self, image_batch, verbose=0):
            return np.array([[0.7, 0.2, 0.1]])

    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"test image")

    monkeypatch.setattr(
        inference,
        "preprocess_image",
        lambda path: np.zeros((128, 448, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda: FakeModel(),
    )
    monkeypatch.setattr(
        inference,
        "load_class_names",
        lambda: ["Bacterial leaf blight", "Brown spot"],
    )

    with pytest.raises(ValueError, match="probabilities"):
        inference.predict_image(image_path)


def test_predict_image_returns_expected_result(tmp_path, monkeypatch):
    class FakeModel:
        def predict(self, image_batch, verbose=0):
            return np.array([[0.8, 0.15, 0.05]])

    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"test image")

    class_names = [
        "Bacterial leaf blight",
        "Brown spot",
        "Leaf smut",
    ]

    monkeypatch.setattr(
        inference,
        "preprocess_image",
        lambda path: np.zeros((128, 448, 3), dtype=np.float32),
    )
    monkeypatch.setattr(inference, "load_model", lambda: FakeModel())
    monkeypatch.setattr(
        inference, "load_class_names", lambda: class_names
    )

    result = inference.predict_image(image_path)

    assert result["predicted_class"] == "Bacterial leaf blight"
    assert result["confidence"] == pytest.approx(0.8)
    assert result["probabilities"] == {
        "Bacterial leaf blight": pytest.approx(0.8),
        "Brown spot": pytest.approx(0.15),
        "Leaf smut": pytest.approx(0.05),
    }


def test_predict_image_rejects_missing_image(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.jpg"

    def fail_if_preprocessed(image_path):
        pytest.fail("Preprocessing should not run for a missing image.")

    monkeypatch.setattr(
        inference,
        "preprocess_image",
        fail_if_preprocessed,
    )

    with pytest.raises(FileNotFoundError, match="Image not found"):
        inference.predict_image(missing_path)