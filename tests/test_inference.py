

from src.inference.predict import load_class_names


def test_class_names_exist():
    class_names = load_class_names()

    assert isinstance(class_names, list)
    assert len(class_names) == 3


def test_class_names_are_expected():
    class_names = load_class_names()

    assert class_names == [
        "Bacterial leaf blight",
        "Brown spot",
        "Leaf smut",
    ]