import tensorflow as tf


def create_augmentation_pipeline() -> tf.keras.Sequential:
    """Create the training-only image augmentation pipeline."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal"
            ),
            tf.keras.layers.RandomRotation(
                factor=0.05
            ),
            tf.keras.layers.RandomZoom(
                height_factor=0.10,
                width_factor=0.10
            ),
            tf.keras.layers.RandomContrast(
                factor=0.10
            ),
            tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, 0.0, 1.0)
            )
        ],
        name="data_augmentation",
    )