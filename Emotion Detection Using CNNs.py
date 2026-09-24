"""
Emotion Detection Using CNNs

CNN-based facial emotion classification pipeline supporting:
- Conv2D
- MaxPooling
- BatchNormalization
- Dropout
- Adam
- Categorical cross-entropy
- Optional MobileNetV2 transfer learning

Dataset format:
data/
    train/
        angry/
        happy/
        ...
    validation/
        angry/
        happy/
        ...

Install:
    pip install tensorflow

The model supports seven classes by default.
"""

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


IMG_SIZE = (96, 96)
BATCH_SIZE = 32
NUM_CLASSES = 7

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def load_data(data_dir="data"):
    root = Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        root / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        root / "validation",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    return train_ds, validation_ds


def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_mobilenet_transfer_model():
    """Transfer-learning alternative using MobileNetV2."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 127.5, offset=-1),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(model, train_ds, validation_ds, epochs=10):
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
    )

    return history


def predict_emotion(model, image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE,
    )

    array = tf.keras.utils.img_to_array(image)
    array = tf.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    index = int(tf.argmax(probabilities))

    print(f"Emotion: {EMOTIONS[index]}")
    print(f"Confidence: {probabilities[index]:.2%}")


def main():
    train_ds, validation_ds = load_data()

    print("\nBuilding CNN...")
    model = build_cnn()
    model.summary()

    train_model(
        model,
        train_ds,
        validation_ds,
        epochs=10,
    )

    loss, accuracy = model.evaluate(validation_ds, verbose=0)

    print(f"\nValidation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.2%}")

    model.save("emotion_cnn.keras")
    print("Saved model as emotion_cnn.keras")

    # Optional transfer-learning model:
    # transfer_model = build_mobilenet_transfer_model()
    # train_model(transfer_model, train_ds, validation_ds)


if __name__ == "__main__":
    main()
