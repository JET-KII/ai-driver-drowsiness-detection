"""Train the compact grayscale CNN described in the project report."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train(data_dir: Path, output_dir: Path, image_size: int, epochs: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    common = {
        "directory": str(data_dir),
        "target_size": (image_size, image_size),
        "color_mode": "grayscale",
        "class_mode": "categorical",
        "batch_size": 32,
        "seed": 42,
    }
    training = generator.flow_from_directory(subset="training", shuffle=True, **common)
    validation = generator.flow_from_directory(subset="validation", shuffle=False, **common)

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(training.num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        training,
        validation_data=validation,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = output_dir / f"driver_drowsiness_model_{stamp}.h5"
    model.save(model_path)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("Model accuracy over time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"accuracy_plot_{stamp}.png")
    print(f"Saved model to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=244)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.image_size, args.epochs)


if __name__ == "__main__":
    main()

