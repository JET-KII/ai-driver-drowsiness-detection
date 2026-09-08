"""Convert a Keras H5 model to TensorFlow Lite."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def convert(source: Path, destination: Path) -> None:
    model = tf.keras.models.load_model(source)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    destination.write_bytes(converter.convert())
    print(f"Saved TensorFlow Lite model to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()

