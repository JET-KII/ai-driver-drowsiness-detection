"""Balance image classes by reproducible random undersampling."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def balance_dataset(source_dir: Path, output_dir: Path, seed: int) -> None:
    categories = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not categories:
        raise ValueError(f"No class directories found in {source_dir}")

    images_by_class = {
        category.name: [
            path for path in category.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for category in categories
    }
    empty_classes = [name for name, images in images_by_class.items() if not images]
    if empty_classes:
        raise ValueError(f"Classes contain no images: {', '.join(empty_classes)}")

    sample_size = min(len(images) for images in images_by_class.values())
    rng = random.Random(seed)

    for class_name, images in images_by_class.items():
        destination = output_dir / class_name
        destination.mkdir(parents=True, exist_ok=True)
        for image in rng.sample(images, sample_size):
            shutil.copy2(image, destination / image.name)

    print(f"Balanced {len(categories)} classes to {sample_size} images each")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    balance_dataset(args.source_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()

