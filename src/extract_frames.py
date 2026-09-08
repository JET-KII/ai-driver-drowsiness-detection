"""Extract resized frames from videos arranged in labeled directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


def extract_frames(dataset_dir: Path, output_dir: Path, size: tuple[int, int]) -> None:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for category in sorted(dataset_dir.iterdir()):
        if not category.is_dir():
            continue

        label = category.name.strip().lower().replace(" ", "_")
        destination = output_dir / label
        destination.mkdir(parents=True, exist_ok=True)

        for video_path in sorted(category.iterdir()):
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                print(f"Skipping unreadable video: {video_path}")
                continue

            frame_count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame = cv2.resize(frame, size)
                filename = f"{label}_{video_path.stem}_f{frame_count:06d}.jpg"
                cv2.imwrite(str(destination / filename), frame)
                frame_count += 1

            capture.release()
            print(f"Extracted {frame_count} frames from {video_path.name} into {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--width", type=int, default=244)
    parser.add_argument("--height", type=int, default=244)
    args = parser.parse_args()
    extract_frames(args.dataset_dir, args.output_dir, (args.width, args.height))


if __name__ == "__main__":
    main()

