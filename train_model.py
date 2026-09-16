"""
Fine-tune YOLOv8 on daily-use objects (pen, toothbrush, keys, etc.).

Usage:
    python setup_dataset.py          # create folder structure
    # add labeled images under data/daily_objects/
    python train_model.py            # train with defaults
    python train_model.py --epochs 100 --imgsz 1280
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("data/daily_objects/data.yaml")
DEFAULT_BASE_MODEL = "yolov8s.pt"
DEFAULT_PROJECT = "runs/detect"
DEFAULT_NAME = "daily_objects"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on daily-use objects")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_YAML,
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help="Base weights (yolov8s.pt or yolov8m.pt recommended)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (640–1280 helps small objects)",
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="cuda, cpu, or 0 for first GPU")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Output project dir")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Run name")
    return parser.parse_args()


def count_labeled_images(data_yaml: Path) -> int:
    """Rough count of images in train/val folders."""
    root = data_yaml.parent
    total = 0
    for split in ("train", "val"):
        split_path = root / "images" / split
        if split_path.is_dir():
            total += len(list(split_path.glob("*.jpg")))
            total += len(list(split_path.glob("*.jpeg")))
            total += len(list(split_path.glob("*.png")))
    return total


def main():
    args = parse_args()

    if not args.data.is_file():
        raise SystemExit(
            f"Dataset config not found: {args.data}\n"
            "Run: python setup_dataset.py"
        )

    image_count = count_labeled_images(args.data)
    if image_count == 0:
        raise SystemExit(
            "No training images found.\n"
            "Add labeled images to data/daily_objects/images/train/ "
            "and data/daily_objects/labels/train/ (see data/daily_objects/README.md)."
        )

    print(f"Found ~{image_count} images. Training {args.model} for {args.epochs} epochs...")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        # Small-object friendly settings
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        patience=20,
        save=True,
        plots=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    print("In detectU.py, select 'Daily Objects (custom)' to use the trained model.")


if __name__ == "__main__":
    main()
