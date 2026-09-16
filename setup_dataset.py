"""Create the daily_objects dataset folder structure."""

from pathlib import Path

ROOT = Path("data/daily_objects")

FOLDERS = [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val",
]


def main():
    for folder in FOLDERS:
        path = ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        gitkeep = path / ".gitkeep"
        gitkeep.touch(exist_ok=True)

    yaml_path = ROOT / "data.yaml"
    if not yaml_path.is_file():
        print(f"Warning: {yaml_path} not found — create it before training.")

    print(f"Dataset folders ready under {ROOT}/")
    print("Next steps:")
    print("  1. Add photos to images/train and images/val")
    print("  2. Add YOLO labels to labels/train and labels/val")
    print("  3. Run: python train_model.py")
    print("See data/daily_objects/README.md for details.")


if __name__ == "__main__":
    main()
