# VisionAI — Surveillance & Inventory Detection

Computer vision application built with **YOLOv8**, **PyTorch**, **OpenCV**, and **CustomTkinter**. Detects COCO objects for live webcam surveillance and static image inventory counting, with optional **custom daily-object** fine-tuning.

## Features

- **Fast live webcam**: YOLOv8s + threaded capture/inference (latest-frame-only)
- **High-quality static images**: YOLOv8x on COCO preset
- **Custom daily objects**: pen, toothbrush, keys, phone, and more (after training)
- Dual input: live webcam and static image file
- Multi-object detection with adjustable confidence (default 0.35 webcam / 0.50 images)
- GPU acceleration (CUDA + FP16 when available)

## Setup

```bash
pip install -r requirements.txt
python detectU.py
```

On first run, YOLOv8 weights download automatically.

## Performance tips

| Setting | Default | Why |
|---------|---------|-----|
| Live model | YOLOv8s | ~5× faster than YOLOv8x on CPU |
| Webcam imgsz | 480 | Lower latency; raise to 640 if GPU allows |
| Confidence | 0.35 (webcam) | Catches more small objects |
| Threading | Capture + inference | UI stays responsive |

For best FPS, install PyTorch with CUDA support. CPU-only works but expect ~5–15 FPS depending on hardware.

## Custom daily-object training

COCO confuses similar small items (e.g. pen vs toothbrush). Train a custom model:

```bash
python setup_dataset.py
# Label images — see data/daily_objects/README.md
python train_model.py
```

Then in the app, select **Daily Objects (custom)**. Weights load from `runs/detect/daily_objects/weights/best.pt`.

Recommended: 50+ images per class, `--imgsz 1280` for small objects.
