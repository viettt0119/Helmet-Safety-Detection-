"""Huan luyen mo hinh Helmet Safety Detection bang YOLOv10n tren moi truong local (Python 3.14)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_PRETRAINED = MODELS_DIR / "yolov10n.pt"
WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10n.pt"


def ensure_weights(weights_path: Path) -> None:
    # Tai trong so pretrain neu chua co.
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        print(f"[INFO] Da co trong so: {weights_path}")
        return

    print(f"[INFO] Dang tai trong so tu: {WEIGHT_URL}")
    urlretrieve(WEIGHT_URL, weights_path)
    print(f"[INFO] Tai xong: {weights_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning YOLOv10n for Helmet Safety Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 or 16)")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', or CUDA index like '0'",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_PRETRAINED, help="Path to pretrained weights")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs", help="Output project directory")
    parser.add_argument("--name", type=str, default="helmet_yolov10n", help="Run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data.yaml: {DATA_YAML}")

    ensure_weights(args.weights)

    # API on dinh tren ultralytics (Python 3.14): class YOLO.
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    # Tu dong chon thiet bi de tranh loi khi may khong co CUDA.
    requested_device = args.device.lower()
    if requested_device == "auto":
        try:
            import torch

            runtime_device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            runtime_device = "cpu"
    else:
        runtime_device = args.device
        if runtime_device != "cpu":
            try:
                import torch

                if not torch.cuda.is_available():
                    print("[WARN] CUDA khong kha dung, fallback sang CPU.")
                    runtime_device = "cpu"
            except Exception:
                print("[WARN] Khong the kiem tra CUDA, fallback sang CPU.")
                runtime_device = "cpu"

    print(f"[INFO] Start training on device={runtime_device}...")
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=runtime_device,
        project=str(args.project),
        name=args.name,
    )

    best_weight = args.project / args.name / "weights" / "best.pt"
    print(f"[INFO] Training done. Best weight should be at: {best_weight}")


if __name__ == "__main__":
    # Bat buoc tren Windows de tranh loi multiprocessing cua PyTorch DataLoader.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
