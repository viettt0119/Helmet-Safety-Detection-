"""Danh gia mo hinh da huan luyen tren tap test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"
DEFAULT_BEST = PROJECT_ROOT / "runs" / "helmet_yolov10n" / "weights" / "best.pt"
DEFAULT_FALLBACK = PROJECT_ROOT / "models" / "yolov10n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model on test split")
    parser.add_argument("--weights", type=Path, default=DEFAULT_BEST, help="Path to best.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or CUDA index like '0'")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split name")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    requested = device_arg.lower()
    if requested == "auto":
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    if requested != "cpu":
        try:
            import torch

            if not torch.cuda.is_available():
                print("[WARN] CUDA not available, fallback to CPU.")
                return "cpu"
        except Exception:
            print("[WARN] Cannot verify CUDA, fallback to CPU.")
            return "cpu"
    return device_arg


def find_latest_best(project_root: Path) -> Path | None:
    """Tim file best.pt moi nhat trong runs/*/weights/."""
    candidates = list((project_root / "runs").glob("*/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing data.yaml: {DATA_YAML}")

    weights_path = args.weights
    if not weights_path.exists():
        latest_best = find_latest_best(PROJECT_ROOT)
        if latest_best is not None:
            print(f"[WARN] Missing {weights_path}, using latest best: {latest_best}")
            weights_path = latest_best
        elif (PROJECT_ROOT / "best.pt").exists():
            weights_path = PROJECT_ROOT / "best.pt"
            print(f"[WARN] Missing default weights, using {weights_path}")
        elif (PROJECT_ROOT / "models" / "best.pt").exists():
            weights_path = PROJECT_ROOT / "models" / "best.pt"
            print(f"[WARN] Missing default weights, using {weights_path}")
        elif DEFAULT_FALLBACK.exists():
            print(f"[WARN] Missing {weights_path}, fallback to {DEFAULT_FALLBACK}")
            print("[WARN] Fallback weights are pretrained. Train first for accurate validation.")
            weights_path = DEFAULT_FALLBACK
        else:
            raise FileNotFoundError(
                f"Missing weights: {weights_path}. Run train.py first to create best.pt"
            )

    runtime_device = resolve_device(args.device)

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        split=args.split,
        device=runtime_device,
    )

    print("\n[INFO] Evaluation result:")
    print(metrics)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
