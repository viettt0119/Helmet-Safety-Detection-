"""Realtime inference: hien thi khung cho 3 class helmet, person, head."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BEST = PROJECT_ROOT / "runs" / "helmet_yolov10n" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on image/video/webcam")
    parser.add_argument("--weights", type=Path, default=DEFAULT_BEST, help="Path to best.pt")
    parser.add_argument("--source", type=str, default="0", help="Image/video path or webcam index (default: 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.10, help="Base confidence for model predict")
    parser.add_argument("--person-conf", type=float, default=0.35, help="Min confidence for person")
    parser.add_argument("--helmet-conf", type=float, default=0.35, help="Min confidence for helmet")
    parser.add_argument("--head-conf", type=float, default=0.35, help="Min confidence for head")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or CUDA index like '0'")
    return parser.parse_args()


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def is_image_file(source: str) -> bool:
    suffix = Path(source).suffix.lower()
    return suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
    candidates = list((project_root / "runs").glob("*/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def names_to_map(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def draw_detections(result, names_map: dict[int, str], args: argparse.Namespace):
    frame = result.orig_img.copy()

    if result.boxes is None or len(result.boxes) == 0:
        cv2.putText(frame, "NO DETECTION", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2, cv2.LINE_AA)
        return frame

    cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()

    # Mau theo class
    colors = {
        "person": (255, 0, 0),     # Blue
        "helmet": (0, 255, 255),   # Yellow
        "head": (0, 255, 0),       # Green
    }

    counts = {"person": 0, "helmet": 0, "head": 0}

    for i, cid in enumerate(cls_ids):
        name = names_map.get(int(cid), str(cid)).lower()
        conf = float(confs[i])

        if name not in counts:
            continue

        if name == "person" and conf < args.person_conf:
            continue
        if name == "helmet" and conf < args.helmet_conf:
            continue
        if name == "head" and conf < args.head_conf:
            continue

        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        color = colors[name]
        label = f"{name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        counts[name] += 1

    summary = f"person:{counts['person']}  helmet:{counts['helmet']}  head:{counts['head']}"
    cv2.rectangle(frame, (10, 10), (520, 46), (20, 20, 20), -1)
    cv2.putText(frame, summary, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    return frame


def show_single_image(window_name: str, image) -> None:
    cv2.imshow(window_name, image)
    print("[INFO] Press any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_video_stream(window_name: str, result_stream, names_map: dict[int, str], args: argparse.Namespace) -> None:
    print("[INFO] Press 'q' or ESC to exit.")
    for result in result_stream:
        frame = draw_detections(result, names_map, args)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

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
        else:
            raise FileNotFoundError(
                f"Missing weights: {weights_path}. Please train first and use best.pt."
            )

    runtime_device = resolve_device(args.device)

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    names_map = names_to_map(model.names)

    print(f"[INFO] Using weights: {weights_path}")
    print(f"[INFO] Class mapping: {names_map}")

    required = {"person", "helmet", "head"}
    available = {v.lower() for v in names_map.values()}
    missing = required - available
    if missing:
        print(f"[WARN] Missing classes in model: {sorted(missing)}")

    source = parse_source(args.source)
    window_name = "Helmet Safety Detection - YOLO"

    if isinstance(source, str) and Path(source).exists() and is_image_file(source):
        results = model.predict(source=source, imgsz=args.imgsz, conf=args.conf, device=runtime_device)
        frame = draw_detections(results[0], names_map, args)
        show_single_image(window_name, frame)
        return

    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=runtime_device,
        stream=True,
    )
    show_video_stream(window_name, results, names_map, args)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
