"""OpenCV adapters: decode/encode images, draw annotations, read/write video streams."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from helmet_safety.application.ports import Image
from helmet_safety.domain.entities import ComplianceStatus, DetectionClass, OverallStatus, SafetyReport
from helmet_safety.domain.errors import InvalidInputError, UnsupportedMediaError, VideoSourceError
from helmet_safety.infrastructure.logging import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
VIDEO_SUFFIXES: Final = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg"})
ALLOWED_IMAGE_TYPES: Final = frozenset(
    {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff", "application/octet-stream"}
)
MAX_PIXELS: Final = 40_000_000  # decompression-bomb guard (~40 MP)

# BGR colours
CLASS_COLORS: Final[dict[DetectionClass, tuple[int, int, int]]] = {
    DetectionClass.PERSON: (255, 128, 0),
    DetectionClass.HELMET: (0, 220, 255),
    DetectionClass.HEAD: (0, 0, 255),
}
STATUS_COLORS: Final[dict[ComplianceStatus, tuple[int, int, int]]] = {
    ComplianceStatus.COMPLIANT: (0, 200, 0),
    ComplianceStatus.VIOLATION: (0, 0, 255),
    ComplianceStatus.UNKNOWN: (160, 160, 160),
}
OVERALL_COLORS: Final[dict[OverallStatus, tuple[int, int, int]]] = {
    OverallStatus.SAFE: (0, 160, 0),
    OverallStatus.UNSAFE: (0, 0, 220),
    OverallStatus.INCONCLUSIVE: (0, 140, 255),
    OverallStatus.EMPTY: (90, 90, 90),
}


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def decode_image(data: bytes) -> Image:
    """Decode encoded image bytes (JPEG/PNG/...) into a BGR uint8 array."""
    if not data:
        raise InvalidInputError("Empty image payload")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise UnsupportedMediaError("Cannot decode image: unsupported or corrupted file")
    return _normalize(image)


def read_image(path: Path) -> Image:
    if not path.is_file():
        raise InvalidInputError(f"Image file not found: {path}")
    if not is_image_path(path):
        raise UnsupportedMediaError(f"Unsupported image extension: {path.suffix}")
    return decode_image(path.read_bytes())


def _normalize(image: np.ndarray) -> Image:
    if image.dtype == np.uint16:
        image = (image / 257).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise UnsupportedMediaError(f"Unsupported image bit depth: {image.dtype}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise UnsupportedMediaError(f"Unsupported image shape: {image.shape}")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise InvalidInputError(f"Image too large: {image.shape[1]}x{image.shape[0]} (max {MAX_PIXELS} px)")
    return np.ascontiguousarray(image)


def encode_jpeg(image: Image, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise InvalidInputError("Failed to encode image as JPEG")
    return bytes(buffer.tobytes())


def write_image(path: Path, image: Image) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise InvalidInputError(f"Failed to encode image for {path}")
    path.write_bytes(buffer.tobytes())
    return path


class OpenCVAnnotator:
    """Draws per-worker compliance boxes, raw detections and a summary banner."""

    def __init__(self, *, line_width: int | None = None, draw_raw_detections: bool = True) -> None:
        self._line_width = line_width
        self._draw_raw = draw_raw_detections

    def annotate(self, image: Image, report: SafetyReport) -> Image:
        canvas = image.copy()
        height, width = canvas.shape[:2]
        thickness = self._line_width or max(1, round(min(height, width) / 320))
        font_scale = max(0.4, min(height, width) / 900)

        if self._draw_raw:
            for det in report.detections:
                if det.label is DetectionClass.PERSON:
                    continue  # persons are drawn via the worker overlay below
                x1, y1, x2, y2 = det.box.as_int_tuple()
                color = CLASS_COLORS[det.label]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
                _label(
                    canvas, f"{det.label.value} {det.confidence:.2f}", (x1, y1), color, font_scale, thickness
                )

        for worker in report.workers:
            x1, y1, x2, y2 = worker.anchor_box.as_int_tuple()
            color = STATUS_COLORS[worker.status]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness + 1)
            _label(
                canvas,
                worker.status.value.upper(),
                (x1, y2 + int(22 * font_scale) + 4),
                color,
                font_scale,
                thickness,
            )

        summary = (
            f"{report.status.value.upper()} | workers:{len(report.workers)} "
            f"ok:{report.compliant_count} no-helmet:{report.violation_count} ?:{report.unknown_count}"
        )
        banner_h = int(34 * font_scale) + 10
        cv2.rectangle(canvas, (0, 0), (width, banner_h), OVERALL_COLORS[report.status], -1)
        cv2.putText(
            canvas,
            summary,
            (8, banner_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness),
            cv2.LINE_AA,
        )
        return canvas


def _label(
    canvas: Image,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y = max(th + baseline, y)
    cv2.rectangle(canvas, (x, y - th - baseline), (x + tw, y), color, -1)
    cv2.putText(
        canvas,
        text,
        (x, y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def parse_source(source: str) -> int | str:
    """``"0"`` -> webcam index 0; anything else is a path / URL."""
    stripped = source.strip()
    return int(stripped) if stripped.isdigit() else stripped


class OpenCVFrameSource:
    """Frames from a webcam index, video file or network stream, with reconnect on live streams."""

    def __init__(
        self, source: int | str, *, reconnect_attempts: int = 3, reconnect_delay_s: float = 1.0
    ) -> None:
        self._source = source
        self._is_live = isinstance(source, int) or str(source).lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )
        if isinstance(source, str) and not self._is_live and not Path(source).is_file():
            raise VideoSourceError(f"Video file not found: {source}")
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay_s = reconnect_delay_s
        self._capture: cv2.VideoCapture | None = None

    @property
    def fps(self) -> float:
        capture = self._open()
        value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return value if 1.0 <= value <= 240.0 else 25.0

    def _open(self) -> cv2.VideoCapture:
        if self._capture is not None and self._capture.isOpened():
            return self._capture
        capture = cv2.VideoCapture(self._source)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Cannot open video source: {self._source}")
        self._capture = capture
        return capture

    def frames(self) -> Iterator[Image]:
        capture = self._open()
        failures = 0
        while True:
            ok, frame = capture.read()
            if ok and frame is not None:
                failures = 0
                yield _normalize(frame)
                continue
            if not self._is_live:
                return  # end of file
            failures += 1
            if failures > self._reconnect_attempts:
                raise VideoSourceError(
                    f"Lost video stream after {self._reconnect_attempts} reconnect attempts"
                )
            logger.warning("video_reconnect", source=str(self._source), attempt=failures)
            time.sleep(self._reconnect_delay_s * failures)
            self.close()
            capture = self._open()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class WindowSink:
    """Shows frames in a GUI window; returns False when the user presses q / ESC."""

    def __init__(self, title: str = "Helmet Safety Detection") -> None:
        self._title = title

    def write(self, frame: Image) -> bool:
        cv2.imshow(self._title, frame)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        with contextlib.suppress(cv2.error):
            cv2.destroyWindow(self._title)


class VideoFileSink:
    """Writes annotated frames to an MP4 file (writer lazily created on first frame)."""

    def __init__(self, path: Path, fps: float = 25.0) -> None:
        self._path = path
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: Image) -> bool:
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (width, height))
            if not self._writer.isOpened():
                raise VideoSourceError(f"Cannot open video writer for {self._path}")
        self._writer.write(frame)
        return True

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
