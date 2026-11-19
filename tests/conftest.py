"""Shared fixtures and test doubles."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from helmet_safety.application.ports import Image, ModelInfo
from helmet_safety.domain.entities import BoundingBox, Detection, DetectionClass
from helmet_safety.infrastructure.config import PROJECT_ROOT, Settings, get_settings

TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "test" / "images"
REAL_WEIGHTS = PROJECT_ROOT / "models" / "helmet_yolov10n_best.pt"


def det(label: str, conf: float, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(DetectionClass(label), conf, BoundingBox(x1, y1, x2, y2))


class FakeDetector:
    """In-memory ObjectDetector returning canned detections."""

    def __init__(self, detections: Sequence[Detection] = (), *, error: Exception | None = None) -> None:
        self.detections = list(detections)
        self.error = error
        self.calls = 0
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(Path("fake.pt"), "cpu", 640, {0: "head", 1: "helmet", 2: "person"})

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def detect(self, image: Image) -> Sequence[Detection]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.detections)


class ListFrameSource:
    def __init__(self, frames: Sequence[Image]) -> None:
        self._frames = list(frames)
        self.closed = False

    def frames(self) -> Iterator[Image]:
        yield from self._frames

    def close(self) -> None:
        self.closed = True


class RecordingSink:
    def __init__(self, stop_after: int | None = None) -> None:
        self.frames: list[Image] = []
        self.closed = False
        self._stop_after = stop_after

    def write(self, frame: Image) -> bool:
        self.frames.append(frame)
        return self._stop_after is None or len(self.frames) < self._stop_after

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Never let a developer's .env / HELMET_* variables leak into tests."""
    import os

    for key in list(os.environ):
        if key.startswith("HELMET_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def image() -> Image:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def worker_with_helmet() -> list[Detection]:
    return [det("person", 0.9, 100, 100, 200, 400), det("helmet", 0.8, 130, 100, 170, 140)]


@pytest.fixture
def worker_without_helmet() -> list[Detection]:
    return [det("person", 0.9, 300, 100, 400, 400), det("head", 0.7, 330, 105, 370, 145)]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test", model={"runs_dir": tmp_path / "runs", "weights_path": tmp_path / "w.pt"}
    )
