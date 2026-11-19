"""Ports (interfaces) the application layer depends on. Infrastructure provides adapters.

Images are ``numpy`` arrays in BGR order with shape ``(H, W, 3)`` and dtype ``uint8``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from helmet_safety.domain.entities import Detection, SafetyReport

type Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ModelInfo:
    weights_path: Path
    device: str
    image_size: int
    class_names: Mapping[int, str]


@runtime_checkable
class ObjectDetector(Protocol):
    """Runs the ML model on one image and returns typed detections."""

    @property
    def info(self) -> ModelInfo: ...

    def load(self) -> None:
        """Eagerly load / warm up the model. Must be idempotent."""

    @property
    def is_loaded(self) -> bool: ...

    def detect(self, image: Image) -> Sequence[Detection]: ...


class ImageAnnotator(Protocol):
    def annotate(self, image: Image, report: SafetyReport) -> Image: ...


class FrameSource(Protocol):
    """Iterable stream of frames (webcam, video file, RTSP...). Must be closed after use."""

    def frames(self) -> Iterator[Image]: ...

    def close(self) -> None: ...


class FrameSink(Protocol):
    """Destination for annotated frames (window, video file...)."""

    def write(self, frame: Image) -> bool:
        """Write a frame. Return ``False`` to request the stream to stop (e.g. user pressed q)."""

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    dataset_yaml: Path
    base_weights: Path
    epochs: int
    image_size: int
    batch: int
    workers: int
    device: str
    project_dir: Path
    run_name: str
    patience: int = 20
    seed: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run_dir: Path
    best_weights: Path
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    dataset_yaml: Path
    weights: Path
    split: str
    image_size: int
    batch: int
    device: str
    project_dir: Path


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    precision: float
    recall: float
    map50: float
    map50_95: float
    per_class_map50_95: Mapping[str, float]
    save_dir: Path | None = None


class ModelTrainer(Protocol):
    def train(self, request: TrainingRequest) -> TrainingResult: ...


class ModelEvaluator(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...
