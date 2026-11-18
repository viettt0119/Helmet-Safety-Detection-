"""Ultralytics YOLO adapters implementing the detector / trainer / evaluator ports."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from helmet_safety.application.ports import (
    EvaluationRequest,
    EvaluationResult,
    Image,
    ModelInfo,
    TrainingRequest,
    TrainingResult,
)
from helmet_safety.domain.entities import BoundingBox, Detection, DetectionClass
from helmet_safety.domain.errors import InferenceError, ModelLoadError, ModelNotFoundError
from helmet_safety.infrastructure.logging import get_logger

logger = get_logger(__name__)

# Windows + conda can ship two OpenMP runtimes (torch & numpy); this avoids the hard crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

type ModelFactory = Callable[[str], Any]


@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Temporarily send Ultralytics console output (logger + progress bars) to stderr.

    Keeps stdout machine-readable for the CLI (``--json``). Everything is restored afterwards.
    """
    target = sys.stderr
    handlers = [h for h in logging.getLogger("ultralytics").handlers if isinstance(h, logging.StreamHandler)]
    previous = [h.setStream(target) for h in handlers]
    try:
        with contextlib.redirect_stdout(target):
            yield
    finally:
        for handler, stream in zip(handlers, previous, strict=True):
            if stream is not None:
                handler.setStream(stream)


def _default_model_factory(weights: str) -> Any:
    from ultralytics import YOLO  # type: ignore[attr-defined]

    return YOLO(weights)


def names_to_map(names: Mapping[Any, Any] | Sequence[Any]) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


class UltralyticsDetector:
    """Thread-safe, lazily-loaded YOLO detector.

    Ultralytics predictors keep mutable state, so concurrent ``predict`` calls on one model
    instance are serialised with a lock. Scale horizontally with more API workers/replicas.
    """

    def __init__(
        self,
        weights: Path,
        *,
        device: str,
        image_size: int = 640,
        confidence: float = 0.10,
        iou: float = 0.7,
        max_detections: int = 300,
        half: bool = False,
        model_factory: ModelFactory = _default_model_factory,
    ) -> None:
        self._weights = weights
        self._device = device
        self._image_size = image_size
        self._confidence = confidence
        self._iou = iou
        self._max_detections = max_detections
        self._half = half
        self._model_factory = model_factory
        self._model: Any | None = None
        self._class_map: dict[int, DetectionClass | None] = {}
        self._names: dict[int, str] = {}
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            weights_path=self._weights,
            device=self._device,
            image_size=self._image_size,
            class_names=dict(self._names),
        )

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self._weights.exists():
                raise ModelNotFoundError(f"Weights not found: {self._weights}")
            try:
                model = self._model_factory(str(self._weights))
            except Exception as exc:
                raise ModelLoadError(f"Cannot load model from {self._weights}: {exc}") from exc
            self._names = names_to_map(getattr(model, "names", {}) or {})
            self._class_map = {idx: DetectionClass.parse(name) for idx, name in self._names.items()}
            unknown = sorted(name for idx, name in self._names.items() if self._class_map[idx] is None)
            missing = sorted(
                {c.value for c in DetectionClass} - {c.value for c in self._class_map.values() if c}
            )
            if unknown:
                logger.warning("model_unknown_classes_ignored", classes=unknown)
            if missing:
                logger.warning("model_missing_expected_classes", classes=missing)
            self._model = model
            logger.info("model_loaded", weights=str(self._weights), device=self._device, classes=self._names)
            self._warmup()

    def _warmup(self) -> None:
        try:
            self._predict(np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8))
        except Exception as exc:
            logger.warning("model_warmup_failed", error=str(exc))

    def _predict(self, image: Image) -> Any:
        assert self._model is not None
        with self._predict_lock:
            kwargs: dict[str, Any] = {
                "imgsz": self._image_size,
                "conf": self._confidence,
                "iou": self._iou,
                "max_det": self._max_detections,
                "device": self._device,
                "verbose": False,
            }
            if self._half:  # only pass when enabled: the flag is deprecated in recent Ultralytics
                kwargs["half"] = True
            return self._model.predict(source=image, **kwargs)

    def detect(self, image: Image) -> list[Detection]:
        self.load()
        try:
            results = self._predict(image)
        except Exception as exc:
            raise InferenceError(f"YOLO prediction failed: {exc}") from exc
        if not results:
            return []
        return self._convert(results[0])

    def _convert(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        cls_ids = _to_numpy(boxes.cls).astype(int)
        confs = _to_numpy(boxes.conf).astype(float)
        xyxy = _to_numpy(boxes.xyxy).astype(float)
        detections: list[Detection] = []
        for cls_id, conf, coords in zip(cls_ids, confs, xyxy, strict=True):
            label = self._class_map.get(int(cls_id))
            if label is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in coords)
            detections.append(
                Detection(
                    label=label,
                    confidence=min(max(float(conf), 0.0), 1.0),
                    box=BoundingBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                )
            )
        return detections


def _to_numpy(value: Any) -> np.ndarray[Any, Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


class UltralyticsTrainer:
    def __init__(self, model_factory: ModelFactory = _default_model_factory) -> None:
        self._model_factory = model_factory

    def train(self, request: TrainingRequest) -> TrainingResult:
        if not request.base_weights.is_file():
            raise ModelNotFoundError(f"Base weights not found: {request.base_weights}")
        model = self._model_factory(str(request.base_weights))
        logger.info("training_started", run=request.run_name, device=request.device, epochs=request.epochs)
        with stdout_to_stderr():
            self._run(model, request)
        trainer = getattr(model, "trainer", None)
        run_dir = Path(getattr(trainer, "save_dir", request.project_dir / request.run_name))
        best = Path(getattr(trainer, "best", run_dir / "weights" / "best.pt"))
        raw_metrics = getattr(trainer, "metrics", {}) or {}
        metrics = {str(k): float(v) for k, v in dict(raw_metrics).items() if _is_number(v)}
        logger.info("training_finished", run_dir=str(run_dir), best=str(best), metrics=metrics)
        return TrainingResult(run_dir=run_dir, best_weights=best, metrics=metrics)

    @staticmethod
    def _run(model: Any, request: TrainingRequest) -> None:
        model.train(
            data=str(request.dataset_yaml),
            epochs=request.epochs,
            imgsz=request.image_size,
            batch=request.batch,
            workers=request.workers,
            device=request.device,
            project=str(request.project_dir),
            name=request.run_name,
            patience=request.patience,
            seed=request.seed,
            exist_ok=False,
            **dict(request.extra),
        )


class UltralyticsEvaluator:
    def __init__(self, model_factory: ModelFactory = _default_model_factory) -> None:
        self._model_factory = model_factory

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        model = self._model_factory(str(request.weights))
        with stdout_to_stderr():
            metrics = model.val(
                data=str(request.dataset_yaml),
                split=request.split,
                imgsz=request.image_size,
                batch=request.batch,
                device=request.device,
                project=str(request.project_dir),
                name=f"eval_{request.split}",
                exist_ok=True,
                plots=True,
                verbose=False,
            )
        box = metrics.box
        names = names_to_map(getattr(metrics, "names", None) or getattr(model, "names", {}) or {})
        maps = list(getattr(box, "maps", []))
        per_class = {names.get(i, str(i)): float(v) for i, v in enumerate(maps)}
        save_dir = getattr(metrics, "save_dir", None)
        return EvaluationResult(
            precision=float(box.mp),
            recall=float(box.mr),
            map50=float(box.map50),
            map50_95=float(box.map),
            per_class_map50_95=per_class,
            save_dir=Path(save_dir) if save_dir else None,
        )


def _is_number(value: object) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True
