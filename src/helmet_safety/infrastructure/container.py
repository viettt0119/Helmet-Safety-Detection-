"""Composition root: builds the object graph from :class:`Settings` (Dependency Injection)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from helmet_safety.application.analyze import AnalyzeImageUseCase, AnalyzeStreamUseCase
from helmet_safety.application.ports import ImageAnnotator, ModelEvaluator, ModelTrainer, ObjectDetector
from helmet_safety.application.training import EvaluateModelUseCase, TrainModelUseCase
from helmet_safety.infrastructure.config import Settings
from helmet_safety.infrastructure.device import resolve_device
from helmet_safety.infrastructure.imaging import OpenCVAnnotator
from helmet_safety.infrastructure.weights import WeightsDownloader, WeightsResolver
from helmet_safety.infrastructure.yolo import UltralyticsDetector, UltralyticsEvaluator, UltralyticsTrainer


@dataclass
class Container:
    """Lazily wires adapters into use cases. Any dependency can be overridden (tests)."""

    settings: Settings
    detector_override: ObjectDetector | None = None
    annotator_override: ImageAnnotator | None = None
    trainer_override: ModelTrainer | None = None
    evaluator_override: ModelEvaluator | None = None

    @cached_property
    def device(self) -> str:
        return resolve_device(self.settings.model.device)

    @cached_property
    def weights_resolver(self) -> WeightsResolver:
        model = self.settings.model
        return WeightsResolver(
            model.weights_path,
            runs_dir=model.runs_dir if model.auto_discover_weights else None,
        )

    @cached_property
    def downloader(self) -> WeightsDownloader:
        d = self.settings.download
        return WeightsDownloader(
            timeout_s=d.timeout_s,
            max_attempts=d.max_attempts,
            backoff_initial_s=d.backoff_initial_s,
            backoff_max_s=d.backoff_max_s,
        )

    @cached_property
    def detector(self) -> ObjectDetector:
        if self.detector_override is not None:
            return self.detector_override
        model = self.settings.model
        return UltralyticsDetector(
            self.weights_resolver.resolve(),
            device=self.device,
            image_size=model.image_size,
            confidence=model.base_confidence,
            iou=model.iou,
            max_detections=model.max_detections,
            half=model.half_precision and self.device not in {"cpu", "mps"},
        )

    @cached_property
    def annotator(self) -> ImageAnnotator:
        return self.annotator_override or OpenCVAnnotator()

    @cached_property
    def analyze_image(self) -> AnalyzeImageUseCase:
        policy = self.settings.policy
        return AnalyzeImageUseCase(
            detector=self.detector,
            policy=policy.to_confidence_policy(),
            analyzer=policy.to_analyzer(),
            annotator=self.annotator,
        )

    @cached_property
    def analyze_stream(self) -> AnalyzeStreamUseCase:
        return AnalyzeStreamUseCase(self.analyze_image)

    @cached_property
    def train_model(self) -> TrainModelUseCase:
        return TrainModelUseCase(self.trainer_override or UltralyticsTrainer())

    @cached_property
    def evaluate_model(self) -> EvaluateModelUseCase:
        return EvaluateModelUseCase(self.evaluator_override or UltralyticsEvaluator())
