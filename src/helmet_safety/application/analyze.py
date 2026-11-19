"""Use cases: analyse a single image or a stream of frames."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from helmet_safety.application.ports import FrameSink, FrameSource, Image, ImageAnnotator, ObjectDetector
from helmet_safety.domain.entities import ImageSize, SafetyReport
from helmet_safety.domain.errors import HelmetSafetyError, InferenceError, InvalidInputError
from helmet_safety.domain.policies import ConfidencePolicy
from helmet_safety.domain.services import ComplianceAnalyzer

MIN_IMAGE_SIDE = 16


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    report: SafetyReport
    inference_ms: float
    annotated: Image | None = None


def validate_image(image: object) -> Image:
    """Defensive check of the image contract used across the application."""
    if not isinstance(image, np.ndarray):
        raise InvalidInputError(f"Image must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise InvalidInputError(f"Image dtype must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidInputError(f"Image must have shape (H, W, 3), got {image.shape}")
    height, width = image.shape[:2]
    if height < MIN_IMAGE_SIDE or width < MIN_IMAGE_SIDE:
        raise InvalidInputError(f"Image too small ({width}x{height}); minimum side is {MIN_IMAGE_SIDE}px")
    return image


class AnalyzeImageUseCase:
    """detect -> filter by confidence policy -> compliance analysis -> (optional) annotate."""

    def __init__(
        self,
        detector: ObjectDetector,
        policy: ConfidencePolicy,
        analyzer: ComplianceAnalyzer,
        annotator: ImageAnnotator | None = None,
    ) -> None:
        self._detector = detector
        self._policy = policy
        self._analyzer = analyzer
        self._annotator = annotator

    def execute(self, image: object, *, annotate: bool = False) -> AnalysisResult:
        frame = validate_image(image)
        height, width = frame.shape[:2]

        try:
            self._detector.load()  # no-op when loaded; keeps model loading out of the latency metric
            started = time.perf_counter()
            raw = self._detector.detect(frame)
        except HelmetSafetyError:
            raise
        except Exception as exc:  # adapter bug or library failure -> typed error
            raise InferenceError(f"Detector failed: {exc}") from exc
        inference_ms = (time.perf_counter() - started) * 1000.0

        detections = [replace(d, box=d.box.clip(width, height)) for d in raw]
        accepted = self._policy.apply(detections)
        report = self._analyzer.analyze(accepted, ImageSize(width=width, height=height))

        annotated: Image | None = None
        if annotate:
            if self._annotator is None:
                raise InvalidInputError("Annotation requested but no annotator is configured")
            annotated = self._annotator.annotate(frame, report)
        return AnalysisResult(report=report, inference_ms=inference_ms, annotated=annotated)


@dataclass(frozen=True, slots=True)
class StreamSummary:
    frames: int
    unsafe_frames: int
    max_violations: int
    avg_inference_ms: float


class AnalyzeStreamUseCase:
    """Processes frames from a :class:`FrameSource`, writing annotated frames to sinks."""

    def __init__(self, analyze_image: AnalyzeImageUseCase) -> None:
        self._analyze_image = analyze_image

    def iterate(
        self, source: FrameSource, *, annotate: bool, max_frames: int | None = None
    ) -> Iterator[AnalysisResult]:
        if max_frames is not None and max_frames <= 0:
            raise InvalidInputError("max_frames must be a positive integer")
        try:
            for index, frame in enumerate(source.frames()):
                if max_frames is not None and index >= max_frames:
                    break
                yield self._analyze_image.execute(frame, annotate=annotate)
        finally:
            source.close()

    def run(
        self,
        source: FrameSource,
        sinks: list[FrameSink],
        *,
        max_frames: int | None = None,
    ) -> StreamSummary:
        frames = unsafe = max_violations = 0
        total_ms = 0.0
        try:
            for result in self.iterate(source, annotate=bool(sinks), max_frames=max_frames):
                frames += 1
                total_ms += result.inference_ms
                max_violations = max(max_violations, result.report.violation_count)
                if result.report.violation_count:
                    unsafe += 1
                if result.annotated is not None:
                    keep_going = [sink.write(result.annotated) for sink in sinks]
                    if not all(keep_going):
                        break
        finally:
            for sink in sinks:
                sink.close()
        return StreamSummary(
            frames=frames,
            unsafe_frames=unsafe,
            max_violations=max_violations,
            avg_inference_ms=total_ms / frames if frames else 0.0,
        )
