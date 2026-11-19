from __future__ import annotations

import numpy as np
import pytest

from helmet_safety.application.analyze import AnalyzeImageUseCase, AnalyzeStreamUseCase, validate_image
from helmet_safety.application.ports import Image
from helmet_safety.domain.entities import ComplianceStatus, Detection, OverallStatus
from helmet_safety.domain.errors import InferenceError, InvalidInputError, ModelNotFoundError
from helmet_safety.domain.policies import ConfidencePolicy
from helmet_safety.domain.services import ComplianceAnalyzer
from helmet_safety.infrastructure.imaging import OpenCVAnnotator
from tests.conftest import FakeDetector, ListFrameSource, RecordingSink, det


def make_use_case(detector: FakeDetector, annotator: OpenCVAnnotator | None = None) -> AnalyzeImageUseCase:
    return AnalyzeImageUseCase(detector, ConfidencePolicy(default=0.35), ComplianceAnalyzer(), annotator)


class TestValidateImage:
    @pytest.mark.parametrize(
        "bad",
        [
            "not an array",
            np.zeros((32, 32, 3), dtype=np.float32),
            np.zeros((32, 32), dtype=np.uint8),
            np.zeros((32, 32, 4), dtype=np.uint8),
            np.zeros((8, 32, 3), dtype=np.uint8),
        ],
    )
    def test_rejects(self, bad: object) -> None:
        with pytest.raises(InvalidInputError):
            validate_image(bad)

    def test_accepts(self, image: Image) -> None:
        assert validate_image(image) is image


class TestAnalyzeImage:
    def test_happy_path(self, image: Image, worker_with_helmet: list[Detection]) -> None:
        detector = FakeDetector(worker_with_helmet)
        result = make_use_case(detector).execute(image)
        assert result.report.status is OverallStatus.SAFE
        assert result.report.image_size.width == 640
        assert result.inference_ms >= 0
        assert result.annotated is None
        assert detector.is_loaded

    def test_low_confidence_detections_are_filtered(self, image: Image) -> None:
        detector = FakeDetector([det("person", 0.9, 0, 0, 100, 300), det("head", 0.2, 30, 0, 70, 40)])
        result = make_use_case(detector).execute(image)
        assert [w.status for w in result.report.workers] == [ComplianceStatus.UNKNOWN]
        assert len(result.report.detections) == 1

    def test_boxes_are_clipped_to_image(self, image: Image) -> None:
        detector = FakeDetector([det("helmet", 0.9, -10, -10, 700, 500)])
        box = make_use_case(detector).execute(image).report.detections[0].box
        assert (box.x1, box.y1, box.x2, box.y2) == (0, 0, 640, 480)

    def test_annotation(self, image: Image, worker_without_helmet: list[Detection]) -> None:
        result = make_use_case(FakeDetector(worker_without_helmet), OpenCVAnnotator()).execute(
            image, annotate=True
        )
        assert result.annotated is not None
        assert result.annotated.shape == image.shape
        assert not np.array_equal(result.annotated, image)

    def test_annotation_without_annotator_fails(self, image: Image) -> None:
        with pytest.raises(InvalidInputError, match="annotator"):
            make_use_case(FakeDetector()).execute(image, annotate=True)

    def test_unexpected_detector_error_is_wrapped(self, image: Image) -> None:
        with pytest.raises(InferenceError, match="boom"):
            make_use_case(FakeDetector(error=RuntimeError("boom"))).execute(image)

    def test_domain_errors_pass_through(self, image: Image) -> None:
        with pytest.raises(ModelNotFoundError):
            make_use_case(FakeDetector(error=ModelNotFoundError("missing"))).execute(image)

    def test_invalid_image_never_reaches_detector(self) -> None:
        detector = FakeDetector()
        with pytest.raises(InvalidInputError):
            make_use_case(detector).execute(np.zeros((4, 4, 3), dtype=np.uint8))
        assert detector.calls == 0


class TestAnalyzeStream:
    def test_run_counts_and_closes(self, image: Image, worker_without_helmet: list[Detection]) -> None:
        source = ListFrameSource([image] * 3)
        sink = RecordingSink()
        use_case = AnalyzeStreamUseCase(make_use_case(FakeDetector(worker_without_helmet), OpenCVAnnotator()))
        summary = use_case.run(source, [sink])
        assert (summary.frames, summary.unsafe_frames, summary.max_violations) == (3, 3, 1)
        assert len(sink.frames) == 3
        assert source.closed
        assert sink.closed

    def test_sink_can_stop_stream(self, image: Image) -> None:
        sink = RecordingSink(stop_after=2)
        use_case = AnalyzeStreamUseCase(make_use_case(FakeDetector(), OpenCVAnnotator()))
        summary = use_case.run(ListFrameSource([image] * 10), [sink])
        assert summary.frames == 2

    def test_max_frames(self, image: Image) -> None:
        use_case = AnalyzeStreamUseCase(make_use_case(FakeDetector()))
        summary = use_case.run(ListFrameSource([image] * 10), [], max_frames=4)
        assert summary.frames == 4
        assert summary.unsafe_frames == 0

    def test_empty_stream(self) -> None:
        summary = AnalyzeStreamUseCase(make_use_case(FakeDetector())).run(ListFrameSource([]), [])
        assert summary.frames == 0
        assert summary.avg_inference_ms == 0.0

    def test_invalid_max_frames(self, image: Image) -> None:
        source = ListFrameSource([image])
        with pytest.raises(InvalidInputError):
            list(
                AnalyzeStreamUseCase(make_use_case(FakeDetector())).iterate(
                    source, annotate=False, max_frames=0
                )
            )

    def test_resources_closed_on_error(self, image: Image) -> None:
        source = ListFrameSource([image])
        sink = RecordingSink()
        use_case = AnalyzeStreamUseCase(
            make_use_case(FakeDetector(error=RuntimeError("x")), OpenCVAnnotator())
        )
        with pytest.raises(InferenceError):
            use_case.run(source, [sink])
        assert source.closed
        assert sink.closed
