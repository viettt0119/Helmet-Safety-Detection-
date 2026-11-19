from __future__ import annotations

import pytest

from helmet_safety.domain.entities import (
    BoundingBox,
    ComplianceStatus,
    Detection,
    DetectionClass,
    ImageSize,
    OverallStatus,
    SafetyReport,
    WorkerAssessment,
)
from tests.conftest import det


class TestBoundingBox:
    def test_geometry(self) -> None:
        box = BoundingBox(10, 20, 30, 60)
        assert (box.width, box.height, box.area) == (20, 40, 800)
        assert box.center == (20, 40)

    @pytest.mark.parametrize("coords", [(10, 0, 5, 10), (0, 10, 10, 5), (float("nan"), 0, 1, 1)])
    def test_rejects_invalid_coordinates(self, coords: tuple[float, float, float, float]) -> None:
        with pytest.raises(ValueError, match="box"):
            BoundingBox(*coords)

    def test_iou_identical_disjoint_partial(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(BoundingBox(20, 20, 30, 30)) == 0.0
        assert a.iou(BoundingBox(5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_iou_degenerate_boxes(self) -> None:
        point = BoundingBox(1, 1, 1, 1)
        assert point.iou(point) == 0.0

    def test_containment(self) -> None:
        outer = BoundingBox(0, 0, 100, 100)
        assert outer.containment_of(BoundingBox(10, 10, 20, 20)) == 1.0
        assert outer.containment_of(BoundingBox(90, 0, 110, 10)) == pytest.approx(0.5)
        assert outer.containment_of(BoundingBox(50, 50, 50, 50)) == 1.0
        assert outer.containment_of(BoundingBox(150, 150, 150, 150)) == 0.0

    def test_top_region(self) -> None:
        assert BoundingBox(0, 100, 50, 300).top_region(0.25) == BoundingBox(0, 100, 50, 150)
        with pytest.raises(ValueError, match="ratio"):
            BoundingBox(0, 0, 1, 1).top_region(0)

    def test_clip(self) -> None:
        assert BoundingBox(-5, -5, 700, 500).clip(640, 480) == BoundingBox(0, 0, 640, 480)

    def test_as_int_tuple(self) -> None:
        assert BoundingBox(1.4, 1.6, 2.5, 3.5).as_int_tuple() == (1, 2, 2, 4)


class TestDetectionClass:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("helmet", DetectionClass.HELMET),
            (" Person ", DetectionClass.PERSON),
            ("HEAD", DetectionClass.HEAD),
            ("hard-hat", DetectionClass.HELMET),
            ("no_helmet", DetectionClass.HEAD),
            ("vest", None),
        ],
    )
    def test_parse(self, raw: str, expected: DetectionClass | None) -> None:
        assert DetectionClass.parse(raw) is expected


def test_detection_confidence_bounds() -> None:
    box = BoundingBox(0, 0, 1, 1)
    with pytest.raises(ValueError, match="confidence"):
        Detection(DetectionClass.HEAD, 1.5, box)
    with pytest.raises(ValueError, match="confidence"):
        Detection(DetectionClass.HEAD, -0.1, box)


def test_image_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        ImageSize(0, 10)


def test_worker_anchor_box_priority() -> None:
    person = det("person", 0.9, 0, 0, 10, 10)
    head = det("head", 0.9, 1, 1, 2, 2)
    assert WorkerAssessment(ComplianceStatus.VIOLATION, person=person, head=head).anchor_box == person.box
    assert WorkerAssessment(ComplianceStatus.VIOLATION, head=head).anchor_box == head.box
    with pytest.raises(ValueError, match="no detection"):
        _ = WorkerAssessment(ComplianceStatus.UNKNOWN).anchor_box


class TestSafetyReport:
    size = ImageSize(100, 100)

    def _report(self, *statuses: ComplianceStatus) -> SafetyReport:
        helmet = det("helmet", 0.9, 0, 0, 1, 1)
        return SafetyReport(self.size, (helmet,), tuple(WorkerAssessment(s, helmet=helmet) for s in statuses))

    def test_empty(self) -> None:
        report = self._report()
        assert report.status is OverallStatus.EMPTY
        assert report.compliance_rate is None

    def test_safe(self) -> None:
        report = self._report(ComplianceStatus.COMPLIANT, ComplianceStatus.COMPLIANT)
        assert report.status is OverallStatus.SAFE
        assert report.compliance_rate == 1.0

    def test_unsafe_wins_over_unknown(self) -> None:
        report = self._report(
            ComplianceStatus.COMPLIANT, ComplianceStatus.VIOLATION, ComplianceStatus.UNKNOWN
        )
        assert report.status is OverallStatus.UNSAFE
        assert report.compliance_rate == 0.5
        assert (report.compliant_count, report.violation_count, report.unknown_count) == (1, 1, 1)

    def test_inconclusive(self) -> None:
        assert (
            self._report(ComplianceStatus.COMPLIANT, ComplianceStatus.UNKNOWN).status
            is OverallStatus.INCONCLUSIVE
        )

    def test_count(self) -> None:
        assert self._report().count(DetectionClass.HELMET) == 1
        assert self._report().count(DetectionClass.HEAD) == 0
