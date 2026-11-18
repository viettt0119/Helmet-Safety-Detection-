from __future__ import annotations

import pytest

from helmet_safety.domain.entities import ComplianceStatus, Detection, ImageSize, OverallStatus
from helmet_safety.domain.services import ComplianceAnalyzer
from tests.conftest import det

SIZE = ImageSize(640, 480)


def analyze(detections: list[Detection], **kwargs: object) -> tuple[ComplianceStatus, ...]:
    report = ComplianceAnalyzer(**kwargs).analyze(detections, SIZE)  # type: ignore[arg-type]
    return tuple(w.status for w in report.workers)


def test_person_with_helmet_is_compliant(worker_with_helmet: list[Detection]) -> None:
    report = ComplianceAnalyzer().analyze(worker_with_helmet, SIZE)
    assert [w.status for w in report.workers] == [ComplianceStatus.COMPLIANT]
    assert report.workers[0].helmet is worker_with_helmet[1]
    assert report.status is OverallStatus.SAFE


def test_person_with_bare_head_is_violation(worker_without_helmet: list[Detection]) -> None:
    report = ComplianceAnalyzer().analyze(worker_without_helmet, SIZE)
    assert [w.status for w in report.workers] == [ComplianceStatus.VIOLATION]
    assert report.status is OverallStatus.UNSAFE


def test_mixed_scene(worker_with_helmet: list[Detection], worker_without_helmet: list[Detection]) -> None:
    report = ComplianceAnalyzer().analyze(worker_with_helmet + worker_without_helmet, SIZE)
    assert sorted(w.status for w in report.workers) == [
        ComplianceStatus.COMPLIANT,
        ComplianceStatus.VIOLATION,
    ]
    assert report.compliance_rate == 0.5


def test_person_without_visible_head_is_unknown() -> None:
    assert analyze([det("person", 0.9, 0, 0, 100, 300)]) == (ComplianceStatus.UNKNOWN,)


def test_helmet_in_lower_body_is_not_matched() -> None:
    # Helmet carried in the hand (lower half of the person box) must not make the worker compliant.
    statuses = analyze(
        [det("person", 0.9, 0, 0, 100, 300), det("helmet", 0.9, 10, 250, 40, 280)],
        count_unmatched_helmets=False,
    )
    assert statuses == (ComplianceStatus.UNKNOWN,)


def test_each_helmet_is_assigned_once() -> None:
    # Two overlapping persons, a single helmet: only one can be compliant.
    statuses = analyze(
        [
            det("person", 0.9, 0, 0, 100, 300),
            det("person", 0.8, 10, 0, 110, 300),
            det("helmet", 0.9, 40, 0, 70, 30),
        ]
    )
    assert sorted(statuses) == [ComplianceStatus.COMPLIANT, ComplianceStatus.UNKNOWN]


def test_conflicting_helmet_and_head_uses_confidence() -> None:
    person = det("person", 0.9, 0, 0, 100, 300)
    helmet = det("helmet", 0.4, 30, 0, 70, 40)
    head = det("head", 0.8, 30, 0, 70, 40)
    assert analyze([person, helmet, head]) == (ComplianceStatus.VIOLATION,)
    assert analyze([person, det("helmet", 0.9, 30, 0, 70, 40), head]) == (ComplianceStatus.COMPLIANT,)


def test_unmatched_head_and_helmet_close_ups() -> None:
    head = det("head", 0.8, 0, 0, 50, 50)
    helmet = det("helmet", 0.8, 300, 0, 350, 50)
    assert sorted(analyze([head, helmet])) == [ComplianceStatus.COMPLIANT, ComplianceStatus.VIOLATION]
    assert analyze([head, helmet], count_unmatched_heads=False) == (ComplianceStatus.COMPLIANT,)
    assert analyze([head, helmet], count_unmatched_helmets=False) == (ComplianceStatus.VIOLATION,)


def test_duplicate_unmatched_head_and_helmet_counted_once() -> None:
    head = det("head", 0.6, 0, 0, 50, 50)
    helmet = det("helmet", 0.9, 2, 2, 52, 52)
    assert analyze([head, helmet]) == (ComplianceStatus.COMPLIANT,)


def test_no_detections_is_empty() -> None:
    report = ComplianceAnalyzer().analyze([], SIZE)
    assert report.workers == ()
    assert report.status is OverallStatus.EMPTY


@pytest.mark.parametrize(("field", "value"), [("head_region_ratio", 0.0), ("min_containment", 1.5)])
def test_invalid_parameters(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ComplianceAnalyzer(**{field: value})  # type: ignore[arg-type]
