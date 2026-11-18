"""Domain service: decide, per worker, whether a helmet is worn.

The model outputs three independent classes (``person``, ``helmet``, ``head`` = bare head).
:class:`ComplianceAnalyzer` associates helmets / bare heads with the person boxes that contain
them (upper part of the body) using a greedy, confidence-ordered matching, so every gear
detection is assigned to at most one person.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from helmet_safety.domain.entities import (
    ComplianceStatus,
    Detection,
    DetectionClass,
    ImageSize,
    SafetyReport,
    WorkerAssessment,
)


@dataclass(frozen=True, slots=True)
class ComplianceAnalyzer:
    """Associates head-level detections to persons and derives compliance.

    Args:
        head_region_ratio: top fraction of a person box where the head is searched for.
        min_containment: minimal fraction of the head/helmet box that must lie inside that region.
        count_unmatched_helmets: treat a helmet that belongs to no detected person as a compliant
            worker (close-up shots where the body is out of frame).
        count_unmatched_heads: treat a bare head without a person box as a violation.
    """

    head_region_ratio: float = 0.5
    min_containment: float = 0.5
    count_unmatched_helmets: bool = True
    count_unmatched_heads: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.head_region_ratio <= 1.0:
            raise ValueError(f"head_region_ratio must be in (0, 1], got {self.head_region_ratio}")
        if not 0.0 < self.min_containment <= 1.0:
            raise ValueError(f"min_containment must be in (0, 1], got {self.min_containment}")

    def analyze(self, detections: Sequence[Detection], image_size: ImageSize) -> SafetyReport:
        persons = _sorted_by_conf(d for d in detections if d.label is DetectionClass.PERSON)
        helmets = _sorted_by_conf(d for d in detections if d.label is DetectionClass.HELMET)
        heads = _sorted_by_conf(d for d in detections if d.label is DetectionClass.HEAD)

        used_helmets: set[int] = set()
        used_heads: set[int] = set()
        workers: list[WorkerAssessment] = []

        for person in persons:
            helmet = self._best_match(person, helmets, used_helmets)
            head = self._best_match(person, heads, used_heads)
            workers.append(self._assess(person, helmet, head))

        workers.extend(
            self._standalone_workers(
                [h for i, h in enumerate(heads) if i not in used_heads],
                [h for i, h in enumerate(helmets) if i not in used_helmets],
            )
        )

        return SafetyReport(image_size=image_size, detections=tuple(detections), workers=tuple(workers))

    def _best_match(
        self, person: Detection, candidates: Sequence[Detection], used: set[int]
    ) -> Detection | None:
        region = person.box.top_region(self.head_region_ratio)
        best_index: int | None = None
        best_score = 0.0
        for index, candidate in enumerate(candidates):
            if index in used:
                continue
            containment = region.containment_of(candidate.box)
            if containment < self.min_containment:
                continue
            score = containment * candidate.confidence
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            return None
        used.add(best_index)
        return candidates[best_index]

    def _standalone_workers(
        self, heads: Sequence[Detection], helmets: Sequence[Detection]
    ) -> list[WorkerAssessment]:
        """Head-level detections without a matching person (close-ups, occluded bodies).

        When a bare head and a helmet overlap strongly they describe the same head: the more
        confident detection wins, so one physical worker is never counted twice.
        """
        result: list[WorkerAssessment] = []
        if self.count_unmatched_heads:
            result.extend(
                WorkerAssessment(status=ComplianceStatus.VIOLATION, head=h)
                for h in heads
                if not _is_dominated(h, helmets)
            )
        if self.count_unmatched_helmets:
            result.extend(
                WorkerAssessment(status=ComplianceStatus.COMPLIANT, helmet=h)
                for h in helmets
                if not _is_dominated(h, heads)
            )
        return result

    @staticmethod
    def _assess(person: Detection, helmet: Detection | None, head: Detection | None) -> WorkerAssessment:
        if helmet is not None and head is not None:
            # Same head detected as both classes: trust the more confident prediction.
            status = (
                ComplianceStatus.COMPLIANT
                if helmet.confidence >= head.confidence
                else ComplianceStatus.VIOLATION
            )
        elif helmet is not None:
            status = ComplianceStatus.COMPLIANT
        elif head is not None:
            status = ComplianceStatus.VIOLATION
        else:
            status = ComplianceStatus.UNKNOWN
        return WorkerAssessment(status=status, person=person, helmet=helmet, head=head)


def _sorted_by_conf(detections: Iterable[Detection]) -> list[Detection]:
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


def _is_dominated(target: Detection, rivals: Sequence[Detection], iou_threshold: float = 0.5) -> bool:
    """True if a more confident rival detection covers the same region (duplicate on one head)."""
    return any(r.confidence > target.confidence and target.box.iou(r.box) >= iou_threshold for r in rivals)
