"""Core value objects of the helmet-compliance domain."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self


class DetectionClass(StrEnum):
    """Classes produced by the detector (see ``data/data.yaml``)."""

    HEAD = "head"  # a bare head, i.e. NO helmet
    HELMET = "helmet"
    PERSON = "person"

    @classmethod
    def parse(cls, raw: str) -> DetectionClass | None:
        """Map a raw model label to a known class (case/space-insensitive). Unknown -> ``None``."""
        normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {"hardhat": cls.HELMET, "hard_hat": cls.HELMET, "no_helmet": cls.HEAD}
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return None


class ComplianceStatus(StrEnum):
    """Compliance state of a single observed worker."""

    COMPLIANT = "compliant"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


class OverallStatus(StrEnum):
    """Aggregated status of a whole image / frame."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    INCONCLUSIVE = "inconclusive"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in absolute pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        coords = (self.x1, self.y1, self.x2, self.y2)
        if any(math.isnan(c) for c in coords):
            raise ValueError(f"Bounding box contains NaN: {coords}")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid bounding box (x2<x1 or y2<y1): {coords}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def intersection_area(self, other: BoundingBox) -> float:
        width = min(self.x2, other.x2) - max(self.x1, other.x1)
        height = min(self.y2, other.y2) - max(self.y1, other.y1)
        if width <= 0 or height <= 0:
            return 0.0
        return width * height

    def iou(self, other: BoundingBox) -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def containment_of(self, other: BoundingBox) -> float:
        """Fraction of ``other``'s area that lies inside ``self`` (0..1)."""
        if other.area <= 0:
            cx, cy = other.center
            return 1.0 if self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2 else 0.0
        return self.intersection_area(other) / other.area

    def top_region(self, ratio: float) -> BoundingBox:
        """The top ``ratio`` part of the box (where a head is expected for a standing person)."""
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        return BoundingBox(self.x1, self.y1, self.x2, self.y1 + self.height * ratio)

    def clip(self, width: int, height: int) -> Self:
        """Clamp the box to image bounds."""
        return type(self)(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        )

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detector output."""

    label: DetectionClass
    confidence: float
    box: BoundingBox

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Image size must be positive, got {self.width}x{self.height}")


@dataclass(frozen=True, slots=True)
class WorkerAssessment:
    """Compliance decision for one worker.

    ``person`` may be ``None`` when only a head/helmet is visible (close-up shots).
    """

    status: ComplianceStatus
    person: Detection | None = None
    helmet: Detection | None = None
    head: Detection | None = None

    @property
    def anchor_box(self) -> BoundingBox:
        """Most representative box for this worker (used for drawing / reporting)."""
        for candidate in (self.person, self.head, self.helmet):
            if candidate is not None:
                return candidate.box
        raise ValueError("WorkerAssessment has no detection attached")


@dataclass(frozen=True, slots=True)
class SafetyReport:
    """Result of analysing one image/frame."""

    image_size: ImageSize
    detections: tuple[Detection, ...]
    workers: tuple[WorkerAssessment, ...] = field(default_factory=tuple)

    def count(self, label: DetectionClass) -> int:
        return sum(1 for d in self.detections if d.label is label)

    def _count_status(self, status: ComplianceStatus) -> int:
        return sum(1 for w in self.workers if w.status is status)

    @property
    def compliant_count(self) -> int:
        return self._count_status(ComplianceStatus.COMPLIANT)

    @property
    def violation_count(self) -> int:
        return self._count_status(ComplianceStatus.VIOLATION)

    @property
    def unknown_count(self) -> int:
        return self._count_status(ComplianceStatus.UNKNOWN)

    @property
    def compliance_rate(self) -> float | None:
        """Compliant / (compliant + violation). ``None`` when nobody could be verified."""
        verified = self.compliant_count + self.violation_count
        return self.compliant_count / verified if verified else None

    @property
    def status(self) -> OverallStatus:
        if not self.workers:
            return OverallStatus.EMPTY
        if self.violation_count:
            return OverallStatus.UNSAFE
        if self.unknown_count:
            return OverallStatus.INCONCLUSIVE
        return OverallStatus.SAFE
