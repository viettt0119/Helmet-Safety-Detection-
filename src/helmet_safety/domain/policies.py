"""Filtering policies applied to raw detector output (Strategy pattern)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from helmet_safety.domain.entities import Detection, DetectionClass


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    """Per-class minimum confidence. Classes missing from ``thresholds`` use ``default``."""

    default: float = 0.35
    thresholds: Mapping[DetectionClass, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in {
            "default": self.default,
            **{c.value: v for c, v in self.thresholds.items()},
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Threshold for {name!r} must be in [0, 1], got {value}")
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))

    def threshold_for(self, label: DetectionClass) -> float:
        return self.thresholds.get(label, self.default)

    def accepts(self, detection: Detection) -> bool:
        return detection.confidence >= self.threshold_for(detection.label)

    def apply(self, detections: Iterable[Detection]) -> list[Detection]:
        return [d for d in detections if self.accepts(d)]
