"""Pydantic DTOs shared by the REST API and the CLI JSON output."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from helmet_safety.domain.entities import (
    BoundingBox,
    ComplianceStatus,
    Detection,
    DetectionClass,
    OverallStatus,
    SafetyReport,
    WorkerAssessment,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class BoxDTO(_Frozen):
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_domain(cls, box: BoundingBox) -> BoxDTO:
        return cls(x1=round(box.x1, 2), y1=round(box.y1, 2), x2=round(box.x2, 2), y2=round(box.y2, 2))


class DetectionDTO(_Frozen):
    label: DetectionClass
    confidence: float = Field(ge=0, le=1)
    box: BoxDTO

    @classmethod
    def from_domain(cls, det: Detection) -> DetectionDTO:
        return cls(label=det.label, confidence=round(det.confidence, 4), box=BoxDTO.from_domain(det.box))


class WorkerDTO(_Frozen):
    status: ComplianceStatus
    box: BoxDTO
    person: DetectionDTO | None = None
    helmet: DetectionDTO | None = None
    head: DetectionDTO | None = None

    @classmethod
    def from_domain(cls, worker: WorkerAssessment) -> WorkerDTO:
        def conv(d: Detection | None) -> DetectionDTO | None:
            return DetectionDTO.from_domain(d) if d is not None else None

        return cls(
            status=worker.status,
            box=BoxDTO.from_domain(worker.anchor_box),
            person=conv(worker.person),
            helmet=conv(worker.helmet),
            head=conv(worker.head),
        )


class SummaryDTO(_Frozen):
    workers: int
    compliant: int
    violations: int
    unknown: int
    compliance_rate: float | None
    counts: dict[DetectionClass, int]


class SafetyReportDTO(_Frozen):
    status: OverallStatus
    image_width: int
    image_height: int
    inference_ms: float
    summary: SummaryDTO
    workers: list[WorkerDTO]
    detections: list[DetectionDTO]

    @classmethod
    def from_domain(cls, report: SafetyReport, inference_ms: float) -> SafetyReportDTO:
        rate = report.compliance_rate
        return cls(
            status=report.status,
            image_width=report.image_size.width,
            image_height=report.image_size.height,
            inference_ms=round(inference_ms, 2),
            summary=SummaryDTO(
                workers=len(report.workers),
                compliant=report.compliant_count,
                violations=report.violation_count,
                unknown=report.unknown_count,
                compliance_rate=round(rate, 4) if rate is not None else None,
                counts={c: report.count(c) for c in DetectionClass},
            ),
            workers=[WorkerDTO.from_domain(w) for w in report.workers],
            detections=[DetectionDTO.from_domain(d) for d in report.detections],
        )
