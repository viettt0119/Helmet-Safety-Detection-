"""HTTP routes."""

from __future__ import annotations

from typing import Annotated

import anyio
from fastapi import APIRouter, File, Query, Response, UploadFile, status

from helmet_safety import __version__
from helmet_safety.application.analyze import AnalysisResult
from helmet_safety.domain.errors import (
    InferenceTimeoutError,
    InvalidInputError,
    PayloadTooLargeError,
    UnsupportedMediaError,
)
from helmet_safety.infrastructure.container import Container
from helmet_safety.infrastructure.imaging import ALLOWED_IMAGE_TYPES, decode_image, encode_jpeg
from helmet_safety.interfaces.api.dependencies import ContainerDep, LimiterDep
from helmet_safety.interfaces.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    ReadinessResponse,
)
from helmet_safety.interfaces.serialization import SafetyReportDTO

health_router = APIRouter(tags=["health"])
v1_router = APIRouter(prefix="/v1", tags=["detection"])

_ERRORS: dict[int | str, dict[str, object]] = {
    code: {"model": ErrorResponse} for code in (413, 415, 422, 500, 503, 504)
}
_READ_CHUNK = 1 << 20


@health_router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@health_router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
async def readiness(container: ContainerDep, response: Response) -> ReadinessResponse:
    loaded = container.detector.is_loaded
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if loaded else "loading", model_loaded=loaded)


@v1_router.get("/model", response_model=ModelInfoResponse)
async def model_info(container: ContainerDep) -> ModelInfoResponse:
    info = container.detector.info
    return ModelInfoResponse(
        weights=info.weights_path.name,
        device=info.device,
        image_size=info.image_size,
        class_names=dict(info.class_names),
        loaded=container.detector.is_loaded,
    )


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    content_type = (file.content_type or "application/octet-stream").split(";")[0].strip().lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise UnsupportedMediaError(f"Unsupported content type: {content_type}")
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_READ_CHUNK):
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLargeError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
        chunks.append(chunk)
    if total == 0:
        raise InvalidInputError("Uploaded file is empty")
    return b"".join(chunks)


async def _analyze(
    container: Container, limiter: anyio.CapacityLimiter, file: UploadFile, *, annotate: bool
) -> AnalysisResult:
    api = container.settings.api
    data = await _read_upload(file, api.max_upload_bytes)
    image = await anyio.to_thread.run_sync(decode_image, data)
    try:
        with anyio.fail_after(api.inference_timeout_s):
            return await anyio.to_thread.run_sync(
                lambda: container.analyze_image.execute(image, annotate=annotate),
                limiter=limiter,
                abandon_on_cancel=True,
            )
    except TimeoutError as exc:
        raise InferenceTimeoutError(f"Inference exceeded {api.inference_timeout_s:.1f}s") from exc


@v1_router.post("/detections", response_model=SafetyReportDTO, responses=_ERRORS)
async def detect(
    container: ContainerDep,
    limiter: LimiterDep,
    file: Annotated[UploadFile, File(description="Image file (JPEG/PNG/BMP/WEBP/TIFF)")],
) -> SafetyReportDTO:
    """Detect workers and decide helmet compliance for one image."""
    result = await _analyze(container, limiter, file, annotate=False)
    return SafetyReportDTO.from_domain(result.report, result.inference_ms)


@v1_router.post(
    "/detections/annotated",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, **_ERRORS},
)
async def detect_annotated(
    container: ContainerDep,
    limiter: LimiterDep,
    file: Annotated[UploadFile, File(description="Image file (JPEG/PNG/BMP/WEBP/TIFF)")],
    quality: Annotated[int | None, Query(ge=1, le=100)] = None,
) -> Response:
    """Same as ``/detections`` but returns the annotated JPEG; summary in ``X-Safety-*`` headers."""
    result = await _analyze(container, limiter, file, annotate=True)
    assert result.annotated is not None
    jpeg_quality = quality or container.settings.api.jpeg_quality
    body = await anyio.to_thread.run_sync(encode_jpeg, result.annotated, jpeg_quality)
    report = result.report
    return Response(
        content=body,
        media_type="image/jpeg",
        headers={
            "X-Safety-Status": report.status.value,
            "X-Safety-Workers": str(len(report.workers)),
            "X-Safety-Violations": str(report.violation_count),
            "X-Inference-Ms": f"{result.inference_ms:.2f}",
        },
    )
