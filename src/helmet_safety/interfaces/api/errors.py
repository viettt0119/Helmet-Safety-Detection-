"""Global exception handling: every error leaves the API as a consistent JSON envelope."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from helmet_safety.domain.errors import (
    ConfigurationError,
    DatasetError,
    DownloadError,
    HelmetSafetyError,
    InferenceError,
    InferenceTimeoutError,
    InvalidInputError,
    ModelLoadError,
    ModelNotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    VideoSourceError,
)
from helmet_safety.infrastructure.logging import get_logger
from helmet_safety.interfaces.api.schemas import ErrorBody, ErrorResponse

logger = get_logger(__name__)

# Order matters: most specific first.
_STATUS_MAP: tuple[tuple[type[HelmetSafetyError], int], ...] = (
    (PayloadTooLargeError, status.HTTP_413_CONTENT_TOO_LARGE),
    (UnsupportedMediaError, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
    (InvalidInputError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (InferenceTimeoutError, status.HTTP_504_GATEWAY_TIMEOUT),
    (ModelNotFoundError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (ModelLoadError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (InferenceError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (VideoSourceError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (DatasetError, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (DownloadError, status.HTTP_502_BAD_GATEWAY),
    (ConfigurationError, status.HTTP_500_INTERNAL_SERVER_ERROR),
)


def status_for(exc: HelmetSafetyError) -> int:
    for exc_type, code in _STATUS_MAP:
        if isinstance(exc, exc_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _envelope(request: Request, code: str, message: str, http_status: int) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    body = ErrorResponse(error=ErrorBody(code=code, message=message, request_id=request_id))
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(status_code=http_status, content=body.model_dump(), headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HelmetSafetyError)
    async def _domain_error(request: Request, exc: HelmetSafetyError) -> JSONResponse:
        http_status = status_for(exc)
        log = logger.error if http_status >= 500 else logger.info
        log("request_failed", code=exc.code, error=exc.message, status=http_status)
        return _envelope(request, exc.code, exc.message, http_status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', '')}" for err in exc.errors()
        )
        return _envelope(request, "validation_error", details or "Invalid request", 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _envelope(request, f"http_{exc.status_code}", str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", error=str(exc))
        return _envelope(request, "internal_error", "Internal server error", 500)
