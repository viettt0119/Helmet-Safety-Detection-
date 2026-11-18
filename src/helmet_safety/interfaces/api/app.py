"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import anyio
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from helmet_safety import __version__
from helmet_safety.infrastructure.config import Settings, get_settings
from helmet_safety.infrastructure.container import Container
from helmet_safety.infrastructure.logging import configure_logging, get_logger
from helmet_safety.interfaces.api.errors import register_exception_handlers
from helmet_safety.interfaces.api.routes import health_router, v1_router

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())
    container = container or Container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container
        app.state.inference_limiter = anyio.CapacityLimiter(settings.api.max_concurrency)
        logger.info("api_starting", environment=settings.environment, version=__version__)
        # Fail fast on missing weights; load + warm up the model off the event loop.
        await anyio.to_thread.run_sync(container.detector.load)
        logger.info(
            "api_ready",
            model=container.detector.info.weights_path.name,
            device=container.detector.info.device,
        )
        yield
        logger.info("api_stopped")

    app = FastAPI(
        title="Helmet Safety Detection API",
        version=__version__,
        description="Detect workers and verify helmet (PPE) compliance using YOLOv10.",
        lifespan=lifespan,
    )
    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Safety-Status", "X-Safety-Workers", "X-Safety-Violations"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if 0 < len(incoming) <= 128 and incoming.isprintable() else uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            request_id=request_id,
        )
        return response

    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(v1_router)
    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory helmet_safety.interfaces.api.app:app_factory``."""
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)
    return create_app(settings)
