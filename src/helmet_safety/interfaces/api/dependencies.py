"""FastAPI dependency providers (thin adapters over the DI container stored on app.state)."""

from __future__ import annotations

from typing import Annotated, cast

import anyio
from fastapi import Depends, Request

from helmet_safety.infrastructure.container import Container


def get_container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


def get_limiter(request: Request) -> anyio.CapacityLimiter:
    return cast(anyio.CapacityLimiter, request.app.state.inference_limiter)


ContainerDep = Annotated[Container, Depends(get_container)]
LimiterDep = Annotated[anyio.CapacityLimiter, Depends(get_limiter)]
