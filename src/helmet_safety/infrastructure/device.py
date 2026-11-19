"""Compute-device resolution with graceful CPU fallback."""

from __future__ import annotations

from collections.abc import Callable

from helmet_safety.infrastructure.logging import get_logger

logger = get_logger(__name__)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch missing or broken driver
        return False


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_device(
    requested: str,
    *,
    cuda_available: Callable[[], bool] = _cuda_available,
    mps_available: Callable[[], bool] = _mps_available,
) -> str:
    """Translate a user device request into a value Ultralytics accepts.

    ``auto`` -> first CUDA GPU, else Apple MPS, else CPU. Explicit GPU requests fall back to CPU
    (with a warning) when CUDA is unavailable instead of crashing.
    """
    value = requested.strip().lower()
    if value == "cpu":
        return "cpu"
    if value == "auto":
        if cuda_available():
            return "0"
        if mps_available():
            return "mps"
        return "cpu"
    if value == "mps":
        if mps_available():
            return "mps"
        logger.warning("device_fallback", requested=requested, reason="MPS not available", using="cpu")
        return "cpu"
    if not cuda_available():
        logger.warning("device_fallback", requested=requested, reason="CUDA not available", using="cpu")
        return "cpu"
    if value.startswith("cuda"):
        _, _, index = value.partition(":")
        return index or "0"
    return value
