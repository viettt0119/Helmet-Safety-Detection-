"""Helmet Safety Detection — PPE (helmet) compliance detection built on YOLOv10."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("helmet-safety")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
