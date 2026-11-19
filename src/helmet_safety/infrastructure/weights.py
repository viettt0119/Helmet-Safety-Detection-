"""Locating model weights on disk and downloading pretrained checkpoints (retry + timeout)."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import httpx
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from helmet_safety.domain.errors import DownloadError, ModelNotFoundError
from helmet_safety.infrastructure.logging import get_logger

logger = get_logger(__name__)

MIN_WEIGHTS_BYTES = 100 * 1024  # anything smaller is certainly not a YOLO checkpoint
WEIGHT_SUFFIXES = frozenset({".pt", ".onnx", ".engine", ".torchscript", ".openvino"})


def find_latest_best(runs_dir: Path) -> Path | None:
    """Newest ``runs/<run>/weights/best.pt`` (by modification time), or ``None``."""
    if not runs_dir.is_dir():
        return None
    candidates = [p for p in runs_dir.glob("*/weights/best.pt") if p.is_file()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


class WeightsResolver:
    """Chain of Responsibility over candidate locations; first existing file wins."""

    def __init__(
        self, primary: Path, *, runs_dir: Path | None = None, fallbacks: Sequence[Path] = ()
    ) -> None:
        self._primary = primary
        self._runs_dir = runs_dir
        self._fallbacks = tuple(fallbacks)

    def resolve(self) -> Path:
        if self._primary.is_file():
            return self._validated(self._primary)
        tried = [str(self._primary)]
        if self._runs_dir is not None:
            latest = find_latest_best(self._runs_dir)
            if latest is not None:
                logger.warning("weights_fallback", missing=str(self._primary), using=str(latest))
                return self._validated(latest)
            tried.append(f"{self._runs_dir}/*/weights/best.pt")
        for candidate in self._fallbacks:
            if candidate.is_file():
                logger.warning("weights_fallback", missing=str(self._primary), using=str(candidate))
                return self._validated(candidate)
            tried.append(str(candidate))
        raise ModelNotFoundError(
            "No model weights found. Tried: " + ", ".join(tried) + ". Train a model first "
            "(`helmet-safety train`) or set HELMET_MODEL__WEIGHTS_PATH."
        )

    @staticmethod
    def _validated(path: Path) -> Path:
        if path.suffix.lower() not in WEIGHT_SUFFIXES and not path.is_dir():
            raise ModelNotFoundError(f"Unsupported weights format: {path.suffix} ({path})")
        return path.resolve()


class _RetryableHTTPError(Exception):
    pass


class WeightsDownloader:
    """Streams a file to disk atomically with exponential back-off retries and timeouts."""

    def __init__(
        self,
        *,
        timeout_s: float = 60.0,
        max_attempts: int = 4,
        backoff_initial_s: float = 1.0,
        backoff_max_s: float = 20.0,
        client: httpx.Client | None = None,
        min_bytes: int = MIN_WEIGHTS_BYTES,
    ) -> None:
        self._timeout = httpx.Timeout(timeout_s, connect=min(10.0, timeout_s))
        self._max_attempts = max_attempts
        self._backoff_initial = backoff_initial_s
        self._backoff_max = backoff_max_s
        self._client = client
        self._min_bytes = min_bytes

    def ensure(self, url: str, destination: Path) -> Path:
        """Return ``destination``, downloading it first if it does not exist yet."""
        if destination.is_file() and destination.stat().st_size >= self._min_bytes:
            return destination
        return self.download(url, destination)

    def download(self, url: str, destination: Path) -> Path:
        if not url.startswith("https://"):
            raise DownloadError(f"Refusing non-HTTPS download URL: {url}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(
                multiplier=self._backoff_initial, max=self._backoff_max, jitter=self._backoff_initial
            ),
            retry=retry_if_exception_type((httpx.TransportError, _RetryableHTTPError)),
            reraise=False,
        )
        try:
            for attempt in retrying:
                with attempt:
                    logger.info("download_attempt", url=url, attempt=attempt.retry_state.attempt_number)
                    self._download_once(url, destination)
        except RetryError as exc:
            cause = exc.last_attempt.exception()
            raise DownloadError(
                f"Download failed after {self._max_attempts} attempts: {url} ({cause})"
            ) from cause
        except httpx.HTTPError as exc:
            raise DownloadError(f"Download failed: {url} ({exc})") from exc
        logger.info("download_complete", url=url, path=str(destination), bytes=destination.stat().st_size)
        return destination

    def _download_once(self, url: str, destination: Path) -> None:
        client = self._client or httpx.Client(timeout=self._timeout, follow_redirects=True)
        fd, tmp_name = tempfile.mkstemp(prefix=destination.name, suffix=".part", dir=destination.parent)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle, client.stream("GET", url, timeout=self._timeout) as response:
                if response.status_code >= 500 or response.status_code == 429:
                    raise _RetryableHTTPError(f"HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise DownloadError(f"HTTP {response.status_code} for {url}")
                for chunk in response.iter_bytes(chunk_size=1 << 16):
                    handle.write(chunk)
            size = tmp_path.stat().st_size
            if size < self._min_bytes:
                raise DownloadError(f"Downloaded file is too small ({size} bytes) — not a valid checkpoint")
            tmp_path.replace(destination)
        finally:
            tmp_path.unlink(missing_ok=True)
            if self._client is None:
                client.close()
