from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pytest
import respx

from helmet_safety.domain.errors import DownloadError, ModelNotFoundError
from helmet_safety.infrastructure.weights import WeightsDownloader, WeightsResolver, find_latest_best

URL = "https://example.com/yolov10n.pt"
PAYLOAD = b"\x00" * 2048


def make_best(runs: Path, name: str, mtime: float) -> Path:
    path = runs / name / "weights" / "best.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"w")
    os.utime(path, (mtime, mtime))
    return path


class TestResolver:
    def test_primary_wins(self, tmp_path: Path) -> None:
        primary = tmp_path / "model.pt"
        primary.write_bytes(b"w")
        make_best(tmp_path / "runs", "a", time.time())
        assert WeightsResolver(primary, runs_dir=tmp_path / "runs").resolve() == primary.resolve()

    def test_latest_run_fallback(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        make_best(runs, "old", 1_000)
        newest = make_best(runs, "new", 2_000)
        assert find_latest_best(runs) == newest
        assert WeightsResolver(tmp_path / "missing.pt", runs_dir=runs).resolve() == newest.resolve()

    def test_explicit_fallbacks(self, tmp_path: Path) -> None:
        fallback = tmp_path / "fallback.pt"
        fallback.write_bytes(b"w")
        resolver = WeightsResolver(
            tmp_path / "missing.pt", runs_dir=tmp_path / "noruns", fallbacks=[fallback]
        )
        assert resolver.resolve() == fallback.resolve()

    def test_nothing_found(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotFoundError, match=r"missing\.pt"):
            WeightsResolver(tmp_path / "missing.pt", runs_dir=tmp_path).resolve()

    def test_unsupported_suffix(self, tmp_path: Path) -> None:
        bad = tmp_path / "model.txt"
        bad.write_text("x")
        with pytest.raises(ModelNotFoundError, match="Unsupported"):
            WeightsResolver(bad).resolve()

    def test_find_latest_best_missing_dir(self, tmp_path: Path) -> None:
        assert find_latest_best(tmp_path / "nope") is None


def downloader(**kwargs: object) -> WeightsDownloader:
    params: dict[str, object] = {
        "max_attempts": 3,
        "backoff_initial_s": 0,
        "backoff_max_s": 0,
        "min_bytes": 1024,
    }
    params.update(kwargs)
    return WeightsDownloader(**params)  # type: ignore[arg-type]


class TestDownloader:
    @respx.mock
    def test_success(self, tmp_path: Path) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, content=PAYLOAD))
        target = tmp_path / "sub" / "w.pt"
        assert downloader().download(URL, target) == target
        assert target.read_bytes() == PAYLOAD
        assert not list(target.parent.glob("*.part"))

    @respx.mock
    def test_retries_transient_errors(self, tmp_path: Path) -> None:
        route = respx.get(URL).mock(
            side_effect=[
                httpx.ConnectError("down"),
                httpx.Response(503),
                httpx.Response(200, content=PAYLOAD),
            ]
        )
        downloader().download(URL, tmp_path / "w.pt")
        assert route.call_count == 3

    @respx.mock
    def test_gives_up_after_max_attempts(self, tmp_path: Path) -> None:
        route = respx.get(URL).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(DownloadError, match="after 3 attempts"):
            downloader().download(URL, tmp_path / "w.pt")
        assert route.call_count == 3
        assert not (tmp_path / "w.pt").exists()

    @respx.mock
    def test_client_error_not_retried(self, tmp_path: Path) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(404))
        with pytest.raises(DownloadError, match="404"):
            downloader().download(URL, tmp_path / "w.pt")
        assert route.call_count == 1

    @respx.mock
    def test_rejects_too_small_payload(self, tmp_path: Path) -> None:
        respx.get(URL).mock(return_value=httpx.Response(200, content=b"<html>"))
        with pytest.raises(DownloadError, match="too small"):
            downloader().download(URL, tmp_path / "w.pt")
        assert not (tmp_path / "w.pt").exists()

    def test_rejects_plain_http(self, tmp_path: Path) -> None:
        with pytest.raises(DownloadError, match="non-HTTPS"):
            downloader().download("http://example.com/w.pt", tmp_path / "w.pt")

    @respx.mock
    def test_ensure_skips_existing(self, tmp_path: Path) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(200, content=PAYLOAD))
        target = tmp_path / "w.pt"
        target.write_bytes(PAYLOAD)
        assert downloader().ensure(URL, target) == target
        assert route.call_count == 0
        target.write_bytes(b"tiny")  # corrupted/partial file is re-downloaded
        downloader().ensure(URL, target)
        assert route.call_count == 1
