from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from helmet_safety.application.ports import Image
from helmet_safety.domain.entities import ComplianceStatus, ImageSize, SafetyReport, WorkerAssessment
from helmet_safety.domain.errors import InvalidInputError, UnsupportedMediaError, VideoSourceError
from helmet_safety.infrastructure.imaging import (
    OpenCVAnnotator,
    OpenCVFrameSource,
    VideoFileSink,
    decode_image,
    encode_jpeg,
    is_image_path,
    parse_source,
    read_image,
    write_image,
)
from tests.conftest import det


def png_bytes(array: np.ndarray) -> bytes:  # type: ignore[type-arg]
    ok, buf = cv2.imencode(".png", array)
    assert ok
    return buf.tobytes()


class TestCodec:
    def test_roundtrip_jpeg(self, image: Image) -> None:
        decoded = decode_image(encode_jpeg(image, 95))
        assert decoded.shape == image.shape
        assert decoded.dtype == np.uint8

    def test_grayscale_and_alpha_are_converted(self) -> None:
        assert decode_image(png_bytes(np.zeros((20, 30), np.uint8))).shape == (20, 30, 3)
        assert decode_image(png_bytes(np.zeros((20, 30, 4), np.uint8))).shape == (20, 30, 3)

    def test_16bit_is_converted(self) -> None:
        decoded = decode_image(png_bytes(np.full((20, 20, 3), 65535, np.uint16)))
        assert decoded.dtype == np.uint8
        assert decoded.max() == 255

    def test_empty_and_garbage(self) -> None:
        with pytest.raises(InvalidInputError, match="Empty"):
            decode_image(b"")
        with pytest.raises(UnsupportedMediaError):
            decode_image(b"definitely not an image")

    def test_read_and_write_files(self, tmp_path: Path, image: Image) -> None:
        path = write_image(tmp_path / "x" / "img.png", image)
        assert np.array_equal(read_image(path), image)
        with pytest.raises(InvalidInputError, match="not found"):
            read_image(tmp_path / "missing.png")
        (tmp_path / "f.gif").write_bytes(b"GIF")
        with pytest.raises(UnsupportedMediaError):
            read_image(tmp_path / "f.gif")

    def test_helpers(self) -> None:
        assert is_image_path("a/B.JPG")
        assert not is_image_path("video.mp4")
        assert parse_source(" 0 ") == 0
        assert parse_source("rtsp://cam") == "rtsp://cam"


def test_annotator_draws_and_keeps_input(image: Image) -> None:
    person = det("person", 0.9, 10, 10, 200, 400)
    head = det("head", 0.8, 50, 12, 90, 60)
    report = SafetyReport(
        ImageSize(640, 480),
        (person, head),
        (WorkerAssessment(ComplianceStatus.VIOLATION, person=person, head=head),),
    )
    original = image.copy()
    annotated = OpenCVAnnotator().annotate(image, report)
    assert np.array_equal(image, original)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    # Banner is drawn in the UNSAFE colour (red channel dominant in BGR).
    b, _g, r = annotated[2, 600]
    assert r > 150
    assert b < 80


def test_annotator_empty_report(image: Image) -> None:
    out = OpenCVAnnotator(line_width=1, draw_raw_detections=False).annotate(
        image, SafetyReport(ImageSize(640, 480), ())
    )
    assert out.shape == image.shape


class TestVideo:
    def _make_video(self, path: Path, frames: int = 5) -> Path:
        sink = VideoFileSink(path, fps=10)
        for i in range(frames):
            assert sink.write(np.full((64, 96, 3), i * 40, np.uint8))
        sink.close()
        sink.close()  # idempotent
        return path

    def test_write_then_read(self, tmp_path: Path) -> None:
        video = self._make_video(tmp_path / "out" / "v.mp4")
        source = OpenCVFrameSource(str(video))
        assert 1.0 <= source.fps <= 240.0
        frames = list(source.frames())
        source.close()
        assert len(frames) == 5
        assert frames[0].shape == (64, 96, 3)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(VideoSourceError, match="not found"):
            OpenCVFrameSource(str(tmp_path / "none.mp4"))

    def test_unopenable_source(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a video")
        with pytest.raises(VideoSourceError, match="Cannot open"):
            list(OpenCVFrameSource(str(bad)).frames())
