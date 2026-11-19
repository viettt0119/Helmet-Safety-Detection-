"""End-to-end tests with the real trained YOLOv10n checkpoint and the real test images.

Marked ``slow``; skipped automatically when the checkpoint is not present.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from helmet_safety.domain.entities import DetectionClass, OverallStatus
from helmet_safety.infrastructure.config import Settings
from helmet_safety.infrastructure.container import Container
from helmet_safety.infrastructure.imaging import read_image
from helmet_safety.interfaces.api.app import create_app
from helmet_safety.interfaces.cli import main as cli
from tests.conftest import REAL_WEIGHTS, TEST_IMAGES_DIR

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not REAL_WEIGHTS.is_file(), reason="trained checkpoint not available"),
]

VIOLATION_IMAGE = TEST_IMAGES_DIR / "person-535-_jpg.rf.629cf0b56b737d6b9833694eb7a50542.jpg"
SAFE_IMAGE = TEST_IMAGES_DIR / "person-9-_jpg.rf.2ee757cbb067a4df46d31028c17270a8.jpg"


@pytest.fixture(scope="module")
def container() -> Container:
    return Container(Settings(environment="test", model={"device": "cpu"}))


def test_known_violation_detected(container: Container) -> None:
    result = container.analyze_image.execute(read_image(VIOLATION_IMAGE), annotate=True)
    assert result.report.status is OverallStatus.UNSAFE
    assert result.report.count(DetectionClass.HEAD) >= 1
    assert result.annotated is not None


def test_known_safe_scene(container: Container) -> None:
    report = container.analyze_image.execute(read_image(SAFE_IMAGE)).report
    assert report.status is OverallStatus.SAFE
    assert report.compliant_count >= 1


def test_detection_quality_on_test_split(container: Container) -> None:
    """Sanity check: on images named helmet-*/person-* the model finds the expected object class."""
    images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    assert len(images) > 50
    hits = 0
    for path in images:
        expected = DetectionClass.HELMET if path.name.startswith("helmet") else DetectionClass.PERSON
        if container.analyze_image.execute(read_image(path)).report.count(expected) > 0:
            hits += 1
    assert hits / len(images) >= 0.75


@pytest.fixture(scope="module")
def api_client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container=container)) as client:
        yield client


def test_api_end_to_end(api_client: TestClient) -> None:
    assert api_client.get("/health/ready").json()["model_loaded"] is True
    with VIOLATION_IMAGE.open("rb") as fh:
        response = api_client.post("/v1/detections", files={"file": ("x.jpg", fh, "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["status"] == "unsafe"
    with SAFE_IMAGE.open("rb") as fh:
        response = api_client.post("/v1/detections/annotated", files={"file": ("x.jpg", fh, "image/jpeg")})
    assert response.status_code == 200
    assert response.headers["X-Safety-Status"] == "safe"
    assert response.content[:2] == b"\xff\xd8"


def test_cli_predict_real(container: Container, tmp_path: Path) -> None:
    cli.state.container = container
    try:
        result = CliRunner().invoke(cli.app, ["predict", str(VIOLATION_IMAGE), "--json", "-o", str(tmp_path)])
    finally:
        cli.state.container = None
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "unsafe"
    assert any(p.suffix == ".jpg" for p in tmp_path.iterdir())


def test_evaluate_on_test_split(tmp_path: Path) -> None:
    settings = Settings(environment="test", model={"device": "cpu", "runs_dir": tmp_path / "runs"})
    cli.state.container = Container(settings)
    try:
        result = CliRunner().invoke(cli.app, ["evaluate", "--split", "test", "--batch", "16"])
    finally:
        cli.state.container = None
    assert result.exit_code == 0, result.output
    metrics = json.loads(result.stdout)
    assert metrics["mAP50"] > 0.6
    assert set(metrics["per_class_mAP50-95"]) == {"head", "helmet", "person"}
