"""End-to-end HTTP tests of the FastAPI app with an in-memory detector (fast, deterministic)."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from helmet_safety.application.ports import Image
from helmet_safety.domain.entities import Detection
from helmet_safety.domain.errors import ModelNotFoundError
from helmet_safety.infrastructure.config import Settings
from helmet_safety.infrastructure.container import Container
from helmet_safety.interfaces.api.app import create_app
from tests.conftest import FakeDetector, det


def jpeg(image: Image) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


@pytest.fixture
def detector(worker_with_helmet: list[Detection], worker_without_helmet: list[Detection]) -> FakeDetector:
    return FakeDetector(worker_with_helmet + worker_without_helmet)


def build_client(settings: Settings, detector: FakeDetector) -> TestClient:
    container = Container(settings, detector_override=detector)
    return TestClient(create_app(container=container), raise_server_exceptions=False)


@pytest.fixture
def client(settings: Settings, detector: FakeDetector) -> Iterator[TestClient]:
    with build_client(settings, detector) as test_client:
        yield test_client


@pytest.fixture
def upload(image: Image) -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("site.jpg", jpeg(image), "image/jpeg")}


def test_liveness_and_readiness(client: TestClient) -> None:
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "model_loaded": True}


def test_readiness_when_not_loaded(settings: Settings) -> None:
    detector = FakeDetector()
    app = create_app(container=Container(settings, detector_override=detector))
    app.state.container = Container(settings, detector_override=detector)
    with TestClient(app) as test_client:
        detector._loaded = False
        assert test_client.get("/health/ready").status_code == 503


def test_model_info(client: TestClient) -> None:
    body = client.get("/v1/model").json()
    assert body["class_names"] == {"0": "head", "1": "helmet", "2": "person"}
    assert body["loaded"] is True


def test_detections_json(client: TestClient, upload: dict[str, tuple[str, bytes, str]]) -> None:
    response = client.post("/v1/detections", files=upload, headers={"X-Request-ID": "abc-123"})
    assert response.status_code == 200, response.text
    assert response.headers["X-Request-ID"] == "abc-123"
    body = response.json()
    assert body["status"] == "unsafe"
    assert body["image_width"] == 640
    assert body["image_height"] == 480
    assert body["summary"]["workers"] == 2
    assert body["summary"]["compliant"] == 1
    assert body["summary"]["violations"] == 1
    assert body["summary"]["compliance_rate"] == 0.5
    assert body["summary"]["counts"] == {"head": 1, "helmet": 1, "person": 2}
    assert {w["status"] for w in body["workers"]} == {"compliant", "violation"}
    assert len(body["detections"]) == 4


def test_annotated_image(client: TestClient, upload: dict[str, tuple[str, bytes, str]]) -> None:
    response = client.post("/v1/detections/annotated?quality=80", files=upload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["X-Safety-Status"] == "unsafe"
    assert response.headers["X-Safety-Violations"] == "1"
    decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (480, 640, 3)


def test_png_upload(client: TestClient, image: Image) -> None:
    _ok, buf = cv2.imencode(".png", image)
    response = client.post("/v1/detections", files={"file": ("a.png", buf.tobytes(), "image/png")})
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("files", "status", "code"),
    [
        ({"file": ("a.txt", b"hello", "text/plain")}, 415, "unsupported_media"),
        ({"file": ("a.jpg", b"not really a jpeg", "image/jpeg")}, 415, "unsupported_media"),
        ({"file": ("a.jpg", b"", "image/jpeg")}, 422, "invalid_input"),
        ({"other": ("a.jpg", b"x", "image/jpeg")}, 422, "validation_error"),
    ],
)
def test_bad_uploads(
    client: TestClient, files: dict[str, tuple[str, bytes, str]], status: int, code: str
) -> None:
    response = client.post("/v1/detections", files=files)
    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    assert body["error"]["request_id"]


def test_tiny_image_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/detections", files={"file": ("a.jpg", jpeg(np.zeros((8, 8, 3), np.uint8)), "image/jpeg")}
    )
    assert response.status_code == 422


def test_payload_too_large(settings: Settings, detector: FakeDetector, image: Image) -> None:
    small = settings.model_copy(update={"api": settings.api.model_copy(update={"max_upload_mb": 0.001})})
    with build_client(small, detector) as test_client:
        response = test_client.post("/v1/detections", files={"file": ("a.jpg", jpeg(image), "image/jpeg")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


class SlowDetector(FakeDetector):
    def detect(self, image: Image) -> Sequence[Detection]:
        time.sleep(0.5)
        return []


def test_inference_timeout(settings: Settings, upload: dict[str, tuple[str, bytes, str]]) -> None:
    fast = settings.model_copy(update={"api": settings.api.model_copy(update={"inference_timeout_s": 0.05})})
    with build_client(fast, SlowDetector()) as test_client:
        response = test_client.post("/v1/detections", files=upload)
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "inference_timeout"


def test_detector_crash_returns_500(settings: Settings, upload: dict[str, tuple[str, bytes, str]]) -> None:
    with build_client(settings, FakeDetector(error=RuntimeError("segfault-ish"))) as test_client:
        response = test_client.post("/v1/detections", files=upload)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "inference_error"


def test_model_missing_fails_startup(settings: Settings, tmp_path: Path) -> None:
    # No override -> real container tries to resolve weights in an empty tmp dir.
    app = create_app(container=Container(settings))
    with pytest.raises(ModelNotFoundError), TestClient(app):
        pass


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_404"


def test_invalid_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "x" * 500})
    assert len(response.headers["X-Request-ID"]) == 32


def test_cors_enabled(settings: Settings, detector: FakeDetector) -> None:
    cors = settings.model_copy(
        update={"api": settings.api.model_copy(update={"cors_origins": ["https://app.example"]})}
    )
    with build_client(cors, detector) as test_client:
        response = test_client.options(
            "/v1/detections",
            headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST"},
        )
    assert response.headers["access-control-allow-origin"] == "https://app.example"


def test_openapi_schema(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/v1/detections",
        "/v1/detections/annotated",
        "/health/live",
        "/health/ready",
        "/v1/model",
    } <= set(paths)


def test_empty_scene(settings: Settings, upload: dict[str, tuple[str, bytes, str]]) -> None:
    with build_client(settings, FakeDetector([det("helmet", 0.1, 0, 0, 10, 10)])) as test_client:
        body = test_client.post("/v1/detections", files=upload).json()
    assert body["status"] == "empty"
    assert body["summary"]["compliance_rate"] is None
