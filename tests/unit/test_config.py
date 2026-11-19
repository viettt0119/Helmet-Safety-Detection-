from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from helmet_safety.domain.entities import DetectionClass
from helmet_safety.infrastructure.config import PROJECT_ROOT, Settings, get_settings


def test_defaults_point_into_project() -> None:
    settings = Settings()
    assert settings.model.weights_path == PROJECT_ROOT / "models" / "helmet_yolov10n_best.pt"
    assert settings.training.dataset_yaml == PROJECT_ROOT / "data" / "data.yaml"
    assert settings.api.max_upload_bytes == 10 * 1024 * 1024


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HELMET_MODEL__DEVICE", "CPU")
    monkeypatch.setenv("HELMET_API__PORT", "9001")
    monkeypatch.setenv("HELMET_POLICY__HEAD_CONFIDENCE", "0.2")
    monkeypatch.setenv("HELMET_LOG_LEVEL", "debug")
    settings = get_settings()
    assert settings.model.device == "cpu"
    assert settings.api.port == 9001
    assert settings.log_level == "DEBUG"
    assert settings.policy.to_confidence_policy().threshold_for(DetectionClass.HEAD) == 0.2


def test_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("HELMET_MODEL__IMAGE_SIZE=320\n")
    assert Settings().model.image_size == 320


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("HELMET_MODEL__DEVICE", "gpu"),
        ("HELMET_MODEL__IMAGE_SIZE", "100"),
        ("HELMET_MODEL__BASE_CONFIDENCE", "1.5"),
        ("HELMET_API__PORT", "70000"),
        ("HELMET_TRAINING__BATCH", "0"),
        ("HELMET_TRAINING__RUN_NAME", "../evil"),
        ("HELMET_LOG_LEVEL", "verbose"),
        ("HELMET_DOWNLOAD__BACKOFF_MAX_S", "0.1"),
    ],
)
def test_invalid_values_rejected(monkeypatch: pytest.MonkeyPatch, env: str, value: str) -> None:
    monkeypatch.setenv(env, value)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda", "cuda:1", "0", "0,1"])
def test_valid_devices(device: str) -> None:
    assert Settings(model={"device": device}).model.device == device


def test_policy_to_analyzer() -> None:
    analyzer = Settings(
        policy={"head_region_ratio": 0.4, "count_unmatched_heads": False}
    ).policy.to_analyzer()
    assert analyzer.head_region_ratio == 0.4
    assert analyzer.count_unmatched_heads is False


def test_settings_singleton() -> None:
    assert get_settings() is get_settings()
