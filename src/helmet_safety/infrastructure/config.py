"""Typed, validated application settings (12-factor: environment variables / ``.env``)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from helmet_safety.domain.entities import DetectionClass
from helmet_safety.domain.policies import ConfidencePolicy
from helmet_safety.domain.services import ComplianceAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[3]

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
_DEVICE_PATTERN = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?|\d+(,\d+)*)$")


class ModelSettings(BaseModel):
    weights_path: Path = PROJECT_ROOT / "models" / "helmet_yolov10n_best.pt"
    pretrained_weights_path: Path = PROJECT_ROOT / "models" / "yolov10n.pt"
    pretrained_weights_url: HttpUrl = HttpUrl(
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10n.pt"
    )
    runs_dir: Path = PROJECT_ROOT / "runs"
    auto_discover_weights: bool = Field(
        default=True, description="Fallback to the newest runs/*/weights/best.pt if weights_path is missing"
    )
    device: str = "auto"
    image_size: int = Field(default=640, ge=32, le=4096)
    base_confidence: Probability = 0.10
    iou: Probability = 0.7
    max_detections: int = Field(default=300, ge=1, le=10_000)
    half_precision: bool = False

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _DEVICE_PATTERN.match(normalized):
            raise ValueError(
                "device must be 'auto', 'cpu', 'mps', 'cuda', 'cuda:N' or GPU indices like '0' / '0,1'"
            )
        return normalized

    @field_validator("image_size")
    @classmethod
    def _validate_image_size(cls, value: int) -> int:
        if value % 32:
            raise ValueError("image_size must be a multiple of 32")
        return value


class PolicySettings(BaseModel):
    """Per-class confidence thresholds and worker-association parameters."""

    default_confidence: Probability = 0.35
    person_confidence: Probability = 0.35
    helmet_confidence: Probability = 0.35
    head_confidence: Probability = 0.35
    head_region_ratio: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    min_containment: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    count_unmatched_helmets: bool = True
    count_unmatched_heads: bool = True

    def to_confidence_policy(self) -> ConfidencePolicy:
        return ConfidencePolicy(
            default=self.default_confidence,
            thresholds={
                DetectionClass.PERSON: self.person_confidence,
                DetectionClass.HELMET: self.helmet_confidence,
                DetectionClass.HEAD: self.head_confidence,
            },
        )

    def to_analyzer(self) -> ComplianceAnalyzer:
        return ComplianceAnalyzer(
            head_region_ratio=self.head_region_ratio,
            min_containment=self.min_containment,
            count_unmatched_helmets=self.count_unmatched_helmets,
            count_unmatched_heads=self.count_unmatched_heads,
        )


class ApiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=64)
    max_upload_mb: float = Field(default=10.0, gt=0, le=200)
    inference_timeout_s: float = Field(default=30.0, gt=0, le=600)
    max_concurrency: int = Field(default=2, ge=1, le=64)
    cors_origins: list[str] = Field(default_factory=list)
    jpeg_quality: int = Field(default=90, ge=1, le=100)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


class TrainingSettings(BaseModel):
    dataset_yaml: Path = PROJECT_ROOT / "data" / "data.yaml"
    epochs: int = Field(default=50, ge=1, le=10_000)
    batch: int = Field(default=8, ge=-1)
    workers: int = Field(default=4, ge=0, le=64)
    patience: int = Field(default=20, ge=0)
    seed: int = 0
    run_name: str = Field(default="helmet_yolov10n", pattern=r"^[A-Za-z0-9_.-]+$")

    @field_validator("batch")
    @classmethod
    def _validate_batch(cls, value: int) -> int:
        if value == 0:
            raise ValueError("batch must be a positive integer or -1 (auto batch)")
        return value


class DownloadSettings(BaseModel):
    timeout_s: float = Field(default=60.0, gt=0)
    max_attempts: int = Field(default=4, ge=1, le=10)
    backoff_initial_s: float = Field(default=1.0, ge=0)
    backoff_max_s: float = Field(default=20.0, ge=0)


class Settings(BaseSettings):
    """Root settings. Env vars use the ``HELMET_`` prefix and ``__`` for nesting.

    Example: ``HELMET_MODEL__DEVICE=cpu``, ``HELMET_API__PORT=9000``.
    """

    model_config = SettingsConfigDict(
        env_prefix="HELMET_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    environment: Literal["development", "staging", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False

    model: ModelSettings = Field(default_factory=ModelSettings)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.download.backoff_max_s < self.download.backoff_initial_s:
            raise ValueError("download.backoff_max_s must be >= download.backoff_initial_s")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cleared in tests via ``get_settings.cache_clear()``)."""
    return Settings()
