"""Domain/application error hierarchy.

Every error raised intentionally by this package derives from :class:`HelmetSafetyError`, so the
interface layer (API / CLI) can map them to user-facing responses in a single place.
"""

from __future__ import annotations


class HelmetSafetyError(Exception):
    """Base class for all expected errors of the application."""

    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(HelmetSafetyError):
    """Invalid or inconsistent configuration."""

    code = "configuration_error"


class InvalidInputError(HelmetSafetyError):
    """User supplied data that cannot be processed (bad image, bad source, ...)."""

    code = "invalid_input"


class UnsupportedMediaError(InvalidInputError):
    """The uploaded content type / file extension is not supported."""

    code = "unsupported_media"


class PayloadTooLargeError(InvalidInputError):
    """The uploaded payload exceeds the configured size limit."""

    code = "payload_too_large"


class ModelNotFoundError(HelmetSafetyError):
    """No usable weights file could be located."""

    code = "model_not_found"


class ModelLoadError(HelmetSafetyError):
    """Weights exist but the model could not be loaded."""

    code = "model_load_error"


class InferenceError(HelmetSafetyError):
    """The detector failed while running a prediction."""

    code = "inference_error"


class InferenceTimeoutError(InferenceError):
    """The prediction did not finish within the configured deadline."""

    code = "inference_timeout"


class DatasetError(HelmetSafetyError):
    """The dataset description (data.yaml) or its folders are invalid."""

    code = "dataset_error"


class DownloadError(HelmetSafetyError):
    """A remote resource could not be downloaded after all retries."""

    code = "download_error"


class VideoSourceError(HelmetSafetyError):
    """A camera / video stream could not be opened or read."""

    code = "video_source_error"
