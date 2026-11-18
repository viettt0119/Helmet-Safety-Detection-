from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helmet_safety.application.ports import (
    EvaluationRequest,
    EvaluationResult,
    TrainingRequest,
    TrainingResult,
)
from helmet_safety.application.training import EvaluateModelUseCase, TrainModelUseCase
from helmet_safety.domain.errors import InvalidInputError, ModelNotFoundError


@pytest.fixture
def data_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text("names: [a]\n")
    return path


def train_request(data_yaml: Path, **overrides: object) -> TrainingRequest:
    values: dict[str, object] = {
        "dataset_yaml": data_yaml,
        "base_weights": data_yaml.parent / "w.pt",
        "epochs": 1,
        "image_size": 640,
        "batch": 8,
        "workers": 0,
        "device": "cpu",
        "project_dir": data_yaml.parent / "runs",
        "run_name": "test",
    }
    values.update(overrides)
    return TrainingRequest(**values)  # type: ignore[arg-type]


class TestTrain:
    def test_delegates_to_trainer(self, data_yaml: Path) -> None:
        trainer = MagicMock()
        trainer.train.return_value = TrainingResult(Path("r"), Path("r/best.pt"), {"map": 0.5})
        request = train_request(data_yaml)
        assert TrainModelUseCase(trainer).execute(request).metrics == {"map": 0.5}
        trainer.train.assert_called_once_with(request)

    @pytest.mark.parametrize(
        "overrides",
        [{"epochs": 0}, {"batch": 0}, {"batch": -2}, {"image_size": 100}, {"image_size": 0}, {"workers": -1}],
    )
    def test_validation(self, data_yaml: Path, overrides: dict[str, object]) -> None:
        trainer = MagicMock()
        with pytest.raises(InvalidInputError):
            TrainModelUseCase(trainer).execute(train_request(data_yaml, **overrides))
        trainer.train.assert_not_called()

    def test_missing_dataset(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidInputError, match="Dataset"):
            TrainModelUseCase(MagicMock()).execute(train_request(tmp_path / "nope.yaml"))


class TestEvaluate:
    def _request(self, data_yaml: Path, weights: Path, split: str = "test") -> EvaluationRequest:
        return EvaluationRequest(data_yaml, weights, split, 640, 16, "cpu", data_yaml.parent)

    def test_delegates(self, data_yaml: Path) -> None:
        weights = data_yaml.parent / "w.pt"
        weights.write_bytes(b"x")
        evaluator = MagicMock()
        evaluator.evaluate.return_value = EvaluationResult(0.1, 0.2, 0.3, 0.4, {})
        assert EvaluateModelUseCase(evaluator).execute(self._request(data_yaml, weights)).map50 == 0.3

    def test_invalid_split(self, data_yaml: Path) -> None:
        with pytest.raises(InvalidInputError, match="split"):
            EvaluateModelUseCase(MagicMock()).execute(self._request(data_yaml, data_yaml, "dev"))

    def test_missing_weights(self, data_yaml: Path) -> None:
        with pytest.raises(ModelNotFoundError):
            EvaluateModelUseCase(MagicMock()).execute(self._request(data_yaml, data_yaml.parent / "none.pt"))

    def test_missing_dataset(self, data_yaml: Path) -> None:
        with pytest.raises(InvalidInputError):
            EvaluateModelUseCase(MagicMock()).execute(self._request(data_yaml.parent / "x.yaml", data_yaml))
