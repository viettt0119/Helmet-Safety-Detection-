"""Use cases: train and evaluate the model."""

from __future__ import annotations

from helmet_safety.application.ports import (
    EvaluationRequest,
    EvaluationResult,
    ModelEvaluator,
    ModelTrainer,
    TrainingRequest,
    TrainingResult,
)
from helmet_safety.domain.errors import InvalidInputError, ModelNotFoundError

VALID_SPLITS = frozenset({"train", "val", "test"})


class TrainModelUseCase:
    def __init__(self, trainer: ModelTrainer) -> None:
        self._trainer = trainer

    def execute(self, request: TrainingRequest) -> TrainingResult:
        if request.epochs < 1:
            raise InvalidInputError("epochs must be >= 1")
        if request.batch == 0 or request.batch < -1:
            raise InvalidInputError("batch must be a positive integer or -1 (auto batch)")
        if request.image_size < 32 or request.image_size % 32:
            raise InvalidInputError("image_size must be a multiple of 32 and >= 32")
        if request.workers < 0:
            raise InvalidInputError("workers must be >= 0")
        if not request.dataset_yaml.is_file():
            raise InvalidInputError(f"Dataset config not found: {request.dataset_yaml}")
        return self._trainer.train(request)


class EvaluateModelUseCase:
    def __init__(self, evaluator: ModelEvaluator) -> None:
        self._evaluator = evaluator

    def execute(self, request: EvaluationRequest) -> EvaluationResult:
        if request.split not in VALID_SPLITS:
            raise InvalidInputError(f"split must be one of {sorted(VALID_SPLITS)}, got {request.split!r}")
        if not request.weights.is_file():
            raise ModelNotFoundError(f"Weights not found: {request.weights}")
        if not request.dataset_yaml.is_file():
            raise InvalidInputError(f"Dataset config not found: {request.dataset_yaml}")
        return self._evaluator.evaluate(request)
