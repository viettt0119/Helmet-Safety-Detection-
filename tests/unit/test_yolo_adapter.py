from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from helmet_safety.application.ports import EvaluationRequest, Image, TrainingRequest
from helmet_safety.domain.entities import DetectionClass
from helmet_safety.domain.errors import InferenceError, ModelLoadError, ModelNotFoundError
from helmet_safety.infrastructure.yolo import (
    UltralyticsDetector,
    UltralyticsEvaluator,
    UltralyticsTrainer,
    names_to_map,
)


class FakeTensor:
    def __init__(self, data: list[Any]) -> None:
        self._data = np.asarray(data)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:  # type: ignore[type-arg]
        return self._data


class FakeBoxes:
    def __init__(self, cls: list[int], conf: list[float], xyxy: list[list[float]]) -> None:
        self.cls, self.conf, self.xyxy = FakeTensor(cls), FakeTensor(conf), FakeTensor(xyxy)

    def __len__(self) -> int:
        return len(self.cls.numpy())


class FakeYOLO:
    def __init__(self, names: Any, boxes: FakeBoxes | None) -> None:
        self.names = names
        self.boxes = boxes
        self.predict_kwargs: list[dict[str, Any]] = []
        self.fail = False

    def predict(self, **kwargs: Any) -> list[Any]:
        self.predict_kwargs.append(kwargs)
        if self.fail:
            raise RuntimeError("cuda oom")
        return [SimpleNamespace(boxes=self.boxes)]


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "best.pt"
    path.write_bytes(b"w")
    return path


def make_detector(weights: Path, model: FakeYOLO, **kwargs: Any) -> UltralyticsDetector:
    return UltralyticsDetector(weights, device="cpu", image_size=320, model_factory=lambda _: model, **kwargs)


def test_names_to_map() -> None:
    assert names_to_map(["a", "b"]) == {0: "a", 1: "b"}
    assert names_to_map({"1": "b", 0: "a"}) == {1: "b", 0: "a"}


def test_detect_converts_and_skips_unknown_classes(weights: Path, image: Image) -> None:
    boxes = FakeBoxes(
        [0, 1, 2, 3], [0.9, 0.8, 1.0000001, 0.5], [[1, 2, 3, 4], [5, 6, 7, 8], [30, 40, 10, 20], [0, 0, 1, 1]]
    )
    model = FakeYOLO({0: "head", 1: "helmet", 2: "person", 3: "vest"}, boxes)
    detector = make_detector(weights, model)
    assert not detector.is_loaded
    detections = detector.detect(image)
    assert detector.is_loaded
    assert [d.label for d in detections] == [
        DetectionClass.HEAD,
        DetectionClass.HELMET,
        DetectionClass.PERSON,
    ]
    assert detections[2].confidence == 1.0
    assert detections[2].box.as_int_tuple() == (10, 20, 30, 40)  # swapped coords normalised
    assert detector.info.class_names[3] == "vest"
    assert "half" not in model.predict_kwargs[-1]
    assert model.predict_kwargs[-1]["imgsz"] == 320


def test_half_precision_flag_forwarded(weights: Path, image: Image) -> None:
    model = FakeYOLO(["head"], None)
    make_detector(weights, model, half=True).detect(image)
    assert model.predict_kwargs[-1]["half"] is True


def test_no_boxes(weights: Path, image: Image) -> None:
    assert make_detector(weights, FakeYOLO(["head"], None)).detect(image) == []
    assert make_detector(weights, FakeYOLO(["head"], FakeBoxes([], [], []))).detect(image) == []


def test_load_is_idempotent(weights: Path) -> None:
    factory = MagicMock(return_value=FakeYOLO(["head"], None))
    detector = UltralyticsDetector(weights, device="cpu", model_factory=factory)
    detector.load()
    detector.load()
    factory.assert_called_once()


def test_missing_weights(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        make_detector(tmp_path / "none.pt", FakeYOLO([], None)).load()


def test_corrupt_weights(weights: Path) -> None:
    def boom(_: str) -> Any:
        raise RuntimeError("bad pickle")

    with pytest.raises(ModelLoadError, match="bad pickle"):
        UltralyticsDetector(weights, device="cpu", model_factory=boom).load()


def test_prediction_failure_is_wrapped(weights: Path, image: Image) -> None:
    model = FakeYOLO(["head"], None)
    detector = make_detector(weights, model)
    detector.load()  # warm-up succeeds
    model.fail = True
    with pytest.raises(InferenceError, match="cuda oom"):
        detector.detect(image)


def test_warmup_failure_does_not_break_loading(weights: Path) -> None:
    model = FakeYOLO(["head"], None)
    model.fail = True
    detector = make_detector(weights, model)
    detector.load()
    assert detector.is_loaded


def test_trainer(tmp_path: Path, weights: Path) -> None:
    model = MagicMock()
    model.trainer = SimpleNamespace(
        save_dir=str(tmp_path / "run"), best=str(tmp_path / "run/best.pt"), metrics={"map": 0.5, "x": "n/a"}
    )
    request = TrainingRequest(
        tmp_path / "d.yaml", weights, 1, 320, 4, 0, "cpu", tmp_path, "r", extra={"cache": True}
    )
    result = UltralyticsTrainer(lambda _: model).train(request)
    assert result.best_weights == tmp_path / "run/best.pt"
    assert result.metrics == {"map": 0.5}
    kwargs = model.train.call_args.kwargs
    assert kwargs["epochs"] == 1
    assert kwargs["cache"] is True
    assert kwargs["exist_ok"] is False


def test_trainer_missing_weights(tmp_path: Path) -> None:
    request = TrainingRequest(tmp_path / "d.yaml", tmp_path / "none.pt", 1, 320, 4, 0, "cpu", tmp_path, "r")
    with pytest.raises(ModelNotFoundError):
        UltralyticsTrainer(lambda _: MagicMock()).train(request)


def test_evaluator(tmp_path: Path, weights: Path) -> None:
    model = MagicMock()
    model.val.return_value = SimpleNamespace(
        box=SimpleNamespace(mp=0.7, mr=0.6, map50=0.8, map=0.4, maps=[0.3, 0.5]),
        names={0: "head", 1: "helmet"},
        save_dir=str(tmp_path / "eval"),
    )
    request = EvaluationRequest(tmp_path / "d.yaml", weights, "test", 320, 8, "cpu", tmp_path)
    result = UltralyticsEvaluator(lambda _: model).evaluate(request)
    assert (result.precision, result.recall, result.map50, result.map50_95) == (0.7, 0.6, 0.8, 0.4)
    assert result.per_class_map50_95 == {"head": 0.3, "helmet": 0.5}
    assert result.save_dir == tmp_path / "eval"
