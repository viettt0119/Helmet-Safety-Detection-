"""CLI tests via Typer's runner with an in-memory detector / mocked trainer."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from helmet_safety.application.ports import EvaluationResult, Image, TrainingResult
from helmet_safety.domain.entities import Detection
from helmet_safety.infrastructure.config import Settings
from helmet_safety.infrastructure.container import Container
from helmet_safety.interfaces.cli import main as cli
from helmet_safety.interfaces.cli.main import EXIT_UNSAFE, app
from tests.conftest import FakeDetector

runner = CliRunner()


@pytest.fixture
def install_container(settings: Settings) -> Iterator[object]:
    def _install(**overrides: object) -> Container:
        container = Container(settings, **overrides)  # type: ignore[arg-type]
        cli.state.container = container
        return container

    yield _install
    cli.state.container = None


@pytest.fixture
def images_dir(tmp_path: Path, image: Image) -> Path:
    folder = tmp_path / "imgs"
    folder.mkdir()
    for name in ("a.jpg", "b.png"):
        cv2.imwrite(str(folder / name), image)
    (folder / "notes.txt").write_text("ignored")
    return folder


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_predict_folder_text_and_outputs(
    install_container, images_dir: Path, tmp_path: Path, worker_without_helmet: list[Detection]
) -> None:  # type: ignore[no-untyped-def]
    install_container(detector_override=FakeDetector(worker_without_helmet))
    out = tmp_path / "out"
    result = runner.invoke(app, ["predict", str(images_dir), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "a.jpg: UNSAFE" in result.stdout
    assert "b.png: UNSAFE" in result.stdout
    assert sorted(p.name for p in out.iterdir()) == ["a.json", "a_annotated.jpg", "b.json", "b_annotated.jpg"]
    assert json.loads((out / "a.json").read_text())["summary"]["violations"] == 1


def test_predict_single_image_json_and_fail_on_violation(
    install_container, images_dir: Path, worker_without_helmet: list[Detection]
) -> None:  # type: ignore[no-untyped-def]
    install_container(detector_override=FakeDetector(worker_without_helmet))
    result = runner.invoke(app, ["predict", str(images_dir / "a.jpg"), "--json", "--fail-on-violation"])
    assert result.exit_code == EXIT_UNSAFE
    assert json.loads(result.stdout)["status"] == "unsafe"


def test_predict_safe_scene_exit_zero(
    install_container, images_dir: Path, worker_with_helmet: list[Detection]
) -> None:  # type: ignore[no-untyped-def]
    install_container(detector_override=FakeDetector(worker_with_helmet))
    result = runner.invoke(app, ["predict", str(images_dir), "--json", "--fail-on-violation"])
    assert result.exit_code == 0
    assert [r["status"] for r in json.loads(result.stdout)] == ["safe", "safe"]


def test_predict_video(install_container, tmp_path: Path, worker_without_helmet: list[Detection]) -> None:  # type: ignore[no-untyped-def]
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (96, 64))
    for _ in range(6):
        writer.write(np.zeros((64, 96, 3), np.uint8))
    writer.release()
    install_container(detector_override=FakeDetector(worker_without_helmet))
    out = tmp_path / "out"
    result = runner.invoke(app, ["predict", str(video), "-o", str(out), "--max-frames", "4", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["frames"] == 4
    assert payload["unsafe_frames"] == 4
    assert (out / "clip_annotated.mp4").stat().st_size > 0


def test_predict_errors(install_container, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    install_container(detector_override=FakeDetector())
    missing = runner.invoke(app, ["predict", str(tmp_path / "missing.jpg")])
    assert missing.exit_code == 1
    assert "invalid_input" in missing.output
    empty = tmp_path / "empty"
    empty.mkdir()
    assert runner.invoke(app, ["predict", str(empty)]).exit_code == 1
    assert runner.invoke(app, ["predict", str(tmp_path / "nope.mp4")]).exit_code == 1
    assert runner.invoke(app, ["predict", "0"]).exit_code == 1  # webcam without any sink/limit


def test_info_without_weights(install_container) -> None:  # type: ignore[no-untyped-def]
    install_container()
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 1
    assert "model_not_found" in result.output


def test_info_with_fake(install_container) -> None:  # type: ignore[no-untyped-def]
    install_container(detector_override=FakeDetector())
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["classes"]["1"] == "helmet"


def _dataset(tmp_path: Path) -> Path:
    for split in ("train", "valid"):
        (tmp_path / "ds" / split / "images").mkdir(parents=True)
    path = tmp_path / "ds" / "data.yaml"
    path.write_text("train: train/images\nval: valid/images\nnames: [head, helmet, person]\n")
    return path


def test_train_with_mock_trainer(install_container, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    trainer = MagicMock()
    trainer.train.return_value = TrainingResult(tmp_path / "run", tmp_path / "run/best.pt", {"mAP50": 0.8})
    install_container(trainer_override=trainer)
    base = tmp_path / "base.pt"
    base.write_bytes(b"w")
    result = runner.invoke(
        app,
        [
            "train",
            "--epochs",
            "2",
            "--batch",
            "4",
            "--data",
            str(_dataset(tmp_path)),
            "--base-weights",
            str(base),
            "--name",
            "t1",
        ],
    )
    assert result.exit_code == 0, result.output
    request = trainer.train.call_args.args[0]
    assert (request.epochs, request.batch, request.run_name, request.device) == (2, 4, "t1", "cpu")
    resolved = yaml.safe_load(request.dataset_yaml.read_text())  # rewritten copy with absolute paths
    assert Path(resolved["path"]).is_absolute()
    assert Path(resolved["train"]).is_absolute()
    assert json.loads(result.stdout)["metrics"] == {"mAP50": 0.8}


def test_train_bad_dataset(install_container, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    install_container(trainer_override=MagicMock())
    base = tmp_path / "base.pt"
    base.write_bytes(b"w")
    result = runner.invoke(app, ["train", "--data", str(tmp_path / "none.yaml"), "--base-weights", str(base)])
    assert result.exit_code == 1
    assert "dataset_error" in result.output


def test_evaluate_with_mock(install_container, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(0.7, 0.8, 0.82, 0.44, {"helmet": 0.5})
    install_container(evaluator_override=evaluator)
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    result = runner.invoke(
        app, ["evaluate", "--weights", str(weights), "--data", str(_dataset(tmp_path)), "--split", "val"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["mAP50"] == 0.82


def test_evaluate_invalid_split(install_container, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    install_container(evaluator_override=MagicMock())
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    result = runner.invoke(
        app, ["evaluate", "--weights", str(weights), "--data", str(_dataset(tmp_path)), "--split", "dev"]
    )
    assert result.exit_code == 1


def test_download_weights_existing(install_container, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    install_container()
    target = tmp_path / "w.pt"
    target.write_bytes(b"\0" * 200_000)
    result = runner.invoke(app, ["download-weights", "--destination", str(target)])
    assert result.exit_code == 0
    assert result.stdout.strip() == str(target)


def test_download_weights_rejects_http(install_container, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    install_container()
    result = runner.invoke(
        app,
        ["download-weights", "--url", "http://x/w.pt", "--destination", str(tmp_path / "w.pt"), "--force"],
    )
    assert result.exit_code == 1
    assert "download_error" in result.output
