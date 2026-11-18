from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from helmet_safety.domain.errors import DatasetError
from helmet_safety.infrastructure.dataset import resolve_dataset


def make_dataset(root: Path, content: str) -> Path:
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
    path = root / "data.yaml"
    path.write_text(content, encoding="utf-8")
    return path


ROBOFLOW = """﻿path: data
train: train/images
val: valid/images
test: test/images
nc: 3
names: ['head', 'helmet', 'person']
"""


def test_resolves_relative_roboflow_layout(tmp_path: Path) -> None:
    yaml_path = make_dataset(tmp_path / "data", ROBOFLOW)
    spec = resolve_dataset(yaml_path, cache_dir=tmp_path / "cache")
    assert spec.class_names == ("head", "helmet", "person")
    assert spec.root == (tmp_path / "data").resolve()
    assert spec.splits["val"] == (tmp_path / "data" / "valid" / "images").resolve()
    written = yaml.safe_load(spec.resolved_yaml.read_text())
    assert Path(written["path"]).is_absolute()
    assert written["nc"] == 3


def test_names_mapping_and_no_path(tmp_path: Path) -> None:
    yaml_path = make_dataset(tmp_path, "train: train/images\nval: valid/images\nnames: {1: b, 0: a}\n")
    spec = resolve_dataset(yaml_path, cache_dir=tmp_path / "c")
    assert spec.class_names == ("a", "b")
    assert "test" not in spec.splits


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("train: train/images\nnames: [a]\n", "Missing required split 'val'"),
        ("train: train/images\nval: valid/images\nnames: 3\n", "'names'"),
        ("train: train/images\nval: valid/images\nnc: 2\nnames: [a]\n", "'nc'"),
        ("train: train/images\nval: valid/images\nnames: []\n", "No class names"),
        ("train: nope/images\nval: valid/images\nnames: [a]\n", "missing location"),
        ("- just\n- a list\n", "mapping"),
        ("names: [unclosed\n", "Invalid YAML"),
    ],
)
def test_invalid(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        resolve_dataset(make_dataset(tmp_path, content), cache_dir=tmp_path / "c")


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        resolve_dataset(tmp_path / "data.yaml")


def test_skip_file_checks(tmp_path: Path) -> None:
    path = tmp_path / "d.yaml"
    path.write_text("train: a\nval: b\nnames: [x]\n")
    assert (
        resolve_dataset(path, check_files=False, cache_dir=tmp_path / "c").splits["train"] == tmp_path / "a"
    )
