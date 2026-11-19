"""Validation and path-normalisation of the YOLO dataset description (``data.yaml``).

The original Roboflow export uses ``path: data`` which Ultralytics resolves against its global
``datasets_dir`` setting — a frequent source of "dataset not found" errors. We rewrite a copy with
an absolute ``path`` so training/evaluation work from any working directory and any OS.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from helmet_safety.domain.errors import DatasetError

REQUIRED_SPLITS = ("train", "val")


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    root: Path
    splits: dict[str, Path]
    class_names: tuple[str, ...]
    resolved_yaml: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetError(f"Dataset config not found: {path}")
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError as exc:
        raise DatasetError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(content, dict):
        raise DatasetError(f"Dataset config must be a mapping: {path}")
    return content


def _class_names(content: dict[str, Any], path: Path) -> tuple[str, ...]:
    names = content.get("names")
    if isinstance(names, dict):
        ordered = [str(names[k]) for k in sorted(names, key=int)]
    elif isinstance(names, list):
        ordered = [str(n) for n in names]
    else:
        raise DatasetError(f"'names' must be a list or mapping in {path}")
    declared = content.get("nc")
    if declared is not None and int(declared) != len(ordered):
        raise DatasetError(f"'nc' ({declared}) does not match number of names ({len(ordered)}) in {path}")
    if not ordered:
        raise DatasetError(f"No class names defined in {path}")
    return tuple(ordered)


def resolve_dataset(
    yaml_path: Path, *, cache_dir: Path | None = None, check_files: bool = True
) -> DatasetSpec:
    """Validate ``yaml_path`` and write an absolute-path copy usable by Ultralytics."""
    yaml_path = yaml_path.resolve()
    content = _load_yaml(yaml_path)
    names = _class_names(content, yaml_path)

    raw_root = content.get("path")
    root = Path(str(raw_root)) if raw_root else yaml_path.parent
    if not root.is_absolute():
        # Interpret relative to the yaml location; Roboflow exports often say ``path: data``
        # meaning "the folder containing this yaml" when run from the project root.
        candidate = (yaml_path.parent / root).resolve()
        root = candidate if candidate.is_dir() else yaml_path.parent

    splits: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        rel = content.get(split)
        if rel is None:
            if split in REQUIRED_SPLITS:
                raise DatasetError(f"Missing required split '{split}' in {yaml_path}")
            continue
        split_path = Path(str(rel))
        split_path = split_path if split_path.is_absolute() else (root / split_path).resolve()
        if check_files and not split_path.exists():
            raise DatasetError(f"Split '{split}' points to a missing location: {split_path}")
        splits[split] = split_path

    resolved = {**content, "path": str(root), **{k: str(v) for k, v in splits.items()}}
    resolved["names"] = list(names)
    resolved["nc"] = len(names)

    target_dir = cache_dir or (yaml_path.parent / ".resolved")
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(yaml_path).encode()).hexdigest()[:10]
    resolved_yaml = target_dir / f"{yaml_path.stem}.{digest}.yaml"
    resolved_yaml.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return DatasetSpec(root=root, splits=splits, class_names=names, resolved_yaml=resolved_yaml)
