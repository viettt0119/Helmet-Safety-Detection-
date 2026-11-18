from __future__ import annotations

import pytest

from helmet_safety.infrastructure.device import resolve_device


def yes() -> bool:
    return True


def no() -> bool:
    return False


@pytest.mark.parametrize(
    ("requested", "cuda", "mps", "expected"),
    [
        ("auto", yes, no, "0"),
        ("auto", no, yes, "mps"),
        ("auto", no, no, "cpu"),
        ("cpu", yes, yes, "cpu"),
        ("0", yes, no, "0"),
        ("0,1", yes, no, "0,1"),
        ("cuda", yes, no, "0"),
        ("cuda:1", yes, no, "1"),
        ("0", no, no, "cpu"),
        ("cuda:0", no, no, "cpu"),
        ("mps", no, yes, "mps"),
        ("mps", no, no, "cpu"),
        (" AUTO ", no, no, "cpu"),
    ],
)
def test_resolve_device(requested: str, cuda: object, mps: object, expected: str) -> None:
    assert resolve_device(requested, cuda_available=cuda, mps_available=mps) == expected  # type: ignore[arg-type]


def test_default_probes_do_not_crash() -> None:
    assert resolve_device("auto") in {"0", "mps", "cpu"}
