from __future__ import annotations

import pytest

from helmet_safety.domain.entities import DetectionClass
from helmet_safety.domain.policies import ConfidencePolicy
from tests.conftest import det


def test_per_class_thresholds() -> None:
    policy = ConfidencePolicy(default=0.5, thresholds={DetectionClass.HEAD: 0.2})
    head = det("head", 0.3, 0, 0, 1, 1)
    helmet = det("helmet", 0.3, 0, 0, 1, 1)
    assert policy.accepts(head)
    assert not policy.accepts(helmet)
    assert policy.apply([head, helmet]) == [head]


def test_threshold_boundary_is_inclusive() -> None:
    assert ConfidencePolicy(default=0.5).accepts(det("person", 0.5, 0, 0, 1, 1))


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError, match="Threshold"):
        ConfidencePolicy(default=bad)
    with pytest.raises(ValueError, match="Threshold"):
        ConfidencePolicy(thresholds={DetectionClass.PERSON: bad})


def test_thresholds_are_immutable() -> None:
    policy = ConfidencePolicy(thresholds={DetectionClass.PERSON: 0.4})
    with pytest.raises(TypeError):
        policy.thresholds[DetectionClass.PERSON] = 0.1  # type: ignore[index]
