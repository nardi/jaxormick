"""Shared strategies and tolerances for the tests."""

import sys
from pathlib import Path

import numpy as np
from hypothesis import strategies as st
from numpy.typing import ArrayLike

sys.path.insert(0, str(Path(__file__).parent))

from functions import DOMAIN_LOWER, DOMAIN_UPPER  # noqa: E402

coordinates = st.floats(
    min_value=DOMAIN_LOWER, max_value=DOMAIN_UPPER, allow_nan=False, width=32
)
"""Strategy for one coordinate inside the test domain."""


def slack(*arrays: ArrayLike) -> np.ndarray:
    """Returns the rounding slack allowed when comparing values of these magnitudes.

    The relaxations and the function are computed by different sequences of `float32`
    operations, so they may differ by rounding errors that grow with the magnitudes of
    the intermediate values.
    """
    magnitude = sum(np.abs(np.asarray(array, dtype=np.float64)) for array in arrays)
    return np.asarray(2e3 * np.finfo(np.float32).eps * (1 + magnitude))


@st.composite
def boxes_with_points(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draws a box in the test domain and eight points inside it."""
    corners = np.array([[draw(coordinates), draw(coordinates)] for _ in range(2)])
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    fractions = np.array(
        [[draw(st.floats(0.0, 1.0)) for _ in range(2)] for _ in range(8)]
    )
    points = np.clip(lower + fractions * (upper - lower), lower, upper)
    return (
        lower.astype(np.float32),
        upper.astype(np.float32),
        points.astype(np.float32),
    )


@st.composite
def boxes_with_boundary_reference(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draws a box, points inside it, and moves the first point onto its boundary.

    Linear programs have their solutions on the boundary of the box, and there the
    convex and concave sides of intermediate values often tie.
    """
    lower, upper, points = draw(boxes_with_points())
    for axis in range(2):
        match draw(st.sampled_from(["lower", "upper", "inside"])):
            case "lower":
                points[0, axis] = lower[axis]
            case "upper":
                points[0, axis] = upper[axis]
    return lower, upper, points
