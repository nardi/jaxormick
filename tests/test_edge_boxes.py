"""Soundness of the relaxations on boxes with subnormal, tiny and one-ulp coordinates."""

import functools

import jax
import numpy as np
import pytest
from conftest import assert_planes_bound, slack
from functions import DOMAIN_LOWER, DOMAIN_UPPER, TEST_FUNCTIONS
from hypothesis import given, settings
from hypothesis import strategies as st

import jaxormick as jmk

NAMES = sorted(TEST_FUNCTIONS)
"""Names of the test functions, in a fixed order."""

SPECIAL_VALUES = [
    0.0,
    float(np.finfo(np.float32).smallest_subnormal),
    float(np.finfo(np.float32).smallest_normal),
    1e-30,
    1e-10,
    0.5,
    1.0,
    1.5,
    2.0,
]
"""Magnitudes near which the functions have kinks, zeros or infinite slopes."""


@st.composite
def edge_coordinates(draw: st.DrawFn) -> np.float32:
    """Draws a coordinate that is a special value, or a few ulps away from one."""
    value = np.float32(draw(st.sampled_from(SPECIAL_VALUES)))
    value = -value if draw(st.booleans()) else value
    for _ in range(draw(st.integers(0, 3))):
        direction = np.float32(draw(st.sampled_from([-np.inf, np.inf])))
        value = np.nextafter(value, direction)
    return np.clip(value, DOMAIN_LOWER, DOMAIN_UPPER)


@st.composite
def edge_boxes_with_points(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draws a box with edge coordinates and points at its corners and in between.

    A box is either two edge coordinates per axis or a few ulps wide.
    """
    lower, upper = [], []
    for _ in range(2):
        first = draw(edge_coordinates())
        if draw(st.booleans()):
            second = draw(edge_coordinates())
        else:
            second = first
            for _ in range(draw(st.integers(0, 4))):
                second = np.nextafter(second, np.float32(np.inf))
        lower.append(min(first, second))
        upper.append(max(first, second))
    lower, upper = np.array(lower, np.float32), np.array(upper, np.float32)
    fractions = np.array(
        [[draw(st.sampled_from([0.0, 0.5, 1.0])) for _ in range(2)] for _ in range(8)]
    )
    points = np.clip(lower + fractions * (upper - lower), lower, upper)
    return lower, upper, points.astype(np.float32)


@functools.cache
def batched_relaxation(name: str):
    """Returns the relaxation of a test function, batched over reference points."""
    return jax.jit(jax.vmap(jmk.relax(TEST_FUNCTIONS[name]), in_axes=(0, None, None)))


@functools.cache
def batched_function(name: str):
    """Returns the test function, batched over points."""
    return jax.jit(jax.vmap(TEST_FUNCTIONS[name]))


@functools.cache
def affine(name: str, mode: jmk.Mode):
    """Returns the affine relaxation of a test function."""
    return jmk.affine_relaxation(TEST_FUNCTIONS[name], mode)


@pytest.mark.parametrize("name", NAMES)
@settings(max_examples=60, deadline=None)
@given(sample=edge_boxes_with_points())
def test_relaxation_bounds_function_on_edge_boxes(name, sample):
    """Checks `lower ≤ convex ≤ f ≤ concave ≤ upper` on edge boxes."""
    lower, upper, points = sample
    relaxation = batched_relaxation(name)(points, lower, upper)
    values = np.asarray(batched_function(name)(points))
    allowed = slack(values, relaxation.lower, relaxation.upper)
    assert np.all(relaxation.lower <= relaxation.convex + allowed)
    assert np.all(relaxation.convex <= values + allowed)
    assert np.all(values <= relaxation.concave + allowed)
    assert np.all(relaxation.concave <= relaxation.upper + allowed)


@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize("name", NAMES)
@settings(max_examples=40, deadline=None)
@given(sample=edge_boxes_with_points())
def test_affine_planes_bound_function_on_edge_boxes(name, mode, sample):
    """Checks that the planes at a corner or midpoint bound `f` on edge boxes."""
    lower, upper, points = sample
    values = np.asarray(batched_function(name)(points))
    planes = affine(name, mode)(points[0], lower, upper)
    assert_planes_bound(planes, points[0], points, values)
