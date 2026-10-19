"""Soundness, convexity and exactness of the relaxations of the test functions."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    boxes_with_boundary_reference,
    boxes_with_points,
    coordinates,
    slack,
)
from functions import TEST_FUNCTIONS
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import jaxormick as jmk

NAMES = sorted(TEST_FUNCTIONS)
"""Names of the test functions, in a fixed order."""


@functools.cache
def batched_relaxation(name: str):
    """Returns the relaxation of a test function, batched over reference points."""
    return jax.jit(jax.vmap(jmk.relax(TEST_FUNCTIONS[name]), in_axes=(0, None, None)))


@functools.cache
def batched_function(name: str):
    """Returns the test function, batched over points."""
    return jax.jit(jax.vmap(TEST_FUNCTIONS[name]))


@functools.cache
def affine(name: str, mode: jmk.Mode = "forward"):
    """Returns the affine relaxation of a test function."""
    return jmk.affine_relaxation(TEST_FUNCTIONS[name], mode)


@pytest.mark.parametrize("name", NAMES)
@settings(max_examples=60, deadline=None)
@given(sample=boxes_with_points())
def test_relaxation_bounds_function(name, sample):
    """Checks `lower ≤ convex ≤ f ≤ concave ≤ upper` at every sampled point."""
    lower, upper, points = sample
    relaxation = batched_relaxation(name)(points, lower, upper)
    values = np.asarray(batched_function(name)(points))
    allowed = slack(values, relaxation.lower, relaxation.upper)
    assert np.all(relaxation.lower <= relaxation.convex + allowed)
    assert np.all(relaxation.convex <= values + allowed)
    assert np.all(values <= relaxation.concave + allowed)
    assert np.all(relaxation.concave <= relaxation.upper + allowed)


@pytest.mark.parametrize("name", NAMES)
@settings(max_examples=60, deadline=None)
@given(sample=boxes_with_points())
def test_relaxations_are_convex_and_concave(name, sample):
    """Checks the midpoint inequalities of convexity and concavity between points."""
    lower, upper, points = sample
    relaxed = batched_relaxation(name)
    first, second = points[:4], points[4:]
    at_first, at_second = relaxed(first, lower, upper), relaxed(second, lower, upper)
    at_middle = relaxed((first + second) / 2, lower, upper)
    allowed = slack(at_first.lower, at_first.upper)
    assert np.all(
        at_middle.convex <= (at_first.convex + at_second.convex) / 2 + allowed
    )
    assert np.all(
        at_middle.concave >= (at_first.concave + at_second.concave) / 2 - allowed
    )


@pytest.mark.parametrize("name", NAMES)
@settings(max_examples=30, deadline=None)
@given(point=st.tuples(coordinates, coordinates))
def test_degenerate_box_is_exact(name, point):
    """Checks that all four fields equal `f(x)` on a box that is a single point."""
    x = np.array(point, dtype=np.float32)
    relaxation = jmk.relax(TEST_FUNCTIONS[name])(x, x, x)
    value = float(TEST_FUNCTIONS[name](jnp.asarray(x)))
    for field in jax.tree.leaves(relaxation):
        assert abs(float(field) - value) <= slack(value)


def assert_planes_bound(
    planes: jmk.AffineRelaxation,
    reference: np.ndarray,
    points: np.ndarray,
    values: np.ndarray,
) -> None:
    """Checks that the planes through `reference` bound `values` at `points`.

    A side may have no finite slope, like `sqrt` at zero. It then has no tangent plane,
    which callers detect from the infinite coefficients.
    """
    allowed = slack(values, planes.convex, planes.concave)
    offsets = (points - reference).astype(np.float64)
    if np.all(np.isfinite(planes.convex_gradient)):
        below = planes.convex + offsets @ np.asarray(planes.convex_gradient)
        assert np.all(below <= values + allowed)
    if np.all(np.isfinite(planes.concave_gradient)):
        above = planes.concave + offsets @ np.asarray(planes.concave_gradient)
        assert np.all(above >= values - allowed)


@pytest.mark.parametrize("name", NAMES)
@settings(max_examples=40, deadline=None)
@given(sample=boxes_with_points())
def test_affine_planes_bound_function(name, sample):
    """Checks that tangent planes at an interior point have finite slopes and bound `f`."""
    lower, upper, points = sample
    # References on the boundary have their own test, where slopes may be infinite.
    assume(np.all((lower < points[0]) & (points[0] < upper)))
    planes = affine(name)(points[0], lower, upper)
    values = np.asarray(batched_function(name)(points))
    assert np.all(np.isfinite(planes.convex_gradient))
    assert np.all(np.isfinite(planes.concave_gradient))
    assert_planes_bound(planes, points[0], points, values)


@pytest.mark.parametrize("name", NAMES)
@settings(max_examples=100, deadline=None)
@given(sample=boxes_with_boundary_reference())
def test_affine_planes_bound_function_at_boundary_reference(name, sample):
    """Checks the tangent planes when the reference point lies on the box boundary."""
    lower, upper, points = sample
    planes = affine(name)(points[0], lower, upper)
    values = np.asarray(batched_function(name)(points))
    assert_planes_bound(planes, points[0], points, values)


def test_affine_planes_bound_clip_when_rounding_crosses_sides():
    """Checks the planes of `clip` at a corner where rounding crosses the relaxations.

    Hypothesis found this box on CI. The inner `max` of the clip has its concave side an
    ulp below its convex side at the lower corner.
    """
    lower = np.full(2, -1.5634626, dtype=np.float32)
    upper = np.array([1.0, 0.0], dtype=np.float32)
    grid = np.linspace(0, 1, 9, dtype=np.float32)
    points = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    points = lower + points * (upper - lower)
    planes = affine("clip")(lower, lower, upper)
    values = np.asarray(batched_function("clip")(points))
    assert_planes_bound(planes, lower, points, values)


@pytest.mark.parametrize("name", NAMES)
@settings(max_examples=20, deadline=None)
@given(sample=boxes_with_boundary_reference())
def test_forward_and_reverse_mode_agree(name, sample):
    """Checks that both modes give the same values and valid planes.

    Relaxations have kinks, where the two modes may pick different subgradients, so
    only the values have to agree.
    """
    lower, upper, points = sample
    values = np.asarray(batched_function(name)(points))
    forward = affine(name, "forward")(points[0], lower, upper)
    reverse = affine(name, "reverse")(points[0], lower, upper)
    for planes in (forward, reverse):
        assert_planes_bound(planes, points[0], points, values)
    np.testing.assert_allclose(forward.convex, reverse.convex, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(forward.concave, reverse.concave, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_relaxation_gap_shrinks_with_box(name):
    """Checks that the gap at a fixed point shrinks as the box around it halves."""
    center = np.array([0.3, -0.7], dtype=np.float32)
    relaxed = jmk.relax(TEST_FUNCTIONS[name])
    gaps = []
    for half_width in (1.0, 0.5, 0.25, 0.125, 0.0625):
        relaxation = relaxed(center, center - half_width, center + half_width)
        gaps.append(float(relaxation.concave - relaxation.convex))
    allowed = float(slack(gaps[0]))
    assert all(later <= earlier + allowed for earlier, later in zip(gaps, gaps[1:]))
    assert gaps[-1] <= 0.2 * gaps[0] + allowed
