"""Intermediate bounds tightened by the subgradients at the middle of the box."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import boxes_with_points, slack
from functions import TEST_FUNCTIONS
from hypothesis import given, settings

import jaxormick as jmk


def exp_of_concave(x: jax.Array) -> jax.Array:
    """Returns `exp(x − x²)`, whose inner interval is much wider than its range."""
    return jnp.exp(x[0] - x[0] ** 2)


@functools.cache
def batched_relaxation(name: str, tighten_bounds: bool = True):
    """Returns the relaxation of a test function, batched over reference points."""
    relaxed = jmk.relax(TEST_FUNCTIONS[name], tighten_bounds=tighten_bounds)
    return jax.jit(jax.vmap(relaxed, in_axes=(0, None, None)))


@pytest.mark.parametrize("name", sorted(TEST_FUNCTIONS))
@settings(max_examples=20, deadline=None)
@given(sample=boxes_with_points())
def test_tightened_relaxation_bounds_function_and_is_convex(name, sample):
    """Checks `lower ≤ convex ≤ f ≤ concave ≤ upper` and the convexity inequalities."""
    lower, upper, points = sample
    relaxed = batched_relaxation(name)
    relaxation = relaxed(points, lower, upper)
    values = np.asarray(jax.vmap(TEST_FUNCTIONS[name])(points))
    allowed = slack(values, relaxation.lower, relaxation.upper)
    assert np.all(relaxation.lower <= values + allowed)
    assert np.all(values <= relaxation.upper + allowed)
    assert np.all(relaxation.convex <= values + allowed)
    assert np.all(values <= relaxation.concave + allowed)

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


@pytest.mark.parametrize("name", sorted(TEST_FUNCTIONS))
@settings(max_examples=20, deadline=None)
@given(sample=boxes_with_points())
def test_tightened_bounds_are_at_least_as_tight(name, sample):
    """Checks that tightening never widens the interval of the output."""
    lower, upper, points = sample
    tightened = batched_relaxation(name)(points, lower, upper)
    plain = batched_relaxation(name, tighten_bounds=False)(points, lower, upper)
    allowed = slack(plain.lower, plain.upper)
    assert np.all(tightened.lower >= plain.lower - allowed)
    assert np.all(tightened.upper <= plain.upper + allowed)


def test_tightening_narrows_inner_interval():
    """Checks that a tighter bound on `x − x²` tightens the exponential around it.

    On `[0, 2]`, the plane of the convex side `−x` at the middle gives `x − x² ≥ −2`,
    and that of the concave side gives `x − x² ≤ 1`. The natural interval is `[−4, 2]`.
    """
    lower, upper = jnp.array([0.0]), jnp.array([2.0])
    bounds = jmk.interval(exp_of_concave, tighten_bounds=True)(lower, upper)
    np.testing.assert_allclose(bounds, (np.exp(-2.0), np.e), rtol=1e-6)
    reference = jnp.array([0.7])
    tightened = jmk.relax(exp_of_concave, tighten_bounds=True)(reference, lower, upper)
    plain = jmk.relax(exp_of_concave)(reference, lower, upper)
    assert tightened.concave < 0.5 * plain.concave


def test_tightening_keeps_integer_intermediates():
    """Checks a function with an integer intermediate, which has no subgradient."""

    def function(x: jax.Array) -> jax.Array:
        return (x[0] * x[1]).astype(jnp.int32).astype(jnp.float32) + x[0]

    reference, lower, upper = jnp.array([0.3, 0.2]), -2 * jnp.ones(2), 2 * jnp.ones(2)
    tightened = jmk.relax(function, tighten_bounds=True)(reference, lower, upper)
    plain = jmk.relax(function)(reference, lower, upper)
    for field in ("lower", "upper", "convex", "concave"):
        np.testing.assert_allclose(getattr(tightened, field), getattr(plain, field))


def test_tightening_falls_back_on_unbounded_box():
    """Checks that a box with an infinite side keeps the natural interval."""
    lower, upper = jnp.array([0.0, -jnp.inf]), jnp.array([1.0, 1.0])
    bounds = jmk.interval(lambda x: x[0] + x[1], tighten_bounds=True)(lower, upper)
    np.testing.assert_allclose(bounds, (-np.inf, 2.0))


@pytest.mark.parametrize("side", [jmk.convex_side, jmk.concave_side])
def test_sides_pass_tighten_bounds_to_relax(side):
    """Checks that one side alone matches the side of the full relaxation."""
    reference, lower, upper = jnp.array([0.7]), jnp.array([0.0]), jnp.array([2.0])
    value, _ = side(exp_of_concave, tighten_bounds=True)(reference, lower, upper)
    relaxation = jmk.relax(exp_of_concave, tighten_bounds=True)(reference, lower, upper)
    expected = relaxation.convex if side is jmk.convex_side else relaxation.concave
    np.testing.assert_allclose(value, expected, rtol=1e-6)


def test_tightened_planes_bound_function():
    """Checks that the planes of the tightened relaxation bound the function."""
    reference, lower, upper = jnp.array([0.7]), jnp.array([0.0]), jnp.array([2.0])
    planes = jmk.affine_relaxation(exp_of_concave, tighten_bounds=True)(
        reference, lower, upper
    )
    points = jnp.linspace(lower, upper, 101)
    values = jax.vmap(exp_of_concave)(points)
    steps = (points - reference) @ planes.convex_gradient
    assert np.all(planes.convex + steps <= values + 1e-5)
    steps = (points - reference) @ planes.concave_gradient
    assert np.all(values <= planes.concave + steps + 1e-5)
