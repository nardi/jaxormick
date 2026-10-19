"""αBB relaxations, which bound the Hessian of a function over the box."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import boxes_with_points, slack
from functions import TEST_FUNCTIONS
from hypothesis import given, settings

import jaxormick as jmk
from jaxormick._alpha_bb import gershgorin_shifts

NONSMOOTH_NAMES = {
    "absolute",
    "maximum",
    "minimum",
    "relu",
    "softplus",
    "clip",
    "reduce_max_min",
}
"""Test functions whose Hessians contain comparisons, which have no relaxation."""

SMOOTH_NAMES = sorted(set(TEST_FUNCTIONS) - NONSMOOTH_NAMES)
"""Test functions whose Hessians can be bounded over every box of the test domain."""


@functools.cache
def batched_relaxation(name: str, alpha_bb: bool = True):
    """Returns the relaxation of a test function, batched over reference points."""
    relaxed = jmk.relax(TEST_FUNCTIONS[name], alpha_bb=alpha_bb)
    return jax.jit(jax.vmap(relaxed, in_axes=(0, None, None)))


@pytest.mark.parametrize("name", SMOOTH_NAMES)
@settings(max_examples=20, deadline=None)
@given(sample=boxes_with_points())
def test_alpha_bb_bounds_function_and_is_convex(name, sample):
    """Checks `convex ≤ f ≤ concave` and the midpoint inequalities of convexity."""
    lower, upper, points = sample
    relaxed = batched_relaxation(name)
    relaxation = relaxed(points, lower, upper)
    values = np.asarray(jax.vmap(TEST_FUNCTIONS[name])(points))
    allowed = slack(values, relaxation.lower, relaxation.upper)
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


@pytest.mark.parametrize("name", SMOOTH_NAMES)
@settings(max_examples=20, deadline=None)
@given(sample=boxes_with_points())
def test_alpha_bb_is_at_least_as_tight_as_mccormick(name, sample):
    """Checks that the intersection never loosens the McCormick relaxation.

    Near a tie, the intersection keeps a bound that is a few ulps looser than a side.
    """
    lower, upper, points = sample
    with_alpha_bb = batched_relaxation(name)(points, lower, upper)
    mccormick = batched_relaxation(name, alpha_bb=False)(points, lower, upper)
    allowed = slack(mccormick.lower, mccormick.upper)
    assert np.all(with_alpha_bb.convex >= mccormick.convex - allowed)
    assert np.all(with_alpha_bb.concave <= mccormick.concave + allowed)


def test_alpha_bb_tightens_quadratic_form():
    """Checks that αBB narrows the gap between the sides of an indefinite quadratic."""
    reference, lower, upper = jnp.array([0.1, -0.2]), -jnp.ones(2), jnp.ones(2)
    with_alpha_bb = jmk.relax(TEST_FUNCTIONS["quadratic_form"], alpha_bb=True)(
        reference, lower, upper
    )
    mccormick = jmk.relax(TEST_FUNCTIONS["quadratic_form"])(reference, lower, upper)
    gap = with_alpha_bb.concave - with_alpha_bb.convex
    assert gap < 0.75 * (mccormick.concave - mccormick.convex)


def test_gershgorin_shifts_of_product():
    """Checks the shifts for the Hessian of `x · y` on a box of widths 2 and 1.

    The off-diagonal entry of 1 is scaled by the ratio of the widths, which gives
    shifts of `(1 · 1/2) / 2` and `(1 · 2/1) / 2`.
    """
    hessian = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    convex_shift, concave_shift = gershgorin_shifts(
        hessian, hessian, jnp.array([2.0, 1.0])
    )
    np.testing.assert_allclose(convex_shift, [0.25, 1.0])
    np.testing.assert_allclose(concave_shift, [0.25, 1.0])


def test_alpha_bb_detects_convex_function():
    """Checks that a convex function is its own convex side, since no shift is needed."""
    reference, lower, upper = jnp.array([0.3, -0.7]), -jnp.ones(2), jnp.ones(2)
    function = TEST_FUNCTIONS["sum_exp"]
    relaxation = jmk.relax(function, alpha_bb=True)(reference, lower, upper)
    np.testing.assert_allclose(relaxation.convex, function(reference), rtol=1e-6)


def test_alpha_bb_falls_back_where_hessian_is_unbounded():
    """Checks that an unbounded Hessian leaves the McCormick relaxation unchanged.

    The second derivative of `√(x + 2)` is unbounded on a box that touches `x = −2`.
    """

    def function(x: jax.Array) -> jax.Array:
        return jnp.sqrt(x[0] + 2) * x[1]

    reference, lower, upper = (
        jnp.array([-1.0, 0.5]),
        jnp.array([-2.0, 0.0]),
        jnp.ones(2),
    )
    with_alpha_bb = jmk.relax(function, alpha_bb=True)(reference, lower, upper)
    mccormick = jmk.relax(function)(reference, lower, upper)
    np.testing.assert_allclose(with_alpha_bb.convex, mccormick.convex)
    np.testing.assert_allclose(with_alpha_bb.concave, mccormick.concave)
    planes = jmk.affine_relaxation(function, alpha_bb=True)(reference, lower, upper)
    assert np.all(np.isfinite(planes.convex_gradient))
    assert np.all(np.isfinite(planes.concave_gradient))


def test_alpha_bb_accepts_box_of_zero_width():
    """Checks a box that fixes one variable, whose quadratic term vanishes."""
    reference, lower, upper = jnp.array([0.5, 0.2]), jnp.array([0.5, -1.0]), jnp.ones(2)
    function = TEST_FUNCTIONS["six_hump_camel"]
    relaxation = jmk.relax(function, alpha_bb=True)(reference, lower, upper)
    assert np.isfinite(relaxation.convex) and np.isfinite(relaxation.concave)
    assert relaxation.convex <= function(reference) <= relaxation.concave


@pytest.mark.parametrize("side", [jmk.convex_side, jmk.concave_side])
def test_sides_pass_alpha_bb_to_relax(side):
    """Checks that one side alone matches the side of the full relaxation."""
    reference, lower, upper = jnp.array([0.1, -0.2]), -jnp.ones(2), jnp.ones(2)
    function = TEST_FUNCTIONS["quadratic_form"]
    value, _ = side(function, alpha_bb=True)(reference, lower, upper)
    relaxation = jmk.relax(function, alpha_bb=True)(reference, lower, upper)
    expected = relaxation.convex if side is jmk.convex_side else relaxation.concave
    np.testing.assert_allclose(value, expected, rtol=1e-6)


def test_alpha_bb_with_several_outputs_raises():
    """Checks that αBB rejects a function with more than one output array."""
    relaxed = jmk.relax(lambda x: (x[0], x[1]), alpha_bb=True)
    with pytest.raises(ValueError, match="alpha_bb"):
        relaxed(jnp.zeros(2), -jnp.ones(2), jnp.ones(2))


def test_alpha_bb_of_nonsmooth_function_raises():
    """Checks that a Hessian without a relaxation names `alpha_bb` in the error."""
    relaxed = jmk.relax(TEST_FUNCTIONS["relu"], alpha_bb=True)
    with pytest.raises(jmk.UnsupportedPrimitiveError, match="alpha_bb"):
        relaxed(jnp.zeros(2), -jnp.ones(2), jnp.ones(2))
