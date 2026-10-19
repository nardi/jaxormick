"""Tightness of the univariate relaxations, compared with numerically computed envelopes.

These tests run in `float64`, since they compare against envelopes to more digits than
`float32` holds.
"""

import functools
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import lax

import jaxormick as jmk
from jaxormick._rules import _composition

GRID_SIZE = 20001
"""Number of grid points on which the numerical envelopes are computed."""


@pytest.fixture(autouse=True)
def float64():
    """Runs each test with 64-bit floating-point numbers enabled."""
    with jax.enable_x64(True):
        yield


def lower_hull(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns the vertices of the lower convex hull of sorted points with Andrew's scan."""
    hull: list[int] = []
    for index in range(len(points)):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            cross = (points[second] - points[first]) * (
                values[index] - values[first]
            ) - (values[second] - values[first]) * (points[index] - points[first])
            if cross > 0:
                break
            hull.pop()
        hull.append(index)
    return points[hull], values[hull]


@functools.cache
def batched_relaxation(name: str):
    """Returns the relaxation of a univariate test function, batched over points."""
    return jax.vmap(jmk.relax(UNIVARIATE[name]), in_axes=(0, None, None))


def numerical_envelopes(function, lower: float, upper: float, at: np.ndarray):
    """Returns the convex and concave envelopes of `function` on a grid, evaluated at `at`."""
    grid = np.linspace(lower, upper, GRID_SIZE)
    values = np.asarray(function(jnp.asarray(grid)))
    convex = np.interp(at, *lower_hull(grid, values))
    hull_points, hull_values = lower_hull(grid, -values)
    concave = -np.interp(at, hull_points, hull_values)
    return convex, concave


UNIVARIATE = {
    "cube": lambda x: x**3,
    "fifth_power": lambda x: x**5,
    "seventh_power": lambda x: lax.integer_pow(x, 7),
    "tanh": jnp.tanh,
    "logistic": jax.nn.sigmoid,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "exp": jnp.exp,
    "square": jnp.square,
    "abs": jnp.abs,
}
"""Univariate functions with exact envelopes, mapped to their definitions."""


@pytest.mark.parametrize("name", sorted(UNIVARIATE))
@settings(max_examples=25, deadline=None)
@given(
    lower=st.floats(-4.0, 4.0),
    width=st.floats(0.05, 8.0),
    fractions=st.lists(st.floats(0.0, 1.0), min_size=5, max_size=5),
)
def test_relaxation_matches_envelope(name, lower, width, fractions):
    """Checks that the relaxation of `f(x)` over `[l, u]` is the envelope of `f`."""
    function = UNIVARIATE[name]
    upper = lower + width
    points = lower + np.array(fractions) * width
    relaxation = batched_relaxation(name)(points, lower, upper)
    convex, concave = numerical_envelopes(function, lower, upper, points)
    step = width / (GRID_SIZE - 1)
    scale = 1 + np.max(np.abs(function(jnp.array([lower, upper]))))
    tolerance = 1e-6 * scale + 10 * step**2 * scale
    np.testing.assert_allclose(relaxation.convex, convex, atol=tolerance)
    np.testing.assert_allclose(relaxation.concave, concave, atol=tolerance)


def test_cubic_envelope_has_closed_form():
    """Checks the convex envelope of `x³` on `[−1, 2]`, which is tangent at `x = 1/2`."""
    points = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    relaxation = jax.vmap(jmk.relax(lambda x: x**3), in_axes=(0, None, None))(
        points, -1.0, 2.0
    )
    line = -1 + 0.75 * (points + 1)
    expected = np.where(points <= 0.5, line, points**3)
    np.testing.assert_allclose(relaxation.convex, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("steps", [0, 1, 3, 8])
@settings(max_examples=20, deadline=None)
@given(lower=st.floats(-6.0, -0.1), upper=st.floats(0.1, 6.0))
def test_envelope_with_few_bisection_steps_is_valid(steps, lower, upper):
    """Checks that an inexact tangent point still gives a convex underestimator."""
    points = np.linspace(lower, upper, 101)
    with mock.patch.object(_composition, "bisection_steps", lambda dtype: steps):
        envelope = np.asarray(
            _composition.convex_envelope_of_convex_concave(
                jnp.tanh,
                0.0,
                jnp.asarray(lower),
                jnp.asarray(upper),
                jnp.asarray(points),
            )
        )
    assert np.all(envelope <= np.tanh(points) + 1e-12)
    second_differences = envelope[:-2] - 2 * envelope[1:-1] + envelope[2:]
    assert np.all(second_differences >= -1e-12)


def test_envelope_with_inflection_outside_interval():
    """Checks the two cases where the interval lies on one side of the inflection point."""
    points = jnp.linspace(-3.0, -1.0, 5)
    left = _composition.convex_envelope_of_convex_concave(
        jnp.tanh, 0.0, jnp.asarray(-3.0), jnp.asarray(-1.0), points
    )
    np.testing.assert_allclose(left, jnp.tanh(points))
    points = jnp.linspace(1.0, 3.0, 5)
    right = _composition.convex_envelope_of_convex_concave(
        jnp.tanh, 0.0, jnp.asarray(1.0), jnp.asarray(3.0), points
    )
    chord = jnp.tanh(1.0) + (jnp.tanh(3.0) - jnp.tanh(1.0)) / 2 * (points - 1)
    np.testing.assert_allclose(right, chord)


def test_cos_over_several_periods():
    """Checks the bounds of `cos` and its envelopes between and beyond its minima."""
    points = np.array([-7.0, -2.0, 0.0, 5.0, 12.0])
    relaxation = batched_relaxation("cos")(points, -7.0, 12.0)
    np.testing.assert_allclose(relaxation.lower, -1)
    np.testing.assert_allclose(relaxation.upper, 1)
    np.testing.assert_allclose(relaxation.convex[1:4], -1)
    np.testing.assert_allclose(
        np.asarray(relaxation.convex)[[0, 4]], np.cos(points[[0, 4]])
    )


def test_mid_with_crossed_sides():
    """Checks that a point between crossed sides is returned with zero gradient.

    Rounding can leave the concave side an ulp below the convex side.
    """
    convex, concave = jnp.float32(1.5000001), jnp.float32(1.5)
    np.testing.assert_equal(np.asarray(_composition.mid(convex, concave, 1.5)), 1.5)
    np.testing.assert_equal(np.asarray(_composition.mid(convex, concave, 1.0)), convex)
    np.testing.assert_equal(np.asarray(_composition.mid(convex, concave, 2.0)), concave)
    gradients = jax.grad(_composition.mid, argnums=(0, 1))(convex, concave, 1.5)
    np.testing.assert_equal(np.asarray(gradients), [0.0, 0.0])
