"""Relaxations of implicit functions, from `lax.custom_root` and linear solves."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import boxes_with_points, coordinates, slack
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import lax

import jaxormick as jmk
from jaxormick import _implicit


def linear_system(x: jax.Array) -> jax.Array:
    """Returns the solution of a diagonally dominant system that depends on `x`."""
    matrix = jnp.array([[6.0 + x[0], 1.0], [1.0, 5.0 - x[1]]])
    return jnp.linalg.solve(matrix, jnp.stack([x[1], 1.0 + x[0] * x[1]]))


def cubic_root(x: jax.Array) -> jax.Array:
    """Returns the real root `y` of `y³ + (3 + x₀) y = x₁`, found by Newton's method."""

    def residual(y: jax.Array) -> jax.Array:
        return y**3 + (3.0 + x[0]) * y - x[1]

    def newton(function, guess):
        derivative = jax.grad(function)
        return lax.fori_loop(0, 20, lambda _, y: y - function(y) / derivative(y), guess)

    def tangent_solve(function, value):
        return value / jax.grad(function)(0.0)

    return lax.custom_root(residual, 0.0, newton, tangent_solve)


@st.composite
def small_boxes_with_points(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draws a box of half-width at most 0.25 in the test domain, and points inside it."""
    middle = np.array([draw(coordinates), draw(coordinates)])
    radius = np.array([draw(st.floats(0.0, 0.25)) for _ in range(2)])
    lower, upper = np.clip(middle - radius, -2, 2), np.clip(middle + radius, -2, 2)
    fractions = np.array(
        [[draw(st.floats(0.0, 1.0)) for _ in range(2)] for _ in range(8)]
    )
    points = np.clip(lower + fractions * (upper - lower), lower, upper)
    return (
        lower.astype(np.float32),
        upper.astype(np.float32),
        points.astype(np.float32),
    )


@functools.cache
def batched_relaxation(name: str, tighten_bounds: bool):
    """Returns the relaxation of an implicit test function, batched over points."""
    function = {"linear_system": linear_system, "cubic_root": cubic_root}[name]
    relaxed = jmk.relax(function, tighten_bounds=tighten_bounds)
    return jax.jit(jax.vmap(relaxed, in_axes=(0, None, None))), function


def check_relaxation(name: str, tighten_bounds: bool, sample) -> None:
    """Checks that the relaxation is finite, bounds the function and is convex."""
    lower, upper, points = sample
    relaxed, function = batched_relaxation(name, tighten_bounds)
    relaxation = relaxed(points, lower, upper)
    assert np.all(np.isfinite(relaxation.lower)) and np.all(
        np.isfinite(relaxation.upper)
    )
    values = np.asarray(jax.vmap(function)(points))
    allowed = slack(values, relaxation.lower, relaxation.upper)
    assert np.all(relaxation.lower <= relaxation.convex + allowed)
    assert np.all(relaxation.convex <= values + allowed)
    assert np.all(values <= relaxation.concave + allowed)
    assert np.all(relaxation.concave <= relaxation.upper + allowed)

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


@pytest.mark.parametrize("tighten_bounds", [False, True])
@settings(max_examples=20, deadline=None)
@given(sample=boxes_with_points())
def test_linear_solve_relaxation_is_valid(tighten_bounds, sample):
    """Checks the relaxation of `jnp.linalg.solve` with a matrix that depends on `x`."""
    check_relaxation("linear_system", tighten_bounds, sample)


@pytest.mark.parametrize("tighten_bounds", [False, True])
@settings(max_examples=20, deadline=None)
@given(sample=small_boxes_with_points())
def test_custom_root_relaxation_is_valid(tighten_bounds, sample):
    """Checks the relaxation of a root found by `lax.custom_root` on small boxes."""
    check_relaxation("cubic_root", tighten_bounds, sample)


def test_custom_linear_solve_with_constant_matrix():
    """Checks that a solve with a constant matrix relaxes its right-hand side exactly."""

    def function(x: jax.Array) -> jax.Array:
        return lax.custom_linear_solve(lambda v: 2 * v, x[0], lambda _, b: b / 2)

    reference, lower, upper = jnp.array([0.5, 0.0]), -jnp.ones(2), jnp.ones(2)
    relaxation = jmk.relax(function)(reference, lower, upper)
    np.testing.assert_allclose(
        [relaxation.lower, relaxation.upper, relaxation.convex, relaxation.concave],
        [-0.5, 0.5, 0.25, 0.25],
        rtol=1e-6,
    )


def test_custom_root_planes_follow_implicit_derivative():
    """Checks that the planes at the reference have the derivative of the root."""
    reference = jnp.array([0.1, 0.5])
    lower, upper = reference - 0.01, reference + 0.01
    planes = jmk.affine_relaxation(cubic_root)(reference, lower, upper)
    gradient = jax.grad(cubic_root)(reference)
    np.testing.assert_allclose(planes.convex_gradient, gradient, rtol=1e-2)
    np.testing.assert_allclose(planes.concave_gradient, gradient, rtol=1e-2)


def test_custom_root_without_enclosure_is_unbounded():
    """Checks that a box too wide for the Krawczyk test gives infinite bounds."""
    reference, lower, upper = (
        jnp.zeros(2),
        jnp.array([-2.0, -2.0]),
        jnp.array([2.0, 2.0]),
    )
    relaxation = jmk.relax(cubic_root)(reference, lower, upper)
    assert relaxation.lower == -np.inf and relaxation.upper == np.inf
    assert relaxation.convex == -np.inf and relaxation.concave == np.inf


def test_custom_root_with_auxiliary_output_raises():
    """Checks that `lax.custom_root` with `has_aux=True` raises."""

    def function(x: jax.Array) -> jax.Array:
        root, _ = lax.custom_root(
            lambda y: y - x[0],
            0.0,
            lambda f, guess: (guess - f(guess), x[1]),
            lambda f, value: value,
            has_aux=True,
        )
        return root

    with pytest.raises(jmk.UnsupportedPrimitiveError, match="has_aux"):
        jmk.relax(function)(jnp.zeros(2), -jnp.ones(2), jnp.ones(2))


@pytest.mark.parametrize("size", [1, 2, 5])
def test_inverse_matches_the_library(size):
    """Checks the pivoting inverse against `jnp.linalg.inv`, including a zero on the diagonal."""
    rng = np.random.default_rng(size)
    matrix = jnp.asarray(rng.normal(size=(size, size)))
    if size > 1:
        matrix = matrix.at[0, 0].set(0.0)
    np.testing.assert_allclose(
        _implicit.inverse(matrix),
        jnp.linalg.inv(matrix),
        rtol=100 * jnp.finfo(matrix.dtype).eps,
        atol=100 * jnp.finfo(matrix.dtype).eps,
    )


def test_inverse_of_a_singular_matrix_is_not_finite():
    """Checks that a singular matrix gives non-finite entries, which the preconditioner zeroes."""
    inverse = _implicit.inverse(jnp.array([[1.0, 2.0], [2.0, 4.0]]))
    assert not np.all(np.isfinite(inverse))


def test_relaxed_root_lowers_without_library_calls():
    """Checks that the relaxation of a root needs no LAPACK call, which a PJRT plugin lacks."""
    relaxed = jax.jit(jmk.relax(cubic_root))
    text = relaxed.lower(jnp.zeros(2), -jnp.ones(2), jnp.ones(2)).as_text()
    assert "lapack" not in text
