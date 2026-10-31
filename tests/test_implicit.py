"""Relaxations of implicit functions, from `lax.custom_root` and linear solves."""

import functools
from typing import Any

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

needs_exposed_root = pytest.mark.skipif(
    jax.__version_info__ < (0, 11),
    reason="needs jax>=0.11, where the residual is visible",
)
"""Marks tests of roots, which JAX hides from the interpreter before version 0.11."""


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
    """Draws a box of half-width at most 0.15 in the test domain, and points inside it.

    The midpoint preconditioner of the Krawczyk operator verifies a root of the cubic only
    if the right-hand side has half-width below about `0.22` where the linear coefficient
    is smallest, so wider boxes can legitimately get infinite bounds.
    """
    middle = np.array([draw(coordinates), draw(coordinates)])
    radius = np.array([draw(st.floats(0.0, 0.15)) for _ in range(2)])
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


@needs_exposed_root
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


@needs_exposed_root
def test_custom_root_planes_follow_implicit_derivative():
    """Checks that the planes at the reference have the derivative of the root."""
    reference = jnp.array([0.1, 0.5])
    lower, upper = reference - 0.01, reference + 0.01
    planes = jmk.affine_relaxation(cubic_root)(reference, lower, upper)
    gradient = jax.grad(cubic_root)(reference)
    np.testing.assert_allclose(planes.convex_gradient, gradient, rtol=1e-2)
    np.testing.assert_allclose(planes.concave_gradient, gradient, rtol=1e-2)


@needs_exposed_root
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


@needs_exposed_root
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


@needs_exposed_root
def test_relaxed_root_lowers_without_library_calls():
    """Checks that the relaxation of a root needs no LAPACK call, which a PJRT plugin lacks."""
    relaxed = jax.jit(jmk.relax(cubic_root))
    text = relaxed.lower(jnp.zeros(2), -jnp.ones(2), jnp.ones(2)).as_text()
    assert "lapack" not in text


def two_roots(x: jax.Array) -> jax.Array:
    """Returns the sum of the roots of the cubic for two different right-hand sides."""
    return cubic_root(x) + cubic_root(x * jnp.array([0.5, 1.0]))


def roots_in_calls(x: jax.Array) -> jax.Array:
    """Returns the root of the cubic inside `jax.jit`, which takes no enclosure."""
    return jax.jit(cubic_root)(x)


def box_bounds(size: int = 2):
    """Returns example arguments for the size query: a reference point and a box."""
    return jnp.zeros(size), -jnp.ones(size), jnp.ones(size)


@pytest.mark.parametrize(
    ("function", "size"),
    [
        (cubic_root, 2),
        (two_roots, 4),
        (roots_in_calls, 0),
        (linear_system, 0),
        (lambda x: jnp.sin(x), 0),
    ],
)
@needs_exposed_root
def test_root_enclosure_size(function, size):
    """Checks that only the roots at the top level count, with two numbers per entry."""
    assert jmk.root_enclosure_size(function, *box_bounds()) == size


@needs_exposed_root
def test_size_query_accepts_shape_structs():
    """Checks that the size can be asked with shapes instead of arrays."""
    shape = jax.ShapeDtypeStruct((2,), jnp.float32)
    assert jmk.root_enclosure_size(cubic_root, shape, shape, shape) == 2


@pytest.mark.parametrize(
    "function", [cubic_root, two_roots, roots_in_calls, linear_system]
)
@needs_exposed_root
def test_missing_enclosures_change_nothing(function):
    """Checks that NaN enclosures give the relaxation that `relax` gives without them."""
    reference, lower, upper = (
        jnp.array([0.1, 0.05]),
        jnp.array([-0.1, -0.1]),
        jnp.array([0.2, 0.1]),
    )
    size = jmk.root_enclosure_size(function, reference, lower, upper)
    plain = jmk.relax(function)(reference, lower, upper)
    enclosed, found = jmk.relax(function, enclosures=True)(
        reference, lower, upper, jnp.full(size, jnp.nan)
    )
    for first, second in zip(
        jax.tree.leaves(plain), jax.tree.leaves(enclosed), strict=True
    ):
        np.testing.assert_array_equal(first, second)
    assert found.shape == (size,)


@needs_exposed_root
def test_found_enclosure_holds_the_root():
    """Checks that the verified enclosure is finite and contains the root at every sampled point."""
    lower, upper = jnp.array([-0.5, -0.1]), jnp.array([0.5, 0.1])
    relaxed = jmk.relax(cubic_root, enclosures=True)
    _, found = relaxed(jnp.zeros(2), lower, upper, jnp.full(2, jnp.nan))
    low, high = found[:1], found[1:]
    rng = np.random.default_rng(0)
    points = lower + (upper - lower) * rng.uniform(size=(50, 2))
    roots = np.asarray(jax.vmap(cubic_root)(jnp.asarray(points)))
    assert np.all(np.isfinite(found))
    assert np.all(low <= roots + 1e-12) and np.all(roots <= high + 1e-12)


@needs_exposed_root
def test_unverified_boxes_find_no_enclosure():
    """Checks that a box too wide for the Krawczyk operator gives NaN and infinite bounds."""
    lower, upper = jnp.array([-2.0, -0.25]), jnp.array([-2.0, 0.25])
    relaxation, found = jmk.relax(cubic_root, enclosures=True)(
        jnp.array([-2.0, 0.0]), lower, upper, jnp.full(2, jnp.nan)
    )
    assert np.all(np.isnan(found))
    assert np.isinf(relaxation.lower) and np.isinf(relaxation.upper)


@st.composite
def nested_boxes(draw):
    """Draws a box, a box inside it and points inside the inner one."""
    middle = np.array([draw(coordinates), draw(coordinates)])
    radius = np.array([draw(st.floats(0.0, 0.15)) for _ in range(2)])
    lower, upper = np.clip(middle - radius, -2, 2), np.clip(middle + radius, -2, 2)
    first = np.array([draw(st.floats(0.0, 1.0)) for _ in range(2)])
    second = np.array([draw(st.floats(0.0, 1.0)) for _ in range(2)])
    inner_lower = lower + np.minimum(first, second) * (upper - lower)
    inner_upper = lower + np.maximum(first, second) * (upper - lower)
    fractions = np.array(
        [[draw(st.floats(0.0, 1.0)) for _ in range(2)] for _ in range(6)]
    )
    points = np.clip(
        inner_lower + fractions * (inner_upper - inner_lower), inner_lower, inner_upper
    )
    arrays = (lower, upper, inner_lower, inner_upper, points)
    return tuple(array.astype(np.float32) for array in arrays)


@needs_exposed_root
@settings(max_examples=40, deadline=None)
@given(sample=nested_boxes())
def test_enclosure_of_a_larger_box_gives_valid_relaxations(sample):
    """Checks that relaxations started from the enclosure of an outer box stay valid inside it."""
    lower, upper, inner_lower, inner_upper, points = sample
    relaxed = jax.jit(
        jax.vmap(jmk.relax(cubic_root, enclosures=True), in_axes=(0, None, None, None))
    )
    single = jmk.relax(cubic_root, enclosures=True)
    outer_middle = (lower + upper) / 2
    _, found = single(outer_middle, lower, upper, jnp.full(2, jnp.nan))
    relaxation, inner_found = relaxed(points, inner_lower, inner_upper, found)
    values = np.asarray(jax.vmap(cubic_root)(points))
    allowed = slack(values, relaxation.lower, relaxation.upper)
    if np.all(np.isnan(found)):
        # No enclosure to start from, so the search for one ran and the usual checks apply.
        return
    assert np.all(np.isfinite(relaxation.lower)) and np.all(
        np.isfinite(relaxation.upper)
    )
    assert np.all(relaxation.lower <= relaxation.convex + allowed)
    assert np.all(relaxation.convex <= values + allowed)
    assert np.all(values <= relaxation.concave + allowed)
    assert np.all(relaxation.concave <= relaxation.upper + allowed)
    # The enclosure found for the inner box holds the root at every point of that box.
    assert np.all(inner_found[0, :1] <= values + allowed)
    assert np.all(values <= inner_found[0, 1:] + allowed)


@needs_exposed_root
@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize("side", ["convex", "concave"])
def test_one_sided_functions_with_enclosures(mode, side):
    """Checks that the one-sided functions with NaN enclosures match the ones without."""
    make = jmk.convex_side if side == "convex" else jmk.concave_side
    reference, lower, upper = (
        jnp.array([0.1, 0.05]),
        jnp.array([-0.1, -0.1]),
        jnp.array([0.2, 0.1]),
    )
    value, gradient = make(cubic_root, mode)(reference, lower, upper)
    with_value, with_gradient, found = make(cubic_root, mode, enclosures=True)(
        reference, lower, upper, jnp.full(2, jnp.nan)
    )
    np.testing.assert_array_equal(value, with_value)
    np.testing.assert_array_equal(gradient, with_gradient)
    assert np.all(np.isfinite(found))
    # Started from the enclosure it found, the same call stays valid.
    again_value, again_gradient, _ = make(cubic_root, mode, enclosures=True)(
        reference, lower, upper, found
    )
    assert np.isfinite(again_value) and np.all(np.isfinite(again_gradient))


@needs_exposed_root
def test_interval_with_enclosures():
    """Checks that the interval function with NaN enclosures matches the one without."""
    lower, upper = jnp.array([-0.1, -0.1]), jnp.array([0.2, 0.1])
    low, high = jmk.interval(cubic_root)(lower, upper)
    with_low, with_high, found = jmk.interval(cubic_root, enclosures=True)(
        lower, upper, jnp.full(2, jnp.nan)
    )
    np.testing.assert_array_equal(low, with_low)
    np.testing.assert_array_equal(high, with_high)
    assert np.all(np.isfinite(found))


@needs_exposed_root
def test_enclosure_errors():
    """Checks the errors for enclosures of the wrong length and for tightened bounds."""
    reference, lower, upper = jnp.zeros(2), -jnp.ones(2), jnp.ones(2)
    relaxed = jmk.relax(cubic_root, enclosures=True)
    with pytest.raises(ValueError, match="too few"):
        relaxed(reference, lower, upper, jnp.zeros(0))
    with pytest.raises(ValueError, match="roots take"):
        relaxed(reference, lower, upper, jnp.full(6, jnp.nan))
    with pytest.raises(ValueError, match="tighten_bounds"):
        jmk.relax(cubic_root, tighten_bounds=True, enclosures=True)
    sideways: Any = "sideways"
    with pytest.raises(ValueError, match="mode"):
        jmk.convex_side(cubic_root, sideways, enclosures=True)


@pytest.mark.skipif(
    jax.__version_info__ >= (0, 11), reason="JAX 0.11 exposes the residual of roots"
)
def test_custom_root_before_jax_011_raises():
    """Checks that `lax.custom_root` raises where JAX hides its residual."""
    with pytest.raises(jmk.UnsupportedPrimitiveError, match="custom_root"):
        jmk.relax(cubic_root)(jnp.zeros(2), -jnp.ones(2), jnp.ones(2))
