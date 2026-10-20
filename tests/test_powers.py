"""Powers with an array of constant exponents."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxormick as jmk

POSITIVE_LOWER = np.linspace(0.4, 1.1, 8, dtype=np.float32)
"""Lower bounds of a box that is positive in every coordinate."""

POSITIVE_UPPER = POSITIVE_LOWER + np.linspace(0.3, 2.0, 8, dtype=np.float32)
"""Upper bounds of the positive box."""

POSITIVE_REFERENCE = (POSITIVE_LOWER + POSITIVE_UPPER) / 2
"""Reference point in the middle of the positive box."""

MIXED_EXPONENTS = np.array([0.0, 1.0, 2.0, 3.0, -2.0, -1.7, 0.3, 1.5])
"""Exponents that cover every integer and fractional class, one for each coordinate."""


def assert_same_relaxation(first: jmk.Relaxation, second: jmk.Relaxation) -> None:
    """Checks that two relaxations have the same shape and agree in every field."""
    assert first.shape == second.shape
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)


def stacked_scalar_powers(point: jax.Array, exponents: np.ndarray) -> jax.Array:
    """Returns the powers of the coordinates of `point`, each with its own scalar."""
    return jnp.stack(
        [point[index] ** float(value) for index, value in enumerate(exponents)]
    )


def test_array_exponent_matches_scalar_exponents():
    """Checks that an exponent array relaxes like the stack of its scalar powers."""
    array_form = jmk.relax(lambda x: x**MIXED_EXPONENTS)
    scalar_form = jmk.relax(lambda x: stacked_scalar_powers(x, MIXED_EXPONENTS))
    arguments = (POSITIVE_REFERENCE, POSITIVE_LOWER, POSITIVE_UPPER)
    assert_same_relaxation(array_form(*arguments), scalar_form(*arguments))


def test_integer_exponents_on_box_around_zero():
    """Checks that integer exponents on a box that contains zero match their scalars."""
    exponents = np.array([2, 5, 3, 4, 1, 0, 3, 2])
    lower, upper = (
        np.full(8, -1.0, np.float32),
        np.linspace(0.5, 2, 8, dtype=np.float32),
    )
    reference = (lower + upper) / 2
    assert_same_relaxation(
        jmk.relax(lambda x: x**exponents)(reference, lower, upper),
        jmk.relax(lambda x: stacked_scalar_powers(x, exponents))(
            reference, lower, upper
        ),
    )


def test_uniform_exponent_array_matches_scalar_exponent():
    """Checks that an array of equal exponents takes the path of one exponent."""
    arguments = (POSITIVE_REFERENCE, POSITIVE_LOWER, POSITIVE_UPPER)
    for value in (2.5, 3.0):
        assert_same_relaxation(
            jmk.relax(lambda x: x ** np.full(8, value))(*arguments),
            jmk.relax(lambda x: x**value)(*arguments),
        )


def test_scalar_base_with_array_exponent():
    """Checks that a relaxed scalar raised to an array gives an array of powers."""
    exponents = np.array([0.5, 2.0, 3.0])
    arguments = (POSITIVE_REFERENCE[:1], POSITIVE_LOWER[:1], POSITIVE_UPPER[:1])
    assert_same_relaxation(
        jmk.relax(lambda x: x[0] ** exponents)(*arguments),
        jmk.relax(lambda x: jnp.stack([x[0] ** float(value) for value in exponents]))(
            *arguments
        ),
    )


def test_scalar_base_with_uniform_array_exponent():
    """Checks that a uniform exponent array still gives the shape of the array."""
    arguments = (POSITIVE_REFERENCE[:1], POSITIVE_LOWER[:1], POSITIVE_UPPER[:1])
    relaxation = jmk.relax(lambda x: x[0] ** np.full(3, 2.0))(*arguments)
    assert relaxation.shape == (3,)
    assert_same_relaxation(
        relaxation, jmk.relax(lambda x: jnp.stack([x[0] ** 2.0] * 3))(*arguments)
    )


def test_matrix_base_with_row_of_exponents():
    """Checks that a row of exponents applies to every row of a matrix."""
    exponents = np.array([0.5, 1.5, 2.0, 3.0])
    shape = (2, 4)
    arguments = (
        POSITIVE_REFERENCE.reshape(shape),
        POSITIVE_LOWER.reshape(shape),
        POSITIVE_UPPER.reshape(shape),
    )
    assert_same_relaxation(
        jmk.relax(lambda x: x**exponents)(*arguments),
        jmk.relax(
            lambda x: jnp.stack([stacked_scalar_powers(row, exponents) for row in x])
        )(*arguments),
    )


def test_empty_exponent_array():
    """Checks that a power of no elements is an empty relaxation."""
    relaxation = jmk.relax(lambda x: x[:0] ** np.zeros(0))(
        POSITIVE_REFERENCE, POSITIVE_LOWER, POSITIVE_UPPER
    )
    assert relaxation.shape == (0,)


def test_integer_dtype_exponent_array():
    """Checks that an exponent array of integers is accepted."""
    arguments = (POSITIVE_REFERENCE[:3], POSITIVE_LOWER[:3], POSITIVE_UPPER[:3])
    assert_same_relaxation(
        jmk.relax(lambda x: x ** jnp.array([2, 3, -1]))(*arguments),
        jmk.relax(lambda x: stacked_scalar_powers(x, np.array([2, 3, -1])))(*arguments),
    )


def test_exponent_array_that_is_traced_raises():
    """Checks that an exponent array that depends on a traced argument is rejected."""
    relaxed = jmk.relax(lambda x, exponents: x**exponents)
    with pytest.raises(jmk.UnsupportedPrimitiveError, match="known while tracing"):
        relaxed(
            POSITIVE_REFERENCE,
            POSITIVE_LOWER,
            POSITIVE_UPPER,
            jnp.asarray(MIXED_EXPONENTS, jnp.float32),
        )


def product(x: jax.Array, scale: jax.Array) -> jax.Array:
    """Returns a product of powers with a different exponent for each factor."""
    return jnp.prod((scale * x) ** jnp.linspace(0.01, 0.06, x.shape[0]))


def test_product_of_powers_with_exponent_array():
    """Checks the convex side of a product with an exponent for each factor."""
    size = 8
    arguments = (
        jnp.full(size, 1.0),
        jnp.full(size, 0.1),
        jnp.full(size, 10.0),
        jnp.ones(size),
    )
    value, gradient = jmk.convex_side(product)(*arguments)
    assert value <= product(arguments[0], arguments[3]) + 1e-6
    assert np.all(np.isfinite(gradient))


def lowered_lines(size: int) -> int:
    """Returns the number of lines of StableHLO of the relaxed product of a size."""
    arguments = (
        jnp.full(size, 1.0),
        jnp.full(size, 0.1),
        jnp.full(size, 10.0),
        jnp.ones(size),
    )
    return len(
        jax.jit(jmk.convex_side(product)).lower(*arguments).as_text().splitlines()
    )


def test_program_size_grows_slowly_with_factor_count():
    """Checks that four times as many factors need less than twice the program."""
    assert lowered_lines(32) < 2 * lowered_lines(8)
