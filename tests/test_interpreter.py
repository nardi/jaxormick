"""How the interpreter handles call primitives, control flow and unsupported operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import jaxormick as jmk

LOWER = jnp.array([-1.0, 0.5])
"""Lower bounds of the box used in these tests."""

UPPER = jnp.array([1.5, 2.0])
"""Upper bounds of the box used in these tests."""

REFERENCE = jnp.array([0.25, 1.0])
"""Reference point inside the box."""


def assert_same_relaxation(first: jmk.Relaxation, second: jmk.Relaxation) -> None:
    """Checks that two relaxations agree in every field."""
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)


def test_scan_matches_unrolled_loop():
    """Checks that a relaxed `fori_loop` gives the relaxation of the unrolled loop."""

    def step(guess: jax.Array, target: jax.Array) -> jax.Array:
        return (guess + target / guess) / 2

    def looped(x: jax.Array) -> jax.Array:
        target = x[1] + 1
        return lax.fori_loop(0, 4, lambda i, guess: step(guess, target), 2.0)

    def unrolled(x: jax.Array) -> jax.Array:
        guess = jnp.asarray(2.0)
        for _ in range(4):
            guess = step(guess, x[1] + 1)
        return guess

    assert_same_relaxation(
        jmk.relax(looped)(REFERENCE, LOWER, UPPER),
        jmk.relax(unrolled)(REFERENCE, LOWER, UPPER),
    )


def test_scan_promotes_carry_that_becomes_relaxed_later():
    """Checks a carry that only becomes relaxed through another carry."""

    def step(carry, _):
        first, second, third = carry
        return (first * 1.5, first + second, second - third), first

    def function(x: jax.Array) -> jax.Array:
        (first, second, third), stacked = lax.scan(
            step, (x[0], jnp.asarray(1.0), jnp.asarray(0.0)), length=3
        )
        return first + second + third + jnp.sum(stacked)

    relaxation = jmk.relax(function)(REFERENCE, LOWER, UPPER)
    value = function(REFERENCE)
    assert relaxation.convex <= value <= relaxation.concave
    assert relaxation.lower < relaxation.upper


def test_scan_with_plain_carry_and_relaxed_sequence():
    """Checks a `scan` whose carry stays plain while its outputs are relaxed."""

    def function(x: jax.Array) -> jax.Array:
        count, squares = lax.scan(lambda n, y: (n + 1, y * y), 0, x)
        return jnp.sum(squares) * count

    relaxation = jmk.relax(function)(REFERENCE, LOWER, UPPER)
    assert relaxation.convex <= function(REFERENCE) <= relaxation.concave


def test_custom_vjp_relaxes_primal():
    """Checks that a `custom_vjp` function is relaxed through its primal computation."""

    @jax.custom_vjp
    def softened(y: jax.Array) -> jax.Array:
        return jnp.exp(y) - y

    softened.defvjp(lambda y: (softened(y), y), lambda y, g: (g * (jnp.exp(y) - 1),))
    assert_same_relaxation(
        jmk.relax(lambda x: softened(x[0]))(REFERENCE, LOWER, UPPER),
        jmk.relax(lambda x: jnp.exp(x[0]) - x[0])(REFERENCE, LOWER, UPPER),
    )


def test_cond_with_traced_index():
    """Checks a `cond` whose branch index is an argument, which runs as `lax.switch`."""

    def function(x: jax.Array, index: jax.Array | int) -> jax.Array:
        branches = [lambda y: y[0] * y[1], lambda y: jnp.exp(y[0]), lambda y: 2.0]
        return lax.switch(index, branches, x)

    relaxed = jmk.relax(function)
    for index in range(3):
        relaxation = relaxed(REFERENCE, LOWER, UPPER, index)
        value = function(REFERENCE, index)
        assert relaxation.convex <= value <= relaxation.concave
    assert_same_relaxation(
        relaxed(REFERENCE, LOWER, UPPER, 1),
        jmk.relax(lambda x: jnp.exp(x[0]))(REFERENCE, LOWER, UPPER),
    )


def test_softplus_ignores_nan_check():
    """Checks that the NaN check inside `jax.nn.softplus` does not stop the relaxation."""
    relaxation = jmk.relax(lambda x: jax.nn.softplus(x[0]))(REFERENCE, LOWER, UPPER)
    assert relaxation.convex <= jax.nn.softplus(REFERENCE[0]) <= relaxation.concave


UNSUPPORTED = {
    "while": lambda x: lax.while_loop(lambda y: y < 3, lambda y: y * 2, x[1]),
    "reduce_max": lambda x: jnp.max(x),
    "comparison": lambda x: jnp.where(x[0] < x[1], x[0], x[1]),
    "equality": lambda x: jnp.where(x[0] == x[1], x[0], x[1]),
    "relaxed_predicate": lambda x: jnp.where(x[0].astype(bool), x[0], x[1]),
    "relaxed_branch_index": lambda x: lax.cond(
        x[0].astype(bool), lambda y: y, lambda y: -y, x[1]
    ),
    "relaxed_exponent": lambda x: x[0] ** x[1],
}
"""Functions whose relaxation is not supported, mapped to a name for the test ID."""


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unsupported_operation_raises(name):
    """Checks that unsupported operations raise `UnsupportedPrimitiveError`."""
    with pytest.raises(jmk.UnsupportedPrimitiveError):
        jmk.relax(UNSUPPORTED[name])(REFERENCE, LOWER, UPPER)


def test_exponent_unknown_while_tracing_raises():
    """Checks that `pow` with an exponent passed as an argument raises."""
    with pytest.raises(jmk.UnsupportedPrimitiveError):
        jmk.relax(lambda x, exponent: x[0] ** exponent)(REFERENCE, LOWER, UPPER, 1.5)
