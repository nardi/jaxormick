"""Rewrites of repeated equations and products of powers before relaxation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import jaxormick as jmk
from jaxormick._rewrite import EquationTable

LOWER = jnp.array([-1.0, -0.5])
"""Lower bounds of the box used in these tests, which contains zero."""

UPPER = jnp.array([1.0, 2.0])
"""Upper bounds of the box used in these tests."""

REFERENCE = jnp.array([0.2, 0.3])
"""Reference point inside the box."""

EQUIVALENT_FORMS = {
    "product of a slice with itself": (lambda x: x[0] * x[0], lambda x: x[0] ** 2),
    "product of three factors": (lambda x: x[1] * x[1] * x[1], lambda x: x[1] ** 3),
    "power of a power": (lambda x: (x[0] ** 2) ** 2, lambda x: x[0] ** 4),
    "product of powers": (lambda x: x[0] ** 2 * x[0] ** 3, lambda x: x[0] ** 5),
    "square times its base": (lambda x: jnp.square(x[1]) * x[1], lambda x: x[1] ** 3),
    "square of a power": (lambda x: jnp.square(x[0] ** 3), lambda x: x[0] ** 6),
    "inverse of a power": (lambda x: 1 / x[1] ** 2, lambda x: x[1] ** -2),
}
"""Pairs of a function and the single power that it is rewritten to."""


def assert_same_relaxation(first: jmk.Relaxation, second: jmk.Relaxation) -> None:
    """Checks that two relaxations agree in every field."""
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", sorted(EQUIVALENT_FORMS))
def test_rewritten_form_matches_power(name):
    """Checks that a product or composition is relaxed like the equivalent power."""
    function, power = EQUIVALENT_FORMS[name]
    assert_same_relaxation(
        jmk.relax(function)(REFERENCE, LOWER, UPPER),
        jmk.relax(power)(REFERENCE, LOWER, UPPER),
    )


def test_square_of_slice_has_exact_bounds():
    """Checks that `x[0] * x[0]` has a nonnegative lower bound on a box around zero."""
    lower, upper = jmk.interval(lambda x: x[0] * x[0])(LOWER, UPPER)
    assert lower == 0.0
    assert upper == 1.0


def test_product_of_different_slices_is_bilinear():
    """Checks that a product of different variables keeps the bilinear relaxation."""
    relaxation = jmk.relax(lambda x: x[0] * x[1])(REFERENCE, LOWER, UPPER)
    assert relaxation.lower == -2.0
    assert relaxation.upper == 2.0


def test_repeated_equations_inside_loop():
    """Checks that repeated equations in a loop body are merged as well."""

    def looped(x: jax.Array) -> jax.Array:
        return lax.fori_loop(0, 2, lambda i, total: total + x[0] * x[0], 0.0)

    lower, _ = jmk.interval(looped)(LOWER, UPPER)
    assert lower == 0.0


def test_product_with_constant_is_scaled():
    """Checks that a product with a literal constant is not taken for a power."""
    relaxation = jmk.relax(lambda x: x[0] * x[0] * 2.0)(REFERENCE, LOWER, UPPER)
    assert relaxation.lower == 0.0
    np.testing.assert_allclose(relaxation.convex, 2 * 0.2**2, rtol=1e-6)


def test_effectful_equations_are_not_merged():
    """Checks that two equal debug prints have no key, so both run."""

    def function(x: jax.Array) -> None:
        jax.debug.print("{}", x[0])
        jax.debug.print("{}", x[0])

    table = EquationTable()
    printed = [
        equation
        for equation in jax.make_jaxpr(function)(REFERENCE).eqns
        if equation.effects
    ]
    assert len(printed) == 2
    for equation in printed:
        assert table.key(equation) is None
        table.record(equation, equation)
        assert table.earlier_outputs(equation) is None


def test_unhashable_parameters_are_not_merged():
    """Checks that an equation whose parameters are not hashable has no key."""
    (equation,) = jax.make_jaxpr(lambda x: x**2)(1.0).eqns
    assert EquationTable().key(equation.replace(params={"y": [2]})) is None
