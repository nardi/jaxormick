"""Intersections of relaxations of equivalent forms of a function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxormick as jmk

LOWER = jnp.array([0.0])
"""Lower bound of the box used in these tests."""

UPPER = jnp.array([1.0])
"""Upper bound of the box used in these tests."""


def product_form(x: jax.Array) -> jax.Array:
    """Returns `x (1 - x)` as a product."""
    return x[0] * (1 - x[0])


def square_form(x: jax.Array) -> jax.Array:
    """Returns `x (1 - x)` as a square subtracted from a constant."""
    return 0.25 - (x[0] - 0.5) ** 2


@pytest.mark.parametrize("reference", [0.1, 0.5, 0.9])
def test_intersection_is_valid_and_tighter(reference):
    """Checks that the intersection bounds the function and beats both forms."""
    point = jnp.array([reference])
    product = jmk.relax(product_form)(point, LOWER, UPPER)
    square = jmk.relax(square_form)(point, LOWER, UPPER)
    both = jmk.intersect_relaxations(product, square)
    value = product_form(point)
    tolerance = 1e-6
    assert both.lower <= both.convex <= value + tolerance
    assert value - tolerance <= both.concave <= both.upper
    assert both.convex >= max(product.convex, square.convex)
    assert both.concave <= min(product.concave, square.concave)
    assert both.lower >= max(product.lower, square.lower)
    assert both.upper <= min(product.upper, square.upper)


def test_intersection_of_trees():
    """Checks that relaxations in matching trees are intersected leaf by leaf."""

    def pair(x: jax.Array) -> dict[str, jax.Array]:
        return {"product": product_form(x), "square": square_form(x)}

    def swapped(x: jax.Array) -> dict[str, jax.Array]:
        return {"product": square_form(x), "square": product_form(x)}

    point = jnp.array([0.3])
    first = jmk.relax(pair)(point, LOWER, UPPER)
    second = jmk.relax(swapped)(point, LOWER, UPPER)
    both = jmk.intersect_relaxations(first, second)
    for left, right in zip(
        jax.tree.leaves(both["product"]), jax.tree.leaves(both["square"])
    ):
        np.testing.assert_allclose(left, right)


def test_intersection_derivative_is_subgradient():
    """Checks that the derivative of the intersected convex side is a subgradient."""

    def convex_side(point: jax.Array) -> jax.Array:
        return jmk.intersect_relaxations(
            jmk.relax(product_form)(point, LOWER, UPPER),
            jmk.relax(square_form)(point, LOWER, UPPER),
        ).convex

    reference = jnp.array([0.3])
    value, gradient = jax.value_and_grad(convex_side)(reference)
    for other in np.linspace(0.0, 1.0, 21):
        point = jnp.array([other])
        plane = value + gradient @ (point - reference)
        assert plane <= convex_side(point) + 1e-6


def test_intersection_of_nothing_raises():
    """Checks that an intersection needs at least one relaxation."""
    with pytest.raises(ValueError, match="at least one"):
        jmk.intersect_relaxations()
