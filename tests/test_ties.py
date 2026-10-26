"""Tests of the choice between two arguments that may tie, which keeps subgradients valid."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxormick._relaxation import larger, scale_tolerance, smaller, tie_tolerance

INFINITY = jnp.inf


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (-INFINITY, 5.0, 5.0),
        (5.0, -INFINITY, 5.0),
        (INFINITY, 5.0, INFINITY),
        (5.0, INFINITY, INFINITY),
        (INFINITY, INFINITY, INFINITY),
        (-INFINITY, -INFINITY, -INFINITY),
        (0.0, 3.0, 3.0),
    ],
)
def test_larger_with_infinite_arguments(
    first: float, second: float, expected: float
) -> None:
    """Checks that infinite arguments tie only with themselves."""
    result = larger(jnp.asarray(first), jnp.asarray(second))
    assert float(result) == expected


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (INFINITY, 5.0, 5.0),
        (5.0, INFINITY, 5.0),
        (-INFINITY, 5.0, -INFINITY),
        (5.0, -INFINITY, -INFINITY),
        (-INFINITY, -INFINITY, -INFINITY),
        (INFINITY, INFINITY, INFINITY),
        (4.0, 3.0, 3.0),
    ],
)
def test_smaller_with_infinite_arguments(
    first: float, second: float, expected: float
) -> None:
    """Checks that `smaller` mirrors `larger` at infinite arguments."""
    result = smaller(jnp.asarray(first), jnp.asarray(second))
    assert float(result) == expected


def test_a_near_tie_keeps_the_first_argument() -> None:
    """Checks that arguments a few ulps apart give the first one, whichever is larger."""
    first = jnp.asarray(1.0)
    nudged = jnp.asarray(1.0 + 4 * np.finfo(np.float64).eps)
    assert float(larger(first, nudged)) == 1.0
    assert float(smaller(nudged, first)) == float(nudged)
    assert float(larger(first, jnp.asarray(1.5))) == 1.5


def test_the_gradient_follows_the_first_argument_at_a_tie() -> None:
    """Checks that the gradient goes to one argument and not half to each."""
    gradient = jax.grad(lambda first, second: larger(first, second), argnums=(0, 1))
    assert tuple(float(entry) for entry in gradient(2.0, 2.0)) == (1.0, 0.0)


def test_tolerance_is_finite_and_scales_with_the_arguments() -> None:
    """Checks the tolerance near zero, for large arguments and for infinite ones."""
    limits = jnp.finfo(jnp.asarray(1.0).dtype)
    small = tie_tolerance(jnp.asarray(0.0), jnp.asarray(0.0))
    assert float(small) > 0.0
    large = tie_tolerance(jnp.asarray(1.0), jnp.asarray(2.0))
    assert float(large) > 16 * limits.eps
    assert float(tie_tolerance(jnp.asarray(INFINITY), jnp.asarray(1.0))) == limits.max
    with_scale = tie_tolerance(jnp.asarray(0.0), jnp.asarray(0.0), 1e6)
    assert float(with_scale) > float(large)
    assert float(scale_tolerance(jnp.asarray(-INFINITY))) == limits.max
    assert float(scale_tolerance(jnp.asarray(1.0))) == pytest.approx(
        float(tie_tolerance(jnp.asarray(1.0), jnp.asarray(1.0)))
    )
