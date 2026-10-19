from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


class Relaxation(eqx.Module):
    """Interval bounds and McCormick relaxations of an array over a box.

    All four fields have the shape of the relaxed value. `convex` and `concave` are the
    values of the relaxations at the reference point, so they depend on that point,
    while `lower` and `upper` depend only on the box.
    """

    lower: Array
    """Lower bound of the value over the box."""
    upper: Array
    """Upper bound of the value over the box."""
    convex: Array
    """Convex underestimator of the value, evaluated at the reference point."""
    concave: Array
    """Concave overestimator of the value, evaluated at the reference point."""

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the relaxed value."""
        return jnp.shape(self.lower)

    @property
    def dtype(self) -> jnp.dtype:
        """The dtype of the relaxed value."""
        return jnp.result_type(self.lower)

    @classmethod
    def constant(cls, value: ArrayLike) -> Self:
        """Returns the exact relaxation of a value that does not depend on the variables."""
        value = jnp.asarray(value)
        return cls(lower=value, upper=value, convex=value, concave=value)

    @classmethod
    def variable(cls, reference: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> Self:
        """Returns the relaxation of the variables themselves.

        A variable is linear in itself, so both relaxations equal the reference point.
        """
        reference = jnp.asarray(reference)
        return cls(
            lower=jnp.asarray(lower),
            upper=jnp.asarray(upper),
            convex=reference,
            concave=reference,
        )

    def tightened(self) -> Self:
        """Returns the relaxation with `convex` and `concave` clipped to the bounds.

        The maximum of a convex function and a constant is still convex, so the clipped
        relaxations stay valid. Near a tie, the bound wins, even if the side is a few
        ulps tighter. Its gradient of zero gives a valid plane even where rounding has
        pushed the side past the bound.
        """
        return type(self)(
            lower=self.lower,
            upper=self.upper,
            convex=larger(self.lower, self.convex),
            concave=smaller(self.upper, self.concave),
        )


TIE_TOLERANCE = 16
"""Ulps of the arguments within which `larger` and `smaller` keep the first one."""


def tie_tolerance(first: Array, second: Array) -> Array:
    """Returns a few ulps of the larger magnitude of the arguments, or zero if infinite."""
    epsilon = jnp.finfo(jnp.result_type(first, second)).eps
    tolerance = TIE_TOLERANCE * epsilon * jnp.maximum(jnp.abs(first), jnp.abs(second))
    return jnp.where(jnp.isfinite(tolerance), tolerance, 0.0)


def larger(first: Array, second: Array) -> Array:
    """Returns the elementwise maximum, differentiated along the argument it picks.

    Where the arguments tie, `jnp.maximum` splits the gradient between them. A plain
    choice with `jnp.where` does not fix this either. XLA may evaluate the comparison
    in several fused kernels with different rounding, so that the paths of the gradient
    pick different arguments, and their sum is no subgradient. This function keeps the
    first argument unless the second is larger by more than a few ulps. Near a tie,
    rounding then cannot flip the choice. The result may fall short of the maximum by
    these few ulps, which is still valid, since each argument is.
    """
    return jnp.where(first >= second - tie_tolerance(first, second), first, second)


def smaller(first: Array, second: Array) -> Array:
    """Returns the elementwise minimum, differentiated along the argument it picks.

    This mirrors `larger`.
    """
    return jnp.where(first <= second + tie_tolerance(first, second), first, second)
