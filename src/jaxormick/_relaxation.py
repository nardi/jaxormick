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
        relaxations stay valid and are never looser.
        """
        return type(self)(
            lower=self.lower,
            upper=self.upper,
            convex=jnp.maximum(self.convex, self.lower),
            concave=jnp.minimum(self.concave, self.upper),
        )
